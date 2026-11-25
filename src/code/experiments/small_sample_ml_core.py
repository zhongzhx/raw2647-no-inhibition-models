"""Fold-safe utilities for the RAW264.7 small-sample ML evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_path(path: str | Path, forbidden_tokens: Sequence[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = str(resolved).lower()
    if any(str(token).lower() in lowered for token in forbidden_tokens):
        raise RuntimeError(f"Forbidden locked/exposed test path: {resolved}")
    return resolved


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def graph_feature_matrices(graphs: Sequence[Any]) -> dict[str, np.ndarray]:
    def matrix(attribute: str, dtype: Any) -> np.ndarray:
        values = np.concatenate(
            [getattr(graph, attribute).detach().cpu().reshape(1, -1).numpy() for graph in graphs], axis=0
        ).astype(dtype, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {attribute}")
        return values

    result = {
        "rdkit": matrix("global_feat", np.float32),
        "morgan": matrix("morgan_fp", np.float32),
        "maccs": matrix("maccs_fp", np.float32),
    }
    expected = {"rdkit": 171, "morgan": 2048, "maccs": 167}
    for name, dimension in expected.items():
        if result[name].shape[1] != dimension:
            raise ValueError(f"{name} dimension mismatch: {result[name].shape}")
    return result


def assemble_feature_set(
    components: Sequence[str],
    indices: np.ndarray,
    base: dict[str, np.ndarray],
    representation: dict[str, np.ndarray],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for component in components:
        source = base if component in base else representation
        if component not in source:
            raise KeyError(f"Missing feature component: {component}")
        values.append(np.asarray(source[component][indices], dtype=np.float32))
    output = np.concatenate(values, axis=1)
    if output.shape[0] != len(indices) or not np.isfinite(output).all():
        raise RuntimeError("Assembled feature matrix is invalid")
    return output


class FoldSafePreprocessor(BaseEstimator, TransformerMixin):
    """Training-fold-only imputation, filtering, scaling and reduction."""

    def __init__(
        self,
        reducer: str = "none",
        scale: bool = False,
        variance_threshold: float = 1e-8,
        correlation_threshold: float = 0.95,
        random_state: int = 13,
    ) -> None:
        self.reducer = reducer
        self.scale = scale
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

    @staticmethod
    def _finite(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        output[~np.isfinite(output)] = np.nan
        return output

    def fit(self, values: np.ndarray, labels: np.ndarray | None = None) -> "FoldSafePreprocessor":
        matrix = self._finite(values)
        self.fit_sample_count_ = int(matrix.shape[0])
        self.input_dim_ = int(matrix.shape[1])
        with np.errstate(all="ignore"):
            median = np.nanmedian(matrix, axis=0)
        median[~np.isfinite(median)] = 0.0
        self.median_ = median
        matrix = np.where(np.isnan(matrix), median, matrix)

        variance = matrix.var(axis=0)
        kept = np.flatnonzero(np.isfinite(variance) & (variance > float(self.variance_threshold)))
        if not len(kept):
            raise ValueError("All features removed by variance filter")
        matrix = matrix[:, kept]
        self.variance_columns_ = kept

        # Exact duplicate feature columns are removed using their byte identity.
        seen: dict[bytes, int] = {}
        unique_local: list[int] = []
        contiguous = np.ascontiguousarray(matrix.T)
        for index, column in enumerate(contiguous):
            key = hashlib.blake2b(column.tobytes(), digest_size=16).digest()
            if key not in seen or not np.array_equal(column, contiguous[seen[key]]):
                seen[key] = index
                unique_local.append(index)
        self.unique_columns_ = np.asarray(unique_local, dtype=int)
        matrix = matrix[:, self.unique_columns_]

        self.correlation_columns_ = np.arange(matrix.shape[1], dtype=int)
        if self.reducer == "corr" and matrix.shape[1] > 1:
            correlation = np.corrcoef(matrix, rowvar=False)
            correlation = np.nan_to_num(np.abs(correlation), nan=0.0)
            keep = np.ones(matrix.shape[1], dtype=bool)
            for right in range(1, matrix.shape[1]):
                if np.any(correlation[:right, right][keep[:right]] > float(self.correlation_threshold)):
                    keep[right] = False
            self.correlation_columns_ = np.flatnonzero(keep)
            matrix = matrix[:, self.correlation_columns_]

        self.mi_columns_ = np.arange(matrix.shape[1], dtype=int)
        if self.reducer.startswith("mi"):
            if labels is None:
                raise ValueError("Mutual-information selection requires labels")
            requested = int(self.reducer.removeprefix("mi"))
            score = mutual_info_classif(matrix, np.asarray(labels), random_state=self.random_state)
            count = min(requested, matrix.shape[1])
            self.mi_columns_ = np.sort(np.argsort(score, kind="stable")[-count:])
            matrix = matrix[:, self.mi_columns_]

        needs_scaling = bool(self.scale or self.reducer.startswith("pca"))
        self.scaler_ = StandardScaler().fit(matrix) if needs_scaling else None
        scaled = self.scaler_.transform(matrix) if self.scaler_ is not None else matrix

        self.pca_ = None
        if self.reducer.startswith("pca"):
            suffix = self.reducer.removeprefix("pca")
            if suffix == "95":
                components: int | float = 0.95
            else:
                components = min(int(suffix), scaled.shape[1], max(scaled.shape[0] - 1, 1))
            self.pca_ = PCA(n_components=components, svd_solver="full", random_state=self.random_state).fit(scaled)
            scaled = self.pca_.transform(scaled)
        self.output_dim_ = int(scaled.shape[1])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = self._finite(values)
        matrix = np.where(np.isnan(matrix), self.median_, matrix)
        matrix = matrix[:, self.variance_columns_]
        matrix = matrix[:, self.unique_columns_]
        matrix = matrix[:, self.correlation_columns_]
        matrix = matrix[:, self.mi_columns_]
        if self.scaler_ is not None:
            matrix = self.scaler_.transform(matrix)
        if self.pca_ is not None:
            matrix = self.pca_.transform(matrix)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Preprocessor produced non-finite values")
        return np.asarray(matrix, dtype=np.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "fit_sample_count": int(self.fit_sample_count_),
            "input_dim": int(self.input_dim_),
            "post_variance_dim": int(len(self.variance_columns_)),
            "post_duplicate_dim": int(len(self.unique_columns_)),
            "post_correlation_dim": int(len(self.correlation_columns_)),
            "post_mi_dim": int(len(self.mi_columns_)),
            "output_dim": int(self.output_dim_),
            "scaled": bool(self.scaler_ is not None),
            "pca": bool(self.pca_ is not None),
            "reducer": self.reducer,
        }


def reducer_options(feature_set: str, model_name: str) -> list[str]:
    if model_name == "Gaussian Process":
        return ["pca16", "pca32", "pca64", "pca128", "pca95"]
    if feature_set == "F1_RDKit":
        return ["none", "corr", "mi32", "mi64", "mi128", "pca32", "pca64", "pca128", "pca95"]
    if feature_set in {"F2_Morgan", "F3_MACCS"}:
        return ["none", "mi64", "mi128"]
    if feature_set in {"F5_FrozenGINE", "F6_Adapter", "F7_GatedFusion"}:
        return ["none", "mi64", "mi128", "pca32", "pca64", "pca128", "pca95"]
    return ["mi64", "mi128", "pca32", "pca64", "pca128", "pca95"]


def deterministic_candidates(
    model_name: str,
    feature_set: str,
    search_space: dict[str, Sequence[Any]],
    trial_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    # ``pca`` is represented by the fold-safe reducer below.  Keeping it in
    # the estimator parameter grid as well would both duplicate candidates and
    # accidentally pass an unsupported argument to GaussianProcessClassifier.
    effective_space = dict(search_space)
    effective_space.pop("pca", None)
    keys = list(effective_space)
    parameter_values = [list(effective_space[key]) for key in keys]
    reducers = reducer_options(feature_set, model_name)
    all_candidates = [
        {**dict(zip(keys, values)), "reducer": reducer}
        for values in itertools.product(*parameter_values)
        for reducer in reducers
    ]
    rng = np.random.default_rng(seed)
    if len(all_candidates) <= trial_count:
        selected = np.arange(len(all_candidates))
    else:
        selected = np.sort(rng.choice(len(all_candidates), size=trial_count, replace=False))
    candidates = [all_candidates[int(index)] for index in selected]
    # Always include a simple, stable candidate from the allowed range.
    simple = {key: list(effective_space[key])[0] for key in keys}
    simple["reducer"] = reducers[0]
    candidates[0] = simple
    return candidates


def model_requires_scaling(model_name: str, reducer: str) -> bool:
    return model_name in {"RBF-SVM", "Elastic Net Logistic Regression", "Gaussian Process"} or reducer.startswith("pca")


def build_estimator(model_name: str, params: dict[str, Any], seed: int) -> Any:
    values = {key: value for key, value in params.items() if key != "reducer"}
    if model_name == "Random Forest":
        return RandomForestClassifier(**values, random_state=seed, n_jobs=-1)
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(**values, random_state=seed, n_jobs=-1)
    if model_name == "RBF-SVM":
        return SVC(**values, kernel="rbf", probability=True, random_state=seed, cache_size=1024)
    if model_name == "Elastic Net Logistic Regression":
        return LogisticRegression(
            **values,
            penalty="elasticnet",
            max_iter=10000,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "Gaussian Process":
        kernel_name = str(values["kernel"])
        if kernel_name.startswith("RBF"):
            kernel: Any = RBF(length_scale=1.0)
        elif kernel_name.startswith("Matern32"):
            kernel = Matern(length_scale=1.0, nu=1.5)
        else:
            kernel = Matern(length_scale=1.0, nu=2.5)
        if "+White" in kernel_name:
            kernel = kernel + WhiteKernel(noise_level=1e-3)
        return GaussianProcessClassifier(
            kernel=kernel,
            random_state=seed,
            n_restarts_optimizer=0,
            max_iter_predict=100,
            warm_start=False,
        )
    raise ValueError(model_name)


def model_complexity_score(model_name: str, params: dict[str, Any]) -> float:
    reducer = str(params.get("reducer", "none"))
    reduction = 0.0 if reducer == "none" else 0.1
    if model_name in {"Random Forest", "ExtraTrees"}:
        depth = 50 if params["max_depth"] is None else int(params["max_depth"])
        return float(params["n_estimators"]) / 1000 + depth / 100 + 1 / float(params["min_samples_leaf"]) + reduction
    if model_name == "XGBoost":
        return float(params["n_estimators"]) / 1000 + float(params["max_depth"]) / 10 + reduction
    if model_name == "RBF-SVM":
        return math.log10(float(params["C"]) + 1e-12) + 5 + reduction
    if model_name == "Elastic Net Logistic Regression":
        return math.log10(float(params["C"]) + 1e-12) + 5 + (1 - float(params["l1_ratio"])) + reduction
    if model_name == "Gaussian Process":
        return 1 + reduction
    return reduction


@dataclass
class FitResult:
    probability: np.ndarray
    preprocessor_audit: dict[str, Any]
    runtime_seconds: float
    peak_memory_bytes: int
    complexity: float
    log_marginal_likelihood: float


def fit_predict(
    model_name: str,
    params: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    seed: int,
    xgboost_runner: Any | None = None,
    force_scale: bool = False,
) -> FitResult:
    started = time.perf_counter()
    reducer = str(params.get("reducer", "none"))
    preprocessor = FoldSafePreprocessor(
        reducer=reducer,
        scale=bool(force_scale or model_requires_scaling(model_name, reducer)),
        random_state=seed,
    ).fit(train_x, train_y)
    transformed_train = preprocessor.transform(train_x)
    transformed_valid = preprocessor.transform(valid_x)
    log_marginal = float("nan")
    if model_name == "XGBoost":
        if xgboost_runner is None:
            raise RuntimeError("XGBoost requires the isolated worker")
        model_params = {key: value for key, value in params.items() if key != "reducer"}
        probability = xgboost_runner(
            "predict", transformed_train, train_y, transformed_valid, [model_params], seed
        )
        complexity = float(model_params["n_estimators"] * (2 ** (int(model_params["max_depth"]) + 1) - 1))
    else:
        estimator = build_estimator(model_name, params, seed)
        estimator.fit(transformed_train, train_y)
        probability = estimator.predict_proba(transformed_valid)[:, 1]
        if model_name in {"Random Forest", "ExtraTrees"}:
            complexity = float(sum(tree.tree_.node_count for tree in estimator.estimators_))
        elif model_name == "RBF-SVM":
            complexity = float(estimator.n_support_.sum())
        elif model_name == "Elastic Net Logistic Regression":
            complexity = float(np.count_nonzero(estimator.coef_))
        else:
            complexity = float(len(train_y))
            log_marginal = float(estimator.log_marginal_likelihood_value_)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if probability.shape != (len(valid_x),) or not np.isfinite(probability).all():
        raise RuntimeError(f"Invalid {model_name} probabilities")
    return FitResult(
        probability=probability,
        preprocessor_audit=preprocessor.audit(),
        runtime_seconds=time.perf_counter() - started,
        peak_memory_bytes=peak_rss_bytes(),
        complexity=complexity,
        log_marginal_likelihood=log_marginal,
    )


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            mask = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return value


def enrichment_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    order = np.argsort(-probability, kind="stable")
    prevalence = max(float(labels.mean()), 1e-12)
    output: dict[str, float] = {}
    for fraction, suffix in ((0.05, "5pct"), (0.10, "10pct")):
        count = max(1, int(math.ceil(len(labels) * fraction)))
        selected = labels[order[:count]]
        precision = float(selected.mean())
        output[f"precision_at_{suffix}"] = precision
        output[f"ef_at_{suffix}"] = precision / prevalence
    for count in (10, 20):
        output[f"active_at_{count}"] = float(labels[order[: min(count, len(labels))]].sum())
    return output


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "pr_auc": float(average_precision_score(labels, clipped)),
        "roc_auc": float(roc_auc_score(labels, clipped)),
        "brier": float(brier_score_loss(labels, clipped)),
        "ece": expected_calibration_error(labels, clipped),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        **enrichment_metrics(labels, clipped),
    }


def threshold_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (np.asarray(probability) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
    }


def choose_thresholds(
    labels: np.ndarray,
    probability: np.ndarray,
    minimum: float = 0.10,
    maximum: float = 0.90,
    step: float = 0.01,
) -> dict[str, float]:
    thresholds = np.round(np.arange(minimum, maximum + step / 2, step), 10)
    rows = [{**threshold_metrics(labels, probability, value)} for value in thresholds]
    frame = pd.DataFrame(rows)
    youden = frame["sensitivity"] + frame["specificity"] - 1.0
    objectives = {
        "balanced_accuracy": frame["balanced_accuracy"],
        "youden_j": youden,
        "mcc": frame["mcc"],
        "f1": frame["f1"],
    }
    selected: dict[str, float] = {}
    for name, values in objectives.items():
        best = float(values.max())
        candidates = frame.loc[np.isclose(values, best), "threshold"]
        selected[name] = float(candidates.iloc[np.argmin(np.abs(candidates.to_numpy() - 0.5))])
    return selected


class ProbabilityCalibrator:
    def __init__(self, method: str) -> None:
        self.method = method
        self.model: Any | None = None

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "none":
            self.model = None
        elif self.method == "platt":
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(logits, labels)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(clipped, labels)
        else:
            raise ValueError(self.method)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "none":
            output = clipped
        elif self.method == "platt":
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            output = self.model.predict_proba(logits)[:, 1]
        else:
            output = self.model.predict(clipped)
        return np.clip(np.asarray(output, dtype=float), 0.0, 1.0)


def select_calibration(
    labels: np.ndarray,
    probability: np.ndarray,
    methods: Sequence[str],
    tolerance: float = 0.002,
    fold_ids: np.ndarray | None = None,
) -> tuple[str, ProbabilityCalibrator, list[dict[str, Any]]]:
    rows = []
    fitted: dict[str, ProbabilityCalibrator] = {}
    selection_predictions: dict[str, np.ndarray] = {}
    for method in methods:
        calibrator = ProbabilityCalibrator(method).fit(probability, labels)
        fitted[method] = calibrator
        if fold_ids is None:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=130013)
            splits = list(splitter.split(np.zeros((len(labels), 1)), labels))
        else:
            ids = np.asarray(fold_ids, dtype=int)
            if ids.shape != np.asarray(labels).shape:
                raise ValueError("Calibration fold IDs do not match labels")
            splits = [(np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)) for fold in np.unique(ids)]
        calibrated = np.full(len(labels), np.nan, dtype=float)
        for train, valid in splits:
            local = ProbabilityCalibrator(method).fit(np.asarray(probability)[train], np.asarray(labels)[train])
            calibrated[valid] = local.predict(np.asarray(probability)[valid])
        if not np.isfinite(calibrated).all():
            raise RuntimeError(f"Incomplete cross-fitted {method} calibration")
        selection_predictions[method] = calibrated
        rows.append({"calibration": method, "selection_predictions_cross_fitted": True, **probability_metrics(labels, calibrated)})
    best_brier = min(row["brier"] for row in rows)
    preference = {"none": 0, "platt": 1, "isotonic": 2}
    eligible = [row for row in rows if row["brier"] <= best_brier + tolerance]
    selected = min(eligible, key=lambda row: preference[row["calibration"]])["calibration"]
    chosen = fitted[str(selected)]
    chosen.selection_probability_ = selection_predictions[str(selected)]
    return str(selected), chosen, rows


def select_candidate(
    search_rows: Sequence[dict[str, Any]],
    tie_tolerance: float,
) -> dict[str, Any]:
    best_mean = max(float(row["inner_mean_pr_auc"]) for row in search_rows)
    eligible = [row for row in search_rows if float(row["inner_mean_pr_auc"]) >= best_mean - tie_tolerance]
    return min(
        eligible,
        key=lambda row: (
            float(row["complexity_score"]),
            float(row["inner_pr_auc_sd"]),
            float(row["inner_mean_brier"]),
            float(row["runtime_seconds"]),
            int(row["candidate_index"]),
        ),
    )


def max_morgan_similarity(
    morgan: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> np.ndarray:
    train = sparse.csr_matrix(morgan[train_indices].astype(np.uint8))
    valid = sparse.csr_matrix(morgan[valid_indices].astype(np.uint8))
    intersection = valid.dot(train.T).toarray().astype(np.float64)
    valid_sum = np.asarray(valid.sum(axis=1)).reshape(-1, 1)
    train_sum = np.asarray(train.sum(axis=1)).reshape(1, -1)
    union = valid_sum + train_sum - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return similarity.max(axis=1)


def similarity_bin(values: np.ndarray) -> np.ndarray:
    return np.asarray(pd.cut(
        values,
        bins=[-np.inf, 0.30, 0.50, 0.70, np.inf],
        right=False,
        labels=["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"],
    ).astype(str))


def validate_outer_isolation(
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    inchikeys: np.ndarray,
    scaffolds: np.ndarray,
) -> dict[str, Any]:
    """Fail closed when an outer split leaks identity or scaffold groups."""
    train = np.asarray(train_indices, dtype=int)
    valid = np.asarray(valid_indices, dtype=int)
    index_overlap = np.intersect1d(train, valid)
    inchikey_overlap = set(inchikeys[train].astype(str)) & set(inchikeys[valid].astype(str))
    scaffold_overlap = set(scaffolds[train].astype(str)) & set(scaffolds[valid].astype(str))
    audit = {
        "train_count": int(len(train)),
        "valid_count": int(len(valid)),
        "index_overlap": int(len(index_overlap)),
        "inchikey_overlap": int(len(inchikey_overlap)),
        "scaffold_overlap": int(len(scaffold_overlap)),
    }
    if any(audit[key] for key in ("index_overlap", "inchikey_overlap", "scaffold_overlap")):
        raise RuntimeError(f"Outer-fold leakage: {audit}")
    return audit


def assert_inner_only(
    fitting_indices: np.ndarray,
    outer_validation_indices: np.ndarray,
    operation: str,
) -> dict[str, Any]:
    """Audit guard shared by threshold and calibration selection."""
    fitted = np.asarray(fitting_indices, dtype=int)
    outer = np.asarray(outer_validation_indices, dtype=int)
    overlap = np.intersect1d(fitted, outer)
    if len(overlap):
        raise RuntimeError(f"{operation} used outer-validation samples")
    return {
        "operation": operation,
        "fit_sample_count": int(len(fitted)),
        "fit_index_sha256": stable_hash(sorted(int(value) for value in fitted)),
        "outer_overlap": 0,
    }


def validate_oof_uniqueness(frame: pd.DataFrame) -> None:
    required = {"model", "feature_set", "seed", "development_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF columns missing: {sorted(required - set(frame.columns))}")
    if bool(frame.duplicated(list(required)).any()):
        raise RuntimeError("Duplicate model-feature-seed-sample OOF prediction")


def paired_cluster_bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    reference_threshold: float,
    candidate_threshold: float,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    positions = {str(group): np.flatnonzero(groups == group) for group in np.unique(groups.astype(str))}
    clusters = np.asarray(sorted(positions), dtype=object)
    rng = np.random.default_rng(seed)

    def values(index: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
        selected_labels = labels[index]
        selected_probability = probability[index]
        predicted = selected_probability >= threshold
        positive = selected_labels == 1
        negative = ~positive
        return {
            "pr_auc": float(average_precision_score(selected_labels, selected_probability)),
            "roc_auc": float(roc_auc_score(selected_labels, selected_probability)),
            "brier": float(np.mean((selected_probability - selected_labels) ** 2)),
            "accuracy": float(np.mean(predicted == selected_labels)),
            "balanced_accuracy": float(
                0.5 * (np.mean(predicted[positive]) + np.mean(~predicted[negative]))
            ),
        }

    observed_reference = values(np.arange(len(labels)), reference_probability, reference_threshold)
    observed_candidate = values(np.arange(len(labels)), candidate_probability, candidate_threshold)
    samples = {metric: [] for metric in observed_reference}
    for _ in range(repetitions):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        index = np.concatenate([positions[str(group)] for group in selected])
        if len(np.unique(labels[index])) < 2:
            continue
        reference = values(index, reference_probability, reference_threshold)
        candidate = values(index, candidate_probability, candidate_threshold)
        for metric in samples:
            samples[metric].append(candidate[metric] - reference[metric])
    rows = []
    for metric, distribution in samples.items():
        delta = observed_candidate[metric] - observed_reference[metric]
        low, high = np.quantile(distribution, [0.025, 0.975]) if distribution else (float("nan"), float("nan"))
        rows.append(
            {
                "metric": metric,
                "delta": float(delta),
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "valid_repetitions": int(len(distribution)),
            }
        )
    return rows


def entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
