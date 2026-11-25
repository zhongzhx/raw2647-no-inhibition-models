"""Small fail-closed helpers for the current rebuild launch audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateRecorder:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        gate: int,
        name: str,
        passed: bool,
        evidence: Any,
        *,
        required: bool = True,
    ) -> None:
        self.rows.append(
            {
                "gate": int(gate),
                "name": str(name),
                "required": bool(required),
                "passed": bool(passed),
                "evidence": evidence,
            }
        )

    @property
    def all_required_passed(self) -> bool:
        return all(row["passed"] for row in self.rows if row["required"])

    @property
    def failed_required_gates(self) -> list[int]:
        return sorted(
            {
                int(row["gate"])
                for row in self.rows
                if row["required"] and not row["passed"]
            }
        )
