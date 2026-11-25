#!/usr/bin/env python3
"""Shared final-pass styling for the manuscript's publication figures."""

from __future__ import annotations

from matplotlib import rcParams
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation, Text


FONT_FAMILY = "Times New Roman"
LINE_WIDTH_PT = 1.0


def configure_publication_style() -> None:
    """Set defaults that apply before artists are created."""
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": LINE_WIDTH_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "patch.linewidth": LINE_WIDTH_PT,
            "grid.linewidth": LINE_WIDTH_PT,
            "xtick.major.width": LINE_WIDTH_PT,
            "ytick.major.width": LINE_WIDTH_PT,
            "xtick.minor.width": LINE_WIDTH_PT,
            "ytick.minor.width": LINE_WIDTH_PT,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig) -> None:
    """Enforce typography, line widths, and transparent canvas before saving."""
    configure_publication_style()
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    for axis in fig.axes:
        axis.set_facecolor("none")
        axis.patch.set_facecolor("none")
        axis.patch.set_alpha(0)
        axis.tick_params(which="both", width=LINE_WIDTH_PT)
        for spine in axis.spines.values():
            spine.set_linewidth(LINE_WIDTH_PT)

    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontweight("bold")
        bbox_patch = text.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_linewidth(LINE_WIDTH_PT)
        if isinstance(text, Annotation) and text.arrow_patch is not None:
            text.arrow_patch.set_linewidth(LINE_WIDTH_PT)

    for line in fig.findobj(match=Line2D):
        line.set_linewidth(LINE_WIDTH_PT)
        line.set_markeredgewidth(LINE_WIDTH_PT)

    for patch in fig.findobj(match=Patch):
        patch.set_linewidth(LINE_WIDTH_PT)

    for collection in fig.findobj(match=Collection):
        try:
            collection.set_linewidth(LINE_WIDTH_PT)
        except (AttributeError, TypeError, ValueError):
            pass


def transparent_savefig(fig, path, **kwargs) -> None:
    """Save a normalized figure with a transparent page/canvas."""
    finalize_figure(fig)
    kwargs.update({"transparent": True, "facecolor": "none", "edgecolor": "none"})
    fig.savefig(path, **kwargs)
