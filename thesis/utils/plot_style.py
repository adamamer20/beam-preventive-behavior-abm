"""Shared Matplotlib styling for thesis figures."""

from __future__ import annotations

import matplotlib as mpl

THESIS_PLOT_FONT_SIZE_PT = 12.0


def apply_thesis_matplotlib_style() -> None:
    """Align default Matplotlib figure text with the thesis 12pt body text."""

    mpl.rcParams.update(
        {
            "font.size": THESIS_PLOT_FONT_SIZE_PT,
            "axes.titlesize": THESIS_PLOT_FONT_SIZE_PT,
            "axes.labelsize": THESIS_PLOT_FONT_SIZE_PT,
            "xtick.labelsize": THESIS_PLOT_FONT_SIZE_PT,
            "ytick.labelsize": THESIS_PLOT_FONT_SIZE_PT,
            "legend.fontsize": THESIS_PLOT_FONT_SIZE_PT,
            "legend.title_fontsize": THESIS_PLOT_FONT_SIZE_PT,
            "figure.titlesize": THESIS_PLOT_FONT_SIZE_PT,
        }
    )
