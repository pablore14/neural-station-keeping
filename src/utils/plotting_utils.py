"""
Publication-quality plotting helpers shared across subsystems.

Centralizes figure styling so every plot in the project has a consistent,
paper-ready look. Change a value here (font size, DPI, palette) and every
figure updates. Import set_paper_style() once at the top of a plotting run.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Single source of truth for output resolution and palette.
FIGURE_DPI = 300                         # 300 dpi is the usual journal minimum

# A restrained, colour-blind-friendly palette (Okabe-Ito subset).
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "grey":   "#999999",
    "black":  "#000000",
}

# Semantic colours reused across regressor plots.
COLOR_ACTIVE   = PALETTE["red"]
COLOR_INACTIVE = PALETTE["blue"]
COLOR_TRAIN    = PALETTE["blue"]
COLOR_VAL      = PALETTE["orange"]
COLOR_TEST     = PALETTE["green"]

IDENTITY_STYLE = dict(color=PALETTE["grey"], linestyle="--", linewidth=1.2, zorder=1)


def set_paper_style():
    """Apply global matplotlib settings for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi":        110,          # on-screen; savefig uses FIGURE_DPI
        "savefig.dpi":       FIGURE_DPI,
        "savefig.bbox":      "tight",
        "font.family":       "serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "axes.spines.top":   False,        # cleaner frame: drop top/right spines
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.6,
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "lines.linewidth":   1.8,
    })


def save_figure(fig, out_dir, filename):
    """Save a figure to out_dir as both PNG (preview) and PDF (vector, for LaTeX).

    The PDF is resolution-independent, so it can be zoomed in the paper without
    pixelation. `filename` may be given with or without a .png extension.
    """
    base = os.path.splitext(filename)[0]   # strip any extension
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{base}.{ext}"),
                    dpi=200, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)


def add_identity_line(ax, lim):
    """Draw the y = x reference diagonal and set square [-lim, lim] limits."""
    ax.plot([-lim, lim], [-lim, lim], **IDENTITY_STYLE)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")