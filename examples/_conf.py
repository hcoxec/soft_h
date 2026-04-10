import matplotlib.pyplot as plt

BLUE     = "#00f5ff"   # neon cyan
GREEN    = "#39ff14"   # neon green
RED      = "#ff2d55"   # neon pink-red
ORANGE   = "#ff9500"   # neon amber
PINK     = "#da00ff"   # neon magenta
OLIVE    = "#aaff00"   # electric lime
BURNT    = "#ff6b00"   # hot orange
LAVENDER = "#7b61ff"   # electric violet
PALETTE  = [BLUE, GREEN, ORANGE, RED, PINK, LAVENDER, OLIVE, BURNT, "#555577", "#8888aa"]

BG       = "#0a0a1a"
PANEL_BG = "#0f0f2a"
GRAY_TEXT  = "#8888bb"
GRAY_LIGHT = "#1a1a3a"
GRAY_MID   = "#2a2a4a"
GRID_COLOR = "#1e1e3a"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "text.color":        GRAY_TEXT,
    "font.family":       "monospace",
    "font.size":         10,
    "figure.dpi":        130,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "axes.edgecolor":    GRAY_MID,
    "xtick.color":       GRAY_TEXT,
    "ytick.color":       GRAY_TEXT,
})


def apply_style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY_MID)
    ax.tick_params(length=0, colors=GRAY_TEXT)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY_TEXT, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY_TEXT, labelpad=6)
    if title:
        ax.set_title(title, color=BLUE, fontsize=11, pad=10, loc="left")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(GRAY_TEXT)
