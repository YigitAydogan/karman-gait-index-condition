# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 13: Publication figures and tables from the saved outputs of
#            Scripts 01 to 12.
#
# Colab ready. Needs numpy, matplotlib and JAX (only Figure 6c, a short
# simulation past a torus bifurcation). Runtime about 2 to 5 min.
#
# Nothing in the analysis is recomputed. Each figure and table is made
# independently; a missing input skips that item with a message.
#
# Division of labour (no information appears in both a figure and a table):
#   Figures carry shapes, structure and distributions:
#     1  model schematic; wake profiles U_loc(x), W(x)
#     2  passive bodies: force landscapes; stability vs vortex peak (95% CI);
#        time to loss vs growth time
#     3  fixed position: rotation-number map; devil's staircases; phase slips
#     4  free position: holding branches as closed loops by fixed-position branch
#     5  index condition: a_L vs det(I - M_int), Scripts 08, 09, 10
#     6  bifurcations: two-parameter holding regions; surging past a torus point
#     7  predictions: holding probability map; tail-beat phase distributions;
#        holding probability along the flow-speed path
#   Tables carry counts and summary numbers:
#     T1 baseline parameters and trout scaling
#     T2 index-condition verification (all scripts)
#     T3 model-structure robustness (Script 09)
#     T4 body-tuning window (Script 06)
#     T5 continuation: stable areas and bifurcation counts (Script 10)
#     T6 bout durations under noise (Script 11)
#     T7 lateral-line removal (Script 11)
#     T8 numerical convergence (Script 12)
#   Version 2 (after the referee-style review): Figure 6b carries a zoomed
#   inset of the narrow stable band; Figure 1c is regenerated with legend
#   entries that state the direction of each velocity and, when the final
#   hand-assembled Figure 1 is present as script13_figures/Fig1_source.png,
#   pasted into it in place of the old panel (c).
#   Tables are written as CSV, LaTeX (booktabs) and one Markdown file for
#   review. Captions with the numbers each figure needs are written to
#   figure_captions.md.
#
# Figure rules: no panel titles; all four spines; text at least 9 pt at
# print size; every axis and colour bar shows labelled lower and upper
# bounds that are multiples of its tick step; confidence intervals stay
# inside the bounds.
# =============================================================================

import os, sys, json, time, math, traceback, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Ellipse, Polygon, Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, NullFormatter

warnings.filterwarnings("ignore")

MODE = "FULL"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script13_figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TAB_DIR, exist_ok=True)
Q = "" if MODE == "FULL" else "_quick"
CK = f"checkpoints_{MODE.lower()}"

def P(*parts):
    return os.path.join(BASE_DIR, *parts)

def jload(*parts):
    p_ = P(*parts)
    if not os.path.exists(p_):
        raise FileNotFoundError(p_)
    return json.load(open(p_))

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
W1, W2 = 3.37, 6.69
plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "lines.linewidth": 1.3,
    "axes.spines.top": True, "axes.spines.right": True, "xtick.top": False, "ytick.right": False,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "#252525", "legend.framealpha": 1.0,
    "legend.borderpad": 0.45, "patch.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
C_NODE, C_SAD, C_NONE = "#2166ac", "#b2182b", "#e08214"
C_ST, C_UN, C_INK, C_MUTE = "#1a9850", "#bdbdbd", "#252525", "#636363"
BRANCH_COL = {"node": C_NODE, "saddle-real": C_SAD, "none": C_NONE}
BRANCH_LAB = {"node": "node branch", "saddle-real": "saddle branch", "none": "no fixed lock"}

PANELS = []

def panel(ax, s, x=None, y=None):
    """Register a panel letter. Without x, y it is placed by align(): centred on the
    y-axis title (or the tick labels), just above the axes, and aligned per column."""
    PANELS.append((ax.figure, ax, s, None if x is None else (x, y)))

def align(fig):
    """Put each panel letter directly above its y-axis title and align both within
    every column of panels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wp, Hp = fig.get_size_inches() * fig.dpi
    pad = 5 / 72 * fig.dpi
    items = [p_ for p_ in PANELS if p_[0] is fig]
    info = {}
    for _, ax, s_, man in items:
        if man is not None:
            fig.text(*ax.transAxes.transform(man) / [Wp, Hp], s_, fontsize=12, fontweight="bold", va="bottom", ha="left")
            continue
        bb = ax.get_window_extent(r)
        tl = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_text()]
        left = min([b.x0 for b in tl], default=bb.x0 - pad)
        lab = ax.yaxis.label
        lw = lab.get_window_extent(r).width if lab.get_text() else 0.0
        info[ax] = dict(bb=bb, xc=left - pad - lw / 2, lw=lw, s=s_)
    cols = {}
    for ax, v in info.items():
        cols.setdefault(round(v["bb"].x0), []).append(ax)
    for axs in cols.values():
        xc = min(info[a]["xc"] for a in axs)
        for a in axs:
            v = info[a]; bb = v["bb"]
            if v["lw"]:
                a.yaxis.set_label_coords((xc + v["lw"] / 2 - bb.x0) / bb.width, 0.5)
            fig.text(xc / Wp, (bb.y1 + 0.07 * fig.dpi) / Hp, v["s"], fontsize=12, fontweight="bold",
                     va="bottom", ha="center")

def tri_layout(cbar_left=False, pw=2.2, ph=1.8):
    """Panels (a), (b) in the first row and (c) centred in the second row, all the same size (inches)."""
    gap = 1.35 if cbar_left else 1.05
    left, bottom, vgap, top = 0.72, 0.55, 1.0, 0.38
    W = W2; H = bottom + ph + vgap + ph + top
    fig = plt.figure(figsize=(W, H))
    y1 = bottom + ph + vgap
    xa, xb = left, left + pw + gap
    xc = (xa + xb + pw) / 2 - pw / 2
    mk = lambda x, y: fig.add_axes([x / W, y / H, pw / W, ph / H])
    return fig, [mk(xa, y1), mk(xb, y1), mk(xc, bottom)]

def save(fig, name):
    for ax in fig.axes:                       # legends inside the plot area: thinner outline than the spines
        lg = ax.get_legend()
        if lg is not None:
            lg.get_frame().set_linewidth(0.5)
    align(fig)
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=600)
    plt.close(fig)
    print(f"  saved {name}")

# -----------------------------------------------------------------------------
# Axis bounds: labelled ends that are multiples of the tick step
# -----------------------------------------------------------------------------
def nice_step(span, n=5):
    raw = max(span, 1e-12) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw * 0.999:
            return m * mag
    return 10 * mag

def decimals(step):
    for d in range(0, 8):
        if abs(round(step * 10 ** d) - step * 10 ** d) < 1e-9:
            return d
    return 6

def lin_axis(ax, which, lo, hi, n=5, step=None, fmt=None):
    """Set limits to [floor(lo), ceil(hi)] on the tick grid and label every tick."""
    step = step or nice_step(hi - lo, n)
    a = math.floor(lo / step + 1e-9) * step
    b = math.ceil(hi / step - 1e-9) * step
    if b <= a:
        b = a + step
    ticks = np.round(np.arange(a, b + step / 2, step), 10)
    d = decimals(step)
    f = fmt or (lambda v, _: f"{(v if abs(v) > step * 1e-6 else 0.0):.{d}f}")
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xlim if which == "x" else ax.set_ylim)(a, b)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(f))
    axis.set_minor_locator(NullLocator())
    return a, b, step

def log_axis(ax, which, lo, hi):
    """Log axis from the decade below lo to the decade above hi, labelled at every decade."""
    a = 10 ** math.floor(math.log10(lo) + 1e-9)
    b = 10 ** math.ceil(math.log10(hi) - 1e-9)
    if b <= a:
        b = a * 10
    ticks = 10.0 ** np.arange(round(math.log10(a)), round(math.log10(b)) + 1)
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    (ax.set_xlim if which == "x" else ax.set_ylim)(a, b)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{int(round(math.log10(v)))}}}$"))
    axis.set_minor_formatter(NullFormatter())

def cbar(fig, im, ax, lo, hi, step, label):
    """Colour bar with exactly the height of its axes."""
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    cax = fig.add_axes([pos.x1 + 0.08 / W, pos.y0, 0.11 / W, pos.height])
    cb = fig.colorbar(im, cax=cax)
    t = np.round(np.arange(lo, hi + step / 2, step), 10)
    d = decimals(step)
    cb.set_ticks(t); cb.set_ticklabels([f"{v:.{d}f}" for v in t])
    cb.set_label(label); cb.outline.set_linewidth(0.8)
    return cb

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(c - h, 0.0), min(c + h, 1.0)

# baseline wake (Scripts 05b to 12)
U_INF, DELTA0, L_DELTA, W0, X_R, L_W = 1.0 / 0.87, 1.30, 0.90, 0.35, 0.35, 1.50
def U_loc(x):
    return U_INF * (1 - DELTA0 * np.exp(-x / L_DELTA))
def W_env(x):
    return W0 * (1 - np.exp(-(x / X_R) ** 2)) * np.exp(-x / L_W)

def s10_classes():
    PTS = np.load(P("script10_continuation", f"points_script10{Q}.npz"))["points"]
    BR = np.load(P("script10_continuation", CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
    CL = dict(np.load(P("script10_continuation", CK, "stageD2_index.npz")))
    key = lambda R: [tuple(k_) for k_ in np.stack([R[:, 2], R[:, 3], np.round(R[:, 4] / 0.001),
                                                   np.round(R[:, 5] / 0.01), np.round(R[:, 6] / 0.1)], 1)]
    kc = dict(zip(key(PTS[CL["sel"].astype(int)]), CL["ftype"].astype(str)))
    stab = np.where(PTS[:, 7] > 0)[0]
    cls = {int(i): kc.get(k_) for i, k_ in zip(stab, key(PTS[stab]))}
    return PTS, BR, CL, cls

CAPTIONS = {}
ITEMS = {}
def item(fn):
    ITEMS[fn.__name__] = fn
    return fn

# =============================================================================
# FIGURES
# =============================================================================
@item
def fig1():
    """With the hand-drawn panels on Drive (script13_figures/fig1a_scene.png and
    fig1b_diagram.png, no panel letters), compose (a) scene, (b) block diagram,
    (c) wake profiles. Without them, fall back to the schematic drawn in code."""
    src = os.path.join(OUT_DIR, "Fig1_source.png")
    if os.path.exists(src):
        return fig1_replace_c(src)
    fa, fb = os.path.join(OUT_DIR, "fig1a_scene.png"), os.path.join(OUT_DIR, "fig1b_diagram.png")
    if os.path.exists(fa) and os.path.exists(fb):
        return fig1_composed(fa, fb)
    fig = plt.figure(figsize=(W2, 2.6))
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.set_xlim(-0.3, 2.2); ax.set_ylim(-0.75, 0.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), 0.16, fc="#d9d9d9", ec=C_INK, lw=0.9))
    for k in range(4):
        xv = 0.45 + 0.5 * k; yv = 0.24 if k % 2 == 0 else -0.24; sg = 1 if yv > 0 else -1
        ax.add_patch(Circle((xv, yv), 0.08, fc="none", ec=C_MUTE, lw=0.8))
        ax.annotate("", xy=(xv + 0.06 * sg, yv + 0.05), xytext=(xv - 0.06 * sg, yv + 0.05),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_MUTE))
    xf = 1.0
    ax.add_patch(Ellipse((xf, 0), 0.42, 0.11, angle=8, fc="#9ecae1", ec=C_INK, lw=0.8))
    ax.add_patch(Polygon([[xf + 0.2, 0.03], [xf + 0.34, 0.12], [xf + 0.34, -0.02]], closed=True, fc="#9ecae1", ec=C_INK, lw=0.8))
    ax.annotate("", xy=(xf + 0.4, 0.17), xytext=(xf + 0.4, -0.08), arrowprops=dict(arrowstyle="<->", lw=0.8, color=C_SAD))
    ax.text(xf + 0.45, 0.03, r"$\theta$", color=C_SAD, fontsize=10)
    ax.annotate("", xy=(xf - 0.05, -0.42), xytext=(0.0, -0.42), arrowprops=dict(arrowstyle="<->", lw=0.8, color=C_INK))
    ax.text(xf / 2, -0.48, r"$x$", ha="center", va="top", fontsize=10)
    ax.annotate("", xy=(2.15, -0.62), xytext=(1.7, -0.62), arrowprops=dict(arrowstyle="->", lw=0.9, color=C_INK))
    ax.text(1.92, -0.56, r"$U_\infty$", ha="center", fontsize=10)
    bx = dict(boxstyle="round,pad=0.3", fc="white", ec=C_INK, lw=0.8)
    ax.text(0.45, 0.62, r"CPG $z$", ha="center", fontsize=9.5, bbox=bx)
    ax.text(1.45, 0.62, r"body $\theta$", ha="center", fontsize=9.5, bbox=bx)
    ax.annotate("", xy=(1.18, 0.66), xytext=(0.72, 0.66), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.95, 0.73, r"$g\,\mathrm{Re}(z\,e^{-i\Delta\phi})$", ha="center", fontsize=9)
    ax.annotate("", xy=(0.5, 0.52), xytext=(1.4, 0.52),
                arrowprops=dict(arrowstyle="->", lw=0.8, connectionstyle="arc3,rad=0.3"))
    ax.text(0.95, 0.36, r"$\varepsilon\,\theta$", ha="center", fontsize=9.5)
    panel(ax, "(a)", x=0.0, y=0.93)
    ax2 = fig.add_axes([0.66, 0.2, 0.32, 0.72])
    xx = np.linspace(0, 3.5, 400)
    ax2.plot(xx, U_loc(xx) / U_INF, color=C_NODE, label=r"$U_{loc}$")
    ax2.plot(xx, W_env(xx) / U_INF, color=C_SAD, label=r"$W$")
    ax2.axhline(0, color=C_UN, lw=0.6)
    try:
        PTS = np.load(P("script10_continuation", f"points_script10{Q}.npz"))["points"]
        xs = PTS[PTS[:, 7] > 0, 5]
        ax2.axvspan(xs.min(), xs.max(), color=C_ST, alpha=0.15, lw=0)
        CAPTIONS["holding_range"] = (float(xs.min()), float(xs.max()))
    except Exception:
        pass
    lin_axis(ax2, "x", 0, 3.5, step=0.5)
    y_all = np.concatenate([U_loc(xx), W_env(xx)]) / U_INF
    lin_axis(ax2, "y", y_all.min(), y_all.max(), step=0.4)
    ax2.set_xlabel(r"$x/\lambda$"); ax2.set_ylabel(r"velocity$/U_\infty$")
    ax2.legend(loc="lower right", handlelength=1.4)
    panel(ax2, "(b)")
    save(fig, "Fig1_model")
    hr = CAPTIONS.get("holding_range")
    CAPTIONS["Fig1"] = ("Model. (a) A fish (body pitch theta, rhythm z from a Stuart-Landau CPG with proprioceptive feedback "
                        "eps*theta and neural delay Delta-phi) at distance x behind a cylinder shedding a Karman street. "
                        "(b) Prescribed mean-flow deficit U_loc and vortex velocity envelope W, in units of the wake wavelength lambda."
                        + (f" Shaded: positions of stable station holding found by continuation (x/lambda = {hr[0]:.2f} to {hr[1]:.2f})." if hr else ""))

def wake_profiles(ax2):
    """Figure 1c: mean streamwise flow and transverse envelope, with the stable holding range."""
    xx = np.linspace(0, 3.5, 400)
    ax2.plot(xx, U_loc(xx) / U_INF, color=C_NODE, label=r"$U_{loc}$, along $x$")
    ax2.plot(xx, W_env(xx) / U_INF, color=C_SAD, label=r"$W$, envelope along $y$")
    ax2.axhline(0, color=C_UN, lw=0.6)
    try:
        PTS = np.load(P("script10_continuation", f"points_script10{Q}.npz"))["points"]
        xs = PTS[PTS[:, 7] > 0, 5]
        ax2.axvspan(xs.min(), xs.max(), color=C_ST, alpha=0.15, lw=0)
        CAPTIONS["holding_range"] = (float(xs.min()), float(xs.max()))
    except Exception:
        pass
    lin_axis(ax2, "x", 0, 3.5, step=0.5)
    y_all = np.concatenate([U_loc(xx), W_env(xx)]) / U_INF
    lin_axis(ax2, "y", -0.8, y_all.max(), step=0.4)          # room below zero for the legend
    ax2.set_xlabel(r"$x/\lambda$"); ax2.set_ylabel(r"velocity$/U_\infty$")
    lg = ax2.legend(loc="lower right", handlelength=1.4)
    lg.get_frame().set_linewidth(0.5)

def fig1_replace_c(src):
    """Paste a regenerated panel (c) into the final hand-assembled Figure 1.
    The layout constants are fractions of the source image measured on the
    3719 x 2970 pixel version: panel (c) occupies the region right of 53.5% of
    the width and below 55.6% of the height; its letter is kept as drawn."""
    from PIL import Image
    img = Image.open(src).convert("RGB")
    Wpx, Hpx = img.size
    dpi = Wpx / W2
    x0, y0 = int(0.5351 * Wpx), int(0.5556 * Hpx)        # region to redraw (letter "(c)" lies above y0)
    rw, rh = Wpx - x0, Hpx - y0
    fig = plt.figure(figsize=(rw / dpi, rh / dpi), dpi=dpi)
    fw, fh = rw / dpi, rh / dpi
    left, top, aw, ah = 0.87 * (Wpx / 3719), 0.18 * (Hpx / 2970), 1.98, 1.79
    ax2 = fig.add_axes([left / fw, (fh - top - ah) / fh, aw / fw, ah / fh])
    wake_profiles(ax2)
    tmp = os.path.join(OUT_DIR, "_fig1c_tmp.png")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(tmp, dpi=dpi, facecolor="white")
    plt.close(fig)
    pc = Image.open(tmp).convert("RGB").resize((rw, rh))
    out = img.copy()
    out.paste(Image.new("RGB", (rw, rh), "white"), (x0, y0))
    out.paste(pc, (x0, y0))
    out.save(os.path.join(OUT_DIR, "Fig1_model.png"), dpi=(dpi, dpi))
    out.save(os.path.join(OUT_DIR, "Fig1_model.pdf"), resolution=dpi)
    os.remove(tmp)
    print("  saved Fig1_model (panel c regenerated inside Fig1_source.png)")
    hr = CAPTIONS.get("holding_range")
    CAPTIONS["Fig1"] = ("Model. (a) Top view of the fish behind the cylinder. (b) Block diagram. (c) Mean streamwise flow "
                        "U_loc, along x, and envelope W of the transverse velocity, along y, both divided by U_inf."
                        + (f" Shaded: stable station holding, x/lambda = {hr[0]:.2f} to {hr[1]:.2f}." if hr else ""))

def fig1_composed(fa, fb):
    import matplotlib.image as mpimg
    A, B = mpimg.imread(fa), mpimg.imread(fb)
    ra, rb = A.shape[0] / A.shape[1], B.shape[0] / B.shape[1]      # height / width
    W = W2
    wa = W - 0.1; ha = wa * ra                                       # scene: full width
    pw, ph = 2.2, 1.8                                                # profile axes, same size as other panels
    wb = W - pw - 0.72 - 0.35 - 0.1; hb = wb * rb                    # diagram: remaining width
    if hb > ph + 0.55:                                               # keep the bottom row compact
        hb = ph + 0.55; wb = hb / rb
    row2 = max(hb, ph + 0.55)
    gap, top = 0.35, 0.3
    H = row2 + gap + ha + top
    fig = plt.figure(figsize=(W, H))
    axa = fig.add_axes([0.05 / W, (row2 + gap) / H, wa / W, ha / H]); axa.imshow(A); axa.axis("off")
    axb = fig.add_axes([0.05 / W, (row2 - hb) / H + (hb < row2) * 0.0, wb / W, hb / H]); axb.imshow(B); axb.axis("off")
    ax2 = fig.add_axes([(W - pw - 0.1) / W, 0.55 / H, pw / W, ph / H])
    panel(axa, "(a)", x=0.0, y=1.0)
    panel(axb, "(b)", x=0.0, y=1.0)
    xx = np.linspace(0, 3.5, 400)
    ax2.plot(xx, U_loc(xx) / U_INF, color=C_NODE, label=r"$U_{loc}$")
    ax2.plot(xx, W_env(xx) / U_INF, color=C_SAD, label=r"$W$")
    ax2.axhline(0, color=C_UN, lw=0.6)
    try:
        PTS = np.load(P("script10_continuation", f"points_script10{Q}.npz"))["points"]
        xs = PTS[PTS[:, 7] > 0, 5]
        ax2.axvspan(xs.min(), xs.max(), color=C_ST, alpha=0.15, lw=0)
        CAPTIONS["holding_range"] = (float(xs.min()), float(xs.max()))
    except Exception:
        pass
    lin_axis(ax2, "x", 0, 3.5, step=0.5)
    y_all = np.concatenate([U_loc(xx), W_env(xx)]) / U_INF
    lin_axis(ax2, "y", y_all.min(), y_all.max(), step=0.4)
    ax2.set_xlabel(r"$x/\lambda$"); ax2.set_ylabel(r"velocity$/U_\infty$")
    ax2.legend(loc="lower right", handlelength=1.4)
    panel(ax2, "(c)")
    save(fig, "Fig1_model")
    hr = CAPTIONS.get("holding_range")
    CAPTIONS["Fig1"] = ("Model. (a) Top view: a fish at distance x behind a cylinder in a Karman street of wavelength lambda "
                        "(upper row clockwise, lower row counterclockwise vortices, fading downstream). The fish yaws by theta, "
                        "feels the transverse vortex velocity w and moves streamwise under the net force F_x. "
                        "(b) Model structure. CPG: Stuart-Landau oscillator with intrinsic frequency omega_c; body: damped mode "
                        "with natural frequency Omega and damping zeta; wake: prescribed, sampled at the fish position x; "
                        "surge: streamwise motion under F_x. Dashed: lateral-line input, absent in the baseline model and used "
                        "only in the model-structure and lateral-line tests. "
                        "(c) Prescribed mean-flow deficit U_loc and vortex velocity envelope W in units of U_inf."
                        + (f" Shaded: positions of stable station holding found by continuation (x/lambda = {hr[0]:.2f} to {hr[1]:.2f})." if hr else ""))

@item
def fig2():
    d1 = np.load(P("script01_passive_baseline", "results_script01.npz"))
    d2 = np.load(P("script02_passive_robustness", "results_script02.npz"))
    fig, ax = tri_layout()
    RATIO, ZETA, XG, F = d1["RATIO"], d1["ZETA"], d1["X_GRID"], d1["F_land"]
    iz = int(np.argmin(np.abs(ZETA - 0.1)))
    cols = ["#1b9e77", "#d95f02", "#7570b3"]
    sel = []
    for c_, r_ in zip(cols, (0.5, 0.95, 3.0)):
        ir = int(np.argmin(np.abs(RATIO - r_))); sel.append(RATIO[ir])
        ax[0].plot(XG, F[ir * ZETA.size + iz], color=c_, label=rf"{RATIO[ir]:.2f}")
    ax[0].plot(XG, F[-1], color=C_INK, ls="--", label="rigid")
    ax[0].axhline(0, color=C_UN, lw=0.6)
    lin_axis(ax[0], "x", 0, 4, step=1)
    Fs = np.concatenate([F[[int(np.argmin(np.abs(RATIO - r_))) * ZETA.size + iz for r_ in (0.5, 0.95, 3.0)]].ravel(), F[-1]])
    lin_axis(ax[0], "y", np.nanmin(Fs), np.nanmax(Fs), step=0.05)
    ax[0].set_xlabel(r"$x/\lambda$"); ax[0].set_ylabel(r"$\bar F_x$")
    ax[0].legend(title=r"$\Omega/\omega_w$", loc="lower right", fontsize=8.5, title_fontsize=8.5, handlelength=1.2)
    panel(ax[0], "(a)")
    r = d2["lhs_eq_x"] / d2["lhs_xpeak"]
    stable = d2["lhs_cls"] == 0
    f2b = P("script02b_passive_cases", "results_script02b.npz")
    if os.path.exists(f2b):
        d2b = np.load(f2b)
        chk, conv, cf = d2b["chk"].astype(int), d2b["conv"].astype(bool), d2b["cls_f"]
        stable = stable.copy(); stable[chk[conv]] = cf[conv] == 0
    bins = np.arange(0, 2.5001, 0.25); cen = 0.5 * (bins[1:] + bins[:-1])
    est = [wilson(int(stable[(r >= a) & (r < b)].sum()), int(((r >= a) & (r < b)).sum())) for a, b in zip(bins[:-1], bins[1:])]
    p_, lo_, hi_ = (np.array(v) * 100 for v in zip(*est))
    ok = np.isfinite(p_)
    ax[1].errorbar(cen[ok], p_[ok], yerr=[p_[ok] - lo_[ok], hi_[ok] - p_[ok]], fmt="o", ms=3.5, color=C_ST,
                   ecolor=C_ST, elinewidth=0.9, capsize=2, label="stable (95% CI)")
    ax[1].axvline(1, color=C_INK, lw=0.8, ls="--", label="vortex peak")
    ax[1].legend(loc="upper right")
    lin_axis(ax[1], "x", 0, 2.5, step=0.5)
    lin_axis(ax[1], "y", 0, np.nanmax(hi_[ok]) * 1.02 + 1e-9, step=nice_step(np.nanmax(hi_[ok]), 4))
    ax[1].set_xlabel(r"$x^*/x_{peak}$"); ax[1].set_ylabel("stable equilibria (%)")
    panel(ax[1], "(b)")
    lam = d1["lam_max"]; tl = np.nanmin(np.stack([d1["tloss_plus"], d1["tloss_minus"]]), 0)
    m = (lam > 1e-3) & np.isfinite(tl) & d1["converged"].astype(bool)
    x_, y_ = 1 / lam[m], tl[m]
    ax[2].plot(x_, y_, ".", ms=3, color=C_SAD, alpha=0.6, label="unstable equilibria")
    slope = None
    if m.sum() > 3:
        c_ = np.polyfit(np.log(x_), np.log(y_), 1); slope = float(c_[0])
        xs_ = np.geomspace(x_.min(), x_.max(), 20)
        ax[2].plot(xs_, np.exp(np.polyval(c_, np.log(xs_))), color=C_INK, lw=0.9, label="power-law fit")
    log_axis(ax[2], "x", x_.min(), x_.max()); log_axis(ax[2], "y", y_.min(), y_.max())
    ax[2].set_xlabel(r"$1/\ln|\mu|_{max}$ (periods)"); ax[2].set_ylabel("time to loss (periods)")
    ax[2].legend(loc="upper left", markerscale=3)
    panel(ax[2], "(c)")
    save(fig, "Fig2_passive")
    CAPTIONS["Fig2"] = ("Passive bodies (no CPG). (a) Period-averaged streamwise force (positive downstream) for body stiffness "
                        f"Omega/omega_w = {', '.join(f'{v:.2f}' for v in sel)} (damping zeta = {ZETA[iz]:.2f}) and a rigid body. "
                        "(b) Fraction of Floquet-stable equilibria against position relative to the vortex-amplitude peak, Latin "
                        "hypercube over four wake families; bars are 95% Wilson intervals; dashed line: vortex peak. "
                        "(c) Time to loss of position after a 0.01 lambda kick against the growth time 1/ln|mu|max"
                        + (f"; line: power-law fit, slope {slope:.2f}." if slope is not None else "."))

@item
def fig3():
    d = np.load(P("script03_cpg_tongues", "results_script03.npz"))
    fig, ax = tri_layout(cbar_left=True)
    EPS = d["EPS"]; RF = d["c0_rho_free"]; RHO = d["c0_rho"]; LK = d["c0_locked11"].astype(float)
    EE = np.broadcast_to(EPS[None, :], RF.shape)
    im = ax[0].pcolormesh(RF, EE, RHO, shading="nearest", cmap="twilight_shifted", vmin=0.4, vmax=2.0, rasterized=True)
    ax[0].contour(RF, EE, LK, levels=[0.5], colors="k", linewidths=1.3)
    ax[0].legend(handles=[plt.Line2D([], [], color="k", lw=1.3, label="1:1 locking")], loc="upper right")
    lin_axis(ax[0], "x", 0.4, 2.0, step=0.4); lin_axis(ax[0], "y", 0, EPS.max(), step=5)
    ax[0].set_xlabel(r"$f_{free}/f_{shed}$"); ax[0].set_ylabel(r"$\varepsilon$")
    cbar(fig, im, ax[0], 0.4, 2.0, 0.4, r"$\rho$")
    panel(ax[0], "(a)")
    cols = ["#1b9e77", "#d95f02", "#7570b3"]
    for c_, e in zip(cols, (2, 6, 12)):
        if f"stair_{e}_rho" in d:
            ax[1].plot(d["RAT_STAIR"], d[f"stair_{e}_rho"], color=c_, label=rf"{e}")
    ax[1].plot(d["RAT_STAIR"], d["RAT_STAIR"], color=C_UN, lw=0.8, ls=":")
    lin_axis(ax[1], "x", 0.4, 2.0, step=0.4); lin_axis(ax[1], "y", 0.4, 2.0, step=0.4)
    ax[1].set_xlabel(r"$\omega_c/\omega_w$"); ax[1].set_ylabel(r"$\rho$")
    ax[1].legend(title=r"$\varepsilon$", loc="upper left", title_fontsize=9, handlelength=1.2)
    panel(ax[1], "(b)")
    sl = jload("script03_cpg_tongues", "summary_script03.json").get("C_phase_slips")
    if sl:
        dl, Ts, ns = np.array(sl["delta"]), np.array(sl["T_slip"]), np.array(sl["n_slips"])
        ok = ns >= 3
        ax[2].plot(dl[ok], Ts[ok], "o", ms=4, color=C_SAD, label="simulations")
        xs_ = np.geomspace(dl[ok].min(), dl[ok].max(), 10)
        c0 = np.exp(np.mean(np.log(Ts[ok]) - sl["exponent"] * np.log(dl[ok])))
        ax[2].plot(xs_, c0 * xs_ ** sl["exponent"], color=C_INK, lw=0.9, label="power-law fit")
        ax[2].legend(loc="upper right")
        log_axis(ax[2], "x", dl[ok].min(), dl[ok].max()); log_axis(ax[2], "y", Ts[ok].min(), Ts[ok].max())
        CAPTIONS["slip"] = sl["exponent"]
    ax[2].set_xlabel(r"distance past edge $\delta$"); ax[2].set_ylabel("time between slips")
    panel(ax[2], "(c)")
    save(fig, "Fig3_fixed_position")
    CAPTIONS["Fig3"] = ("Entrainment at fixed position. (a) Rotation number rho of the rhythm relative to shedding, against the "
                        "free-stream tail-beat frequency f_free/f_shed and proprioceptive coupling eps; black contour: 1:1 locking. "
                        "(b) Devil's staircases at three couplings; dotted: rho = omega_c/omega_w. (c) Time between phase slips "
                        "against distance past the tongue edge"
                        + (f"; line: fitted exponent {CAPTIONS['slip']:.3f} (saddle-node on invariant circle: -0.5)." if "slip" in CAPTIONS else "."))

@item
def fig4():
    PTS, BR, CL, cls = s10_classes()
    bif = jload("script10_continuation", f"bifurcations_script10{Q}.json")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(wspace=0.35))
    pidx = {(int(r_[0]), int(r_[1])): k_ for k_, r_ in enumerate(PTS)}
    picks = []
    for c_, dp in enumerate((0.0, np.pi / 2)):
        ax = axs[c_]
        m0 = (np.abs(PTS[:, 3] - dp) < 1e-6) & (PTS[:, 7] > 0)
        rows = np.unique(PTS[m0, 2])
        if rows.size == 0:
            ax.text(0.5, 0.5, "no stable holding", transform=ax.transAxes, ha="center"); continue
        e_best = rows[int(np.argmax([(m0 & (PTS[:, 2] == e_)).sum() for e_ in rows]))]
        picks.append((dp, e_best))
        X, Y = [], []
        for bi, b in enumerate(BR):
            if abs(b["eps"] - e_best) > 1e-6 or abs(b["dphi"] - dp) > 1e-6 or len(b["Z"]) < 2:
                continue
            Z = b["Z"]; X.append(Z[:, 6]); Y.append(Z[:, 0])
            for j in range(len(Z) - 1):
                k1, k2 = pidx.get((bi, j)), pidx.get((bi, j + 1))
                if k1 is None or k2 is None:
                    continue
                st = PTS[k1, 7] > 0 and PTS[k2, 7] > 0
                c = BRANCH_COL.get(cls.get(k1) or cls.get(k2), C_ST) if st else C_UN
                ax.plot(Z[j:j + 2, 6], Z[j:j + 2, 0], color=c, lw=2.2 if st else 0.9, solid_capstyle="round")
        for b_ in bif:
            if abs(b_["eps"] - e_best) < 1e-6 and abs(b_["dphi"] - dp) < 1e-6:
                mk = {"fold": "o", "loss-torus": "^"}.get(b_["kind"])
                if mk:
                    ax.plot(b_["ratio"], b_["x"], mk, ms=5, mfc="white", mec=C_INK, mew=0.9, zorder=5)
        X, Y = np.concatenate(X), np.concatenate(Y)
        lin_axis(ax, "x", X.min(), X.max(), n=4); lin_axis(ax, "y", Y.min(), Y.max(), step=0.2)
        ax.set_xlabel(r"$\omega_c/\omega_w$"); ax.set_ylabel(r"holding position $x/\lambda$")
        panel(ax, "(a)" if c_ == 0 else "(b)")
    h = [plt.Line2D([], [], color=BRANCH_COL[k_], lw=2.2, label=BRANCH_LAB[k_]) for k_ in ("saddle-real", "node", "none")]
    h += [plt.Line2D([], [], color=C_UN, lw=0.9, label="unstable"),
          plt.Line2D([], [], ls="", marker="o", mfc="white", mec=C_INK, ms=5, label="fold"),
          plt.Line2D([], [], ls="", marker="^", mfc="white", mec=C_INK, ms=5, label="torus")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    save(fig, "Fig4_branches")
    CAPTIONS["Fig4"] = ("Station-holding branches at fixed coupling form closed loops in (omega_c/omega_w, x). Thick: stable, "
                        "coloured by the fixed-position branch reached by homotopy; thin grey: unstable; circles: folds; "
                        "triangles: torus bifurcations. " +
                        " ".join(f"({'ab'[i]}) Delta-phi = {dp:.2f}, eps = {e_:.3f}." for i, (dp, e_) in enumerate(picks)))

@item
def fig5():
    srcs = []
    f8 = P("script08_homotopy", "records_script08.json")
    f9 = P("script09_model_structure", f"records_script09_ALL{Q}.json")
    if os.path.exists(f8):
        srcs.append(("08", [r_ for r_ in json.load(open(f8)) if r_.get("index") is not None and r_.get("kept", True)]))
    if os.path.exists(f9):
        srcs.append(("09", [r_ for r_ in json.load(open(f9)) if r_.get("index") is not None]))
    try:
        PTS, BR, CL, cls = s10_classes()
        ok = np.isfinite(CL["index"])
        srcs.append(("10", [dict(det_I_minus_Mint=float(a), aL=float(b), stable=True, fixed_type=str(t))
                            for a, b, t in zip(CL["dI"][ok], CL["aL"][ok], CL["ftype"][ok])]))
    except Exception:
        pass
    if not srcs:
        raise FileNotFoundError("no index-test records")
    tr = lambda d_: np.sign(d_) * np.log10(1 + 1e3 * np.abs(d_))
    allx = np.concatenate([tr(np.array([r_["det_I_minus_Mint"] for r_ in R])) for _, R in srcs])
    ally = np.concatenate([np.array([r_["aL"] for r_ in R]) for _, R in srcs])
    xm = np.nanmax(np.abs(allx)); ym = np.nanmax(np.abs(ally))
    sx, sy = nice_step(2 * xm, 4), nice_step(2 * ym, 4)
    XB, YB = math.ceil(xm / sx - 1e-9) * sx, math.ceil(ym / sy - 1e-9) * sy
    fig, axs = plt.subplots(1, len(srcs), figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.35))
    axs = np.atleast_1d(axs)
    for i, (ax, (name, R)) in enumerate(zip(axs, srcs)):
        dI = np.array([r_["det_I_minus_Mint"] for r_ in R]); aL = np.array([r_["aL"] for r_ in R])
        st = np.array([r_["stable"] for r_ in R]); ft = np.array([r_["fixed_type"] for r_ in R])
        ax.add_patch(plt.Rectangle((-XB, 0), XB, YB, color=C_ST, alpha=0.08, lw=0))
        ax.add_patch(plt.Rectangle((0, -YB), XB, YB, color=C_ST, alpha=0.08, lw=0))
        ax.scatter(tr(dI[~st]), aL[~st], s=4, c=C_UN, lw=0, rasterized=True)
        for k_ in ("saddle-real", "node"):
            m = st & (ft == k_)
            ax.scatter(tr(dI[m]), aL[m], s=6, c=BRANCH_COL[k_], lw=0, rasterized=True)
        ax.axhline(0, color=C_INK, lw=0.7); ax.axvline(0, color=C_INK, lw=0.7)
        lin_axis(ax, "x", -XB, XB, step=sx); lin_axis(ax, "y", -YB, YB, step=sy)
        ax.set_xlabel(r"$s(\det(I-M_{int}))$")
        if i == 0:
            ax.set_ylabel(r"$a_L$")
        panel(ax, "(" + "abc"[i] + ")")
    h = [plt.Line2D([], [], ls="", marker="o", ms=5, color=BRANCH_COL[k_], label="stable, " + BRANCH_LAB[k_]) for k_ in ("saddle-real", "node")]
    h += [plt.Line2D([], [], ls="", marker="o", ms=5, color=C_UN, label="unstable"),
          Patch(color=C_ST, alpha=0.25, label="index > 0 (allowed)")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, "Fig5_index_condition")
    names = {"08": "26 parameter sets of the baseline structure (Script 08)", "09": "36 model structures and parameter samples (Script 09)",
             "10": "all continued stable branch points (Script 10)"}
    CAPTIONS["Fig5"] = ("The index condition. Relocked force gradient a_L against s(d) = sgn(d) log10(1 + 1000|d|), "
                        "d = det(I - M_int), for every orbit whose fixed-position branch was identified by homotopy. Stable "
                        "orbits are allowed only in the shaded quadrants (index > 0). "
                        + " ".join(f"({'abc'[i]}) {names[n]}." for i, (n, _) in enumerate(srcs)) + " Counts are in Table T2.")

def surging_example(BR):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    C_L, C_D, SIG, OMB, ZEB, MU, G = 3.0, 0.15, 8.0, 1.2 * 2 * np.pi, 0.2, 2 * np.pi, 16.0
    OW = 2 * np.pi
    def rhs(t, s, p):
        x, u, th, om, zr, zi = s
        V = U_INF * (1 - DELTA0 * jnp.exp(-x / L_DELTA)) - u
        w = W0 * (1 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * jnp.sin(OW * (x - t))
        beta = jnp.arctan2(w, jnp.abs(V)); al = beta - th
        gate = 0.5 * (1 + jnp.tanh(V / 0.05))
        L = C_L * (V * V + w * w) * jnp.sin(al) * jnp.cos(al) * gate
        Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
        drive = zr * jnp.cos(p[2]) + zi * jnp.sin(p[2])
        r2 = zr * zr + zi * zi
        return jnp.stack([u, Fx, om, -2 * ZEB * OMB * om - OMB ** 2 * th + SIG * L + G * drive,
                          MU * (1 - r2) * zr - p[0] * OW * zi + p[1] * th, MU * (1 - r2) * zi + p[0] * OW * zr])
    def period(s, p):
        dt = 0.01
        def st(i, s_):
            t = i * dt
            k1 = rhs(t, s_, p); k2 = rhs(t + dt / 2, s_ + dt / 2 * k1, p)
            k3 = rhs(t + dt / 2, s_ + dt / 2 * k2, p); k4 = rhs(t + dt, s_ + dt * k3, p)
            return s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return jax.lax.fori_loop(0, 100, st, s)
    def body(c, _):
        n = period(c, P_)
        return n, n[0]
    for b in BR:
        if abs(b["dphi"]) > 1e-6 or len(b["Z"]) < 3:
            continue
        am, ld = b["amax"], b["lead"]
        j = np.where((am > 1.0005) & (am < 1.01) & (np.abs(ld.imag) > 1e-6))[0]
        if j.size == 0:
            continue
        z = b["Z"][j[0]]
        P_ = jnp.array([z[6], b["eps"], b["dphi"]])
        _, xs = jax.lax.scan(body, jnp.asarray(z[:6] + np.array([1e-3, 0, 0, 0, 0, 0])), None, length=2000)
        xs = np.asarray(xs)
        if np.all(np.isfinite(xs)) and xs.min() > 0.15 and xs.max() < 4.5:
            return dict(x=xs, eps=b["eps"], ratio=float(z[6]), amax=float(am[j[0]]))
    return None

def zero_one_K(phi, n_c=100, seed=0):
    r_ = np.random.default_rng(seed)
    N = phi.size; ncut = N // 10; phi = phi - phi.mean()
    Ks = []
    for c in r_.uniform(np.pi / 5, 4 * np.pi / 5, n_c):
        j = np.arange(1, N + 1)
        p = np.cumsum(phi * np.cos(j * c)); q = np.cumsum(phi * np.sin(j * c))
        M = np.array([np.mean((p[n:] - p[:-n]) ** 2 + (q[n:] - q[:-n]) ** 2) for n in range(1, ncut + 1)])
        Ks.append(np.corrcoef(np.arange(1, ncut + 1), M)[0, 1])
    return float(np.median(Ks))

@item
def fig6():
    PTS, BR, CL, cls = s10_classes()
    bif = jload("script10_continuation", f"bifurcations_script10{Q}.json")
    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.9, wspace=0.35, height_ratios=[1.15, 1])
    EPS_ROWS = np.unique(PTS[:, 2]); d_eps = np.diff(EPS_ROWS).min() if EPS_ROWS.size > 1 else 0.5
    BINS = np.arange(0.90, 1.35 + 1e-9, 0.0025)
    pidx = {(int(r_[0]), int(r_[1])): k_ for k_, r_ in enumerate(PTS)}
    cmap = ListedColormap(["#ffffff", "#e0e0e0", C_NONE, C_SAD, C_NODE])
    occs = []
    for c_, dp in enumerate((0.0, np.pi / 2)):
        occ = {k_: np.zeros((EPS_ROWS.size, BINS.size - 1), bool) for k_ in ("any", "node", "saddle-real", "none")}
        for bi, b in enumerate(BR):
            if abs(b["dphi"] - dp) > 1e-6 or len(b["Z"]) < 2:
                continue
            ei = int(np.argmin(np.abs(EPS_ROWS - b["eps"]))); Z = b["Z"]
            for j in range(len(Z) - 1):
                k1, k2 = pidx.get((bi, j)), pidx.get((bi, j + 1))
                if k1 is None or k2 is None or not (PTS[k1, 7] > 0 and PTS[k2, 7] > 0):
                    continue
                lo, hi = sorted((Z[j, 6], Z[j + 1, 6]))
                cells = np.where((BINS[1:] > lo) & (BINS[:-1] < hi))[0]
                if cells.size == 0:
                    cells = np.array([np.clip(np.searchsorted(BINS, lo) - 1, 0, BINS.size - 2)])
                occ["any"][ei, cells] = True
                c = cls.get(k1) or cls.get(k2)
                if c in occ:
                    occ[c][ei, cells] = True
        occs.append(occ)
    sel = np.where(occs[0]["any"].any(0) | occs[1]["any"].any(0))[0]
    rlo, rhi = (BINS[sel.min()], BINS[sel.max() + 1]) if sel.size else (1.0, 1.1)
    erows = np.where(occs[0]["any"].any(1) | occs[1]["any"].any(1))[0]
    elo, ehi = (EPS_ROWS[erows.min()] - d_eps / 2, EPS_ROWS[erows.max()] + d_eps / 2) if erows.size else (0, 1)
    for c_, dp in enumerate((0.0, np.pi / 2)):
        ax = fig.add_subplot(gs[0, c_]); occ = occs[c_]
        img = np.zeros(occ["any"].shape)
        img[occ["any"]] = 1; img[occ["none"]] = 2; img[occ["saddle-real"]] = 3; img[occ["node"]] = 4
        e_edges = np.concatenate([EPS_ROWS - d_eps / 2, [EPS_ROWS[-1] + d_eps / 2]])
        ax.pcolormesh(BINS, e_edges, img, cmap=cmap, vmin=-0.5, vmax=4.5, rasterized=True)
        for b_ in bif:
            if abs(b_["dphi"] - dp) > 1e-6:
                continue
            if b_["kind"] == "loss-fold":
                ax.plot(b_["ratio"], b_["eps"], "o", ms=2.6, mfc=C_INK, mec="none")
            elif b_["kind"] == "loss-torus":
                ax.plot(b_["ratio"], b_["eps"], "^", ms=3.2, mfc="none", mec=C_INK, mew=0.7)
        lin_axis(ax, "x", rlo, rhi, n=4); lin_axis(ax, "y", max(elo, 0), ehi, n=5)
        ax.set_xlabel(r"$\omega_c/\omega_w$"); ax.set_ylabel(r"$\varepsilon$")
        panel(ax, "(a)" if c_ == 0 else "(b)")
        if c_ == 1:
            # zoomed inset of the narrow band, placed in the empty upper-left corner
            zx, zy = (1.04, 1.08), (1.5, 2.5)
            axi = ax.inset_axes([0.09, 0.56, 0.35, 0.29])
            axi.pcolormesh(BINS, e_edges, img, cmap=cmap, vmin=-0.5, vmax=4.5, rasterized=True)
            for b_ in bif:
                if abs(b_["dphi"] - dp) > 1e-6:
                    continue
                if b_["kind"] == "loss-fold":
                    axi.plot(b_["ratio"], b_["eps"], "o", ms=3.2, mfc=C_INK, mec="none")
                elif b_["kind"] == "loss-torus":
                    axi.plot(b_["ratio"], b_["eps"], "^", ms=3.8, mfc="none", mec=C_INK, mew=0.7)
            lin_axis(axi, "x", zx[0], zx[1], step=0.02); lin_axis(axi, "y", zy[0], zy[1], step=0.5)
            axi.tick_params(labelsize=8, length=2.5)
            axi.yaxis.tick_right(); axi.xaxis.tick_top()
            for sp in axi.spines.values():
                sp.set_linewidth(0.8)
            ax.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=C_INK, lw=0.6))
            ax.indicate_inset_zoom(axi, edgecolor=C_INK, alpha=0.6)
    h = [Patch(color=BRANCH_COL[k_], label=BRANCH_LAB[k_]) for k_ in ("saddle-real", "node", "none")]
    h += [plt.Line2D([], [], ls="", marker="o", ms=4, mfc=C_INK, mec="none", label="loss by fold"),
          plt.Line2D([], [], ls="", marker="^", ms=4.5, mfc="none", mec=C_INK, label="loss by torus")]
    fig.legend(handles=h, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.455))
    ax = fig.add_subplot(gs[1, :])
    ex = surging_example(BR)
    if ex is not None:
        xs = ex["x"]; K = zero_one_K(xs[1000:])
        tt = np.arange(1, xs.size + 1)
        ax.plot(tt, xs, ".", ms=1.2, color=C_INK, rasterized=True)
        lin_axis(ax, "x", 0, xs.size, step=500); lin_axis(ax, "y", xs.min(), xs.max(), n=4)
        CAPTIONS["surge"] = (ex["eps"], ex["ratio"], ex["amax"], K)
    ax.set_xlabel("time (shedding periods)"); ax.set_ylabel(r"$x/\lambda$ (strobed)")
    panel(ax, "(c)")
    save(fig, "Fig6_bifurcations")
    s_ = CAPTIONS.get("surge")
    CAPTIONS["Fig6"] = ("Bifurcations of station holding. Stable holding in (omega_c/omega_w, eps) for neural delay (a) 0 and "
                        "(b) pi/2, coloured by the fixed-position branch (where branches overlap, node is drawn on top); "
                        "symbols: loss of stability by fold or torus bifurcation along continued branches. No stable holding "
                        "exists for delays pi and 3 pi/2. "
                        + (f"(c) Strobed position just past a torus point (eps = {s_[0]:.3f}, omega_c/omega_w = {s_[1]:.4f}, "
                           f"max|mu| = {s_[2]:.4f}): bounded quasiperiodic surging; 0-1 test K = {s_[3]:.3f} over the last "
                           "1000 periods." if s_ else ""))

@item
def fig7():
    d11 = P("script11_noise_predictions", CK)
    MA = np.load(os.path.join(d11, "partA_map.npz")); CD = np.load(os.path.join(d11, "partCD2.npz"))
    s11 = jload("script11_noise_predictions", f"summary_script11{Q}.json")
    PTS, BR, CL, cls = s10_classes()
    fig, ax = tri_layout(cbar_left=True)
    PH = MA["alive"].mean(-1)
    R_ = np.round(np.arange(0.95, 1.2501, 0.005), 6) if MODE == "FULL" else np.round(np.arange(1.00, 1.0601, 0.02), 6)
    E_ = np.round(np.arange(0.25, 6.001, 0.25), 6) if MODE == "FULL" else np.array([1.0, 2.0])
    di = 1
    def edges(v):
        d_ = np.diff(v).min() if v.size > 1 else 0.5
        return np.concatenate([v - d_ / 2, [v[-1] + d_ / 2]])
    vmax_s = nice_step(max(PH[:, :, di].max(), 1e-3), 3)
    vmax = math.ceil(max(PH[:, :, di].max(), 1e-3) / vmax_s - 1e-9) * vmax_s
    im = ax[0].pcolormesh(edges(R_), edges(E_), PH[:, :, di].T, cmap="Greens", vmin=0, vmax=vmax, rasterized=True)
    m = (PTS[:, 7] > 0) & (np.abs(PTS[:, 3] - np.pi / 2) < 1e-6)
    ax[0].scatter(PTS[m, 4], PTS[m, 2], s=0.8, c=C_INK, lw=0, rasterized=True)
    ax[0].legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3, color=C_INK, label="stable orbits")], loc="lower right")
    hold = np.where(PH[:, :, di].max(1) > 0)[0]
    rlo, rhi = (edges(R_)[max(hold.min() - 2, 0)], edges(R_)[min(hold.max() + 3, R_.size)]) if hold.size else (R_[0], R_[-1])
    lin_axis(ax[0], "x", rlo, rhi, n=4); lin_axis(ax[0], "y", 0, 6, step=1)
    cbar(fig, im, ax[0], 0, vmax, vmax_s, r"$P_{hold}$")
    ax[0].set_xlabel(r"$\omega_c/\omega_w$"); ax[0].set_ylabel(r"$\varepsilon$")
    panel(ax[0], "(a)")
    bins = np.linspace(-np.pi, np.pi, 49)
    SEL = CL["sel"].astype(int); FT = CL["ftype"].astype(str)
    ymax = 0
    for k_ in ("saddle-real", "node"):
        mm = (np.abs(PTS[SEL, 3] - np.pi / 2) < 1e-6) & (FT == k_)
        h_, _, _ = ax[1].hist(CD["phase_det"][mm], bins=bins, color=BRANCH_COL[k_], alpha=0.7, density=True)
        ymax = max(ymax, h_.max() if h_.size else 0)
    for g_, k_ in ((2, "saddle-real"), (3, "node")):
        mm = (CD["b_group"] == g_) & (CD["b_level"] == 0) & (CD["b_nacc"] > 0)
        if mm.any():
            h_, _, _ = ax[1].hist(CD["b_phase"][mm], bins=bins, histtype="step", color=BRANCH_COL[k_], lw=1.2, density=True)
            ymax = max(ymax, h_.max())
    lin_axis(ax[1], "x", -np.pi, np.pi, step=np.pi,
             fmt=lambda v, _: {-1: r"$-\pi$", 0: "0", 1: r"$\pi$"}.get(int(round(v / np.pi)), ""))
    lin_axis(ax[1], "y", 0, ymax, n=4)
    ax[1].set_xlabel(r"$\varphi_\theta - \varphi_w$ (rad)"); ax[1].set_ylabel("density")
    ax[1].legend(handles=[Patch(color=C_SAD, alpha=0.7, label="saddle"), Patch(color=C_NODE, alpha=0.7, label="node"),
                          plt.Line2D([], [], color=C_INK, lw=1.2, label="with noise")], loc="upper left", fontsize=8.5)
    panel(ax[1], "(b)")
    tro = s11["trout"]; Uref = tro["free_stream_body_lengths_s"]
    s_ = np.round(np.arange(0.95, 1.0501, 0.0025), 6) if MODE == "FULL" else np.array([0.99, 1.0, 1.01])
    pk = 0
    for f_ in s11.get("flow_speed_fine", []):
        if f_["eps"] not in (1.0, 2.0):
            continue
        col = C_SAD if abs(f_["dphi"]) < 1e-6 else C_NODE
        ax[2].plot(100 * (s_ - 1), f_["p_hold"], "-" if f_["eps"] == 2.0 else "--", color=col,
                   label=rf"{'0' if abs(f_['dphi']) < 1e-6 else r'$\pi/2$'}, {f_['eps']:.0f}")
        pk = max(pk, max(f_["p_hold"]))
    lin_axis(ax[2], "x", 100 * (s_.min() - 1), 100 * (s_.max() - 1), n=2); lin_axis(ax[2], "y", 0, max(pk, 0.05), n=3)
    ax[2].set_xlabel(r"$U_\infty/U_{ref}-1$ (%)"); ax[2].set_ylabel(r"$P_{hold}$")
    ax[2].legend(title=r"$\Delta\phi$, $\varepsilon$", fontsize=8.5, title_fontsize=8.5, loc="upper left", handlelength=1.4)
    panel(ax[2], "(c)")
    save(fig, "Fig7_predictions")
    CAPTIONS["Fig7"] = ("Predictions. (a) Probability of reaching station holding from random initial conditions (x uniform in "
                        "[0.3, 1.5] lambda, random rhythm phase), delay pi/2; dots: stable period-1 holding from continuation. "
                        "(b) Tail-beat phase relative to the local transverse wake velocity on the saddle and node branches at "
                        f"delay pi/2; filled: deterministic, all classified stable points; lines: during holding bouts with joint "
                        f"noise D = {s11['noise_levels'][0]}. (c) Holding probability as flow speed changes with the fish's "
                        f"intrinsic frequencies fixed in physical time (trout scaling, U_ref = {Uref:.2f} body lengths per second). "
                        "Bout durations are in Table T6. P_hold: holding probability.")

# =============================================================================
# TABLES
# =============================================================================
TABLES = {}
def table(name, title, header, rows, notes=""):
    TABLES[name] = (title, header, rows, notes)
    with open(os.path.join(TAB_DIR, f"{name}.csv"), "w") as f:
        f.write(",".join(f'"{h}"' for h in header) + "\n")
        for r_ in rows:
            f.write(",".join(f'"{c}"' for c in r_) + "\n")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    tex = [r"\begin{table}", rf"\caption{{{esc(title)}}}", rf"\label{{tab:{name}}}", r"\begin{ruledtabular}",
           r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}", " & ".join(esc(h) for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(esc(c) for c in r_) + r" \\" for r_ in rows]
    tex += [r"\end{tabular}", r"\end{ruledtabular}"]
    if notes:
        tex.append(rf"\footnotetext{{{esc(notes)}}}")
    tex.append(r"\end{table}")
    open(os.path.join(TAB_DIR, f"{name}.tex"), "w").write("\n".join(tex))
    print(f"  saved table {name}")

def f2(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"

def rng_txt(r_, d=2):
    return "n/a" if not r_ else f"{r_[0]:.{d}f} to {r_[1]:.{d}f}"

@item
def tab1():
    s11 = jload("script11_noise_predictions", f"summary_script11{Q}.json")
    tr = s11["trout"]
    lam, fs = 0.208, 2.74
    rows = [
        ["wake wavelength lambda", "1", f"{lam * 100:.1f} cm"],
        ["shedding period T", "1", f"{tr['shedding_period_s']:.3f} s"],
        ["free stream U_inf", f"{U_INF:.3f}", f"{tr['free_stream_m_s']:.3f} m/s ({tr['free_stream_body_lengths_s']:.2f} L/s)"],
        ["velocity deficit delta_0, decay l_delta", f"{DELTA0}, {L_DELTA}", "-"],
        ["vortex velocity W_0, rise x_r, decay l_w", f"{W0}, {X_R}, {L_W}", "-"],
        ["lift c_L, drag c_D, lift-to-pitch sigma", "3.0, 0.15, 8.0", "-"],
        ["body natural frequency Omega/omega_w", "1.2", f"{tr['body_natural_frequency_Hz']:.2f} Hz"],
        ["body damping ratio zeta", f"{tr['body_damping_ratio']}", "-"],
        ["CPG amplitude rate mu, shear c, gain g", "2 pi, 0, 16", "-"],
        ["holding position (stable, continuation)",
         f"{tr['holding_position_range_m'][0] / lam:.2f} to {tr['holding_position_range_m'][1] / lam:.2f}",
         f"{tr['holding_position_range_body_lengths'][0]:.2f} to {tr['holding_position_range_body_lengths'][1]:.2f} L"],
    ]
    table("T1_parameters", "Baseline parameters (units of wake wavelength and shedding period) and trout scaling "
          "(Beal et al. 2006: lambda = 20.8 cm, f = 2.74 Hz, L = 17.8 cm).", ["quantity", "model", "trout"], rows)

@item
def tab2():
    rows = []
    s8 = jload("script08_homotopy", "summary_script08.json")
    bt = s8["by_fixed_type"]
    rows.append(["parameter sets (08)", s8["n_stable_fixed"], s8["violations_nonmarginal"], s8["index_negative_n"], s8["index_negative_stable"],
                 f"{bt['node']['stable_aL_neg']}/{bt['node']['stable']}", f"{bt['saddle-real']['stable_aL_pos']}/{bt['saddle-real']['stable']}"])
    s9 = jload("script09_model_structure", f"summary_script09_ALL{Q}.json")
    t9 = s9["totals"]
    nn = sum(r_.get("by_fixed_type", {}).get("node", {}).get("stable", 0) for r_ in s9["results"])
    nna = sum(r_.get("by_fixed_type", {}).get("node", {}).get("stable_aL_neg", 0) for r_ in s9["results"])
    ns = sum(r_.get("by_fixed_type", {}).get("saddle-real", {}).get("stable", 0) for r_ in s9["results"])
    nsa = sum(r_.get("by_fixed_type", {}).get("saddle-real", {}).get("stable_aL_pos", 0) for r_ in s9["results"])
    rows.append(["model structures (09)", t9["stable_fixed"], t9["violations_nonmarginal"], t9["index_negative"], t9["index_negative_stable"],
                 f"{nna}/{nn}", f"{nsa}/{ns}"])
    try:
        s10 = jload("script10_continuation", f"summary_script10{Q}.json")["classified"]
        rows.append(["continued branches (10)", s10["evaluated"], s10["violations_after_2x_steps_check"], "-", "-",
                     f"{s10['node_aL_neg']}/{s10['node']}", f"{s10['saddle_aL_pos']}/{s10['saddle_real']}"])
    except FileNotFoundError:
        pass
    try:
        s12 = jload("script12_verification", f"summary_script12{Q}.json")["reevaluation"]
        for k_, lab in (("08", "re-evaluated 08 (12)"), ("09", "re-evaluated 09 (12)")):
            if k_ in s12:
                v = s12[k_]
                rows.append([lab, v["stable_evaluated"], v["violations_nonmarginal"], v["index_negative"], v["index_negative_stable"], "-", "-"])
    except FileNotFoundError:
        pass
    table("T2_index_verification", "Verification of the index condition sgn det(I - M_int) sgn(-a_L) > 0. Branch identity by "
          "homotopy in all rows; rows (12) use a fine homotopy and a_L fitted over x_L +/- 0.005.",
          ["source", "stable orbits", "violations", "index < 0", "of which stable", "node: a_L<0", "saddle: a_L>0"], rows,
          "Violations exclude marginal cases |a_L| < 0.02; there were none. Rows (10) have no unstable orbits (only stable points were classified).")

@item
def tab3():
    s9 = jload("script09_model_structure", f"summary_script09_ALL{Q}.json")
    rows = []
    lhs = [r_ for r_ in s9["results"] if r_["name"].startswith("LHS")]
    for r_ in s9["results"]:
        if r_["name"].startswith("LHS"):
            continue
        bt = r_.get("by_fixed_type", {})
        rows.append([r_["name"], len(r_["state_layout"]), r_["n_stable"],
                     bt.get("saddle-real", {}).get("stable", 0), bt.get("node", {}).get("stable", 0), r_.get("violations_nonmarginal", 0),
                     rng_txt(r_.get("stable_eps_range")), rng_txt(r_.get("stable_ratio_range"), 3), rng_txt(r_.get("stable_x_range"))])
    if lhs:
        rows.append([f"joint LHS ({len(lhs)} samples, {sum(r_['holding'] for r_ in lhs)} with holding)", 6,
                     sum(r_["n_stable"] for r_ in lhs),
                     sum(r_.get("by_fixed_type", {}).get("saddle-real", {}).get("stable", 0) for r_ in lhs),
                     sum(r_.get("by_fixed_type", {}).get("node", {}).get("stable", 0) for r_ in lhs),
                     sum(r_.get("violations_nonmarginal", 0) for r_ in lhs), "-", "-", "-"])
    table("T3_model_structures", "Robustness to model structure (Script 09): stable station-holding orbits found, of which "
          "tested on the saddle and node branches, index violations, and the ranges where stable holding occurs.",
          ["variant", "states", "stable orbits", "tested saddle", "tested node", "violations", "eps", "omega_c/omega_w", "x/lambda"], rows)

@item
def tab4():
    grid = {}
    for tag in ("BODY", "ALL"):
        p_ = P("script06_robustness", f"summary_script06_{tag}.json")
        if not os.path.exists(p_):
            continue
        for r_ in json.load(open(p_))["results"]:
            ov = r_.get("override", {}); c_ = r_.get("counts")
            if not c_ or set(ov) - {"body_ratio", "zeta"}:
                continue
            grid[(round(ov.get("body_ratio", 1.2), 2), round(ov.get("zeta", 0.2), 2))] = (
                sum(v_["stable"] for v_ in c_.values()), c_["node"]["stable"])
    if not grid:
        raise FileNotFoundError("Script 06 summaries")
    zes = sorted({k_[1] for k_ in grid}); brs = sorted({k_[0] for k_ in grid})
    rows = [[f"{b_:.1f}"] + [(f"{grid[(b_, z_)][0]} ({grid[(b_, z_)][1]})" if (b_, z_) in grid else "-") for z_ in zes] for b_ in brs]
    table("T4_body_tuning", "Body-tuning window (Script 06): stable station-holding orbits by body stiffness and damping "
          "(in brackets: on the node branch).", ["Omega/omega_w"] + [f"zeta = {z_:.1f}" for z_ in zes], rows)

@item
def tab5():
    s = jload("script10_continuation", f"summary_script10{Q}.json")
    rows = []
    for dp, a_ in s["stable_area_by_delay"].items():
        if not a_["any"]:
            rows.append([dp, "0", "-", "-", "-", "-", "-"]); continue
        rows.append([dp, f2(a_["any"], 4), f2(a_["saddle_only"], 4), f2(a_["node_only"], 4), f2(a_["none_only"], 4),
                     f2(a_["node_and_saddle"], 4), f2(a_["node_and_none"], 4)])
    e = s["stable_segment_ends"]["by_type"]
    table("T5_continuation", "Stable station-holding area in the (omega_c/omega_w, eps) plane by neural delay and fixed-position "
          "branch (Script 10). Ends of stable segments: "
          + ", ".join(f"{k_} {v_}" for k_, v_ in e.items()) + f"; all {s['continuation']['half_branches']} half-branches close into loops.",
          ["delay", "total", "saddle only", "node only", "no lock only", "node and saddle", "node and no lock"], rows)

@item
def tab6():
    s11 = jload("script11_noise_predictions", f"summary_script11{Q}.json")
    conds = s11["noise_conditions"]
    groups = [("0.000", "saddle-real", "delay 0, saddle"), ("1.571", "saddle-real", "delay pi/2, saddle"),
              ("1.571", "node", "delay pi/2, node")]
    rows = []
    for c_ in conds:
        row = [c_]
        for dp, ft, _ in groups:
            v = s11["bouts"].get(f"{dp}/{ft}/{c_}")
            row.append("-" if not v else (f"{v['median_bout_T']:.0f} ({v['surviving_fraction']:.2f})"
                                         if v["median_bout_T"] is not None else f"> 2000 ({v['surviving_fraction']:.2f})"))
        rows.append(row)
    row = ["exponent of median vs D"]
    for dp, ft, _ in groups:
        v = s11.get("bout_scaling", {}).get(f"{dp}/{ft}")
        row.append(f2(v["exponent"]) if v else "-")
    rows.append(row)
    table("T6_bouts", "Station-holding bouts under noise (Script 11): median bout in shedding periods (fraction still holding "
          "after 2000 periods), starting on stable orbits; D is the phase diffusion per period (wake amplitude noise has "
          "standard deviation sqrt(D)).", ["noise"] + [g[2] for g in groups], rows)

@item
def tab7():
    s11 = jload("script11_noise_predictions", f"summary_script11{Q}.json")
    v = s11["lateral_line"]
    rows = [["runs holding with lateral line", v["holding_with_lateral_line"]],
            ["still holding after removal", f"{v['retained_after_removal']} ({100 * v['retention_fraction']:.0f}%)"],
            ["tail-beat amplitude ratio, off/on (median)", f2(v["tailbeat_amp_ratio_median"])],
            ["tail-beat phase shift relative to wake (median, rad)", f2(v["tailbeat_phase_shift_median_rad"])],
            ["frequency locked before / after", f"{f2(v['frequency_locked_before'])} / {f2(v['frequency_locked_after'])}"]]
    table("T7_lateral_line", "Lateral-line removal (Script 11): fish holding station with lateral-line plus proprioceptive "
          "input, then lateral line switched off for 400 periods.", ["quantity", "value"], rows)

@item
def tab8():
    c = jload("script12_verification", f"summary_script12{Q}.json")["convergence"]
    lv = c["spp_levels"]
    rows = [["points (stratified)", c["n"]],
            [f"max change of max|mu|, {lv[0]} to {lv[-1]} steps", f"{c['max_abs_mu_change_100_to_top']:.1e}"],
            [f"median change of max|mu|", f"{c['median_abs_mu_change_100_to_top']:.1e}"],
            ["stability flips", c["stability_flips"]],
            ["sign flips: det(I - M), det(I - M_int), a_L, index",
             f"{c['detF_sign_flips'][0]}, {c['detInt_sign_flips'][0]}, {c['aL_sign_flips'][0]}, {c['index_sign_flips'][0]}"],
            ["branch-type changes", c["branch_type_changes"]]]
    for h, v in c["ad_vs_fd_max"].items():
        rows.append([f"exact vs finite-difference multipliers, step {float(h):.0e} (max)", f"{v:.1e}"])
    rows.append(["multiplier change, Newton 15 vs 40 iterations (max)", f"{c['newton_tolerance_mu_change_max']:.1e}"])
    table("T8_convergence", "Numerical convergence (Script 12).", ["check", "result"], rows)

# =============================================================================
# Run
# =============================================================================
t0 = time.time()
made, skipped = [], []
for name, fn in ITEMS.items():
    print(f"{name} ...")
    try:
        fn(); made.append(name)
    except Exception as e:
        skipped.append((name, f"{type(e).__name__}: {e}"))
        print(f"  SKIPPED {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)

with open(os.path.join(OUT_DIR, "figure_captions.md"), "w") as f:
    for k_ in sorted(k_ for k_ in CAPTIONS if k_.startswith("Fig")):
        f.write(f"**{k_}.** {CAPTIONS[k_]}\n\n")
with open(os.path.join(TAB_DIR, "all_tables.md"), "w") as f:
    for name, (title, header, rows, notes) in TABLES.items():
        f.write(f"### {name}\n\n{title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r_ in rows:
            f.write("| " + " | ".join(str(c) for c in r_) + " |\n")
        if notes:
            f.write(f"\n{notes}\n")
        f.write("\n")
print(f"\nMade {len(made)} items, skipped {len(skipped)} ({time.time() - t0:.0f} s). Outputs: {OUT_DIR}")
for s_ in skipped:
    print("  skipped:", s_)
