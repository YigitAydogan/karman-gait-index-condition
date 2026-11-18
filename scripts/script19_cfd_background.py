# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 19: Illustrative flow simulation for the background of Fig. 1a.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX; about 5 to 10 min on a CPU, about 1 min on a GPU. Independent of
# the other scripts.
#
# A two-dimensional lattice Boltzmann simulation (D2Q9, BGK collision) of flow
# past a circular cylinder at Reynolds number 100, at which the cylinder sheds
# a regular von Karman vortex street. The geometry matches panel (a) of the
# Fig. 1 drawing: the image is 3719 x 1395 pixels, the cylinder centre lies at
# pixel (650, 774) from the top left, and its diameter is 185 pixels. Flow is
# from left to right. Top and bottom boundaries are periodic, the inlet has a
# fixed velocity and the outlet a zero-gradient condition.
#
# The simulation is an illustration of the flow, not part of the model: the
# model uses a prescribed wake, and the trout experiments are at Reynolds
# numbers of about 10^4, where the wake is turbulent but the shedding is still
# periodic.
#
# Outputs (script19_cfd_background/):
#   fig1a_vorticity_rgba.png    vorticity, transparent where the flow is calm
#   fig1a_vorticity_white.png   the same on a white background
#   fig1a_guide.png             alignment guide: cylinder outline, vortex
#                               centres of the simulation (crosses) and the
#                               vortex markers of the current drawing (circles)
#   vortex_centres.json         vortex centres in pixels, sense of rotation,
#                               and the wavelength and row spacing in pixels
# =============================================================================

import os, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script19_cfd_background")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}")

# ---- geometry of the drawing (pixels)
IMG_W, IMG_H = 3719, 1395
CYL_X, CYL_Y, CYL_D = 650.0, 774.0, 185.0
DRAWN_UPPER = [(1505, 389), (2519, 389), (3533, 391)]      # current vortex markers (clockwise)
DRAWN_LOWER = [(2075, 1117), (2992, 1115)]                 # current vortex markers (counterclockwise)

# ---- lattice
D_CELLS = 50                                   # cylinder diameter in lattice cells
PX = CYL_D / D_CELLS                           # pixels per cell
NX, NY = int(round(IMG_W / PX)), int(round(IMG_H / PX))
CX, CY = CYL_X / PX, CYL_Y / PX
RE, U0 = 100.0, 0.1
NU = U0 * D_CELLS / RE
TAU = 3.0 * NU + 0.5
N_STEPS = 60000
SAVE_EVERY, N_SAVE = 100, 40                   # frames over the last 4000 steps
print(f"grid {NX} x {NY}, {PX:.2f} px per cell, Re {RE}, tau {TAU:.3f}")

# D2Q9
E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
ex, ey, w_ = jnp.array(E[:, 0], jnp.float32), jnp.array(E[:, 1], jnp.float32), jnp.array(W, jnp.float32)

yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")     # row = image y (downward)
solid = ((xx + 0.5 - CX) ** 2 + (yy + 0.5 - CY) ** 2) <= (D_CELLS / 2) ** 2
solid_j = jnp.array(solid)

def feq(rho, ux, uy):
    eu = ex[:, None, None] * ux + ey[:, None, None] * uy
    uu = ux * ux + uy * uy
    return w_[:, None, None] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uu)

def macro(f):
    rho = f.sum(0)
    ux = (f * ex[:, None, None]).sum(0) / rho
    uy = (f * ey[:, None, None]).sum(0) / rho
    return rho, ux, uy

@jax.jit
def step(f, kick):
    rho, ux, uy = macro(f)
    ux = jnp.where(solid_j, 0.0, ux); uy = jnp.where(solid_j, 0.0, uy)
    uy = uy + kick                                              # transient asymmetry to start shedding
    fpost = f - (f - feq(rho, ux, uy)) / TAU
    fpost = jnp.where(solid_j[None], f[OPP], fpost)             # bounce-back inside the cylinder
    fs = jnp.stack([jnp.roll(jnp.roll(fpost[i], int(E[i, 0]), axis=1), int(E[i, 1]), axis=0) for i in range(9)])
    # inlet (x = 0): equilibrium at the free-stream velocity; outlet: zero gradient
    fin = feq(jnp.ones((NY,)), jnp.full((NY,), U0), jnp.zeros((NY,)))
    fs = fs.at[:, :, 0].set(fin)
    fs = fs.at[:, :, -1].set(fs[:, :, -2])
    return fs

@jax.jit
def vorticity(f):
    rho, ux, uy = macro(f)
    # image coordinates: y downward. Vorticity with y upward: dv_up/dx - du/dy_up = -dv/dx + du/dy
    dvdx = (jnp.roll(uy, -1, 1) - jnp.roll(uy, 1, 1)) / 2
    dudy = (jnp.roll(ux, -1, 0) - jnp.roll(ux, 1, 0)) / 2
    return jnp.where(solid_j, 0.0, -dvdx + dudy)

t0 = time.time()
f = feq(jnp.ones((NY, NX)), jnp.full((NY, NX), U0) * (1 - solid_j), jnp.zeros((NY, NX)))
frames = []
for n in range(N_STEPS):
    kick = jnp.where((n < 2000) & (jnp.array(yy) < CY), 0.01, 0.0) * jnp.float32(n < 2000)
    f = step(f, kick)
    if n >= N_STEPS - SAVE_EVERY * N_SAVE and (n % SAVE_EVERY == 0):
        frames.append(np.asarray(vorticity(f)))
    if n % 10000 == 0:
        print(f"   step {n}, {time.time() - t0:.0f} s", flush=True)
rho, ux, uy = macro(f)
assert np.all(np.isfinite(np.asarray(ux))), "simulation became unstable"
print(f"simulation done in {time.time() - t0:.0f} s")

# ---- vortex centres in each frame, choose the frame that best matches the first drawn vortex
from scipy import ndimage
def centres(om):
    sm = ndimage.gaussian_filter(om, 2.0)
    out = []
    for sgn in (-1, 1):
        a = sgn * sm
        thr = 0.25 * a.max()
        lab, n = ndimage.label(a > thr)
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() < 20:
                continue
            wgt = a * m
            cy_, cx_ = ndimage.center_of_mass(wgt)
            if cx_ * PX < CYL_X + CYL_D:                         # skip the attached shear layers
                continue
            out.append(dict(x=float(cx_ * PX), y=float(cy_ * PX), sense="clockwise" if sgn < 0 else "counterclockwise"))
    return out
best, best_d = None, 1e9
for k, om in enumerate(frames):
    cs = [c for c in centres(om) if c["sense"] == "clockwise" and c["y"] < CYL_Y]
    if not cs:
        continue
    d = min(abs(c["x"] - DRAWN_UPPER[0][0]) for c in cs)
    if d < best_d:
        best, best_d = k, d
om = frames[best]
cs = centres(om)
up = sorted([c for c in cs if c["sense"] == "clockwise"], key=lambda c: c["x"])
lo = sorted([c for c in cs if c["sense"] == "counterclockwise"], key=lambda c: c["x"])
lam = float(np.median(np.diff([c["x"] for c in up]))) if len(up) > 1 else None
h = float(np.median([c["y"] for c in lo]) - np.median([c["y"] for c in up])) if up and lo else None
info = dict(reynolds=RE, frame=int(best), pixels_per_cell=PX, cylinder=dict(x=CYL_X, y=CYL_Y, d=CYL_D),
            wavelength_px=lam, wavelength_over_D=lam / CYL_D if lam else None, row_spacing_px=h,
            row_spacing_over_wavelength=h / lam if (h and lam) else None, vortices=cs)
json.dump(info, open(os.path.join(OUT_DIR, "vortex_centres.json"), "w"), indent=2)
print(f"chosen frame {best}; wavelength {lam:.0f} px = {lam / CYL_D:.2f} D; row spacing {h:.0f} px")

# ---- render at the pixel size of the drawing
from PIL import Image
lim = np.percentile(np.abs(om[~solid]), 99.5)
z = np.clip(om / lim, -1, 1)
zi = np.asarray(Image.fromarray(z.astype(np.float32)).resize((IMG_W, IMG_H), Image.BICUBIC))
zi = np.clip(zi, -1, 1)
Y, X = np.mgrid[0:IMG_H, 0:IMG_W]
inside = (X - CYL_X) ** 2 + (Y - CYL_Y) ** 2 <= (CYL_D / 2) ** 2
cmap = plt.get_cmap("RdBu_r")
rgba = cmap(0.5 + 0.5 * zi)
alpha = np.clip(np.abs(zi) / 0.6, 0, 1) ** 0.8 * 0.85
alpha[inside] = 0.0
rgba[..., 3] = alpha
Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT_DIR, "fig1a_vorticity_rgba.png"))
white = rgba[..., :3] * alpha[..., None] + (1 - alpha[..., None])
Image.fromarray((white * 255).astype(np.uint8), "RGB").save(os.path.join(OUT_DIR, "fig1a_vorticity_white.png"))

fig = plt.figure(figsize=(IMG_W / 300, IMG_H / 300), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(white); ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.axis("off")
ax.add_patch(plt.Circle((CYL_X, CYL_Y), CYL_D / 2, fill=False, color="k", lw=1.2))
for c in cs:
    ax.plot(c["x"], c["y"], "x", color="k", ms=10, mew=2)
    ax.text(c["x"] + 30, c["y"] - 30, f"({c['x']:.0f}, {c['y']:.0f})", fontsize=6)
for (x_, y_) in DRAWN_UPPER + DRAWN_LOWER:
    ax.add_patch(plt.Circle((x_, y_), 60, fill=False, color="0.4", lw=1.0, ls="--"))
fig.savefig(os.path.join(OUT_DIR, "fig1a_guide.png"), dpi=300)
plt.close(fig)
print(f"Outputs: {OUT_DIR}; runtime {time.time() - t0:.0f} s")
