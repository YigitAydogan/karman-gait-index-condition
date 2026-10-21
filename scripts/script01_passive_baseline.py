# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 01: Passive baseline. Equilibria, stability map and fates of a
#            compliant passive body holding station in a cylinder drag wake.
#
# Colab ready. Runtime: about 7 to 10 min on a standard CPU runtime (FULL mode),
# about 20 s in QUICK mode. Needs only numpy and matplotlib.
#
# Units (fixed for the whole project):
#   length  : wake wavelength lambda           (lambda = 1)
#   time    : vortex shedding period T         (T = 1)
#   speed   : vortex convection speed lambda/T (u_c = 1)
#   x       : streamwise distance downstream of the cylinder centre
#   forces  : per unit (body + added) mass
#
# Model (state s = [x, u, th, om]):
#   x' = u
#   u' = c_D V|V| - L sin(beta)                           (streamwise, + = downstream)
#   th'' + 2 zeta Omega th' + Omega^2 th = sigma L          (passive body/pitch mode)
#   V    = U_loc(x) - u                                   (relative streamwise flow)
#   w    = W(x) sin(2 pi (x - t))                         (transverse wake velocity)
#   beta = atan2(w, |V|),  alpha = beta - th
#   L    = c_L (V^2 + w^2) sin(alpha) cos(alpha) g(V)       (quasi-steady lift)
#   g(V) = 0.5 (1 + tanh(V / 0.05))                       (no lift in reversed flow)
#
# Questions answered by this script:
#   Q1. Where are the passive station-holding equilibria, and how do body
#       stiffness (Omega) and damping (zeta) move them?
#   Q2. Are they stable? (averaged slow dynamics AND full Floquet analysis)
#   Q3. What happens to a passive body released at the equilibrium?
#       (hold / collide with cylinder / escape downstream)
#   Q4. Decision gate for Claim 1 (can passive elasticity stabilise holding?)
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 0. Run configuration
# -----------------------------------------------------------------------------
MODE = "FULL"            # "FULL" for the paper grid, "QUICK" for a smoke test
SEED = 1
np.random.seed(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script01_passive_baseline")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Locked baseline parameters (shared by all later scripts)
# -----------------------------------------------------------------------------
PARAMS = dict(
    U_inf=1.0 / 0.87,  # free stream; vortices convect at about 0.87 U_inf
    delta0=1.30,       # velocity deficit at the cylinder (>1 means recirculation)
    l_delta=0.90,      # decay length of the deficit (lambda)
    W0=0.35,           # transverse wake velocity scale
    x_r=0.35,          # vortex formation (rise) length (lambda)
    l_w=1.50,          # vortex decay length (lambda)
    c_L=3.0,           # lift coefficient per unit mass (nondim)
    c_D=0.15,          # streamwise drag coefficient per unit mass (nondim)
    sigma=8.0,         # lift-to-pitch torque coupling (weathervane > 0)
    x_coll=0.15,       # cylinder rear surface (collision boundary)
    x_esc=4.5,         # escape boundary (left the wake)
)

if MODE == "FULL":
    N_RATIO, N_ZETA, N_X = 40, 30, 110
    SPP = 200               # RK4 steps per forcing period
    T_FATE = 300            # periods for direct simulations
else:
    N_RATIO, N_ZETA, N_X = 10, 8, 50
    SPP = 120
    T_FATE = 150

RATIO = np.geomspace(0.3, 3.5, N_RATIO)          # Omega / omega_wake
ZETA = np.geomspace(0.02, 1.0, N_ZETA)           # damping ratio
X_GRID = np.linspace(0.18, 3.6, N_X)             # candidate positions
OMEGA_W = 2.0 * np.pi

# -----------------------------------------------------------------------------
# 2. Wake fields and right-hand side (vectorised over a batch)
# -----------------------------------------------------------------------------
def W_env(x, p):
    return p["W0"] * (1.0 - np.exp(-(x / p["x_r"]) ** 2)) * np.exp(-x / p["l_w"])

def U_loc(x, p):
    return p["U_inf"] * (1.0 - p["delta0"] * np.exp(-x / p["l_delta"]))

def forces(t, x, u, th, p):
    """Return streamwise force F_x and lift L for a batch."""
    V = U_loc(x, p) - u
    w = W_env(x, p) * np.sin(OMEGA_W * (x - t))
    beta = np.arctan2(w, np.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    L = p["c_L"] * (V * V + w * w) * np.sin(alpha) * np.cos(alpha) * gate
    Fx = p["c_D"] * V * np.abs(V) - L * np.sin(beta)
    return Fx, L

def rhs_full(t, s, Om, ze, p, rigid):
    x, u, th, om = s
    Fx, L = forces(t, x, u, th, p)
    dom = np.where(rigid, 0.0, -2.0 * ze * Om * om - Om ** 2 * th + p["sigma"] * L)
    dth = np.where(rigid, 0.0, om)
    return np.array([u, Fx, dth, dom])

def rhs_frozen(t, s, x0, u0, Om, ze, p, rigid):
    """Body mode only; fish at x0 + u0 t (envelope frozen at x0, Doppler kept)."""
    th, om = s
    V_shift = u0
    xs = x0 + u0 * t
    V = U_loc(x0, p) - V_shift
    w = W_env(x0, p) * np.sin(OMEGA_W * (xs - t))
    beta = np.arctan2(w, np.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    L = p["c_L"] * (V * V + w * w) * np.sin(alpha) * np.cos(alpha) * gate
    Fx = p["c_D"] * V * np.abs(V) - L * np.sin(beta)
    dom = np.where(rigid, 0.0, -2.0 * ze * Om * om - Om ** 2 * th + p["sigma"] * L)
    dth = np.where(rigid, 0.0, om)
    return np.array([dth, dom]), Fx

def rk4_period(f, s0, t0, T, n, *args, return_mean_force=False):
    """Integrate a batch over one period T with n RK4 steps."""
    s = s0.copy()
    dt = T / n
    t = t0
    Fsum = 0.0
    def ev(tt, ss):
        r = f(tt, ss, *args)
        return r if isinstance(r, tuple) else (r, 0.0)
    for _ in range(n):
        k1, F1 = ev(t, s)
        k2, F2 = ev(t + dt / 2, s + dt / 2 * k1)
        k3, F3 = ev(t + dt / 2, s + dt / 2 * k2)
        k4, F4 = ev(t + dt, s + dt * k3)
        if return_mean_force:
            Fsum = Fsum + (F1 + 2 * F2 + 2 * F3 + F4) / 6.0
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    if return_mean_force:
        return s, Fsum / n
    return s

# -----------------------------------------------------------------------------
# 3. Period-averaged streamwise force F(x, u) via periodic body response
#    (Newton shooting on the 2D body-mode Poincare map, x frozen)
# -----------------------------------------------------------------------------
def averaged_force(x0, u0, Om, ze, p, rigid, n_newton=8, eps=1e-6):
    x0, u0, Om, ze, rigid = np.broadcast_arrays(x0, u0, Om, ze, rigid)
    shape = x0.shape
    x0, u0, Om, ze, rigid = [a.ravel().astype(float) if a.dtype != bool else a.ravel()
                             for a in (x0, u0, Om, ze, rigid)]
    T = 1.0 / (1.0 - u0)  # encounter period under drift u0
    n = SPP
    s = np.zeros((2, x0.size))
    args = (x0, u0, Om, ze, p, rigid)
    for _ in range(n_newton):
        s1 = rk4_period(rhs_frozen, s, 0.0, T, n, *args)
        r = s1 - s
        J = np.zeros((x0.size, 2, 2))
        for j in range(2):
            ds = np.zeros_like(s)
            ds[j] = eps
            s1p = rk4_period(rhs_frozen, s + ds, 0.0, T, n, *args)
            J[:, :, j] = ((s1p - s1) / eps).T
        A = J - np.eye(2)[None]
        A[rigid] = -np.eye(2)
        step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
        step = np.clip(step, -0.5, 0.5)
        s = s + step
        if np.nanmax(np.abs(r)) < 1e-9:
            break
    s_end, Fbar = rk4_period(rhs_frozen, s, 0.0, T, n, *args, return_mean_force=True)
    resid = np.abs(s_end - s).max(axis=0)
    amp = np.abs(s[0])  # body angle at phase 0 (diagnostic only)
    return Fbar.reshape(shape), resid.reshape(shape), s.reshape((2,) + shape)

# -----------------------------------------------------------------------------
# 4. Full 4D periodic orbits (station holding) and Floquet multipliers
# -----------------------------------------------------------------------------
def floquet_full(s_guess, Om, ze, p, rigid, n_newton=10, eps=1e-6):
    s = s_guess.copy()
    B = s.shape[1]
    args = (Om, ze, p, rigid)
    ok = np.ones(B, bool)
    for _ in range(n_newton):
        s1 = rk4_period(rhs_full, s, 0.0, 1.0, SPP, *args)
        r = s1 - s
        J = np.zeros((B, 4, 4))
        for j in range(4):
            ds = np.zeros_like(s)
            ds[j] = eps
            J[:, :, j] = ((rk4_period(rhs_full, s + ds, 0.0, 1.0, SPP, *args) - s1) / eps).T
        A = J - np.eye(4)[None]
        A[rigid, 2:, 2:] = -np.eye(2)
        A[rigid, 2:, :2] = 0.0
        A[rigid, :2, 2:] = 0.0
        try:
            step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
        except np.linalg.LinAlgError:
            step = np.zeros_like(s)
        step = np.clip(step, -0.2, 0.2)
        s = s + step
    s1 = rk4_period(rhs_full, s, 0.0, 1.0, SPP, *args)
    resid = np.abs(s1 - s).max(axis=0)
    J = np.zeros((B, 4, 4))
    for j in range(4):
        ds = np.zeros_like(s)
        ds[j] = eps
        J[:, :, j] = ((rk4_period(rhs_full, s + ds, 0.0, 1.0, SPP, *args) - s1) / eps).T
    J[rigid, 2:, :] = 0.0
    J[rigid, :, 2:] = 0.0
    mu = np.linalg.eigvals(J)
    return s, mu, resid

def classify_floquet(mu, tol=1e-6):
    """0 stable, 1 saddle (real mu>1), 2 oscillatory (complex |mu|>1), 3 flip (mu<-1)."""
    a = np.abs(mu)
    k = np.argmax(a, axis=1)
    m = mu[np.arange(len(mu)), k]
    cls = np.zeros(len(mu), int)
    unst = a.max(axis=1) > 1 + tol
    real = np.abs(m.imag) < 1e-8
    cls[unst & real & (m.real > 0)] = 1
    cls[unst & ~real] = 2
    cls[unst & real & (m.real < 0)] = 3
    return cls, a.max(axis=1)

# -----------------------------------------------------------------------------
# 5. Direct simulations from each equilibrium: fates
# -----------------------------------------------------------------------------
def fate_sim(s0, Om, ze, p, rigid, T_max, kick=0.01):
    """Perturb x by +kick and -kick. 0 hold, 1 collide, 2 escape. Returns time."""
    out = {}
    for sign in (+1, -1):
        s = s0.copy()
        s[0] = s[0] + sign * kick
        B = s.shape[1]
        fate = np.zeros(B, int)
        t_loss = np.full(B, np.nan)
        alive = np.ones(B, bool)
        x_star = s0[0].copy()
        x_hist_max = np.zeros(B)
        for k in range(T_max):
            s = rk4_period(rhs_full, s, 0.0, 1.0, SPP, Om, ze, p, rigid)
            coll = alive & (s[0] < p["x_coll"])
            esc = alive & (s[0] > p["x_esc"])
            fate[coll] = 1
            fate[esc] = 2
            t_loss[coll | esc] = k + 1
            alive &= ~(coll | esc)
            s[:, ~alive] = np.where(np.isfinite(s[:, ~alive]), s[:, ~alive], 0.0)
            x_hist_max = np.maximum(x_hist_max, np.where(alive, np.abs(s[0] - x_star), 0))
            if not alive.any():
                break
        out[sign] = dict(fate=fate, t_loss=t_loss, excursion=x_hist_max)
    return out

# -----------------------------------------------------------------------------
# 6. Run: landscape over (ratio, zeta, x) plus the rigid limit
# -----------------------------------------------------------------------------
t_start = time.time()
p = PARAMS
print(f"Mode {MODE}: grid {N_RATIO} x {N_ZETA} x {N_X}, {SPP} steps per period")

RR, ZZ = np.meshgrid(RATIO, ZETA, indexing="ij")
OM = RR * OMEGA_W
n_par = RR.size
om_flat = np.concatenate([OM.ravel(), [OMEGA_W]])     # last entry = rigid limit
ze_flat = np.concatenate([ZZ.ravel(), [1.0]])
rigid_flat = np.zeros(n_par + 1, bool)
rigid_flat[-1] = True

Xb = np.broadcast_to(X_GRID[None, :], (n_par + 1, N_X))
OMb = np.broadcast_to(om_flat[:, None], Xb.shape)
ZEb = np.broadcast_to(ze_flat[:, None], Xb.shape)
RGb = np.broadcast_to(rigid_flat[:, None], Xb.shape)

print("Step 1/5: averaged force landscape ...")
F_land, F_res, _ = averaged_force(Xb, 0.0, OMb, ZEb, p, RGb)
print(f"   done in {time.time() - t_start:.1f} s, max shooting residual {np.nanmax(F_res):.2e}")

# -- equilibria: sign changes of F along x, linear interpolation
eq_par, eq_x = [], []
for i in range(n_par + 1):
    F = F_land[i]
    sgn = np.sign(F)
    idx = np.where((sgn[:-1] != sgn[1:]) & np.isfinite(F[:-1]) & np.isfinite(F[1:]))[0]
    for j in idx:
        x0, x1, f0, f1 = X_GRID[j], X_GRID[j + 1], F[j], F[j + 1]
        eq_par.append(i)
        eq_x.append(x0 - f0 * (x1 - x0) / (f1 - f0))
eq_par = np.array(eq_par, int)
eq_x = np.array(eq_x)
print(f"   equilibria found: {len(eq_x)} (parameter sets with none: "
      f"{(n_par + 1) - len(np.unique(eq_par))})")

# -- slow-dynamics derivatives at each equilibrium
print("Step 2/5: averaged stability (F_x, F_u) ...")
hx, hu = 1e-3, 1e-3
Om_e, ze_e, rg_e = om_flat[eq_par], ze_flat[eq_par], rigid_flat[eq_par]
Xd = np.stack([eq_x + hx, eq_x - hx, eq_x, eq_x])
Ud = np.stack([0 * eq_x, 0 * eq_x, 0 * eq_x + hu, 0 * eq_x - hu])
Fd, _, body = averaged_force(Xd, Ud, Om_e[None], ze_e[None], p, rg_e[None])
F_x = (Fd[0] - Fd[1]) / (2 * hx)
F_u = (Fd[2] - Fd[3]) / (2 * hu)
avg_cls = np.where(F_x > 0, 1, np.where(F_u > 0, 2, 0))   # 0 stable, 1 saddle, 2 oscillatory
growth_avg = np.where(F_x > 0, np.sqrt(np.maximum(F_x, 0) + (F_u / 2) ** 2) + F_u / 2,
                      F_u / 2)

# -- full Floquet at each equilibrium
print("Step 3/5: full 4D periodic orbits and Floquet multipliers ...")
_, _, s_body = averaged_force(eq_x, 0.0, Om_e, ze_e, p, rg_e)
s_guess = np.stack([eq_x, np.zeros_like(eq_x), s_body[0], s_body[1]])
s_orb, mu, orb_res = floquet_full(s_guess, Om_e, ze_e, p, rg_e)
flo_cls, mu_max = classify_floquet(mu)
lam_max = np.log(mu_max)  # growth rate per forcing period
converged = (orb_res < 1e-6) & (np.abs(s_orb[0] - eq_x) < 0.1)
agree = (flo_cls[converged] == avg_cls[converged]).mean() if converged.any() else np.nan
print(f"   orbits converged: {converged.sum()}/{len(eq_x)}, "
      f"averaged vs Floquet class agreement: {agree:.3f}")

# -- direct simulations
print("Step 4/5: direct simulations (fates) ...")
fates = fate_sim(s_orb, Om_e, ze_e, p, rg_e, T_FATE)

# -- body response diagnostics at the equilibrium orbit
th_amp = np.zeros(len(eq_x))
th_phase = np.zeros(len(eq_x))
n_samp = SPP
s = s_orb.copy()
th_series = np.zeros((n_samp, len(eq_x)))
w_series = np.zeros((n_samp, len(eq_x)))
dt = 1.0 / n_samp
for k in range(n_samp):
    t = k * dt
    th_series[k] = s[2]
    w_series[k] = W_env(s[0], p) * np.sin(OMEGA_W * (s[0] - t))
    k1 = rhs_full(t, s, Om_e, ze_e, p, rg_e)
    k2 = rhs_full(t + dt / 2, s + dt / 2 * k1, Om_e, ze_e, p, rg_e)
    k3 = rhs_full(t + dt / 2, s + dt / 2 * k2, Om_e, ze_e, p, rg_e)
    k4 = rhs_full(t + dt, s + dt * k3, Om_e, ze_e, p, rg_e)
    s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
c_th = np.fft.rfft(th_series, axis=0)[1]
c_w = np.fft.rfft(w_series, axis=0)[1]
th_amp = 2 * np.abs(c_th) / n_samp
th_phase = np.angle(c_th / c_w)          # body lag relative to local transverse flow
spec = np.abs(np.fft.rfft(th_series - th_series.mean(0), axis=0)) ** 2
lock_ratio = np.argmax(spec[1:], axis=0) + 1  # harmonic index of dominant response

# -----------------------------------------------------------------------------
# 7. Assemble maps (one equilibrium per parameter set: the most downstream)
# -----------------------------------------------------------------------------
print("Step 5/5: maps, figures and decision gate ...")
n_eq_map = np.zeros(n_par + 1, int)
prim = np.full(n_par + 1, -1)
for i in range(n_par + 1):
    k = np.where(eq_par == i)[0]
    n_eq_map[i] = len(k)
    if len(k):
        prim[i] = k[np.argmax(eq_x[k])]
def to_map(v, fill=np.nan):
    """Map the primary equilibrium of each set; unconverged orbits are masked (NaN)."""
    m = np.full(n_par + 1, fill, dtype=float)
    ok = prim >= 0
    ok[ok] = converged[prim[ok]]
    m[ok] = v[prim[ok]]
    return m[:n_par].reshape(N_RATIO, N_ZETA), m[n_par]

xstar_map, xstar_rigid = to_map(eq_x)
cls_map, cls_rigid = to_map(flo_cls.astype(float))
lam_map, lam_rigid = to_map(lam_max)
amp_map, _ = to_map(th_amp)
fp, fm = fates[+1]["fate"], fates[-1]["fate"]
fate_code = np.where((fp == 0) & (fm == 0), 0, np.where((fp == 1) | (fm == 1), np.where((fp == 2) | (fm == 2), 3, 1), 2))
fate_map, fate_rigid = to_map(fate_code.astype(float))
tl = np.nanmin(np.stack([fates[+1]["t_loss"], fates[-1]["t_loss"]]), axis=0)
tloss_map, tloss_rigid = to_map(tl)

stable_any = np.any(flo_cls[converged] == 0)
frac_stable = np.mean(flo_cls[converged] == 0) if converged.any() else np.nan

# -----------------------------------------------------------------------------
# 8. Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)

# Fig 1: wake fields and force landscapes
xx = np.linspace(0.0, 3.6, 400)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(xx, U_loc(xx, p) / p["U_inf"], color="#2a5d9f", label=r"$U_{loc}/U_\infty$")
ax[0].plot(xx, W_env(xx, p) / p["U_inf"], color="#c2410c", label=r"$W/U_\infty$")
ax[0].axhline(0, color="0.7", lw=0.6)
ax[0].axvline(p["x_coll"], color="0.5", ls=":", lw=0.8)
ax[0].set_xlabel(r"$x/\lambda$")
ax[0].set_title("Prescribed drag wake", fontsize=9)
ax[0].legend(frameon=False)
sel = [np.argmin(np.abs(RATIO - r)) for r in (0.5, 0.95, 1.5, 3.0)]
iz = np.argmin(np.abs(ZETA - 0.1))
cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
for c, ir in zip(cols, sel):
    ax[1].plot(X_GRID, F_land[ir * N_ZETA + iz], color=c,
               label=rf"$\Omega/\omega_w$={RATIO[ir]:.2f}")
ax[1].plot(X_GRID, F_land[-1], color="k", ls="--", label="rigid")
ax[1].axhline(0, color="0.7", lw=0.6)
ax[1].set_xlabel(r"$x/\lambda$")
ax[1].set_ylabel(r"$\bar F_x$ (+ downstream)")
ax[1].set_title(r"Averaged force, $\zeta$=%.2f" % ZETA[iz], fontsize=9)
ax[1].legend(frameon=False, fontsize=7)
save(fig, "fig01_wake_and_landscape")

# Fig 2: maps over (ratio, zeta)
ext = [np.log10(RATIO[0]), np.log10(RATIO[-1]), np.log10(ZETA[0]), np.log10(ZETA[-1])]
def show(axh, M, title, cmap, norm=None, cbl=""):
    im = axh.imshow(M.T, origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=norm,
                    interpolation="nearest")
    axh.set_xlabel(r"$\log_{10}(\Omega/\omega_w)$")
    axh.set_ylabel(r"$\log_{10}\zeta$")
    axh.set_title(title, fontsize=9)
    cb = plt.colorbar(im, ax=axh, fraction=0.046)
    cb.set_label(cbl)
    return cb
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
show(ax[0, 0], xstar_map, r"Equilibrium position $x^*/\lambda$", "viridis")
cmap_c = ListedColormap(["#1a9850", "#d73027", "#fdae61", "#4575b4"])
cb = show(ax[0, 1], cls_map, "Floquet class", cmap_c, BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
cb.set_ticks([0, 1, 2, 3]); cb.set_ticklabels(["stable", "saddle", "osc.", "flip"])
show(ax[1, 0], lam_map, r"Growth rate $\ln|\mu|_{max}$ per period", "magma")
cmap_f = ListedColormap(["#1a9850", "#7b3294", "#e08214", "#878787"])
cb = show(ax[1, 1], fate_map, "Fate after release", cmap_f, BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
cb.set_ticks([0, 1, 2, 3]); cb.set_ticklabels(["hold", "collide", "escape", "both"])
for a in ax.ravel():
    a.axvline(0, color="w", lw=0.6, ls="--")
fig.tight_layout()
save(fig, "fig02_stability_maps")

# Fig 3: example trajectories (released from equilibrium)
ex_idx = [prim[sel[0] * N_ZETA + iz], prim[sel[1] * N_ZETA + iz], prim[-1]]
ex_idx = [k for k in ex_idx if k >= 0]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for c, k in zip(cols, ex_idx):
    for sign, ls in ((+1, "-"), (-1, "--")):
        s = s_orb[:, [k]].copy()
        s[0] += sign * 0.01
        traj = [s[0, 0]]
        tt = [0]
        for n in range(T_FATE):
            s = rk4_period(rhs_full, s, 0.0, 1.0, SPP, Om_e[[k]], ze_e[[k]], p, rg_e[[k]])
            traj.append(s[0, 0]); tt.append(n + 1)
            if s[0, 0] < p["x_coll"] or s[0, 0] > p["x_esc"]:
                break
        lab = ("rigid" if rg_e[k] else rf"$\Omega/\omega_w$={Om_e[k]/OMEGA_W:.2f}") if sign > 0 else None
        ax[0].plot(tt, traj, color=c, ls=ls, label=lab)
ax[0].axhline(p["x_coll"], color="0.5", ls=":")
ax[0].set_xlabel("time (shedding periods)")
ax[0].set_ylabel(r"$x/\lambda$")
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("Release at equilibrium, x kick of 0.01", fontsize=9)
ok = converged & ~rg_e
ax[1].scatter(F_x[ok], lam_max[ok], s=5, c=np.log10(ze_e[ok]), cmap="viridis")
ax[1].set_xlabel(r"averaged $\partial\bar F/\partial x$")
ax[1].set_ylabel(r"Floquet $\ln|\mu|_{max}$")
ax[1].axvline(0, color="0.7", lw=0.6)
ax[1].set_title("Averaged slope vs full Floquet", fontsize=9)
fig.tight_layout()
save(fig, "fig03_fates_and_agreement")

# -----------------------------------------------------------------------------
# 9. Save results and print decision gate
# -----------------------------------------------------------------------------
np.savez_compressed(
    os.path.join(OUT_DIR, "results_script01.npz"),
    RATIO=RATIO, ZETA=ZETA, X_GRID=X_GRID, F_land=F_land,
    eq_par=eq_par, eq_x=eq_x, F_x=F_x, F_u=F_u, avg_cls=avg_cls,
    s_orb=s_orb, mu=mu, flo_cls=flo_cls, lam_max=lam_max, converged=converged,
    fate_plus=fates[+1]["fate"], fate_minus=fates[-1]["fate"],
    tloss_plus=fates[+1]["t_loss"], tloss_minus=fates[-1]["t_loss"],
    th_amp=th_amp, th_phase=th_phase, lock_ratio=lock_ratio,
    xstar_map=xstar_map, cls_map=cls_map, lam_map=lam_map, fate_map=fate_map,
)
summary = dict(
    mode=MODE, params=PARAMS, grid=dict(n_ratio=N_RATIO, n_zeta=N_ZETA, n_x=N_X, spp=SPP),
    runtime_s=round(time.time() - t_start, 1),
    n_parameter_sets=int(n_par + 1),
    n_equilibria=int(len(eq_x)),
    max_equilibria_per_set=int(n_eq_map.max()),
    sets_without_equilibrium=int((n_eq_map == 0).sum()),
    orbits_converged=int(converged.sum()),
    averaged_vs_floquet_agreement=float(agree),
    fraction_stable_equilibria=float(frac_stable),
    floquet_class_counts={k: int((flo_cls[converged] == v).sum())
                          for k, v in (("stable", 0), ("saddle", 1), ("oscillatory", 2), ("flip", 3))},
    fate_counts={k: int((fate_code == v).sum())
                 for k, v in (("hold", 0), ("collide", 1), ("escape", 2), ("both", 3))},
    dominant_response_harmonic_all_one=bool(np.all(lock_ratio[converged] == 1)),
    rigid_limit=dict(x_star=float(xstar_rigid), floquet_class=float(cls_rigid),
                     growth_rate=float(lam_rigid), fate=float(fate_rigid),
                     t_loss=float(tloss_rigid)),
    growth_rate_range=[float(np.nanmin(lam_max[converged])), float(np.nanmax(lam_max[converged]))],
    median_time_to_loss=float(np.nanmedian(tl)),
)
with open(os.path.join(OUT_DIR, "summary_script01.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n================ SCRIPT 01 SUMMARY ================")
print(json.dumps(summary, indent=2))
print("\n================ DECISION GATE (Claim 1) ================")
if stable_any:
    print(f"PASS: {100*frac_stable:.1f}% of passive equilibria are Floquet stable.")
    print("Claim 1 stays as stated: elasticity/damping can stabilise passive holding.")
    print("Script 02 maps the stable region and its bifurcation boundaries.")
else:
    print("NO STABLE PASSIVE EQUILIBRIUM in the explored (stiffness, damping) plane.")
    print("Claim 1 becomes a negative baseline: passive bodies frequency-lock but cannot")
    print("hold position. Script 02 tests whether this survives wake-model variants;")
    print("if it does, the intrinsic rhythm (Scripts 03 to 04) is the core mechanism.")
print(f"\nOutputs written to: {OUT_DIR}")
print(f"Total runtime: {time.time() - t_start:.1f} s")
