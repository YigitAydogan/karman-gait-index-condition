# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 02: Robustness of the passive result. Is the station-holding saddle of
#            Script 01 generic, or an artefact of one wake model and one set of
#            hydrodynamic coefficients?
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime: about 12 to 20 min (FULL), about 1 min (QUICK).
#
# Same units and equations as Script 01 (lambda = 1, T = 1, u_c = 1).
# Extensions in this script:
#   (a) all wake and force parameters become per-sample arrays
#   (b) four wake-shape families (exponential, plateau, algebraic, bump)
#   (c) optional lateral heave degree of freedom y (fish free to translate)
#
# Tests:
#   T1. One-at-a-time sweeps of every wake/force parameter around baseline
#   T2. Latin hypercube over all parameters, in four wake-shape families
#   T3. Lateral heave degree of freedom (stiffness and damping of y)
#   T4. Force-gradient decomposition at every equilibrium (drag vs thrust)
#       and position of x* relative to the peak of the vortex envelope
#   T5. Full Floquet spot check of the averaged classification
#   T6. Time-step convergence (100 / 200 / 400 steps per period)
#   T7. Time-to-loss scaling against the Floquet growth rate (uses Script 01)
#
# Decision gate:
#   Fraction of stable passive equilibria over the plausible parameter space,
#   and the conditions (if any) under which they occur.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 0. Run configuration
# -----------------------------------------------------------------------------
MODE = "FULL"            # "FULL" or "QUICK"
SEED = 2
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script02_passive_robustness")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
S01_FILE = os.path.join(BASE_DIR, "script01_passive_baseline", "results_script01.npz")

if MODE == "FULL":
    N_X = 80
    SPP = 200
    N_OAT = 12
    N_LHS = {0: 300, 1: 100, 2: 100, 3: 100}
    N_HEAVE_RATIO, N_HEAVE_ZETA = 10, 6
    N_SPOT = 200
else:
    N_X = 50
    SPP = 120
    N_OAT = 5
    N_LHS = {0: 30, 1: 12, 2: 12, 3: 12}
    N_HEAVE_RATIO, N_HEAVE_ZETA = 4, 3
    N_SPOT = 40

OMEGA_W = 2.0 * np.pi
X_GRID = np.linspace(0.16, 3.6, N_X)
N_FALLBACK_TRANS, N_FALLBACK_AVG = 60, 20   # periods, for non-period-1 responses
MARGINAL = 2e-3                             # |growth rate| below this = marginal

# Locked baseline (identical to Script 01)
BASE = dict(U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35,
            l_w=1.50, c_L=3.0, c_D=0.15, sigma=8.0, shape=0)
X_COLL, X_ESC = 0.15, 4.5

# Plausible ranges (log-uniform where marked)
RANGES = dict(
    U_inf=(1 / 0.95, 1 / 0.75, False),   # convection speed 0.75 to 0.95 U_inf
    delta0=(0.8, 1.6, False),           # <1: no reverse flow; >1: recirculation
    l_delta=(0.4, 2.0, True),
    W0=(0.15, 0.6, True),
    x_r=(0.15, 1.6, True),              # formation length; long at low Re
    l_w=(0.6, 4.0, True),
    c_L=(1.0, 6.0, True),
    c_D=(0.03, 0.6, True),
    sigma=(1.0, 20.0, True),
)
PNAMES = list(RANGES.keys())
SHAPE_NAMES = {0: "exponential", 1: "plateau", 2: "algebraic", 3: "bump"}

# Body configurations tested for every wake sample: (ratio Omega/omega_w, zeta)
BODIES = [("rigid", None, None), ("soft", 0.5, 0.1), ("resonant", 0.9, 0.05),
          ("resonant-damped", 0.9, 0.3), ("stiff", 1.5, 0.1), ("very stiff", 3.0, 0.1)]

# -----------------------------------------------------------------------------
# 1. Wake families and right-hand sides (all parameters are batch arrays)
# -----------------------------------------------------------------------------
def deficit(x, p):
    s = p["shape"]
    d_exp = p["delta0"] * np.exp(-x / p["l_delta"])
    d_pla = p["delta0"] * np.exp(-(x / p["l_delta"]) ** 2)
    d_alg = p["delta0"] / np.sqrt(1.0 + (x / p["l_delta"]) ** 2)
    return np.where(s == 1, d_pla, np.where(s == 2, d_alg, d_exp))

def W_env(x, p):
    s = p["shape"]
    rise = 1.0 - np.exp(-(x / p["x_r"]) ** 2)
    w_exp = p["W0"] * rise * np.exp(-x / p["l_w"])
    w_alg = p["W0"] * rise / (1.0 + x / p["l_w"])
    w_bmp = p["W0"] * np.exp(-((x - p["x_r"]) / (0.5 * p["l_w"])) ** 2)
    return np.where(s == 3, w_bmp, np.where(s == 2, w_alg, w_exp))

def U_loc(x, p):
    return p["U_inf"] * (1.0 - deficit(x, p))

def hydro(t, xe, xph, u, th, vy, p):
    """xe: position for envelopes, xph: position for wake phase.
    Returns streamwise force Fx, lift L, beta, and components."""
    V = U_loc(xe, p) - u
    w = W_env(xe, p) * np.sin(OMEGA_W * (xph - t))
    vrel = w - vy
    beta = np.arctan2(vrel, np.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    L = p["c_L"] * (V * V + vrel * vrel) * np.sin(alpha) * np.cos(alpha) * gate
    drag = p["c_D"] * V * np.abs(V)
    liftx = -L * np.sin(beta)
    return drag + liftx, L, beta, drag, liftx

def rhs_frozen(t, s, p):
    """Body (and optional heave) dynamics with x frozen at p['x0'], drift p['u0']."""
    th, om, y, vy = s
    xph = p["x0"] + p["u0"] * t
    Fx, L, beta, drag, liftx = hydro(t, p["x0"], xph, p["u0"], th, vy, p)
    rig = p["rigid"]
    dom = np.where(rig, 0.0, -2 * p["ze"] * p["Om"] * om - p["Om"] ** 2 * th + p["sigma"] * L)
    dth = np.where(rig, 0.0, om)
    hv = p["heave"]
    dvy = np.where(hv, -2 * p["ze_y"] * p["Om_y"] * vy - p["Om_y"] ** 2 * y + L * np.cos(beta), 0.0)
    dy = np.where(hv, vy, 0.0)
    return np.array([dth, dom, dy, dvy]), np.array([Fx, drag, liftx])

def rhs_full(t, s, p):
    x, u, th, om = s
    Fx, L, beta, _, _ = hydro(t, x, x, u, th, 0.0, p)
    rig = p["rigid"]
    dom = np.where(rig, 0.0, -2 * p["ze"] * p["Om"] * om - p["Om"] ** 2 * th + p["sigma"] * L)
    dth = np.where(rig, 0.0, om)
    return np.array([u, Fx, dth, dom]), None

def rk4_period(f, s0, T, n, p, mean_force=False):
    s = s0.copy()
    dt = T / n
    t = 0.0 * T
    Fs = 0.0
    for _ in range(n):
        k1, F1 = f(t, s, p)
        k2, F2 = f(t + dt / 2, s + dt / 2 * k1, p)
        k3, F3 = f(t + dt / 2, s + dt / 2 * k2, p)
        k4, F4 = f(t + dt, s + dt * k3, p)
        if mean_force:
            Fs = Fs + (F1 + 2 * F2 + 2 * F3 + F4) / 6.0
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return (s, Fs / n) if mean_force else s

# -----------------------------------------------------------------------------
# 2. Periodic body response and averaged forces (batch Newton shooting)
# -----------------------------------------------------------------------------
def averaged_force(p, spp=SPP, n_newton=8, eps=1e-6):
    """p: dict of 1D arrays (length B) incl. x0, u0. Returns mean [Fx, drag, liftx]."""
    B = p["x0"].size
    T = 1.0 / (1.0 - p["u0"])
    s = np.zeros((4, B))
    active = np.stack([~p["rigid"], ~p["rigid"], p["heave"], p["heave"]])
    for _ in range(n_newton):
        s1 = rk4_period(rhs_frozen, s, T, spp, p)
        r = np.where(active, s1 - s, 0.0)
        if np.nanmax(np.abs(r)) < 1e-9:
            break
        J = np.zeros((B, 4, 4))
        for j in range(4):
            if not active[j].any():
                continue
            ds = np.zeros_like(s)
            ds[j] = eps
            J[:, :, j] = ((rk4_period(rhs_frozen, s + ds, T, spp, p) - s1) / eps).T
        A = J - np.eye(4)[None]
        inact = ~active.T
        for j in range(4):
            A[inact[:, j], j, :] = 0.0
            A[inact[:, j], :, j] = 0.0
            A[inact[:, j], j, j] = -1.0
        step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
        s = s + np.clip(np.nan_to_num(step), -0.5, 0.5)
    s_end, Fm = rk4_period(rhs_frozen, s, T, spp, p, mean_force=True)
    resid = np.abs(np.where(active, s_end - s, 0.0)).max(axis=0)
    # Fallback: where no period-1 response was found (large near-resonant
    # responses, possibly subharmonic), use a brute-force long time average.
    bad = ~(resid < 1e-6)
    if bad.any():
        idx = np.where(bad)[0]
        pb = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape == (B,) else v)
              for k, v in p.items()}
        Tb = T[idx]
        sb = np.zeros((4, idx.size))
        for _ in range(N_FALLBACK_TRANS):
            sb = rk4_period(rhs_frozen, sb, Tb, spp, pb)
        Fa = 0.0
        for _ in range(N_FALLBACK_AVG):
            sb, Fk = rk4_period(rhs_frozen, sb, Tb, spp, pb, mean_force=True)
            Fa = Fa + Fk / N_FALLBACK_AVG
        Fm[:, idx] = Fa
    return Fm, resid, s

def tile(p, n):
    return {k: np.repeat(v, n) for k, v in p.items()}

def make_sets(rows):
    """rows: list of dicts with wake/force params plus body spec. Returns arrays."""
    keys = PNAMES + ["shape", "Om", "ze", "rigid", "heave", "Om_y", "ze_y"]
    out = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(r[k])
    out = {k: np.array(v) for k, v in out.items()}
    out["shape"] = out["shape"].astype(int)
    out["rigid"] = out["rigid"].astype(bool)
    out["heave"] = out["heave"].astype(bool)
    return out

def body_rows(wake, heave=None):
    rows = []
    for name, ratio, ze in BODIES:
        r = dict(wake)
        r["rigid"] = ratio is None
        r["Om"] = OMEGA_W * (ratio if ratio else 1.0)
        r["ze"] = ze if ze else 1.0
        r["heave"] = False if heave is None else True
        r["Om_y"] = 1.0 if heave is None else heave[0]
        r["ze_y"] = 1.0 if heave is None else heave[1]
        r["body"] = name
        rows.append(r)
    return rows

def analyse_sets(P, label, spp=SPP):
    """Landscape -> equilibria -> averaged derivatives and decomposition."""
    t0 = time.time()
    n_set = P["Om"].size
    Pl = tile(P, N_X)
    Pl["x0"] = np.tile(X_GRID, n_set)
    Pl["u0"] = np.zeros_like(Pl["x0"])
    Fm, res, _ = averaged_force(Pl, spp=spp)
    F = Fm[0].reshape(n_set, N_X)
    eq_set, eq_x = [], []
    for i in range(n_set):
        Fi = F[i]
        sg = np.sign(Fi)
        idx = np.where((sg[:-1] != sg[1:]) & np.isfinite(Fi[:-1]) & np.isfinite(Fi[1:]))[0]
        for j in idx:
            x0, x1, f0, f1 = X_GRID[j], X_GRID[j + 1], Fi[j], Fi[j + 1]
            eq_set.append(i)
            eq_x.append(x0 - f0 * (x1 - x0) / (f1 - f0))
    eq_set = np.array(eq_set, int)
    eq_x = np.array(eq_x)
    nonper = ~(res < 1e-6)
    out = dict(F=F, eq_set=eq_set, eq_x=eq_x, n_set=n_set,
               n_nonperiodic_points=int(nonper.sum()),
               frac_nonperiodic_points=float(nonper.mean()))
    if len(eq_x) == 0:
        out.update(F_x=np.array([]), F_u=np.array([]), dD=np.array([]), dL=np.array([]),
                   cls=np.array([], int))
        return out
    h = 1e-3
    Pe = {k: v[eq_set] for k, v in P.items()}
    Pd = {k: np.concatenate([v] * 4) for k, v in Pe.items()}
    n = len(eq_x)
    Pd["x0"] = np.concatenate([eq_x + h, eq_x - h, eq_x, eq_x])
    Pd["u0"] = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, h), np.full(n, -h)])
    Fd, _, _ = averaged_force(Pd, spp=spp)
    Fd = Fd.reshape(3, 4, n)
    F_x = (Fd[0, 0] - Fd[0, 1]) / (2 * h)
    F_u = (Fd[0, 2] - Fd[0, 3]) / (2 * h)
    dD = (Fd[1, 0] - Fd[1, 1]) / (2 * h)       # drag gradient
    dL = (Fd[2, 0] - Fd[2, 1]) / (2 * h)       # lift (thrust) gradient
    cls = np.where(F_x > 0, 1, np.where(F_u > 0, 2, 0))   # 0 stable, 1 saddle, 2 oscillatory
    out.update(F_x=F_x, F_u=F_u, dD=dD, dL=dL, cls=cls)
    print(f"   [{label}] {n_set} sets, {n} equilibria, "
          f"stable {np.sum(cls == 0)}, saddle {np.sum(cls == 1)}, osc {np.sum(cls == 2)}, "
          f"{time.time() - t0:.0f} s")
    return out

def w_peak(P):
    xf = np.linspace(0.0, 4.0, 801)
    n = P["Om"].size
    Pf = tile(P, xf.size)
    Wf = W_env(np.tile(xf, n), Pf).reshape(n, xf.size)
    return xf[np.argmax(Wf, axis=1)]

# -----------------------------------------------------------------------------
# 3. Full Floquet (for spot checks)
# -----------------------------------------------------------------------------
def floquet_full(P, s_guess, spp=SPP, n_newton=10, eps=1e-6):
    s = s_guess.copy()
    B = s.shape[1]
    rig = P["rigid"]
    for _ in range(n_newton):
        s1 = rk4_period(rhs_full, s, 1.0, spp, P)
        r = s1 - s
        r[2:, rig] = 0.0
        J = np.zeros((B, 4, 4))
        for j in range(4):
            ds = np.zeros_like(s)
            ds[j] = eps
            J[:, :, j] = ((rk4_period(rhs_full, s + ds, 1.0, spp, P) - s1) / eps).T
        A = J - np.eye(4)[None]
        A[rig, 2:, :] = 0.0
        A[rig, :, 2:] = 0.0
        A[rig, 2, 2] = A[rig, 3, 3] = -1.0
        step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
        s = s + np.clip(np.nan_to_num(step), -0.2, 0.2)
    s1 = rk4_period(rhs_full, s, 1.0, spp, P)
    res = np.abs(s1 - s)
    res[2:, rig] = 0.0
    res = res.max(axis=0)
    J = np.zeros((B, 4, 4))
    for j in range(4):
        ds = np.zeros_like(s)
        ds[j] = eps
        J[:, :, j] = ((rk4_period(rhs_full, s + ds, 1.0, spp, P) - s1) / eps).T
    J[rig, 2:, :] = 0.0
    J[rig, :, 2:] = 0.0
    mu = np.linalg.eigvals(J)
    a = np.abs(mu)
    k = np.argmax(a, axis=1)
    m = mu[np.arange(B), k]
    cls = np.where(a.max(1) <= 1 + 1e-6, 0, np.where(np.abs(m.imag) > 1e-8, 2, 1))
    return s, cls, np.log(a.max(1)), res

# -----------------------------------------------------------------------------
# 4. Build experiments
# -----------------------------------------------------------------------------
t_start = time.time()
print(f"Mode {MODE}")

# T1: one-at-a-time sweeps
oat_rows, oat_meta = [], []
for name in PNAMES:
    lo, hi, lg = RANGES[name]
    vals = np.geomspace(lo, hi, N_OAT) if lg else np.linspace(lo, hi, N_OAT)
    for v in vals:
        wk = dict(BASE)
        wk[name] = v
        for r in body_rows(wk):
            oat_rows.append(r)
            oat_meta.append((name, v, r["body"]))
P_oat = make_sets(oat_rows)

# T2: Latin hypercube in four shape families
def lhs(n):
    u = (rng.permuted(np.tile(np.arange(n), (len(PNAMES), 1)), axis=1).T
         + rng.random((n, len(PNAMES)))) / n
    out = []
    for i in range(n):
        d = {}
        for j, k in enumerate(PNAMES):
            lo, hi, lg = RANGES[k]
            d[k] = np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo))) if lg \
                else lo + u[i, j] * (hi - lo)
        out.append(d)
    return out
lhs_rows, lhs_meta = [], []
for shp, n in N_LHS.items():
    for k, wk in enumerate(lhs(n)):
        wk["shape"] = shp
        for r in body_rows(wk):
            lhs_rows.append(r)
            lhs_meta.append((shp, k, r["body"]))
P_lhs = make_sets(lhs_rows)

# T3: lateral heave on the baseline wake
heave_rows, heave_meta = [], []
for ry in (0.3, 1.0, 3.0):
    for zy in (0.1, 0.5):
        for ratio in np.geomspace(0.3, 3.5, N_HEAVE_RATIO):
            for ze in np.geomspace(0.03, 1.0, N_HEAVE_ZETA):
                r = dict(BASE)
                r.update(rigid=False, Om=OMEGA_W * ratio, ze=ze, heave=True,
                         Om_y=OMEGA_W * ry, ze_y=zy, body="heave")
                heave_rows.append(r)
                heave_meta.append((ry, zy, ratio, ze))
P_hv = make_sets(heave_rows)

# -----------------------------------------------------------------------------
# 5. Run
# -----------------------------------------------------------------------------
print("T1: one-at-a-time sweeps ...")
R_oat = analyse_sets(P_oat, "OAT")
print("T2: Latin hypercube (4 wake families) ...")
R_lhs = analyse_sets(P_lhs, "LHS")
print("T3: lateral heave ...")
R_hv = analyse_sets(P_hv, "heave")

# T4 diagnostics: x* relative to vortex-envelope peak
xpk_lhs = w_peak(P_lhs)[R_lhs["eq_set"]] if len(R_lhs["eq_x"]) else np.array([])
up_of_peak = R_lhs["eq_x"] < xpk_lhs

# T5: Floquet spot check (random LHS equilibria + every stable one, rigid excluded
#     from heave since the full model here has no heave)
print("T5: Floquet spot check ...")
ne = len(R_lhs["eq_x"])
pick = np.unique(np.concatenate([
    rng.choice(ne, size=min(N_SPOT, ne), replace=False) if ne else np.array([], int),
    np.where(R_lhs["cls"] == 0)[0][:N_SPOT]])).astype(int)
if len(pick):
    Pe = {k: v[R_lhs["eq_set"][pick]] for k, v in P_lhs.items()}
    Pb = dict(Pe)
    Pb["x0"] = R_lhs["eq_x"][pick]
    Pb["u0"] = np.zeros(len(pick))
    _, _, sb = averaged_force(Pb)
    s_guess = np.stack([R_lhs["eq_x"][pick], np.zeros(len(pick)), sb[0], sb[1]])
    s_orb, cls_f, lam_f, res_f = floquet_full(Pe, s_guess)
    conv = (res_f < 1e-6) & (np.abs(s_orb[0] - R_lhs["eq_x"][pick]) < 0.1)
    nonmarg = conv & (np.abs(lam_f) > MARGINAL)
    stab_f = cls_f == 0
    stab_a = R_lhs["cls"][pick] == 0
    spot_agree = float(np.mean(stab_f[nonmarg] == stab_a[nonmarg])) if nonmarg.any() else np.nan
    type_agree = float(np.mean(cls_f[nonmarg] == R_lhs["cls"][pick][nonmarg])) if nonmarg.any() else np.nan
    n_marg = int((conv & ~nonmarg).sum())
else:
    conv = np.array([], bool); spot_agree = type_agree = np.nan; n_marg = 0
    cls_f = lam_f = np.array([])
print(f"   converged {conv.sum()}/{len(pick)}, marginal {n_marg}, "
      f"stability agreement {spot_agree:.3f}, type agreement {type_agree:.3f}")

# T6: time-step convergence on the baseline bodies
print("T6: time-step convergence ...")
P_ref = make_sets(body_rows(dict(BASE)))
conv_tab = {}
for spp in (100, 200, 400):
    R = analyse_sets(P_ref, f"spp={spp}", spp=spp)
    conv_tab[spp] = dict(x=R["eq_x"].tolist(), F_x=R["F_x"].tolist())
x200 = np.array(conv_tab[200]["x"]); x400 = np.array(conv_tab[400]["x"])
fx200 = np.array(conv_tab[200]["F_x"]); fx400 = np.array(conv_tab[400]["F_x"])
step_err = dict(max_dx=float(np.max(np.abs(x200 - x400))) if len(x200) == len(x400) else np.nan,
                max_rel_dFx=float(np.max(np.abs(fx200 - fx400) / np.abs(fx400)))
                if len(fx200) == len(fx400) else np.nan)

# T7: time to loss vs growth rate (Script 01 results)
t7 = None
if os.path.exists(S01_FILE):
    d = np.load(S01_FILE)
    ok = d["converged"] & np.isfinite(d["tloss_plus"]) & np.isfinite(d["tloss_minus"])
    lam = d["lam_max"][ok]
    tl = np.minimum(d["tloss_plus"][ok], d["tloss_minus"][ok])
    A = np.vstack([1 / lam, np.ones_like(lam)]).T
    coef, *_ = np.linalg.lstsq(A, tl, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((tl - pred) ** 2) / np.sum((tl - tl.mean()) ** 2)
    t7 = dict(slope=float(coef[0]), intercept=float(coef[1]), r2=float(r2),
              implied_amplification=float(np.exp(coef[0])), lam=lam, tl=tl)
    print(f"T7: t_loss = {coef[0]:.2f}/lambda + {coef[1]:.2f}, R^2 = {r2:.3f}")
else:
    print("T7: Script 01 results not found, skipped.")

# -----------------------------------------------------------------------------
# 6. Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)
CLS_COL = {0: "#1a9850", 1: "#d73027", 2: "#fdae61"}
body_names = [b[0] for b in BODIES]

# Fig 1: OAT sweeps, max F_x over bodies per parameter value (saddle if > 0)
fig, axs = plt.subplots(3, 3, figsize=(7.4, 6.4))
oat_meta_arr = np.array(oat_meta, dtype=object)
for ax, name in zip(axs.ravel(), PNAMES):
    for b_i, b in enumerate(body_names):
        sel_sets = [i for i, m in enumerate(oat_meta) if m[0] == name and m[2] == b]
        xs, ys, cs = [], [], []
        for i in sel_sets:
            ks = np.where(R_oat["eq_set"] == i)[0]
            for k in ks:
                xs.append(oat_meta[i][1]); ys.append(R_oat["F_x"][k]); cs.append(R_oat["cls"][k])
        ax.plot(xs, ys, "-", lw=0.6, color=plt.cm.viridis(b_i / (len(body_names) - 1)), alpha=0.8)
        ax.scatter(xs, ys, s=6, c=[CLS_COL[c] for c in cs], zorder=3)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axvline(BASE[name], color="0.7", ls=":", lw=0.8)
    if RANGES[name][2]:
        ax.set_xscale("log")
    ax.set_title(name, fontsize=8)
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\partial\bar F/\partial x$ at $x^*$")
fig.suptitle("T1 one-at-a-time sweeps (green: stable, red: saddle)", fontsize=9)
fig.tight_layout()
save(fig, "fig01_oat_sweeps")

# Fig 2: LHS gradient decomposition and position relative to vortex peak
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
if len(R_lhs["eq_x"]):
    shp = P_lhs["shape"][R_lhs["eq_set"]]
    for s_ in SHAPE_NAMES:
        m = shp == s_
        ax[0].scatter(R_lhs["dD"][m], -R_lhs["dL"][m], s=4, alpha=0.6, label=SHAPE_NAMES[s_])
    lim = np.nanpercentile(np.abs(np.concatenate([R_lhs["dD"], R_lhs["dL"]])), 99)
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.7)
    ax[0].set_xlabel("drag gradient")
    ax[0].set_ylabel("thrust gradient")
    ax[0].set_title("Stable only above dashed line", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6, markerscale=2)
    rel = R_lhs["eq_x"] - xpk_lhs
    for c in (1, 0, 2):
        m = R_lhs["cls"] == c
        ax[1].hist(rel[m], bins=40, color=CLS_COL[c], alpha=0.7,
                   label={0: "stable", 1: "saddle", 2: "osc."}[c])
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_xlabel(r"$x^* - x_{peak}$ of vortex envelope")
    ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title("Equilibrium vs vortex peak", fontsize=8)
    frac_by = []
    for s_ in SHAPE_NAMES:
        m = shp == s_
        frac_by.append(np.mean(R_lhs["cls"][m] == 0) if m.any() else np.nan)
    ax[2].bar(list(SHAPE_NAMES.values()), np.array(frac_by) * 100, color="#4575b4")
    ax[2].set_ylabel("% stable equilibria")
    ax[2].set_title("T2 by wake family", fontsize=8)
    ax[2].tick_params(axis="x", rotation=30)
fig.tight_layout()
save(fig, "fig02_lhs_decomposition")

# Fig 3: heave map + Floquet spot check + time to loss
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
hm = np.array(heave_meta)
if len(R_hv["eq_x"]):
    fx = R_hv["F_x"]
    sets = R_hv["eq_set"]
    for ry, mk in zip((0.3, 1.0, 3.0), ("o", "s", "^")):
        m = hm[sets, 0] == ry
        ax[0].scatter(hm[sets[m], 2], fx[m], s=6, marker=mk, label=rf"$\Omega_y/\omega_w$={ry}",
                      c=[CLS_COL[c] for c in R_hv["cls"][m]])
    ax[0].set_xscale("log")
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_xlabel(r"$\Omega/\omega_w$")
    ax[0].set_ylabel(r"$\partial\bar F/\partial x$")
    ax[0].set_title("T3 lateral heave", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6)
if conv.any():
    ax[1].scatter(R_lhs["F_x"][pick][conv], lam_f[conv], s=5, c="#4575b4")
    ax[1].axvline(0, color="0.6", lw=0.6); ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].set_xlabel(r"averaged $\partial\bar F/\partial x$")
    ax[1].set_ylabel(r"Floquet $\ln|\mu|_{max}$")
    ax[1].set_title(f"T5 agreement {spot_agree:.3f}", fontsize=8)
if t7 is not None:
    ax[2].scatter(1 / t7["lam"], t7["tl"], s=4, c="#7b3294")
    xx = np.linspace((1 / t7["lam"]).min(), (1 / t7["lam"]).max(), 50)
    ax[2].plot(xx, t7["slope"] * xx + t7["intercept"], "k--", lw=0.8)
    ax[2].set_xlabel(r"$1/\ln|\mu|_{max}$ (periods)")
    ax[2].set_ylabel("time to loss (periods)")
    ax[2].set_title(f"T7 $R^2$={t7['r2']:.3f}", fontsize=8)
fig.tight_layout()
save(fig, "fig03_heave_floquet_tloss")

# -----------------------------------------------------------------------------
# 7. Save and decision gate
# -----------------------------------------------------------------------------
def frac(c, v):
    return float(np.mean(c == v)) if len(c) else float("nan")

lhs_stable_idx = np.where(R_lhs["cls"] == 0)[0]
stable_cases = []
for k in lhs_stable_idx[:50]:
    i = R_lhs["eq_set"][k]
    stable_cases.append(dict(shape=SHAPE_NAMES[int(P_lhs["shape"][i])],
                             body=lhs_meta[i][2], x_star=float(R_lhs["eq_x"][k]),
                             x_peak=float(xpk_lhs[k]),
                             **{n: float(P_lhs[n][i]) for n in PNAMES}))
no_eq = {SHAPE_NAMES[s_]: int(np.sum([(P_lhs["shape"][i] == s_) and (i not in set(R_lhs["eq_set"]))
                                      for i in range(R_lhs["n_set"])]))
         for s_ in SHAPE_NAMES}

summary = dict(
    mode=MODE, runtime_s=round(time.time() - t_start, 1), spp=SPP, n_x=N_X,
    ranges={k: list(v[:2]) for k, v in RANGES.items()},
    T1_oat=dict(n_sets=int(R_oat["n_set"]), n_eq=int(len(R_oat["eq_x"])),
                frac_stable=frac(R_oat["cls"], 0), frac_saddle=frac(R_oat["cls"], 1),
                frac_osc=frac(R_oat["cls"], 2)),
    T2_lhs=dict(n_sets=int(R_lhs["n_set"]), n_eq=int(len(R_lhs["eq_x"])),
                frac_stable=frac(R_lhs["cls"], 0), frac_saddle=frac(R_lhs["cls"], 1),
                frac_osc=frac(R_lhs["cls"], 2),
                frac_stable_by_family={SHAPE_NAMES[s_]: frac(R_lhs["cls"][P_lhs["shape"][R_lhs["eq_set"]] == s_], 0)
                                       for s_ in SHAPE_NAMES},
                sets_without_equilibrium_by_family=no_eq,
                max_eq_per_set=int(np.bincount(R_lhs["eq_set"]).max()) if len(R_lhs["eq_set"]) else 0,
                frac_eq_upstream_of_vortex_peak=float(np.mean(up_of_peak)) if len(up_of_peak) else np.nan,
                frac_stable_given_upstream=frac(R_lhs["cls"][up_of_peak], 0),
                frac_stable_given_downstream=frac(R_lhs["cls"][~up_of_peak], 0)),
    T3_heave=dict(n_sets=int(R_hv["n_set"]), n_eq=int(len(R_hv["eq_x"])),
                  frac_stable=frac(R_hv["cls"], 0), frac_osc=frac(R_hv["cls"], 2)),
    T5_floquet=dict(n_checked=int(len(pick)), n_converged=int(conv.sum()),
                    n_marginal_excluded=n_marg, marginal_threshold=MARGINAL,
                    stability_agreement=spot_agree, type_agreement=type_agree),
    T6_convergence=step_err,
    T7_time_to_loss=None if t7 is None else {k: t7[k] for k in ("slope", "intercept", "r2", "implied_amplification")},
    nonperiodic_body_response_points={k: R[f] for k, R in (("oat", R_oat), ("lhs", R_lhs), ("heave", R_hv))
                                      for f in ("frac_nonperiodic_points",)},
    stable_cases_first_50=stable_cases,
)
np.savez_compressed(os.path.join(OUT_DIR, "results_script02.npz"),
                    **{f"oat_{k}": v for k, v in R_oat.items() if isinstance(v, np.ndarray)},
                    **{f"lhs_{k}": v for k, v in R_lhs.items() if isinstance(v, np.ndarray)},
                    **{f"hv_{k}": v for k, v in R_hv.items() if isinstance(v, np.ndarray)},
                    **{f"Plhs_{k}": v for k, v in P_lhs.items()},
                    **{f"Poat_{k}": v for k, v in P_oat.items()},
                    **{f"Phv_{k}": v for k, v in P_hv.items()},
                    lhs_xpeak=xpk_lhs)
with open(os.path.join(OUT_DIR, "summary_script02.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n================ SCRIPT 02 SUMMARY ================")
print(json.dumps({k: v for k, v in summary.items() if k != "stable_cases_first_50"}, indent=2))
print(f"\nStable LHS cases listed in summary file: {len(stable_cases)} (first 50)")

print("\n================ DECISION GATE (passive instability generic?) ================")
fs = summary["T2_lhs"]["frac_stable"]
fo = summary["T2_lhs"]["frac_osc"]
fh = summary["T3_heave"]["frac_stable"]
if (fs < 0.01) and (fh < 0.01):
    print(f"GENERIC: {100*fs:.2f}% stable over the plausible space, {100*fh:.2f}% with heave.")
    print("Passive positional instability is robust. Proceed to Script 03 (intrinsic rhythm)")
    print("with the negative passive result as the paper's baseline.")
elif fs < 0.10:
    print(f"MOSTLY GENERIC: {100*fs:.2f}% stable. Inspect the stable cases (summary file):")
    print("if they share a condition (e.g. long formation length, x* upstream of the vortex")
    print("peak), that condition becomes a testable Reynolds-number prediction (Eldredge vs")
    print("Beal). Proceed to Script 03.")
else:
    print(f"NOT GENERIC: {100*fs:.2f}% stable. Passive stabilisation is common in some wakes.")
    print("Claim 1 is revived; map the stable region before adding the rhythm.")
if fo > 0:
    print(f"Note: {100*fo:.2f}% oscillatory (Hopf-type) instabilities found; inspect them.")
print(f"\nOutputs written to: {OUT_DIR}")
print(f"Total runtime: {time.time() - t_start:.1f} s")
