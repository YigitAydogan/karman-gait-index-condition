# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 04: Free position + intrinsic rhythm. Test the entrainment versus
#            station-keeping prediction.
#
# Colab ready. Needs only numpy and matplotlib. Reads Script 03 results (to place
# the scan inside the 1:1 tongue) if available.
# Runtime: about 8 to 15 min (FULL), about 30 s (QUICK).
#
# Same units and equations as Scripts 01 to 03. State s = [x, u, th, om, zr, zi].
#   x' = u,  u' = c_D V|V| - L sin(beta)                 (V, w, beta at x; Doppler kept)
#   th'' + 2 zeta Omega th' + Omega^2 th = sigma L + g Re(z)
#   z' = [mu(1 + i c)(1 - |z|^2) + i omega_c] z + eps th
# Prescribed-rhythm limit (Newbolt): Re(z) = cos(omega_w t + phi0), no CPG dynamics.
#
# Reduced theory (to be tested). Linearise about station holding in
# (dx, du, dchi), chi = CPG phase relative to the wake at the fish:
#   s^3 + (k + gamma) s^2 + (gamma k - a_P) s - a_L k = 0
#   a_L : d<F>/dx after the CPG re-locks          (slow, relocked gradient)
#   a_P : d<F>/dx at frozen CPG phase             (fast, positional-phase gradient)
#   k   : phase-locking stiffness (Floquet, fixed position)
#   gamma: streamwise damping -d<F>/du at fixed x and CPG phase (measured)
# Prediction: if a_L > 0 and k > 0, holding is unstable (constant term < 0),
# whatever the positional well; with k = 0 (prescribed), stable iff a_P < 0.
#
# Control: a neural phase delay dphi between CPG and muscle (drive Re(z e^{-i dphi}))
# shifts the locked phase relative to the thrust phase, to test whether locking
# selects the unstable positional phase structurally or by accident.
#
# Parts:
#   A. Locked force landscapes <F>(x) for (omega_c, eps) inside the tongue
#   B. Prescribed-rhythm landscapes <F>(x; phi0) at three gains
#   C. Equilibria, a_L, a_P, k, gamma; reduced-theory growth rates
#   D. Full 6D Floquet at every equilibrium (4D for prescribed)
#   E. Direct simulations: time to loss vs predicted and Floquet rates
#   F. c = 1 robustness subset
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"            # "FULL" or "QUICK"
SEED = 5
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
S03_FILE = os.path.join(BASE_DIR, "script03_cpg_tongues", "results_script03.npz")
OUT_DIR = os.path.join(BASE_DIR, "script04_free_position")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OMEGA_W = 2.0 * np.pi
WAKE = dict(U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35,
            l_w=1.50, c_L=3.0, c_D=0.15, sigma=8.0)
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU = 2 * np.pi
G_LOCK = 8.0                    # default gain (make_P)
X_COLL, X_ESC = 0.15, 4.5

if MODE == "FULL":
    SPP = 100
    EPS_LIST = [2.0, 4.0, 6.0, 10.0, 15.0, 20.0]
    N_WC = 20
    G_LOCK_LIST = [8.0, 16.0]
    N_WC_WIDE = 40
    LOCK_RANGE_WIDE = (0.85, 1.45)
    DPHI_LIST = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    G_PRESC = [8.0, 16.0, 32.0]
    N_PHI = 16
    X_GRID = np.linspace(0.3, 2.2, 56)
    T_TR, T_MS = 80, 10
    T_SIM = 600
else:
    SPP = 80
    EPS_LIST = [6.0, 15.0]
    N_WC = 5
    G_LOCK_LIST = [8.0, 16.0]
    N_WC_WIDE = 12
    LOCK_RANGE_WIDE = (0.85, 1.45)
    DPHI_LIST = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    G_PRESC = [16.0, 32.0]
    N_PHI = 10
    X_GRID = np.linspace(0.3, 2.2, 30)
    T_TR, T_MS = 50, 6
    T_SIM = 200
LOCK_TOL = 0.3          # max strobe excursion (rad) for a 1:1 lock
MARGINAL = 2e-3

# -----------------------------------------------------------------------------
# 1. Model
# -----------------------------------------------------------------------------
def U_loc(x):
    return WAKE["U_inf"] * (1.0 - WAKE["delta0"] * np.exp(-x / WAKE["l_delta"]))

def W_env(x):
    return WAKE["W0"] * (1.0 - np.exp(-(x / WAKE["x_r"]) ** 2)) * np.exp(-x / WAKE["l_w"])

def rhs(t, s, P):
    """P: dict of batch arrays: wc, eps, c, g, presc (bool), phi0, freeze (bool)."""
    x, u, th, om, zr, zi = s
    V = U_loc(x) - u - P["ub"]
    w = W_env(x) * np.sin(OMEGA_W * (x - t))
    beta = np.arctan2(w, np.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    L = WAKE["c_L"] * (V * V + w * w) * np.sin(alpha) * np.cos(alpha) * gate
    Fx = WAKE["c_D"] * V * np.abs(V) - L * np.sin(beta)
    # CPG-to-muscle drive with neural phase delay dphi: Re(z exp(-i dphi))
    zdrv = zr * np.cos(P["dphi"]) + zi * np.sin(P["dphi"])
    drive = np.where(P["presc"], np.cos(OMEGA_W * t + P["phi0"] - P["dphi"]), zdrv)
    dom = -2 * ZE_B * OM_B * om - OM_B ** 2 * th + WAKE["sigma"] * L + P["g"] * drive
    r2 = zr * zr + zi * zi
    a = MU * (1.0 - r2)
    b = P["c"] * MU * (1.0 - r2) + P["wc"]
    dzr = np.where(P["presc"], 0.0, a * zr - b * zi + P["eps"] * th)
    dzi = np.where(P["presc"], 0.0, a * zi + b * zr)
    fr = P["freeze"]
    dx = np.where(fr, 0.0, u)
    du = np.where(fr, 0.0, Fx)
    return np.array([dx, du, om, dom, dzr, dzi]), Fx

def integrate(s, P, n_per, t0=0.0, spp=SPP, mean_last=0, strobe=False):
    dt = 1.0 / spp
    t = t0
    Fs = 0.0
    phs = []
    for n in range(n_per):
        for _ in range(spp):
            k1, F1 = rhs(t, s, P)
            k2, F2 = rhs(t + dt / 2, s + dt / 2 * k1, P)
            k3, F3 = rhs(t + dt / 2, s + dt / 2 * k2, P)
            k4, F4 = rhs(t + dt, s + dt * k3, P)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if n >= n_per - mean_last:
                Fs = Fs + (F1 + 2 * F2 + 2 * F3 + F4) / 6
            t += dt
        if strobe:
            phs.append(np.arctan2(s[5], s[4]))
    out = dict(s=s, t=t)
    if mean_last:
        out["F"] = Fs / (mean_last * spp)
    if strobe:
        ph = np.unwrap(np.array(phs), axis=0)
        out["span"] = ph.max(0) - ph.min(0)
        out["phase"] = ph[-1]
    return out

def make_P(B, **kw):
    P = dict(wc=np.full(B, OMEGA_W), eps=np.zeros(B), c=np.zeros(B), g=np.full(B, G_LOCK),
             presc=np.zeros(B, bool), phi0=np.zeros(B), freeze=np.zeros(B, bool),
             ub=np.zeros(B), dphi=np.zeros(B))
    for k, v in kw.items():
        P[k] = np.broadcast_to(np.asarray(v), (B,)).copy()
    return P

def init_state(x, phase=None):
    B = x.size
    ph = rng.uniform(0, 2 * np.pi, B) if phase is None else phase
    return np.array([x, np.zeros(B), np.zeros(B), np.zeros(B), np.cos(ph), np.sin(ph)], float)

def sub(P, idx):
    return {k: v[idx] for k, v in P.items()}

# -----------------------------------------------------------------------------
# 2. Scan design
# -----------------------------------------------------------------------------
t_start = time.time()
if os.path.exists(S03_FILE):
    d3 = np.load(S03_FILE)
    R3, E3, L3 = d3["RATIO"], d3["EPS"], d3["c0_locked11"]
    print("Using Script 03 tongue to place omega_c values for g = 8.")
else:
    R3 = E3 = L3 = None
    print("Script 03 results not found: using a default range around 1.")
pairs = []   # dicts: kind, eps, ratio (omega_c/omega_w), phi0, c, g
def add(kind_, eps_, ratio_, phi0_, c_, g_, dphi_=0.0):
    pairs.append(dict(kind=kind_, eps=eps_, ratio=ratio_, phi0=phi0_, c=c_, g=g_, dphi=dphi_))
for gl in G_LOCK_LIST:
    for e in EPS_LIST:
        if R3 is not None and gl == 8.0:
            j = np.argmin(np.abs(E3 - e))
            rl = R3[L3[:, j]]
            lo, hi = (rl.min() - 0.01, rl.max() + 0.01) if rl.size else (0.97, 1.08)
        else:
            lo, hi = LOCK_RANGE_WIDE            # tongue moves with gain: wide scan
        n_ = N_WC if gl == 8.0 else N_WC_WIDE
        dlist = [0.0] if gl == 8.0 else DPHI_LIST     # neural delay control at g = 16
        for dp in dlist:
            for r in np.linspace(lo, hi, n_):
                add("lock", e, r, 0.0, 0.0, gl, dp)
for g in G_PRESC:
    for ph in np.linspace(0, 2 * np.pi, N_PHI, endpoint=False):
        add("presc", 0.0, 1.0, ph, 0.0, g)
for r in np.linspace(0.95, 1.15, max(5, N_WC // 2)):   # c = 1 subset
    add("lock_c1", 10.0, r, 0.0, 1.0, 8.0)
NP = len(pairs)
kind = np.array([p["kind"] for p in pairs])
PA = {k: np.array([p[k] for p in pairs]) for k in ("eps", "ratio", "phi0", "c", "g", "dphi")}
print(f"Mode {MODE}: {NP} parameter pairs x {X_GRID.size} positions")

# -----------------------------------------------------------------------------
# 3. Part A/B: frozen-position landscapes
# -----------------------------------------------------------------------------
NX = X_GRID.size
Xb = np.tile(X_GRID, NP)
pi = np.repeat(np.arange(NP), NX)
is_presc = kind[pi] == "presc"
P = make_P(Xb.size,
           wc=(OMEGA_W * PA["ratio"])[pi],
           eps=PA["eps"][pi],
           c=PA["c"][pi],
           g=PA["g"][pi],
           presc=is_presc,
           phi0=PA["phi0"][pi],
           dphi=PA["dphi"][pi],
           freeze=True)
t0 = time.time()
s = init_state(Xb)
R = integrate(s, P, T_TR)
R2 = integrate(R["s"], P, T_MS, t0=R["t"], mean_last=T_MS, strobe=True)
F_land = R2["F"].reshape(NP, NX)
locked = np.where(is_presc, True, R2["span"] < LOCK_TOL).reshape(NP, NX)
F_land = np.where(locked, F_land, np.nan)
print(f"A/B landscapes: {time.time() - t0:.0f} s, locked fraction (CPG cases) "
      f"{locked[kind != 'presc'].mean():.3f}")

# -----------------------------------------------------------------------------
# 4. Part C: equilibria and reduced-theory ingredients
# -----------------------------------------------------------------------------
eq_pair, eq_x, aL_grid = [], [], []
for i in range(NP):
    Fi = F_land[i]
    for j in range(NX - 1):
        f0, f1 = Fi[j], Fi[j + 1]
        if np.isfinite(f0) and np.isfinite(f1) and np.sign(f0) != np.sign(f1):
            x0, x1 = X_GRID[j], X_GRID[j + 1]
            eq_pair.append(i)
            eq_x.append(x0 - f0 * (x1 - x0) / (f1 - f0))
            aL_grid.append((f1 - f0) / (x1 - x0))
eq_pair = np.array(eq_pair, int); eq_x = np.array(eq_x); aL_grid = np.array(aL_grid)
NE = eq_x.size
print(f"C: {NE} equilibria")
if NE == 0:
    raise SystemExit("No equilibria found: check parameters.")
ek = kind[eq_pair]
Pe = make_P(NE,
            wc=(OMEGA_W * PA["ratio"])[eq_pair],
            eps=PA["eps"][eq_pair],
            c=PA["c"][eq_pair],
            g=PA["g"][eq_pair],
            presc=ek == "presc",
            phi0=PA["phi0"][eq_pair],
            dphi=PA["dphi"][eq_pair],
            freeze=True)

# (i) converged frozen state at x*, locked CPG phase (at integer time)
h = 2e-3
Xd = np.concatenate([eq_x, eq_x + h, eq_x - h])
Pd = {k: np.concatenate([v] * 3) for k, v in Pe.items()}
Rd = integrate(init_state(Xd), Pd, T_TR)
Rd2 = integrate(Rd["s"], Pd, T_MS, t0=Rd["t"], mean_last=T_MS, strobe=True)
Fd = Rd2["F"].reshape(3, NE)
a_L = (Fd[1] - Fd[2]) / (2 * h)                    # relocked gradient
s_star = Rd2["s"][:, :NE]
t_int = np.round(Rd2["t"])
phi_lock = np.arctan2(s_star[5], s_star[4])        # CPG phase at integer time

# (ii) frozen-phase gradient a_P: prescribed rhythm at the locked phase
Pp = {k: np.concatenate([v] * 2) for k, v in Pe.items()}
Pp["presc"] = np.ones(2 * NE, bool)
Pp["phi0"] = np.concatenate([np.where(Pe["presc"], Pe["phi0"], phi_lock)] * 2)
Xp = np.concatenate([eq_x + h, eq_x - h])
Rp = integrate(init_state(Xp), Pp, 40)
Rp2 = integrate(Rp["s"], Pp, T_MS, t0=Rp["t"], mean_last=T_MS)
Fp = Rp2["F"].reshape(2, NE)
a_P = (Fp[0] - Fp[1]) / (2 * h)
a_L = np.where(Pe["presc"], a_P, a_L)              # prescribed: no relocking

# (ii-b) phase scan of a_P at each locked equilibrium: does a well exist at x*
#        for SOME phase, and did locking pick it?
N_SCAN = 12
lk_idx = np.where(~Pe["presc"])[0]
well_capable = np.zeros(NE, bool)
in_well = np.zeros(NE, bool)
dphi_to_well = np.full(NE, np.nan)
aP_scan = np.full((NE, N_SCAN), np.nan)
if lk_idx.size:
    offs = np.linspace(0, 2 * np.pi, N_SCAN, endpoint=False)
    nl = lk_idx.size
    rep = np.repeat(lk_idx, N_SCAN)
    phis = (phi_lock[lk_idx][:, None] + offs[None, :]).ravel()
    Ps_ = {k: np.concatenate([v[rep]] * 2) for k, v in Pe.items()}
    Ps_["presc"] = np.ones(2 * rep.size, bool)
    Ps_["phi0"] = np.concatenate([phis, phis])
    Xs_ = np.concatenate([eq_x[rep] + h, eq_x[rep] - h])
    Rs_ = integrate(init_state(Xs_), Ps_, 40)
    Rs2 = integrate(Rs_["s"], Ps_, T_MS, t0=Rs_["t"], mean_last=T_MS)
    Fs_ = Rs2["F"].reshape(2, nl, N_SCAN)
    aP_scan[lk_idx] = (Fs_[0] - Fs_[1]) / (2 * h)
    well_capable[lk_idx] = np.any(aP_scan[lk_idx] < 0, axis=1)
    in_well[lk_idx] = aP_scan[lk_idx, 0] < 0
    for i in lk_idx:
        neg = np.where(aP_scan[i] < 0)[0]
        if neg.size:
            d_ = np.minimum(offs[neg], 2 * np.pi - offs[neg])
            dphi_to_well[i] = d_.min()

# (iii) locking stiffness k: Floquet of the frozen 4D (body + CPG) map
def period_map(s, P, t0):
    return integrate(s, P, 1, t0=t0)["s"]
def jac(s, P, t0, dims, eps_fd=1e-6):
    s1 = period_map(s, P, t0)
    B = s.shape[1]
    J = np.zeros((B, len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = eps_fd
        J[:, :, a_] = ((period_map(s + ds, P, t0) - s1)[dims] / eps_fd).T
    return J, s1
J4, _ = jac(s_star, Pe, 0.0, [2, 3, 4, 5])
mu4 = np.sort(np.abs(np.linalg.eigvals(J4)), axis=1)[:, ::-1]
k_lock = np.where(Pe["presc"], 0.0, -np.log(mu4[:, 0]))
# (iv) streamwise damping gamma = -d<F>/du at fixed position and fixed CPG phase
#      (relative-flow change only; includes the speed dependence of lift)
hu = 2e-3
Pg = {k: np.concatenate([v] * 2) for k, v in Pe.items()}
Pg["presc"] = np.ones(2 * NE, bool)
Pg["phi0"] = np.concatenate([np.where(Pe["presc"], Pe["phi0"], phi_lock)] * 2)
Pg["ub"] = np.concatenate([np.full(NE, hu), np.full(NE, -hu)])
Rg = integrate(init_state(np.concatenate([eq_x, eq_x])), Pg, 40)
Rg2 = integrate(Rg["s"], Pg, T_MS, t0=Rg["t"], mean_last=T_MS)
Fg = Rg2["F"].reshape(2, NE)
gamma = -(Fg[0] - Fg[1]) / (2 * hu)
gamma_drag_only = 2 * WAKE["c_D"] * U_loc(eq_x)

# (iv) reduced cubic: s^3 + (k+g) s^2 + (g k - a_P) s - a_L k = 0
pred_rate = np.zeros(NE)
pred_stable = np.zeros(NE, bool)
for i in range(NE):
    k_, g_ = k_lock[i], gamma[i]
    if Pe["presc"][i]:
        roots = np.roots([1.0, g_, -a_P[i]])        # k = 0: 2D positional dynamics
    else:
        roots = np.roots([1.0, k_ + g_, g_ * k_ - a_P[i], -a_L[i] * k_])
    pred_rate[i] = roots.real.max()
    pred_stable[i] = pred_rate[i] < 0

# -----------------------------------------------------------------------------
# 5. Part D: full Floquet at each equilibrium (6D, or 4D for prescribed)
# -----------------------------------------------------------------------------
print("D: full Floquet ...")
Pf = dict(Pe); Pf["freeze"] = np.zeros(NE, bool)
s = s_star.copy()
s[0] = eq_x; s[1] = 0.0
dims_all = [0, 1, 2, 3, 4, 5]
for it in range(12):
    J, s1 = jac(s, Pf, 0.0, dims_all)
    r = s1 - s
    pres = Pf["presc"]
    r[4:, pres] = 0.0
    J[pres, 4:, :] = 0.0; J[pres, :, 4:] = 0.0
    A = J - np.eye(6)[None]
    A[pres, 4, 4] = A[pres, 5, 5] = -1.0
    step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
    s = s + np.clip(np.nan_to_num(step), -0.1, 0.1)
J, s1 = jac(s, Pf, 0.0, dims_all)
res = np.abs(s1 - s); res[4:, Pf["presc"]] = 0.0; res = res.max(0)
J[Pf["presc"], 4:, :] = 0.0; J[Pf["presc"], :, 4:] = 0.0
mu = np.linalg.eigvals(J)
amu = np.abs(mu)
flo_rate = np.log(amu.max(1))
flo_conv = (res < 1e-6) & (np.abs(s[0] - eq_x) < 0.1)
flo_stable = flo_rate < 0
lead = mu[np.arange(NE), np.argmax(amu, 1)]
flo_type = np.where(flo_stable, "stable", np.where(np.abs(lead.imag) > 1e-8, "oscillatory", "saddle"))
print(f"   converged {flo_conv.sum()}/{NE}")

# -----------------------------------------------------------------------------
# 6. Part E: direct simulations (time to loss)
# -----------------------------------------------------------------------------
print("E: direct simulations ...")
ok = flo_conv
idx = np.where(ok)[0]
s0 = s[:, idx].copy()
s0[0] += 0.01
Ps = sub(Pf, idx)
fate = np.zeros(idx.size, int); tloss = np.full(idx.size, np.nan)
alive = np.ones(idx.size, bool)
ss = s0
tt = 0.0
xs_hist = np.full((T_SIM, idx.size), np.nan)
for n in range(T_SIM):
    Rn = integrate(ss, Ps, 1, t0=tt)
    ss, tt = Rn["s"], Rn["t"]
    xs_hist[n] = np.where(alive, ss[0], np.nan)
    c_ = alive & (ss[0] < X_COLL); e_ = alive & (ss[0] > X_ESC)
    fate[c_] = 1; fate[e_] = 2; tloss[c_ | e_] = n + 1
    alive &= ~(c_ | e_)
    ss[:, ~alive] = np.array([1.0, 0, 0, 0, 1.0, 0])[:, None]
    if not alive.any():
        break

# -----------------------------------------------------------------------------
# 7. Tests of the prediction
# -----------------------------------------------------------------------------
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if a.size > 2 else np.nan

lockm = ok & (ek == "lock")
prem = ok & (ek == "presc")
c1m = ok & (ek == "lock_c1")
nonmarg = ok & (np.abs(flo_rate) > MARGINAL)
tests = dict(
    lock_n=int(lockm.sum()),
    lock_frac_aL_positive=float(np.mean(a_L[lockm] > 0)) if lockm.any() else np.nan,
    lock_frac_floquet_stable=float(np.mean(flo_stable[lockm])) if lockm.any() else np.nan,
    lock_frac_stable_when_aL_positive=float(np.mean(flo_stable[lockm & (a_L > 0)]))
        if (lockm & (a_L > 0)).any() else np.nan,
    prediction_stability_agreement=float(np.mean(pred_stable[nonmarg] == flo_stable[nonmarg]))
        if nonmarg.any() else np.nan,
    pred_vs_floquet_rate_spearman=spearman(pred_rate[ok], flo_rate[ok]) if ok.sum() > 2 else np.nan,
    pred_vs_floquet_rate_median_abs_err=float(np.median(np.abs(pred_rate[ok] - flo_rate[ok]))) if ok.any() else np.nan,
    lock_rate_vs_k_spearman=spearman(k_lock[lockm & (a_L > 0)], flo_rate[lockm & (a_L > 0)])
        if (lockm & (a_L > 0)).sum() > 2 else np.nan,
    lock_by_g={str(g): dict(
        n=int((lockm & (Pe["g"] == g)).sum()),
        frac_aP_negative=float(np.mean(a_P[lockm & (Pe["g"] == g)] < 0)) if (lockm & (Pe["g"] == g)).any() else None,
        frac_stable=float(np.mean(flo_stable[lockm & (Pe["g"] == g)])) if (lockm & (Pe["g"] == g)).any() else None)
        for g in G_LOCK_LIST},
    # decisive test: locked equilibria that WOULD sit in a well at frozen phase
    lock_with_well_n=int((lockm & (a_P < 0)).sum()),
    lock_with_well_frac_stable=float(np.mean(flo_stable[lockm & (a_P < 0)])) if (lockm & (a_P < 0)).any() else None,
    lock_with_well_rate_vs_k_spearman=spearman(k_lock[lockm & (a_P < 0) & (a_L > 0)],
                                               flo_rate[lockm & (a_P < 0) & (a_L > 0)])
        if (lockm & (a_P < 0) & (a_L > 0)).sum() > 2 else None,
    lock_with_well_median_rate=float(np.median(flo_rate[lockm & (a_P < 0)])) if (lockm & (a_P < 0)).any() else None,
    lock_by_dphi={f"{dp:.2f}": dict(
        n=int((lockm & (Pe["g"] == 16.0) & np.isclose(Pe["dphi"], dp)).sum()),
        frac_in_well=float(np.mean(a_P[lockm & (Pe["g"] == 16.0) & np.isclose(Pe["dphi"], dp)] < 0))
            if (lockm & (Pe["g"] == 16.0) & np.isclose(Pe["dphi"], dp)).any() else None,
        frac_stable=float(np.mean(flo_stable[lockm & (Pe["g"] == 16.0) & np.isclose(Pe["dphi"], dp)]))
            if (lockm & (Pe["g"] == 16.0) & np.isclose(Pe["dphi"], dp)).any() else None)
        for dp in DPHI_LIST},
    lock_well_capable_n=int((lockm & well_capable).sum()),
    lock_well_capable_frac=float(np.mean(well_capable[lockm])) if lockm.any() else None,
    lock_in_well_given_capable=float(np.mean(in_well[lockm & well_capable])) if (lockm & well_capable).any() else None,
    lock_median_phase_distance_to_well=float(np.nanmedian(dphi_to_well[lockm & well_capable]))
        if (lockm & well_capable).any() else None,
    presc_n=int(prem.sum()),
    presc_hopf_negative_gamma=int(np.sum(prem & (a_P < 0) & (gamma < 0))),
    presc_frac_stable=float(np.mean(flo_stable[prem])) if prem.any() else np.nan,
    presc_frac_stable_by_g={str(g): float(np.mean(flo_stable[prem & (Pe["g"] == g)]))
                            if (prem & (Pe["g"] == g)).any() else None for g in G_PRESC},
    presc_frac_aP_negative=float(np.mean(a_P[prem] < 0)) if prem.any() else np.nan,
    c1_n=int(c1m.sum()),
    c1_frac_floquet_stable=float(np.mean(flo_stable[c1m])) if c1m.any() else np.nan,
    sim_fates={"hold": int(np.sum(fate == 0)), "collide": int(np.sum(fate == 1)),
               "escape": int(np.sum(fate == 2))},
    sim_hold_frac_lock=float(np.mean(fate[ek[idx] == "lock"] == 0)) if (ek[idx] == "lock").any() else np.nan,
    sim_hold_frac_presc=float(np.mean(fate[ek[idx] == "presc"] == 0)) if (ek[idx] == "presc").any() else np.nan,
)
m_ = (fate > 0) & (flo_rate[idx] > MARGINAL)
if m_.sum() > 2:
    tests["tloss_vs_1_over_rate_spearman"] = spearman(1 / flo_rate[idx][m_], tloss[m_])

# -----------------------------------------------------------------------------
# 8. Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
cm = plt.cm.viridis
lock_pairs = [i for i in range(NP) if kind[i] == "lock" and PA["eps"][i] == EPS_LIST[len(EPS_LIST) // 2]
              and PA["g"][i] == G_LOCK_LIST[0]]
for n, i in enumerate(lock_pairs[:: max(1, len(lock_pairs) // 6)]):
    ax[0].plot(X_GRID, F_land[i], color=cm(n / 6), lw=0.8, label=rf"$\omega_c/\omega_w$={PA['ratio'][i]:.3f}")
ax[0].axhline(0, color="0.5", lw=0.6)
ax[0].set_xlabel(r"$x/\lambda$"); ax[0].set_ylabel(r"locked $\bar F_x$")
ax[0].set_title(rf"A: locked landscapes, $\varepsilon$={EPS_LIST[len(EPS_LIST)//2]}", fontsize=8)
ax[0].legend(frameon=False, fontsize=5)
gsel = G_PRESC[len(G_PRESC) // 2]
pp = [i for i in range(NP) if kind[i] == "presc" and PA["g"][i] == gsel]
for n, i in enumerate(pp[:: max(1, len(pp) // 6)]):
    ax[1].plot(X_GRID, F_land[i], color=plt.cm.twilight(n / 6), lw=0.8)
ax[1].axhline(0, color="0.5", lw=0.6)
ax[1].set_xlabel(r"$x/\lambda$"); ax[1].set_ylabel(r"prescribed $\bar F_x$")
ax[1].set_title(f"B: prescribed rhythm, g = {gsel}, varying phase", fontsize=8)
col = np.where(ek == "presc", "#7570b3", np.where(ek == "lock", "#d95f02", "#1b9e77"))
ax[2].scatter(pred_rate[ok], flo_rate[ok], s=8, c=col[ok])
lim = np.nanmax(np.abs(np.concatenate([pred_rate[ok], flo_rate[ok]]))) if ok.any() else 1
ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
ax[2].axhline(0, color="0.7", lw=0.5); ax[2].axvline(0, color="0.7", lw=0.5)
ax[2].set_xlabel("reduced-theory growth rate"); ax[2].set_ylabel("Floquet growth rate")
ax[2].set_title("C/D: theory vs Floquet\n(orange locked, purple prescribed, green c=1)", fontsize=8)
fig.tight_layout()
save(fig, "fig01_landscapes_theory")

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
m = lockm & (a_L > 0)
ax[0].scatter(k_lock[m], flo_rate[m], s=8, c=Pe["eps"][m], cmap="viridis")
ax[0].set_xlabel(r"locking stiffness $k$ (per period)"); ax[0].set_ylabel("Floquet growth rate")
ax[0].set_title(r"Locked, $a_L>0$: rate vs $k$", fontsize=8)
ax[0].axhline(0, color="0.7", lw=0.5)
for g, mk in zip(G_PRESC, ("o", "s", "^")):
    mm = prem & (Pe["g"] == g)
    ax[1].scatter(Pe["phi0"][mm], flo_rate[mm], s=10, marker=mk, label=f"g = {g}")
ax[1].axhline(0, color="0.7", lw=0.5)
ax[1].set_xlabel(r"prescribed phase $\phi_0$"); ax[1].set_ylabel("Floquet growth rate")
ax[1].set_title("Prescribed rhythm (k = 0)", fontsize=8)
ax[1].legend(frameon=False, fontsize=6)
for lab, col_, msk in (("locked", "#d95f02", ek[idx] == "lock"), ("prescribed", "#7570b3", ek[idx] == "presc")):
    sel_ = np.where(msk)[0][:8]
    for n, b in enumerate(sel_):
        ax[2].plot(xs_hist[:, b] - s[0, idx[b]], color=col_, lw=0.6, alpha=0.8, label=lab if n == 0 else None)
ax[2].set_xlabel("time (periods)"); ax[2].set_ylabel(r"$x - x^*$")
ax[2].set_ylim(-1.5, 1.5)
ax[2].set_title("E: release with 0.01 kick", fontsize=8)
ax[2].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig02_k_prescribed_sims")

# -----------------------------------------------------------------------------
# 9. Save and gate
# -----------------------------------------------------------------------------
summary = dict(mode=MODE, runtime_s=round(time.time() - t_start, 1), spp=SPP,
               eps_list=EPS_LIST, g_prescribed=G_PRESC, n_pairs=NP, n_equilibria=NE,
               floquet_converged=int(flo_conv.sum()), tests=tests,
               aL_range_lock=[float(np.nanmin(a_L[lockm])), float(np.nanmax(a_L[lockm]))] if lockm.any() else None,
               aP_range_presc=[float(np.nanmin(a_P[prem])), float(np.nanmax(a_P[prem]))] if prem.any() else None,
               gamma_range=[float(np.nanmin(gamma)), float(np.nanmax(gamma))],
               gamma_drag_only_range=[float(np.nanmin(gamma_drag_only)), float(np.nanmax(gamma_drag_only))],
               k_range=[float(np.nanmin(k_lock[lockm])), float(np.nanmax(k_lock[lockm]))] if lockm.any() else None,
               flo_types={t_: int(np.sum(flo_type[ok] == t_)) for t_ in ("stable", "saddle", "oscillatory")})
np.savez_compressed(os.path.join(OUT_DIR, "results_script04.npz"),
                    X_GRID=X_GRID, F_land=F_land, locked=locked, kind=kind,
                    pairs_eps=PA["eps"], pairs_ratio=PA["ratio"], pairs_g=PA["g"], pairs_dphi=PA["dphi"],
                    pairs_phi=PA["phi0"], eq_pair=eq_pair, eq_x=eq_x,
                    a_L=a_L, a_P=a_P, k_lock=k_lock, gamma=gamma, flo_type=flo_type, pred_rate=pred_rate,
                    flo_rate=flo_rate, flo_conv=flo_conv, aP_scan=aP_scan, phi_lock=phi_lock,
                    well_capable=well_capable, in_well=in_well, fate=fate, tloss=tloss, sim_idx=idx)
with open(os.path.join(OUT_DIR, "summary_script04.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n================ SCRIPT 04 SUMMARY ================")
print(json.dumps(summary, indent=2))
print("\n================ DECISION GATE (entrainment vs station keeping) ================")
t_ = tests
print(f"Locked rhythm: {100*t_['lock_frac_aL_positive']:.1f}% of equilibria have a_L > 0; "
      f"Floquet-stable when a_L > 0: {100*t_['lock_frac_stable_when_aL_positive']:.1f}% (prediction 0%).")
print(f"Prescribed rhythm (k = 0): Floquet-stable {100*t_['presc_frac_stable']:.1f}% "
      f"(prediction: stable iff a_P < 0; a_P < 0 in {100*t_['presc_frac_aP_negative']:.1f}%).")
print(f"Locked equilibria sitting in a frozen-phase well (a_P < 0): {t_['lock_with_well_n']}, "
      f"Floquet-stable fraction {t_['lock_with_well_frac_stable']} (prediction 0 when a_L > 0), "
      f"median growth rate {t_['lock_with_well_median_rate']}, rate vs k Spearman {t_['lock_with_well_rate_vs_k_spearman']}.")
print(f"Prescribed wells destabilised by negative damping (surging Hopf): {t_['presc_hopf_negative_gamma']}")
print(f"Reduced theory vs Floquet: stability agreement {t_['prediction_stability_agreement']:.3f}, "
      f"rate Spearman {t_['pred_vs_floquet_rate_spearman']:.3f}.")
theory_ok = t_["prediction_stability_agreement"] > 0.9
print(f"Locked equilibria where a well exists at SOME phase: {t_['lock_well_capable_n']} "
      f"({t_['lock_well_capable_frac']}); fraction where locking picked a well phase: "
      f"{t_['lock_in_well_given_capable']}; median phase distance to nearest well: "
      f"{t_['lock_median_phase_distance_to_well']} rad")
if not theory_ok:
    print("NOT CONFIRMED: reduced theory disagrees with Floquet; inspect before Script 05.")
elif t_["lock_with_well_n"] >= 5:
    # the theory forbids stability only when a_L > 0; a_L <= 0 cases are checked in 04b
    if t_["lock_frac_stable_when_aL_positive"] < 0.02:
        print("CONFIRMED (decisive): even locked fish sitting in a positional well are unstable;")
        print("locking erases positional memory. Proceed to Script 05 (positional feedback).")
    else:
        print("REFUTED: locked fish with a_L > 0 can be stable. Revisit the reduced theory.")
elif t_["lock_well_capable_n"] > 0:
    print(f"INCONCLUSIVE: wells exist at locked positions, but only {t_['lock_with_well_n']} locked")
    print("equilibria sit in one (need >= 5). Check lock_by_dphi: if a neural delay moves")
    print("fish into wells, extend that delay range; otherwise selection is structural.")
else:
    print("NO WELLS at locked positions for any phase at these gains: the locked test is")
    print("inconclusive. Increase g for locked runs before Script 05.")
print(f"\nOutputs written to: {OUT_DIR}")
