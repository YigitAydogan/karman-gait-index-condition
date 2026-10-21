# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 09: Model-structure robustness of station holding and the index
#            condition.
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime (FULL): about 3 to 5 min per variant, 36 variants, about 2 to 3 h.
# One run does everything; no settings to change. Every finished variant is
# checkpointed to Drive. If Colab disconnects, run the script again: finished
# variants are loaded from the checkpoints and only the rest are computed.
# The final summary, records and figures always cover all variants.
#
# Scripts 06 to 08 varied parameters inside one model structure. A reviewer can
# argue the result is an artefact of that structure (quasi-steady lift, a single
# pitch mode, Stuart-Landau CPG, proprioceptive feedback only, one wake shape).
# Here the structure itself is changed:
#
#   STRUCT  baseline (generic code, reproduces Scripts 05b/06)
#           linear lift + lever arm (L = c_L |V| w_te, w_te the trailing-edge
#             lateral relative velocity, includes body rotation rate)
#           added reactive thrust (Newbolt-type c_T w_te^2 on top of lift tilt)
#           reactive thrust only (lift-tilt thrust removed: mechanism ablation)
#           lift lag (first-order, tau = 0.08 T)
#           lift lag + surge added mass (m_a = 0.5)
#           pitch + heave body: lateral mode at 0.25, 1 and 2 omega_w
#             (near-free, resonant, stiff), zeta_y = 0.3
#           two-mode body (second pitch mode at 2.2 Omega, weight 0.5)
#           van der Pol CPG (mu_v = 1, relaxation-type rhythm)
#           lateral line + proprioception (CPG input eps (theta + w))
#           lateral line only (CPG input eps w, no proprioception)
#   WAKE    plateau deficit, algebraic decay, bump vortex envelope
#           (the Script 02 families)
#   LHS     20-point joint Latin hypercube over body, CPG and wake parameters
#
# For each variant:
#   1. grid search as in Script 06 (prescribed-rhythm seeds, eps rows 0.25 to
#      4.75, wide omega_c scan on the first row, then a window that follows
#      the tilted tongue), full Floquet stability. A second seed family (free
#      relaxation from the same seeds for N_RELAX periods) is added because
#      some structures (van der Pol rhythm) are poorly seeded by a sinusoid.
#   2. the Script 08 homotopy index test on every stable orbit (cap) and a
#      sample of unstable orbits: continue x' = eta u, u' = eta F from eta = 1
#      to 0.01, identify the fixed-position branch, det(I - M_int), a_L,
#      index = sign det(I - M_int) * sign(-a_L)
#
# Gate: 0 non-marginal index violations in every variant. Holding existence
# and the tongue location are reported per variant (no gate: some structures
# may legitimately lose holding; that is a result, not a failure).
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
SEED = 13
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script09_model_structure")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

if MODE == "FULL":
    SPP = 100
    N_PHI_SEED, N_X_SEED = 12, 50
    EPS_ROWS = np.arange(0.25, 4.76, 0.5)
    WIDE = np.linspace(0.70, 1.50, 81)
    N_LOCAL, HALF_W = 21, 0.05
    MAX_STABLE, N_UNST = 120, 40
else:
    SPP = 80
    N_PHI_SEED, N_X_SEED = 6, 30
    EPS_ROWS = np.array([0.75, 1.75])
    WIDE = np.linspace(0.95, 1.15, 9)
    N_LOCAL, HALF_W = 7, 0.04
    MAX_STABLE, N_UNST = 4, 3
DPHI_SEEDS = [0.0, np.pi / 2]
N_RELAX = 40 if MODE == "FULL" else 30
N_LHS = 20
ETAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
OFF = np.linspace(-0.02, 0.02, 7)
RATIO_PROBE = np.array([0.003, 0.006, 0.01, 0.02, 0.035, 0.05])
AL_MARGIN = 0.02
NEWTON_TOL = 1e-8
OMEGA_W = 2.0 * np.pi

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
BASE_CFG = dict(
    # wake (Script 01 baseline) and shape family (0 exp, 1 plateau, 2 algebraic, 3 bump)
    U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35, l_w=1.50, shape=0,
    # hydrodynamics
    c_L=3.0, c_D=0.15, sigma=8.0, lift="qs", tilt=1.0, c_T=0.0, LT=0.5, tau=0.0, m_a=0.0,
    # body
    body_ratio=1.2, zeta=0.2, heave=False, Om_y_ratio=1.0, ze_y=0.3,
    mode2=False, Om2_ratio=2.2, w2=0.5,
    # CPG
    cpg="sl", mu=2 * np.pi, c=0.0, mu_v=1.0, g=16.0, wp=1.0, eLL=0.0,
)

STRUCT = [
    ("baseline", {}),
    ("linear lift + lever arm", dict(lift="linear", LT=0.5)),
    ("added reactive thrust", dict(c_T=0.1)),
    ("reactive thrust only", dict(tilt=0.0, c_T=0.3)),
    ("lift lag", dict(tau=0.08)),
    ("lift lag + added mass", dict(tau=0.08, m_a=0.5)),
    ("pitch + free heave", dict(heave=True, Om_y_ratio=0.25)),
    ("pitch + resonant heave", dict(heave=True, Om_y_ratio=1.0)),
    ("pitch + stiff heave", dict(heave=True, Om_y_ratio=2.0)),
    ("two-mode body", dict(mode2=True)),
    ("van der Pol CPG", dict(cpg="vdp")),
    ("lateral line + proprio", dict(eLL=1.0)),
    ("lateral line only", dict(eLL=1.0, wp=0.0)),
]
WAKES = [
    ("plateau deficit", dict(shape=1)),
    ("algebraic decay", dict(shape=2)),
    ("bump envelope", dict(shape=3)),
]
LHS_RANGES = dict(body_ratio=(1.0, 1.4), zeta=(0.1, 0.3), g=(12.0, 24.0), mu=(np.pi, 4 * np.pi),
                  W0=(0.25, 0.45), c_D=(0.10, 0.25), c_L=(2.5, 3.5), sigma=(6.0, 10.0),
                  x_r=(0.30, 0.45), l_w=(1.2, 1.8), delta0=(1.2, 1.4))

def lhs_variants(n):
    keys = list(LHS_RANGES)
    u = np.zeros((n, len(keys)))
    for j in range(len(keys)):
        u[:, j] = (rng.permutation(n) + rng.uniform(0, 1, n)) / n
    out = []
    for i in range(n):
        over = {k: float(LHS_RANGES[k][0] + u[i, j] * (LHS_RANGES[k][1] - LHS_RANGES[k][0]))
                for j, k in enumerate(keys)}
        out.append((f"LHS {i:02d}", over))
    return out

GROUPS = [("STRUCT", STRUCT), ("WAKE", WAKES), ("LHS", lhs_variants(N_LHS))]
if MODE == "QUICK":
    GROUPS = [(g_, v_[:1]) for g_, v_ in GROUPS]
VARIANTS = [(g_, n_, o_) for g_, v_ in GROUPS for n_, o_ in v_]

CFG = dict(BASE_CFG)
LAY = {}

def apply_cfg(over):
    """Set the active configuration and build the state layout."""
    global CFG, LAY, D, IZ, INT_DIMS, AMP
    CFG = dict(BASE_CFG); CFG.update(over)
    names = ["x", "u", "th", "om"]
    if CFG["heave"]:
        names += ["y", "vy"]
    if CFG["mode2"]:
        names += ["th2", "om2"]
    if CFG["tau"] > 0:
        names += ["Lf"]
    names += ["zr", "zi"]
    LAY = {n: i for i, n in enumerate(names)}
    D = len(names)
    IZ = LAY["zr"]
    INT_DIMS = list(range(2, D))
    AMP = 2.0 if CFG["cpg"] == "vdp" else 1.0
    if CFG["cpg"] == "vdp":
        build_vdp_table()
    return names

VDP_TAB = None

def build_vdp_table():
    """Free van der Pol cycle (unit omega) parametrised by the phase of the
    fundamental of z_r, so that seeds match the prescribed rhythm cos(phi)."""
    global VDP_TAB, AMP
    mv = CFG["mu_v"]
    f = lambda z: np.array([-z[1], z[0] + mv * (1.0 - z[0] ** 2) * z[1]])
    z = np.array([2.0, 0.0]); dt = 2 * np.pi / 4000
    for n in range(30 * 4000):
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    pts, crossings = [z.copy()], []
    n = 0
    while len(crossings) < 2:
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        zn = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        n += 1
        if z[1] < 0 <= zn[1]:
            crossings.append(n)
        z = zn
        pts.append(z.copy())
    pts = np.array(pts)[crossings[0]:crossings[1]]
    th = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)
    c1 = np.mean(pts[:, 0] * np.exp(-1j * th))
    AMP = 2 * np.abs(c1)                      # fundamental amplitude of z_r
    psi = np.angle(c1)                        # z_r ~ AMP cos(th + psi)
    VDP_TAB = (np.angle(np.exp(1j * (th + psi))), pts[:, 0], pts[:, 1])

def cpg_seed(phase):
    if CFG["cpg"] == "vdp":
        a, zr, zi = VDP_TAB
        ph = np.angle(np.exp(1j * phase))
        return (np.interp(ph, a, zr, period=2 * np.pi), np.interp(ph, a, zi, period=2 * np.pi))
    return AMP * np.cos(phase), AMP * np.sin(phase)

# -----------------------------------------------------------------------------
# 2. Model (generic state layout)
# -----------------------------------------------------------------------------
def deficit(x):
    s, d0, ld = CFG["shape"], CFG["delta0"], CFG["l_delta"]
    if s == 1:
        return d0 * np.exp(-(x / ld) ** 2)
    if s == 2:
        return d0 / np.sqrt(1.0 + (x / ld) ** 2)
    return d0 * np.exp(-x / ld)

def U_loc(x):
    return CFG["U_inf"] * (1.0 - deficit(x))

def W_env(x):
    s, W0, xr, lw = CFG["shape"], CFG["W0"], CFG["x_r"], CFG["l_w"]
    rise = 1.0 - np.exp(-(x / xr) ** 2)
    if s == 2:
        return W0 * rise / (1.0 + x / lw)
    if s == 3:
        return W0 * np.exp(-((x - xr) / (0.5 * lw)) ** 2)
    return W0 * rise * np.exp(-x / lw)

def rhs(t, s, P):
    C = CFG
    x, u, th, om = s[0], s[1], s[2], s[3]
    zr, zi = s[IZ], s[IZ + 1]
    d = np.zeros_like(s)
    V = U_loc(x) - u
    w = W_env(x) * np.sin(OMEGA_W * (x - t))
    vy = s[LAY["vy"]] if C["heave"] else 0.0
    if C["mode2"]:
        th_e = th + C["w2"] * s[LAY["th2"]]
        om_e = om + C["w2"] * s[LAY["om2"]]
    else:
        th_e, om_e = th, om
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    beta = np.arctan2(w - vy, np.abs(V))
    # lateral velocity of the trailing edge relative to the flow (lever arm LT)
    w_te = w - vy - np.abs(V) * th_e - C["LT"] * om_e
    if C["lift"] == "qs":
        alpha = beta - th_e
        Lq = C["c_L"] * (V * V + (w - vy) ** 2) * np.sin(alpha) * np.cos(alpha) * gate
    else:
        Lq = C["c_L"] * np.abs(V) * w_te * gate
    if C["tau"] > 0:
        Lf = s[LAY["Lf"]]
        d[LAY["Lf"]] = (Lq - Lf) / C["tau"]
        L = Lf
    else:
        L = Lq
    Fx = C["c_D"] * V * np.abs(V) - C["tilt"] * L * np.sin(beta) - C["c_T"] * w_te * w_te * gate
    # CPG drive with neural delay dphi
    zdrv = (zr * np.cos(P["dphi"]) + zi * np.sin(P["dphi"])) / AMP
    drive = np.where(P["presc"], np.cos(OMEGA_W * t + P["phi0"] - P["dphi"]), zdrv)
    OmB, zeB = C["body_ratio"] * OMEGA_W, C["zeta"]
    d[2] = om
    d[3] = -2 * zeB * OmB * om - OmB ** 2 * th + C["sigma"] * L + P["g"] * drive
    if C["mode2"]:
        Om2 = C["Om2_ratio"] * OmB
        d[LAY["th2"]] = s[LAY["om2"]]
        d[LAY["om2"]] = (-2 * zeB * Om2 * s[LAY["om2"]] - Om2 ** 2 * s[LAY["th2"]]
                         + C["sigma"] * L + P["g"] * drive)
    if C["heave"]:
        Omy = C["Om_y_ratio"] * OMEGA_W
        d[LAY["y"]] = vy
        d[LAY["vy"]] = -2 * C["ze_y"] * Omy * vy - Omy ** 2 * s[LAY["y"]] + L * np.cos(beta)
    inp = P["eps"] * (C["wp"] * th_e + C["eLL"] * w)
    if C["cpg"] == "sl":
        r2 = zr * zr + zi * zi
        a = C["mu"] * (1.0 - r2)
        b = P["c"] * C["mu"] * (1.0 - r2) + P["wc"]
        dzr = a * zr - b * zi + inp
        dzi = a * zi + b * zr
    else:
        dzr = -P["wc"] * zi + inp
        dzi = P["wc"] * (zr + C["mu_v"] * (1.0 - zr * zr) * zi)
    d[IZ] = np.where(P["presc"], 0.0, dzr)
    d[IZ + 1] = np.where(P["presc"], 0.0, dzi)
    fr = P["freeze"]
    eta = P.get("eta", 1.0)
    d[0] = np.where(fr, 0.0, u * eta)
    d[1] = np.where(fr, 0.0, Fx * eta / (1.0 + C["m_a"]))
    return d, Fx

def integrate(s, P, n_per, t0=0.0, spp=None, mean_last=0):
    spp = SPP if spp is None else spp
    dt = 1.0 / spp
    t = t0
    Fs = 0.0
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
    out = dict(s=s, t=t)
    if mean_last:
        out["F"] = Fs / (mean_last * spp)
    return out

def make_P(B, **kw):
    P = dict(wc=np.full(B, OMEGA_W), eps=np.zeros(B), c=np.zeros(B), g=np.full(B, CFG["g"]),
             presc=np.zeros(B, bool), phi0=np.zeros(B), freeze=np.zeros(B, bool), dphi=np.zeros(B))
    for k, v in kw.items():
        P[k] = np.broadcast_to(np.asarray(v), (B,)).copy()
    return P

def PF(B, eps, ratio, dphi, freeze=False, eta=None):
    P = make_P(B, wc=OMEGA_W * np.asarray(ratio), eps=eps, c=CFG["c"], dphi=dphi, freeze=freeze)
    if eta is not None:
        P["eta"] = np.full(B, eta)
    return P

def init_state(x, phase):
    s = np.zeros((D, x.size))
    s[0] = x
    s[IZ], s[IZ + 1] = cpg_seed(phase)
    return s

def cpg_phase(s):
    return np.arctan2(s[IZ + 1], s[IZ])

def valid(s):
    r = np.hypot(s[IZ], s[IZ + 1])
    return (s[0] > 0.2) & (s[0] < 3.5) & (r > 0.3 * AMP) & (r < 2.5 * AMP) & np.isfinite(s).all(0)

def pmap(s, P):
    return integrate(s, P, 1, t0=0.0)["s"]

def jac(s, P, dims, h=1e-6):
    s1 = pmap(s, P)
    J = np.zeros((s.shape[1], len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = h
        J[:, :, a_] = ((pmap(s + ds, P) - s1)[dims] / h).T
    return J, s1

def newton(s, P, dims, n_it=10, clip=0.05):
    s = s.copy(); nd = len(dims)
    for _ in range(n_it):
        J, s1 = jac(s, P, dims)
        r = (s1 - s)[dims]
        ok = np.abs(r).max(0) < NEWTON_TOL
        if ok.all():
            break
        step = np.linalg.solve(J - np.eye(nd)[None], -r.T[:, :, None])[:, :, 0].T
        s[dims] = np.where(ok, s[dims], s[dims] + np.clip(np.nan_to_num(step), -clip, clip))
    J, s1 = jac(s, P, dims)
    res = np.abs((s1 - s)[dims]).max(0)
    return s, np.isfinite(res) & (res < 1e-7), J

def mean_force_period(s, P):
    return integrate(s, P, 1, t0=0.0, mean_last=1)["F"]

def det_I_minus(J):
    mu = np.linalg.eigvals(J)
    return np.real(np.prod(1.0 - mu, axis=1)), mu

# -----------------------------------------------------------------------------
# 3. Grid search (Script 06 method, generic layout)
# -----------------------------------------------------------------------------
def prescribed_seeds():
    PHIS = np.linspace(0, 2 * np.pi, N_PHI_SEED, endpoint=False)
    XG = np.linspace(0.3, 2.2, N_X_SEED)
    combos = [(d_, ph) for d_ in DPHI_SEEDS for ph in PHIS]
    nc = len(combos)
    Xb = np.tile(XG, nc)
    ph_b = np.repeat([c_[1] for c_ in combos], XG.size)
    dp_b = np.repeat([c_[0] for c_ in combos], XG.size)
    Pp = make_P(Xb.size, presc=True, freeze=True, phi0=ph_b, dphi=dp_b)
    Rp = integrate(init_state(Xb, ph_b), Pp, 40)
    Fp = integrate(Rp["s"], Pp, 10, t0=Rp["t"], mean_last=10)["F"].reshape(nc, XG.size)
    sx, sph, sdp = [], [], []
    for ci in range(nc):
        for j in range(XG.size - 1):
            f0, f1 = Fp[ci, j], Fp[ci, j + 1]
            if np.isfinite(f0) and np.isfinite(f1) and np.sign(f0) != np.sign(f1):
                sx.append(XG[j] - f0 * (XG[j + 1] - XG[j]) / (f1 - f0))
                sph.append(combos[ci][1]); sdp.append(combos[ci][0])
    if not sx:
        return None
    sx, sph, sdp = np.array(sx), np.array(sph), np.array(sdp)
    Pq = make_P(sx.size, presc=True, freeze=True, phi0=sph, dphi=sdp)
    S = integrate(init_state(sx, sph), Pq, 40)["s"]
    S[0] = sx; S[1] = 0.0
    S[IZ], S[IZ + 1] = cpg_seed(sph)
    return S, sdp

def grid_search():
    seeds = prescribed_seeds()
    if seeds is None:
        return [], 0
    SEEDS, sdp = seeds
    ns = SEEDS.shape[1]
    orbits = []
    window = WIDE
    for e in EPS_ROWS:
        nq = window.size
        S = np.concatenate([SEEDS] * nq, axis=1)
        Dd = np.tile(sdp, nq)
        R = np.repeat(window, ns)
        origin = np.zeros(S.shape[1], int)
        # second seed family: free (unfrozen) relaxation from the same seeds
        Srel = integrate(S, PF(S.shape[1], e, R, Dd), N_RELAX)["s"]
        m = valid(Srel) & (np.abs(Srel[1]) < 0.3)
        S = np.concatenate([S, Srel[:, m]], axis=1)
        Dd = np.concatenate([Dd, Dd[m]]); R = np.concatenate([R, R[m]])
        origin = np.concatenate([origin, np.ones(m.sum(), int)])
        sn, ok, J = newton(S, PF(S.shape[1], e, R, Dd), list(range(D)))
        ok &= valid(sn)
        mu = np.linalg.eigvals(J)
        a = np.abs(mu); lead_ = mu[np.arange(len(mu)), np.argmax(a, 1)]
        stable = a.max(1) < 1 - 1e-6
        found = []
        for j in np.where(ok)[0]:
            ph = float(cpg_phase(sn[:, j:j + 1])[0])
            dup = any(abs(o["ratio"] - R[j]) < 1e-9 and o["dphi"] == Dd[j] and abs(o["x"] - sn[0, j]) < 0.01
                      and abs(np.angle(np.exp(1j * (o["ph"] - ph)))) < 0.1 for o in orbits if o["eps"] == e)
            if not dup:
                orbits.append(dict(eps=float(e), ratio=float(R[j]), dphi=float(Dd[j]), x=float(sn[0, j]),
                                   ph=ph, stable=bool(stable[j]), rate=float(np.log(a[j].max())),
                                   torus=bool(np.abs(lead_[j].imag) > 1e-6),
                                   seed="relaxed" if origin[j] else "prescribed", state=sn[:, j].copy()))
                found.append(R[j])
        if found:
            centre = float(np.median(found))
            window = np.linspace(centre - HALF_W, centre + HALF_W, N_LOCAL)
    return orbits, ns

# -----------------------------------------------------------------------------
# 4. Homotopy index test (Script 08 method, generic layout)
# -----------------------------------------------------------------------------
def homotopy(S, eps, rat, dph):
    n = S.shape[1]
    s = S.copy()
    alive = np.ones(n, bool)
    fail_eta = np.full(n, np.nan)
    for eta in ETAS[1:]:
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        P = PF(idx.size, eps[idx], rat[idx], dph[idx], eta=eta)
        sn, ok, _ = newton(s[:, idx], P, list(range(D)), n_it=8, clip=0.05)
        ok &= valid(sn)
        s[:, idx[ok]] = sn[:, ok]
        bad = idx[~ok]
        fail_eta[bad] = eta
        alive[bad] = False
    return s, alive, fail_eta

def fixed_branch(s_end, eps, rat, dph):
    n = s_end.shape[1]
    s = s_end.copy(); s[1] = 0.0
    s, ok, J = newton(s, PF(n, eps, rat, dph, freeze=True), INT_DIMS, n_it=12, clip=0.2)
    dI, mu = det_I_minus(J)
    a = np.abs(mu)
    lead_ = mu[np.arange(n), np.argmax(a, 1)]
    typ = np.where(a.max(1) < 1, "node",
                   np.where(np.abs(lead_.imag) < 1e-6, "saddle-real", "unstable-complex"))
    return s, ok, dI, -np.log(a.max(1)), typ

def relocked_gradient(s_fix, eps, rat, dph):
    n = s_fix.shape[1]
    Fo = np.full((n, OFF.size), np.nan)
    for side in (+1, -1):
        s = s_fix.copy()
        for o in sorted([o for o in OFF if (o >= 0 if side > 0 else o <= 0)], key=abs):
            s[0] = s_fix[0] + o
            P = PF(n, eps, rat, dph, freeze=True)
            s, okq, _ = newton(s, P, INT_DIMS, n_it=10, clip=0.2)
            F = mean_force_period(s, P)
            Fo[:, int(np.argmin(np.abs(OFF - o)))] = np.where(okq, F, np.nan)
    aL = np.full(n, np.nan); r2 = np.full(n, np.nan)
    for i in range(n):
        m = np.isfinite(Fo[i])
        if m.sum() >= 5:
            A = np.vstack([OFF[m], np.ones(m.sum())]).T
            coef, *_ = np.linalg.lstsq(A, Fo[i, m], rcond=None)
            pr = A @ coef; ss = np.sum((Fo[i, m] - Fo[i, m].mean()) ** 2)
            aL[i] = coef[0]; r2[i] = 1 - np.sum((Fo[i, m] - pr) ** 2) / ss if ss > 0 else 0.0
    return aL, r2

def tongue_distance(s_end, eps, rat, dph):
    n = s_end.shape[1]
    dist = np.full(n, np.nan)
    for dr in RATIO_PROBE:
        todo = np.where(np.isnan(dist))[0]
        if todo.size == 0:
            break
        for sgn in (+1, -1):
            s = s_end[:, todo].copy(); s[1] = 0.0
            P = PF(todo.size, eps[todo], rat[todo] + sgn * dr, dph[todo], freeze=True)
            _, ok, _ = newton(s, P, INT_DIMS, n_it=12, clip=0.2)
            dist[todo[ok & np.isnan(dist[todo])]] = sgn * dr
    return dist

def index_test(orbs):
    n = len(orbs)
    eps = np.array([o["eps"] for o in orbs]); rat = np.array([o["ratio"] for o in orbs])
    dph = np.array([o["dphi"] for o in orbs])
    S = np.array([o["state"] for o in orbs]).T
    stable = np.array([o["stable"] for o in orbs])
    s_end, reached, fail_eta = homotopy(S, eps, rat, dph)
    s_fix, okfix, dI, kfix, typ = fixed_branch(s_end, eps, rat, dph)
    okfix &= reached
    aL = np.full(n, np.nan); r2 = np.full(n, np.nan)
    idx = np.where(okfix)[0]
    if idx.size:
        aL[idx], r2[idx] = relocked_gradient(s_fix[:, idx], eps[idx], rat[idx], dph[idx])
    dist = np.full(n, np.nan)
    nofix = np.where(reached & ~okfix)[0]
    if nofix.size:
        dist[nofix] = tongue_distance(s_end[:, nofix], eps[nofix], rat[nofix], dph[nofix])
    index = np.sign(dI) * np.sign(-aL)
    recs = []
    for i in range(n):
        status = "fixed" if okfix[i] else ("no-fixed-lock" if reached[i] else "homotopy-fold")
        recs.append(dict(eps=float(eps[i]), ratio=float(rat[i]), dphi=float(dph[i]), x=float(S[0, i]),
                         x_L=float(s_fix[0, i]) if okfix[i] else None, stable=bool(stable[i]),
                         rate=float(orbs[i]["rate"]), torus=bool(orbs[i]["torus"]), seed=orbs[i]["seed"],
                         status=status,
                         fail_eta=None if np.isnan(fail_eta[i]) else float(fail_eta[i]),
                         fixed_type=str(typ[i]) if okfix[i] else "none",
                         det_I_minus_Mint=float(dI[i]) if okfix[i] else None,
                         k=float(kfix[i]) if okfix[i] else None,
                         aL=None if np.isnan(aL[i]) else float(aL[i]),
                         aL_r2=None if np.isnan(r2[i]) else float(r2[i]),
                         index=None if np.isnan(index[i]) else float(index[i]),
                         tongue_dist=None if np.isnan(dist[i]) else float(dist[i])))
    return recs

# -----------------------------------------------------------------------------
# 5. Run
# -----------------------------------------------------------------------------
def run_variant(name, over, vrng):
    t0 = time.time()
    names = apply_cfg(over)
    orbits, n_seeds = grid_search()
    t_grid = time.time() - t0
    res = dict(name=name, override={k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                                    for k, v in over.items()},
               state_layout=names, n_seeds=int(n_seeds), n_orbits=len(orbits))
    if not orbits:
        res.update(n_stable=0, holding=False, runtime_s=round(time.time() - t0, 1),
                   note="no prescribed equilibria" if n_seeds == 0 else "no station-holding orbits")
        return res, []
    st = np.array([o["stable"] for o in orbits])
    i_s = np.where(st)[0]; i_u = np.where(~st)[0]
    if i_s.size > MAX_STABLE:
        i_s = vrng.choice(i_s, MAX_STABLE, replace=False)
    i_u = vrng.choice(i_u, min(N_UNST, i_u.size), replace=False) if i_u.size else i_u
    pick = np.concatenate([i_s, i_u]).astype(int)
    recs = index_test([orbits[i] for i in pick])
    for r_ in recs:
        r_.update(variant=name)
    ev = [r_ for r_ in recs if r_["index"] is not None]
    st_ev = [r_ for r_ in ev if r_["stable"]]
    viol = [r_ for r_ in st_ev if r_["index"] < 0]
    viol_nm = [r_ for r_ in viol if abs(r_["aL"]) >= AL_MARGIN]
    neg = [r_ for r_ in ev if r_["index"] < 0]
    by_type = {}
    for t_ in ("node", "saddle-real", "unstable-complex"):
        rr = [r_ for r_ in ev if r_["fixed_type"] == t_]
        by_type[t_] = dict(n=len(rr), stable=sum(r_["stable"] for r_ in rr),
                           stable_aL_neg=sum(r_["stable"] and r_["aL"] < 0 for r_ in rr),
                           stable_aL_pos=sum(r_["stable"] and r_["aL"] > 0 for r_ in rr))
    so = [o for o in orbits if o["stable"]]
    res.update(
        n_stable=int(st.sum()), holding=bool(st.any()),
        stable_by_seed=dict(prescribed=sum(o["stable"] and o["seed"] == "prescribed" for o in orbits),
                            relaxed=sum(o["stable"] and o["seed"] == "relaxed" for o in orbits)),
        n_torus_unstable=int(sum(1 for o in orbits if (not o["stable"]) and o["torus"])),
        stable_eps_range=[min(o["eps"] for o in so), max(o["eps"] for o in so)] if so else None,
        stable_ratio_range=[min(o["ratio"] for o in so), max(o["ratio"] for o in so)] if so else None,
        stable_x_range=[min(o["x"] for o in so), max(o["x"] for o in so)] if so else None,
        tested=len(recs), n_fixed=len(ev), n_stable_fixed=len(st_ev),
        violations=len(viol), violations_nonmarginal=len(viol_nm),
        index_negative_n=len(neg), index_negative_stable=sum(r_["stable"] for r_ in neg),
        by_fixed_type=by_type,
        no_fixed_lock=sum(r_["status"] == "no-fixed-lock" for r_ in recs),
        homotopy_folds=sum(r_["status"] == "homotopy-fold" for r_ in recs),
        runtime_grid_s=round(t_grid, 1), runtime_s=round(time.time() - t0, 1))
    return res, recs

def ckpt_path(i, name):
    slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return os.path.join(CKPT_DIR, f"{i:02d}_{slug}.json")

t_start = time.time()
n_done = sum(os.path.exists(ckpt_path(i, n_)) for i, (_, n_, _) in enumerate(VARIANTS))
print(f"Mode {MODE}: {len(VARIANTS)} variants ({n_done} already checkpointed, loaded not recomputed)")
results, all_recs = [], []
group_prev = None
for i, (grp, name, over) in enumerate(VARIANTS):
    if grp != group_prev:
        print(f"\n--- {grp} ---"); group_prev = grp
    cp = ckpt_path(i, name)
    if os.path.exists(cp):
        with open(cp) as f:
            d_ = json.load(f)
        r, recs = d_["res"], d_["recs"]
        r["from_checkpoint"] = True
    else:
        r, recs = run_variant(name, over, np.random.default_rng([SEED, i]))
        r["group"] = grp
        for r_ in recs:
            r_["group"] = grp
        with open(cp + ".tmp", "w") as f:
            json.dump(dict(res=r, recs=recs), f)
        os.replace(cp + ".tmp", cp)
    results.append(r); all_recs.extend(recs)
    if r["n_orbits"]:
        print(f"  {name:24s} D={len(r['state_layout']):2d} orbits {r['n_orbits']:4d} stable {r['n_stable']:4d} | "
              f"tested {r['tested']:3d} fixed {r['n_fixed']:3d} no-lock {r['no_fixed_lock']:2d} fold {r['homotopy_folds']:2d} | "
              f"violations {r['violations']} (non-marginal {r['violations_nonmarginal']}) | {r['runtime_s']:.0f} s")
    else:
        print(f"  {name:24s} D={len(r['state_layout']):2d} {r['note']} | {r['runtime_s']:.0f} s")
apply_cfg({})

# -----------------------------------------------------------------------------
# 6. Figure
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
tag = "ALL" if MODE == "FULL" else "ALL_quick"
ev_all = [r_ for r_ in all_recs if r_["index"] is not None]
fig, ax = plt.subplots(1, 2, figsize=(8.0, 0.3 * len(results) + 1.8),
                       gridspec_kw=dict(width_ratios=[1.3, 1]))
names = [r["name"] for r in results]
yy = np.arange(len(names))
ax[0].barh(yy - 0.2, [r["n_stable"] for r in results], height=0.4, color="#1a9850", label="stable holding orbits")
ax[0].barh(yy + 0.2, [r.get("violations", 0) for r in results], height=0.4, color="#d73027", label="index violations")
ax[0].set_yticks(yy); ax[0].set_yticklabels(names); ax[0].invert_yaxis()
ax[0].set_xlabel("count"); ax[0].legend(frameon=False, fontsize=7)
for r_ in sorted(ev_all, key=lambda r: r["stable"]):
    d_ = r_["det_I_minus_Mint"]
    ax[1].scatter(np.sign(d_) * np.log10(1 + abs(d_) * 1e3), r_["aL"], s=5,
                  c="#1a9850" if r_["stable"] else "#cccccc",
                  edgecolors="k" if (r_["stable"] and r_["index"] < 0) else "none", linewidths=0.6)
ax[1].axhline(0, color="k", lw=0.6); ax[1].axvline(0, color="k", lw=0.6)
ax[1].set_xlabel(r"sign$\cdot\log_{10}(1+10^3|\det(I-M_{int})|)$"); ax[1].set_ylabel(r"$a_L$")
ax[1].set_title("All variants\n(green stable; allowed: upper-left, lower-right)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"fig01_structure_{tag}.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, f"fig01_structure_{tag}.svg"))
plt.close(fig)

# -----------------------------------------------------------------------------
# 7. Save and gate
# -----------------------------------------------------------------------------
def totals(rs):
    tot = lambda k: int(sum(r.get(k, 0) for r in rs))
    return dict(variants=len(rs), with_holding=sum(r["holding"] for r in rs),
                tested=tot("tested"), stable_fixed=tot("n_stable_fixed"),
                violations=tot("violations"), violations_nonmarginal=tot("violations_nonmarginal"),
                index_negative=tot("index_negative_n"), index_negative_stable=tot("index_negative_stable"),
                no_fixed_lock=tot("no_fixed_lock"), homotopy_folds=tot("homotopy_folds"))

summary = dict(mode=MODE, eps_rows=EPS_ROWS.tolist(),
               runtime_this_session_s=round(time.time() - t_start, 1),
               totals=totals(results),
               totals_by_group={g_: totals([r for r in results if r.get("group") == g_]) for g_, _ in GROUPS},
               results=results,
               violation_rows=[r_ for r_ in ev_all if r_["stable"] and r_["index"] < 0][:30])
with open(os.path.join(OUT_DIR, f"summary_script09_{tag}.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT_DIR, f"records_script09_{tag}.json"), "w") as f:
    json.dump(all_recs, f)

print("\n================ SCRIPT 09 SUMMARY ================")
print(json.dumps(summary["totals"], indent=2))
print("\nBy group:")
for g_, t_ in summary["totals_by_group"].items():
    print(f"  {g_:7s} variants {t_['variants']:2d}, with holding {t_['with_holding']:2d}, stable tested "
          f"{t_['stable_fixed']:4d}, violations {t_['violations']} (non-marginal {t_['violations_nonmarginal']}), "
          f"stable among index < 0: {t_['index_negative_stable']}/{t_['index_negative']}")
print("\nPer variant (stable eps range | stable omega_c/omega_w range | stable x range):")
for r in results:
    if r.get("holding"):
        print(f"  {r['name']:24s} eps {r['stable_eps_range']} | ratio "
              f"[{r['stable_ratio_range'][0]:.3f}, {r['stable_ratio_range'][1]:.3f}] | x "
              f"[{r['stable_x_range'][0]:.2f}, {r['stable_x_range'][1]:.2f}] | node stable "
              f"{r['by_fixed_type']['node']['stable']}, saddle stable {r['by_fixed_type']['saddle-real']['stable']}")
    else:
        print(f"  {r['name']:24s} no stable holding")
print("\n================ VERDICT ================")
nm = summary["totals"]["violations_nonmarginal"]
print(f"Non-marginal index violations: {nm} over {summary['totals']['stable_fixed']} stable homotopy-identified orbits; "
      f"stable among index < 0: {summary['totals']['index_negative_stable']}/{summary['totals']['index_negative']}")
print(f"Variants with stable holding: {summary['totals']['with_holding']}/{len(results)}")
if nm == 0:
    print("CONFIRMED: the index condition holds across model structures.")
else:
    print("CHECK: inspect violation_rows in the summary before writing.")
print(f"\nOutputs: {OUT_DIR}")
