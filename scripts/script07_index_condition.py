# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 07: Verify the index condition for station holding on all orbits.
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime: about 10 to 20 min per dataset (FULL). DATASETS selects which to run.
#
# Index condition. Linearise the averaged dynamics about station holding with
# positional block (x, u) and internal block A (body + CPG at fixed position).
# The Schur complement gives  det J = -det(A) * a_L,  where a_L is the relocked
# mean-force gradient (internal state re-equilibrated at each x). For the
# stroboscopic map, to leading order in the slow positional dynamics:
#     sign det(I - M_full) = sign det(I - M_int) * sign(-a_L).
# A stable orbit has det(I - M_full) > 0, hence the NECESSARY condition
#     index := sign det(I - M_int) * sign(-a_L) > 0.
#   - stable fixed-position lock (node, det(I - M_int) > 0):  needs a_L < 0
#   - lock with one real multiplier > 1 (saddle):             needs a_L > 0
# No assumption about the speed of the body mode is needed.
#
# Tests over every checked orbit:
#   T1  every stable orbit has index > 0
#   T2  every orbit with index < 0 is unstable
# Marginal cases (|a_L| < AL_MARGIN) are reported separately.
#
# Datasets (outputs of earlier scripts in the Drive project folder):
#   "05b"      script05b_holding_dense/results_script05.npz (baseline, 4 delays)
#   "06_ALL"   script06_robustness/orbits_script06_ALL.json (14 variants)
#   "06_BODY"  script06_robustness/orbits_script06_BODY.json (12 body variants)
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
DATASETS = ["05b", "06_ALL", "06_BODY"]     # run any subset
SEED = 11
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script07_index_condition")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SPP = 100 if MODE == "FULL" else 80
N_UNST = 150 if MODE == "FULL" else 4      # unstable orbits sampled per variant
MAX_STABLE = 400 if MODE == "FULL" else 4  # stable orbits per variant (cap)
N_PH = 8
OFF = np.linspace(-0.02, 0.02, 7)
AL_MARGIN = 0.02
G_HOLD = 16.0

OMEGA_W = 2.0 * np.pi
WAKE = dict(U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35,
            l_w=1.50, c_L=3.0, c_D=0.15, sigma=8.0)
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU = 2 * np.pi
G_LOCK = 8.0                    # default gain (make_P)
X_COLL, X_ESC = 0.15, 4.5

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


LOCK_TOL = 0.3
NEWTON_TOL = 1e-8

def P_for(B, eps, ratio, dphi, freeze=False):
    return make_P(B, wc=OMEGA_W * np.asarray(ratio), eps=eps, g=G_HOLD, dphi=dphi,
                  freeze=freeze)

def pmap(s, P):
    return integrate(s, P, 1, t0=0.0, spp=SPP)["s"]

def jac(s, P, dims, h=1e-6):
    s1 = pmap(s, P)
    J = np.zeros((s.shape[1], len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = h
        J[:, :, a_] = ((pmap(s + ds, P) - s1)[dims] / h).T
    return J, s1

def newton(s, P, dims, n_it=10, clip=0.05):
    """Batch Newton for period-1 orbits of the stroboscopic map in the given dims."""
    s = s.copy()
    nd = len(dims)
    ok = np.zeros(s.shape[1], bool)
    for _ in range(n_it):
        J, s1 = jac(s, P, dims)
        r = (s1 - s)[dims]
        res = np.abs(r).max(0)
        ok = res < NEWTON_TOL
        if ok.all():
            break
        step = np.linalg.solve(J - np.eye(nd)[None], -r.T[:, :, None])[:, :, 0].T
        step = np.clip(np.nan_to_num(step), -clip, clip)
        s[dims] = np.where(ok, s[dims], s[dims] + step)
    J, s1 = jac(s, P, dims)
    res = np.abs((s1 - s)[dims]).max(0)
    mu = np.linalg.eigvals(J)
    return s, res < 1e-7, mu

def lead(mu):
    a = np.abs(mu)
    i = np.argmax(a, axis=1)
    return mu[np.arange(len(mu)), i], a.max(1)

def fixed_k(s, P):
    """Fixed-position lock at the orbit's x: Newton on the frozen 4D map.
    Returns k (nan if no fixed-position lock converges)."""
    Pf = dict(P); Pf["freeze"] = np.ones(s.shape[1], bool)
    sf = s.copy(); sf[1] = 0.0
    sf, ok, mu = newton(sf, Pf, [2, 3, 4, 5], n_it=12, clip=0.2)
    k = -np.log(np.abs(mu).max(1))
    return np.where(ok, k, np.nan)

def valid(s):
    return (s[0] > 0.2) & (s[0] < 3.5) & (np.abs(np.hypot(s[4], s[5]) - 1) < 0.5)


NEWTON_TOL = 1e-8
BASE_WAKE = dict(WAKE)
BASE = dict(g=16.0, mu=2 * np.pi, c=0.0, body_ratio=1.2, zeta=0.2)
VAR06 = {
    "baseline": {}, "c = 1": dict(c=1.0), "mu x 0.5": dict(mu=np.pi), "mu x 2": dict(mu=4 * np.pi),
    "g = 12": dict(g=12.0), "g = 24": dict(g=24.0), "body ratio 0.8": dict(body_ratio=0.8),
    "body ratio 1.6": dict(body_ratio=1.6), "zeta 0.1": dict(zeta=0.1), "zeta 0.4": dict(zeta=0.4),
    "W0 0.25": dict(W0=0.25), "W0 0.45": dict(W0=0.45), "c_D 0.10": dict(c_D=0.10), "c_D 0.25": dict(c_D=0.25),
}
for r_ in (0.9, 1.0, 1.1, 1.3, 1.4, 1.5):
    for z_ in (0.1, 0.2):
        VAR06[f"body {r_:.1f}, zeta {z_:.1f}"] = dict(body_ratio=r_, zeta=z_)

def apply_variant(over):
    global MU, OM_B, ZE_B, WAKE
    cfg = dict(BASE); cfg.update({k: v for k, v in over.items() if k in BASE})
    MU = cfg["mu"]; OM_B = cfg["body_ratio"] * OMEGA_W; ZE_B = cfg["zeta"]
    WAKE = dict(BASE_WAKE); WAKE.update({k: v for k, v in over.items() if k in BASE_WAKE})
    return cfg["g"], cfg["c"]

def PF(B, eps, ratio, dphi, g, c, freeze=False):
    return make_P(B, wc=OMEGA_W * np.asarray(ratio), eps=eps, g=g, c=c, dphi=dphi, freeze=freeze)

def pmap(s, P):
    return integrate(s, P, 1, t0=0.0, spp=SPP)["s"]

def jac(s, P, dims, h=1e-6):
    s1 = pmap(s, P)
    J = np.zeros((s.shape[1], len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = h
        J[:, :, a_] = ((pmap(s + ds, P) - s1)[dims] / h).T
    return J, s1

def newton(s, P, dims, n_it=12, clip=0.1):
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
    return s, res < 1e-7, J

def mean_force_period(s, P):
    return integrate(s, P, 1, t0=0.0, spp=SPP, mean_last=1)["F"]

def circ(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))

def det_I_minus(J):
    mu = np.linalg.eigvals(J)
    return np.real(np.prod(1.0 - mu, axis=1)), mu

def analyse(orbs, g, c, have_state=None):
    """orbs: list of dicts (eps, ratio, dphi, x, ph, stable[, state]). Returns per-orbit records."""
    n = len(orbs)
    eps = np.array([o["eps"] for o in orbs]); rat = np.array([o["ratio"] for o in orbs])
    dph = np.array([o["dphi"] for o in orbs]); x = np.array([o["x"] for o in orbs])
    ph = np.array([o["ph"] for o in orbs]); st_saved = np.array([o["stable"] for o in orbs])
    # 1. full orbit (rebuild if needed) and its Floquet stability
    if have_state is not None:
        S = have_state.copy()
    else:
        Pq = make_P(n, g=g, presc=True, freeze=True, phi0=ph, dphi=dph)
        S = integrate(init_state(x, ph), Pq, 40)["s"]
        S[0] = x; S[1] = 0.0; S[4] = np.cos(ph); S[5] = np.sin(ph)
    Pfull = PF(n, eps, rat, dph, g, c)
    S, okS, Jfull = newton(S, Pfull, list(range(6)), clip=0.05)
    mu_full = np.linalg.eigvals(Jfull)
    rate = np.log(np.abs(mu_full).max(1))
    stable = rate < -1e-6
    keep = okS & (np.abs(S[0] - x) < 0.01) & (stable == st_saved)
    # 2. fixed-position locks at x* from N_PH phases, matched by CPG phase
    phs = np.linspace(0, 2 * np.pi, N_PH, endpoint=False)
    rep = np.repeat(np.arange(n), N_PH)
    S0 = S[:, rep].copy(); S0[1] = 0.0
    S0[4] = np.cos(np.tile(phs, n)); S0[5] = np.sin(np.tile(phs, n))
    Pfix = PF(rep.size, eps[rep], rat[rep], dph[rep], g, c, freeze=True)
    Sf, okf, Jf = newton(S0, Pfix, [2, 3, 4, 5], clip=0.2)
    okf &= np.abs(np.hypot(Sf[4], Sf[5]) - 1) < 0.5
    dI, muf = det_I_minus(Jf)
    phf = np.arctan2(Sf[5], Sf[4]); phS = np.arctan2(S[5], S[4])
    m_state = np.full((6, n), np.nan); m_det = np.full(n, np.nan); m_k = np.full(n, np.nan)
    m_dist = np.full(n, np.nan); m_type = np.array(["none"] * n, dtype=object)
    for i in range(n):
        cand = np.where((rep == i) & okf)[0]
        if cand.size == 0:
            continue
        d_ = circ(phf[cand], phS[i])
        b = cand[np.argmin(d_)]
        m_state[:, i] = Sf[:, b]; m_det[i] = dI[b]; m_dist[i] = d_.min()
        a = np.abs(muf[b]); m_k[i] = -np.log(a.max())
        lead_ = muf[b][np.argmax(a)]
        m_type[i] = ("node" if a.max() < 1 else
                     ("saddle-real" if abs(lead_.imag) < 1e-6 else "unstable-complex"))
    # 3. a_L on the matched branch: Newton continuation over offsets, fit
    has = np.isfinite(m_det)
    aL = np.full(n, np.nan); r2 = np.full(n, np.nan)
    idx = np.where(has)[0]
    if idx.size:
        Fo = np.full((idx.size, OFF.size), np.nan)
        for side in (+1, -1):
            s = m_state[:, idx].copy()
            for o in sorted([o for o in OFF if (o >= 0 if side > 0 else o <= 0)], key=abs):
                s[0] = x[idx] + o
                Pq = PF(idx.size, eps[idx], rat[idx], dph[idx], g, c, freeze=True)
                s, okq, _ = newton(s, Pq, [2, 3, 4, 5], clip=0.2)
                F = mean_force_period(s, Pq)
                Fo[:, int(np.argmin(np.abs(OFF - o)))] = np.where(okq, F, np.nan)
        for j, i in enumerate(idx):
            mm = np.isfinite(Fo[j])
            if mm.sum() >= 5:
                A = np.vstack([OFF[mm], np.ones(mm.sum())]).T
                coef, *_ = np.linalg.lstsq(A, Fo[j, mm], rcond=None)
                pr = A @ coef; ss = np.sum((Fo[j, mm] - Fo[j, mm].mean()) ** 2)
                aL[i] = coef[0]; r2[i] = 1 - np.sum((Fo[j, mm] - pr) ** 2) / ss if ss > 0 else 0.0
    index = np.sign(m_det) * np.sign(-aL)
    recs = []
    for i in range(n):
        recs.append(dict(eps=float(eps[i]), ratio=float(rat[i]), dphi=float(dph[i]), x=float(S[0, i]),
                         kept=bool(keep[i]), stable=bool(stable[i]), rate=float(rate[i]),
                         fixed_type=str(m_type[i]), det_I_minus_Mint=float(m_det[i]), k=float(m_k[i]),
                         phase_dist=float(m_dist[i]), aL=float(aL[i]), aL_r2=float(r2[i]),
                         index=float(index[i])))
    return recs

# -----------------------------------------------------------------------------
# Load datasets
# -----------------------------------------------------------------------------
def load_05b():
    d_ = os.path.join(BASE_DIR, "script05b_holding_dense")
    f_ = next((os.path.join(d_, n_) for n_ in ("results_script05b.npz", "results_script05.npz")
               if os.path.exists(os.path.join(d_, n_))), None)
    if f_ is None:
        return {}
    d = np.load(f_, allow_pickle=True)
    ST = d["states"]
    orbs = [dict(eps=float(e), ratio=float(r), dphi=float(p), x=float(xx), ph=float(np.arctan2(s_[5], s_[4])),
                 stable=(str(t) == "stable"), state=s_)
            for e, r, p, xx, t, s_ in zip(d["eps"], d["ratio"], d["dphi"], d["x"], d["type"], ST)]
    return {"baseline": orbs}

def load_06(tag):
    f_ = os.path.join(BASE_DIR, "script06_robustness", f"orbits_script06_{tag}.json")
    if not os.path.exists(f_):
        return {}
    return {k: v for k, v in json.load(open(f_)).items() if v}

t_start = time.time()
all_recs = []
for ds in DATASETS:
    groups = load_05b() if ds == "05b" else load_06("ALL" if ds == "06_ALL" else "BODY")
    if not groups:
        print(f"{ds}: input not found, skipped"); continue
    for vname, orbs in groups.items():
        over = VAR06.get(vname, {})
        g, c = apply_variant(over)
        st = np.array([o["stable"] for o in orbs])
        i_s = np.where(st)[0]; i_u = np.where(~st)[0]
        if i_s.size > MAX_STABLE:
            i_s = rng.choice(i_s, MAX_STABLE, replace=False)
        i_u = rng.choice(i_u, min(N_UNST, i_u.size), replace=False) if i_u.size else i_u
        pick = np.concatenate([i_s, i_u]).astype(int)
        if pick.size == 0:
            continue
        sub_ = [orbs[i] for i in pick]
        hs = np.array([o["state"] for o in sub_]).T if "state" in sub_[0] else None
        t0 = time.time()
        recs = analyse(sub_, g, c, have_state=hs)
        for r_ in recs:
            r_.update(dataset=ds, variant=vname)
        all_recs.extend(recs)
        kk = [r_ for r_ in recs if r_["kept"]]
        viol = [r_ for r_ in kk if r_["stable"] and np.isfinite(r_["index"]) and r_["index"] < 0]
        print(f"  {ds:8s} {vname:22s} checked {len(kk):4d}/{len(recs):4d}, stable {sum(r_['stable'] for r_ in kk):4d}, "
              f"T1 violations {len(viol)} ({time.time() - t0:.0f} s)")
apply_variant({})

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
R = [r_ for r_ in all_recs if r_["kept"]]
ev = [r_ for r_ in R if np.isfinite(r_["index"]) and np.isfinite(r_["aL"])]
marg = lambda r_: abs(r_["aL"]) < AL_MARGIN
st_ev = [r_ for r_ in ev if r_["stable"]]
t1_viol = [r_ for r_ in st_ev if r_["index"] < 0]
t1_viol_nm = [r_ for r_ in t1_viol if not marg(r_)]
neg = [r_ for r_ in ev if r_["index"] < 0]
t2_viol = [r_ for r_ in neg if r_["stable"]]
by_type = {}
for t_ in ("node", "saddle-real", "unstable-complex"):
    rr = [r_ for r_ in ev if r_["fixed_type"] == t_]
    by_type[t_] = dict(n=len(rr), stable=sum(r_["stable"] for r_ in rr),
                       stable_aL_neg=sum(r_["stable"] and r_["aL"] < 0 for r_ in rr),
                       stable_aL_pos=sum(r_["stable"] and r_["aL"] > 0 for r_ in rr))
no_fixed = [r_ for r_ in R if r_["fixed_type"] == "none"]
summary = dict(
    mode=MODE, datasets=DATASETS, runtime_s=round(time.time() - t_start, 1),
    n_checked=len(R), n_evaluable=len(ev), n_stable_evaluable=len(st_ev),
    T1_stable_with_index_negative=len(t1_viol), T1_nonmarginal=len(t1_viol_nm),
    T2_index_negative_n=len(neg), T2_index_negative_but_stable=len(t2_viol),
    by_fixed_type=by_type,
    no_fixed_lock=dict(n=len(no_fixed), stable=sum(r_["stable"] for r_ in no_fixed)),
    violations=t1_viol[:50],
)
with open(os.path.join(OUT_DIR, "summary_script07.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT_DIR, "records_script07.json"), "w") as f:
    json.dump(all_recs, f)

# -----------------------------------------------------------------------------
# Figure: a_L vs det(I - M_int) coloured by stability
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
if ev:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for r_ in ev:
        ax[0].scatter(np.sign(r_["det_I_minus_Mint"]) * np.log10(1 + abs(r_["det_I_minus_Mint"]) * 1e3),
                      r_["aL"], s=5, c="#1a9850" if r_["stable"] else "#bbbbbb",
                      edgecolors="k" if (r_["stable"] and r_["index"] < 0) else "none", linewidths=0.6,
                      zorder=3 if r_["stable"] else 1)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel(r"sign$\cdot\log_{10}(1+10^3|\det(I-M_{int})|)$")
    ax[0].set_ylabel(r"$a_L$")
    ax[0].set_title("Stable orbits (green) must lie in\nupper-left or lower-right quadrant", fontsize=8)
    labs = list(by_type.keys())
    ax[1].bar(np.arange(len(labs)) - 0.2, [by_type[l_]["stable_aL_neg"] for l_ in labs], width=0.4,
              color="#2166ac", label=r"stable, $a_L<0$")
    ax[1].bar(np.arange(len(labs)) + 0.2, [by_type[l_]["stable_aL_pos"] for l_ in labs], width=0.4,
              color="#b2182b", label=r"stable, $a_L>0$")
    ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs, fontsize=7)
    ax[1].set_ylabel("stable orbits"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("By fixed-position lock type", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_index_condition.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig01_index_condition.svg"))
    plt.close(fig)

print("\n================ SCRIPT 07 SUMMARY ================")
print(json.dumps({k: v for k, v in summary.items() if k != "violations"}, indent=2))
print("\n================ VERDICT ================")
print(f"T1: stable orbits violating the index condition: {len(t1_viol)}/{len(st_ev)} "
      f"({len(t1_viol_nm)} with |a_L| >= {AL_MARGIN})")
print(f"T2: orbits with index < 0 that are stable: {len(t2_viol)}/{len(neg)}")
if len(t1_viol_nm) == 0:
    print("INDEX CONDITION CONFIRMED on all non-marginal orbits: station holding requires")
    print("sign det(I - M_int) * sign(-a_L) > 0 (node branch needs a_L < 0, real-saddle branch a_L > 0).")
else:
    print("VIOLATIONS present: inspect 'violations' in the summary (phase_dist, aL_r2, fixed_type)")
    print("before stating the condition as general.")
print(f"\nOutputs: {OUT_DIR}")
