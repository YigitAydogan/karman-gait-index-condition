# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 08: Homotopy branch identification for the index condition.
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime: about 1 to 2 min per variant (FULL). DATASETS selects which to run.
#
# Script 07 assigned each orbit to a fixed-position lock by matching CPG phase
# (8 Newton starts, nearest phase). A reviewer can call that heuristic. Here the
# branch is identified by continuation instead of matching:
#
#   x' = eta u,  u' = eta F,   body and CPG unchanged.
#
# eta = 1 is the free fish; as eta -> 0 position becomes infinitely slow, and the
# station-holding orbit continues onto the slow manifold: a fixed-position lock
# at the relocked equilibrium x_L, where <F>(x_L) = 0. Following the orbit
# continuously from eta = 1 to eta = ETA_END identifies its fixed-position
# branch with no threshold. Then:
#   - det(I - M_int) and k of the fixed-position orbit reached
#   - a_L on that branch (Newton continuation over x offsets, linear fit)
#   - index = sign det(I - M_int) * sign(-a_L); stable orbits need index > 0
#   - comparison with the Script 07 phase rule (records_script07.json)
#   - orbits whose homotopy fails (a fold in eta) or reaches no fixed lock:
#     distance outside the fixed-position tongue (nearest locking omega_c)
#
# Gate: 0 non-marginal violations among homotopy-identified orbits, and
# branch agreement with the Script 07 phase rule above 95%.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
DATASETS = ["05b", "06_ALL", "06_BODY"]
SEED = 12
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script08_homotopy")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
REC07 = os.path.join(BASE_DIR, "script07_index_condition", "records_script07.json")

SPP = 100 if MODE == "FULL" else 80
N_UNST = 40 if MODE == "FULL" else 3
MAX_STABLE = 400 if MODE == "FULL" else 4
ETAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
OFF = np.linspace(-0.02, 0.02, 7)
RATIO_PROBE = np.array([0.003, 0.006, 0.01, 0.02, 0.035, 0.05])
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


# positional slow-down factor eta (P["eta"]); 1 = free fish
_rhs_base = rhs
def rhs(t, s, P):
    d, F = _rhs_base(t, s, P)
    eta = P.get("eta")
    if eta is not None:
        d = d.copy()
        d[0] = d[0] * eta
        d[1] = d[1] * eta
    return d, F

def homotopy(S, eps, rat, dph, g, c):
    """Continue full orbits from eta = 1 to ETAS[-1]. Returns end states and status."""
    n = S.shape[1]
    s = S.copy()
    alive = np.ones(n, bool)
    fail_eta = np.full(n, np.nan)
    for eta in ETAS[1:]:
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        P = PF(idx.size, eps[idx], rat[idx], dph[idx], g, c)
        P["eta"] = np.full(idx.size, eta)
        sn, ok, _ = newton(s[:, idx], P, list(range(6)), n_it=8, clip=0.05)
        ok &= np.isfinite(sn).all(0) & (np.abs(np.hypot(sn[4], sn[5]) - 1) < 0.5) & (sn[0] > 0.2) & (sn[0] < 3.5)
        s[:, idx[ok]] = sn[:, ok]
        bad = idx[~ok]
        fail_eta[bad] = eta
        alive[bad] = False
    return s, alive, fail_eta

def fixed_branch(s_end, eps, rat, dph, g, c):
    n = s_end.shape[1]
    s = s_end.copy(); s[1] = 0.0
    P = PF(n, eps, rat, dph, g, c, freeze=True)
    s, ok, J = newton(s, P, [2, 3, 4, 5], n_it=12, clip=0.2)
    dI, mu = det_I_minus(J)
    a = np.abs(mu)
    lead_ = mu[np.arange(n), np.argmax(a, 1)]
    typ = np.where(a.max(1) < 1, "node", np.where(np.abs(lead_.imag) < 1e-6, "saddle-real", "unstable-complex"))
    return s, ok, dI, -np.log(a.max(1)), typ

def relocked_gradient(s_fix, eps, rat, dph, g, c):
    n = s_fix.shape[1]
    Fo = np.full((n, OFF.size), np.nan)
    for side in (+1, -1):
        s = s_fix.copy()
        for o in sorted([o for o in OFF if (o >= 0 if side > 0 else o <= 0)], key=abs):
            s[0] = s_fix[0] + o
            P = PF(n, eps, rat, dph, g, c, freeze=True)
            s, okq, _ = newton(s, P, [2, 3, 4, 5], n_it=10, clip=0.2)
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

def tongue_distance(s_end, eps, rat, dph, g, c):
    """For orbits with no fixed lock: smallest |d omega_c/omega_w| at which a fixed
    lock exists at the same x (probing both sides)."""
    n = s_end.shape[1]
    dist = np.full(n, np.nan)
    for dr in RATIO_PROBE:
        todo = np.where(np.isnan(dist))[0]
        if todo.size == 0:
            break
        for sgn in (+1, -1):
            s = s_end[:, todo].copy(); s[1] = 0.0
            P = PF(todo.size, eps[todo], rat[todo] + sgn * dr, dph[todo], g, c, freeze=True)
            _, ok, _ = newton(s, P, [2, 3, 4, 5], n_it=12, clip=0.2)
            dist[todo[ok & np.isnan(dist[todo])]] = sgn * dr
    return dist

def analyse(orbs, g, c, have_state=None):
    n = len(orbs)
    eps = np.array([o["eps"] for o in orbs]); rat = np.array([o["ratio"] for o in orbs])
    dph = np.array([o["dphi"] for o in orbs]); x = np.array([o["x"] for o in orbs])
    ph = np.array([o["ph"] for o in orbs]); st_saved = np.array([o["stable"] for o in orbs])
    if have_state is not None:
        S = have_state.copy()
    else:
        Pq = make_P(n, g=g, presc=True, freeze=True, phi0=ph, dphi=dph)
        S = integrate(init_state(x, ph), Pq, 40)["s"]
        S[0] = x; S[1] = 0.0; S[4] = np.cos(ph); S[5] = np.sin(ph)
    Pfull = PF(n, eps, rat, dph, g, c)
    Pfull["eta"] = np.ones(n)
    S, okS, Jfull = newton(S, Pfull, list(range(6)), clip=0.05)
    rate = np.log(np.abs(np.linalg.eigvals(Jfull)).max(1))
    stable = rate < -1e-6
    keep = okS & (np.abs(S[0] - x) < 0.01) & (stable == st_saved)
    s_end, reached, fail_eta = homotopy(S, eps, rat, dph, g, c)
    s_fix, okfix, dI, kfix, typ = fixed_branch(s_end, eps, rat, dph, g, c)
    okfix &= reached
    aL = np.full(n, np.nan); r2 = np.full(n, np.nan)
    idx = np.where(okfix)[0]
    if idx.size:
        aL[idx], r2[idx] = relocked_gradient(s_fix[:, idx], eps[idx], rat[idx], dph[idx], g, c)
    dist = np.full(n, np.nan)
    nofix = np.where(reached & ~okfix)[0]
    if nofix.size:
        dist[nofix] = tongue_distance(s_end[:, nofix], eps[nofix], rat[nofix], dph[nofix], g, c)
    index = np.sign(dI) * np.sign(-aL)
    recs = []
    for i in range(n):
        status = ("fixed" if okfix[i] else ("no-fixed-lock" if reached[i] else "homotopy-fold"))
        recs.append(dict(eps=float(eps[i]), ratio=float(rat[i]), dphi=float(dph[i]), x=float(S[0, i]),
                         x_L=float(s_fix[0, i]) if okfix[i] else None, kept=bool(keep[i]),
                         stable=bool(stable[i]), rate=float(rate[i]), status=status,
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
# Run
# -----------------------------------------------------------------------------
t_start = time.time()
rec07 = json.load(open(REC07)) if os.path.exists(REC07) else []
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
        viol = [r_ for r_ in kk if r_["stable"] and r_["index"] is not None and r_["index"] < 0]
        print(f"  {ds:8s} {vname:22s} kept {len(kk):4d}, stable {sum(r_['stable'] for r_ in kk):4d}, "
              f"fixed {sum(r_['status'] == 'fixed' for r_ in kk):4d}, no-lock {sum(r_['status'] == 'no-fixed-lock' for r_ in kk):3d}, "
              f"fold {sum(r_['status'] == 'homotopy-fold' for r_ in kk):3d}, violations {len(viol)} ({time.time() - t0:.0f} s)")
apply_variant({})

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
R = [r_ for r_ in all_recs if r_["kept"]]
ev = [r_ for r_ in R if r_["index"] is not None]
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

# agreement with the Script 07 phase rule (same dataset, variant, eps, ratio, dphi, x)
agree, compared, disagree_rows = 0, 0, []
idx07 = {}
for r7 in rec07:
    if r7.get("kept") and r7.get("det_I_minus_Mint") is not None and np.isfinite(r7["det_I_minus_Mint"]):
        key = (r7["dataset"], r7["variant"], round(r7["eps"], 6), round(r7["ratio"], 6), round(r7["dphi"], 6))
        idx07.setdefault(key, []).append(r7)
for r_ in ev:
    key = (r_["dataset"], r_["variant"], round(r_["eps"], 6), round(r_["ratio"], 6), round(r_["dphi"], 6))
    cands = [c7 for c7 in idx07.get(key, []) if abs(c7["x"] - r_["x"]) < 1e-3]
    if not cands:
        continue
    c7 = cands[0]
    if c7["phase_dist"] >= 0.3:
        continue
    compared += 1
    same = np.sign(c7["det_I_minus_Mint"]) == np.sign(r_["det_I_minus_Mint"])
    agree += int(same)
    if not same:
        disagree_rows.append(dict(ours=r_, script07=c7))
agreement = agree / compared if compared else None

nolock = [r_ for r_ in R if r_["status"] == "no-fixed-lock"]
folds = [r_ for r_ in R if r_["status"] == "homotopy-fold"]
nl_d = [abs(r_["tongue_dist"]) for r_ in nolock if r_["tongue_dist"] is not None]
summary = dict(
    mode=MODE, datasets=DATASETS, runtime_s=round(time.time() - t_start, 1),
    n_kept=len(R), n_fixed=len(ev), n_stable_fixed=len(st_ev),
    violations=len(viol), violations_nonmarginal=len(viol_nm),
    index_negative_n=len(neg), index_negative_stable=sum(r_["stable"] for r_ in neg),
    by_fixed_type=by_type,
    phase_rule_compared=compared, phase_rule_agreement=agreement,
    no_fixed_lock=dict(n=len(nolock), stable=sum(r_["stable"] for r_ in nolock),
                       tongue_distance_median=float(np.median(nl_d)) if nl_d else None,
                       tongue_distance_max_found=float(np.max(nl_d)) if nl_d else None,
                       beyond_probe=sum(r_["tongue_dist"] is None for r_ in nolock)),
    homotopy_folds=dict(n=len(folds), stable=sum(r_["stable"] for r_ in folds),
                        fail_eta_counts={str(e): sum(r_["fail_eta"] == e for r_ in folds) for e in ETAS[1:]}),
    violation_rows=viol[:30], disagreement_rows=disagree_rows[:20],
)
with open(os.path.join(OUT_DIR, "summary_script08.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT_DIR, "records_script08.json"), "w") as f:
    json.dump(all_recs, f)

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
if ev:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for r_ in sorted(ev, key=lambda r: r["stable"]):
        d_ = r_["det_I_minus_Mint"]
        ax[0].scatter(np.sign(d_) * np.log10(1 + abs(d_) * 1e3), r_["aL"], s=5,
                      c="#1a9850" if r_["stable"] else "#cccccc",
                      edgecolors="k" if (r_["stable"] and r_["index"] < 0) else "none", linewidths=0.6)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel(r"sign$\cdot\log_{10}(1+10^3|\det(I-M_{int})|)$"); ax[0].set_ylabel(r"$a_L$")
    ax[0].set_title("Homotopy-identified branches\n(green stable; allowed: upper-left, lower-right)", fontsize=8)
    if nl_d:
        ax[1].hist(nl_d, bins=12, color="#999999")
        ax[1].set_xlabel(r"distance outside fixed-position tongue, $|\Delta\omega_c/\omega_w|$")
        ax[1].set_ylabel("no-lock orbits")
        ax[1].set_title("Where no-lock orbits sit", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_homotopy_index.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig01_homotopy_index.svg"))
    plt.close(fig)

print("\n================ SCRIPT 08 SUMMARY ================")
print(json.dumps({k: v for k, v in summary.items() if k not in ("violation_rows", "disagreement_rows")}, indent=2))
print("\n================ VERDICT ================")
print(f"Index violations among stable homotopy-identified orbits: {len(viol)}/{len(st_ev)} "
      f"({len(viol_nm)} non-marginal); stable among index < 0: {summary['index_negative_stable']}/{len(neg)}")
print(f"Branch agreement with Script 07 phase rule (phase distance < 0.3): "
      f"{agreement if agreement is None else round(agreement, 4)} over {compared} orbits")
if len(viol_nm) == 0 and (agreement is None or agreement >= 0.95):
    print("CONFIRMED: the index condition holds with threshold-free branch identification.")
else:
    print("CHECK: inspect violation_rows / disagreement_rows in the summary before writing.")
print(f"\nOutputs: {OUT_DIR}")
