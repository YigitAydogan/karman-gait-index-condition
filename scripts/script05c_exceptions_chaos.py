# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 05c: Resolve the exceptions of Script 05b, and test the surging orbits
#             for chaos.
#
# Loads results_script05b.npz and summary_script05b.json.
# Colab ready. Needs only numpy and matplotlib. Runtime: about 4 to 8 min.
#
# Part 1 (branch assignment). For each selected station-holding orbit:
#   - find ALL fixed-position locked orbits at x* (Newton on the frozen 4D map
#     from 12 CPG phases), with their CPG phase and k
#   - assign the branch whose CPG phase is closest to the free orbit's phase
#   - a_L on the matched branch: Newton continuation of the fixed-position orbit
#     at 9 offsets (works on the repelling branch), mean force fitted vs x
#   - a_P and gamma at the free orbit's phase (prescribed rhythm)
#   - reduced-cubic growth rate vs the Floquet rate from Script 05b
#   Selected: all stable orbits that 05b put on the node branch, all
#   no-fixed-lock orbits, and random stable-saddle and unstable-node controls.
#
# Part 2 (chaos test). Every torus orbit that survived in 05b as bounded
# surging: 3000-period runs; Lyapunov exponent over the last 2500 periods,
# stroboscopic power spectrum (spectral entropy), 0-1 test for chaos (K).
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
SEED = 8
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
IN_DIR = os.path.join(BASE_DIR, "script05b_holding_dense")
OUT_DIR = os.path.join(BASE_DIR, "script05c_exceptions_chaos")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SPP = 100 if MODE == "FULL" else 80
N_CTRL = 20 if MODE == "FULL" else 5
T_LONG = 3000 if MODE == "FULL" else 400
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

def P_for(B, eps, ratio, dphi, freeze=False):
    return make_P(B, wc=OMEGA_W * np.asarray(ratio), eps=eps, g=G_HOLD, dphi=dphi, freeze=freeze)

def pmap(s, P):
    return integrate(s, P, 1, t0=0.0, spp=SPP)["s"]

def jac(s, P, dims, h=1e-6):
    s1 = pmap(s, P)
    J = np.zeros((s.shape[1], len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = h
        J[:, :, a_] = ((pmap(s + ds, P) - s1)[dims] / h).T
    return J, s1

def newton(s, P, dims, n_it=12, clip=0.2):
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
    return s, res < 1e-7, np.linalg.eigvals(J)

def mean_force_period(s, P):
    return integrate(s, P, 1, t0=0.0, spp=SPP, mean_last=1)["F"]

def circ(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
t_start = time.time()
def first_existing(*names):
    for n_ in names:
        f_ = os.path.join(IN_DIR, n_)
        if os.path.exists(f_):
            return f_
    raise FileNotFoundError(f"none of {names} in {IN_DIR}")
d = np.load(first_existing("results_script05b.npz", "results_script05.npz"), allow_pickle=True)
summ = json.load(open(first_existing("summary_script05b.json", "summary_script05.json")))
eps_o, rat_o, x_o = d["eps"], d["ratio"], d["x"]
rate_o, typ_o, k_o, dphi_o = d["rate"], d["type"].astype(str), d["k"], d["dphi"]
ST = d["states"]                      # (n_orbits, 6)
stable = typ_o == "stable"
node_st = np.where(stable & (k_o > 0))[0]
nolock = np.where(np.isnan(k_o))[0]
sad_st = np.where(stable & (k_o < 0))[0]
node_un = np.where(~stable & (k_o > 0))[0]
ctrl1 = rng.choice(sad_st, size=min(N_CTRL, sad_st.size), replace=False) if sad_st.size else np.array([], int)
ctrl2 = rng.choice(node_un, size=min(N_CTRL, node_un.size), replace=False) if node_un.size else np.array([], int)
sel = np.concatenate([node_st, nolock, ctrl1, ctrl2]).astype(int)
group = np.array(["node-stable"] * node_st.size + ["no-lock"] * nolock.size +
                 ["saddle-stable ctrl"] * ctrl1.size + ["node-unstable ctrl"] * ctrl2.size)
NS = sel.size
print(f"Selected {NS}: node-stable {node_st.size}, no-lock {nolock.size}, controls {ctrl1.size}+{ctrl2.size}")

# -----------------------------------------------------------------------------
# Part 1a: all fixed-position locked orbits at x*, from 12 CPG phases
# -----------------------------------------------------------------------------
NPH = 12
phs = np.linspace(0, 2 * np.pi, NPH, endpoint=False)
rep = np.repeat(np.arange(NS), NPH)
S0 = ST[sel][rep].T.copy()
S0[1] = 0.0
S0[4] = np.cos(np.tile(phs, NS)); S0[5] = np.sin(np.tile(phs, NS))
Pf = P_for(rep.size, eps_o[sel][rep], rat_o[sel][rep], dphi_o[sel][rep], freeze=True)
Sf, okf, muf = newton(S0, Pf, [2, 3, 4, 5])
okf &= np.abs(np.hypot(Sf[4], Sf[5]) - 1) < 0.5
kf = -np.log(np.abs(muf).max(1))
phf = np.arctan2(Sf[5], Sf[4])
ph_full = np.arctan2(ST[sel][:, 5], ST[sel][:, 4])
match_k = np.full(NS, np.nan); match_d = np.full(NS, np.nan); n_fixed = np.zeros(NS, int)
match_state = np.full((6, NS), np.nan)
fixed_list = []
for n in range(NS):
    m = np.where((rep == n) & okf)[0]
    # distinct fixed orbits by phase
    uniq = []
    for j in m:
        if all(circ(phf[j], phf[u]) > 0.05 for u in uniq):
            uniq.append(j)
    n_fixed[n] = len(uniq)
    fixed_list.append([dict(phase=float(phf[u]), k=float(kf[u])) for u in uniq])
    if uniq:
        dd = np.array([circ(phf[u], ph_full[n]) for u in uniq])
        b = uniq[int(np.argmin(dd))]
        match_k[n] = kf[b]; match_d[n] = dd.min(); match_state[:, n] = Sf[:, b]
print(f"1a: fixed-position orbits found ({time.time() - t_start:.0f} s); "
      f"median count per position {np.median(n_fixed):.0f}")

# -----------------------------------------------------------------------------
# Part 1b: a_L on the matched branch (Newton continuation at 9 offsets)
# -----------------------------------------------------------------------------
OFF = np.linspace(-0.02, 0.02, 9)
has = np.isfinite(match_k)
idx = np.where(has)[0]
aL = np.full(NS, np.nan); r2 = np.full(NS, np.nan)
if idx.size:
    Fo = np.full((idx.size, OFF.size), np.nan)
    order = np.argsort(np.abs(OFF))                      # continue outward from 0
    cur = {0: match_state[:, idx].copy()}
    states_at = {}
    for side in (+1, -1):
        s = match_state[:, idx].copy()
        offs = [o for o in OFF if (o >= 0 if side > 0 else o <= 0)]
        offs = sorted(offs, key=abs)
        for o in offs:
            s[0] = x_o[sel][idx] + o
            Pq = P_for(idx.size, eps_o[sel][idx], rat_o[sel][idx], dphi_o[sel][idx], freeze=True)
            s, okq, _ = newton(s, Pq, [2, 3, 4, 5])
            F = mean_force_period(s, Pq)
            col = int(np.argmin(np.abs(OFF - o)))
            Fo[:, col] = np.where(okq, F, np.nan)
    for n_, n in enumerate(idx):
        m = np.isfinite(Fo[n_])
        if m.sum() >= 5:
            A = np.vstack([OFF[m], np.ones(m.sum())]).T
            coef, *_ = np.linalg.lstsq(A, Fo[n_, m], rcond=None)
            pr = A @ coef; ss = np.sum((Fo[n_, m] - Fo[n_, m].mean()) ** 2)
            aL[n] = coef[0]; r2[n] = 1 - np.sum((Fo[n_, m] - pr) ** 2) / ss if ss > 0 else 0.0
print(f"1b: a_L on matched branches ({time.time() - t_start:.0f} s)")

# -----------------------------------------------------------------------------
# Part 1c: a_P and gamma at the free orbit's phase; cubic prediction
# -----------------------------------------------------------------------------
h = 2e-3
def presc_force(xs, ub):
    B = xs.size
    reps_ = B // NS
    P = make_P(B, g=G_HOLD, presc=True, freeze=True,
               phi0=np.tile(ph_full, reps_), dphi=np.tile(dphi_o[sel], reps_), ub=ub)
    R = integrate(init_state(xs, np.tile(ph_full, reps_)), P, 40)
    return integrate(R["s"], P, 10, t0=R["t"], mean_last=10)["F"]
xs_ = x_o[sel]
Fp = presc_force(np.concatenate([xs_ + h, xs_ - h, xs_, xs_]),
                 np.concatenate([np.zeros(2 * NS), np.full(NS, h), np.full(NS, -h)])).reshape(4, NS)
aP = (Fp[0] - Fp[1]) / (2 * h)
gam = -(Fp[2] - Fp[3]) / (2 * h)
pred = np.full(NS, np.nan)
for n in range(NS):
    if np.isfinite(aL[n]) and np.isfinite(match_k[n]):
        rts = np.roots([1.0, match_k[n] + gam[n], gam[n] * match_k[n] - aP[n], -aL[n] * match_k[n]])
        pred[n] = rts.real.max()
flo = rate_o[sel]

rows = []
for n in range(NS):
    rows.append(dict(group=str(group[n]), eps=float(eps_o[sel][n]), ratio=float(rat_o[sel][n]),
                     dphi=float(dphi_o[sel][n]), x=float(x_o[sel][n]), floquet_rate=float(flo[n]),
                     k_05b=float(k_o[sel][n]), n_fixed_orbits=int(n_fixed[n]),
                     k_matched=float(match_k[n]), phase_dist=float(match_d[n]),
                     aL=float(aL[n]), aL_r2=float(r2[n]), aP=float(aP[n]), gamma=float(gam[n]),
                     pred_rate=float(pred[n]), fixed_orbits=fixed_list[n]))

def gsum(gname):
    m = group == gname
    if not m.any():
        return None
    return dict(n=int(m.sum()),
                matched_branch=dict(node=int(np.sum(m & (match_k > 0))), saddle=int(np.sum(m & (match_k < 0))),
                                    none=int(np.sum(m & ~np.isfinite(match_k)))),
                aL_negative=int(np.sum(m & (aL < 0))),
                theory_agrees=int(np.sum(m & np.isfinite(pred) & ((pred < 0) == (flo < 0)))),
                theory_evaluable=int(np.sum(m & np.isfinite(pred))))
part1 = {g_: gsum(g_) for g_ in ("node-stable", "no-lock", "saddle-stable ctrl", "node-unstable ctrl")}
print("Part 1:", json.dumps(part1, indent=1))

# -----------------------------------------------------------------------------
# Part 2: chaos test for surging orbits
# -----------------------------------------------------------------------------
surv = [r_ for r_ in summ.get("post_border", []) if r_.get("alive")]
part2 = []
xs_long = None
if surv:
    jj = []
    for r_ in surv:
        m = np.where((np.abs(eps_o - r_["eps"]) < 1e-9) & (np.abs(rat_o - r_["ratio"]) < 1e-9)
                     & (np.abs(dphi_o - r_["dphi"]) < 1e-9) & (np.abs(rate_o - r_["rate"]) < 1e-9))[0]
        if m.size:
            jj.append(int(m[0]))
    jj = np.array(jj, int)
    print(f"Part 2: {jj.size} surging orbits, {T_LONG} periods ...")
    s = ST[jj].T.copy(); s[0] += 1e-3
    P = P_for(jj.size, eps_o[jj], rat_o[jj], dphi_o[jj])
    d0 = 1e-8
    dv = rng.normal(size=s.shape); dv *= d0 / np.linalg.norm(dv, axis=0)
    sp = s + dv
    lsum = np.zeros(jj.size); alive = np.ones(jj.size, bool)
    xs_long = np.full((T_LONG, jj.size), np.nan)
    burn = T_LONG // 6
    for n in range(T_LONG):
        s = pmap(s, P); sp = pmap(sp, P)
        alive &= (s[0] > X_COLL) & (s[0] < X_ESC)
        s[:, ~alive] = np.nan_to_num(s[:, ~alive], nan=1.0)
        dd = sp - s; dn = np.maximum(np.linalg.norm(dd, axis=0), 1e-300)
        if n >= burn:
            lsum += np.log(dn / d0)
        sp = s + dd * (d0 / dn)
        xs_long[n] = np.where(alive, s[0], np.nan)
    lam = lsum / (T_LONG - burn)

    def spectral_entropy(x):
        x = x - x.mean()
        p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        p = p[1:] / p[1:].sum()
        return float(-np.sum(p * np.log(p + 1e-300)) / np.log(p.size))

    def zero_one_K(x, n_c=50):
        """Gottwald-Melbourne 0-1 test (modified, median over c)."""
        x = (x - x.mean()) / (x.std() + 1e-300)
        N = x.size; ncut = N // 10
        Ks = []
        for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c):
            j = np.arange(1, N + 1)
            p = np.cumsum(x * np.cos(j * c)); q = np.cumsum(x * np.sin(j * c))
            M = np.array([np.mean((p[n_:] - p[:-n_]) ** 2 + (q[n_:] - q[:-n_]) ** 2)
                          for n_ in range(1, ncut)])
            nn = np.arange(1, ncut)
            Dm = M - (np.mean(x) ** 2) * (1 - np.cos(nn * c)) / (1 - np.cos(c))
            Ks.append(np.corrcoef(nn, Dm)[0, 1])
        return float(np.median(Ks))

    for n_, j in enumerate(jj):
        rec = dict(eps=float(eps_o[j]), ratio=float(rat_o[j]), dphi=float(dphi_o[j]),
                   alive=bool(alive[n_]))
        if alive[n_]:
            tail = xs_long[burn:, n_]
            rec.update(lyap=float(lam[n_]), surge_amp=float(0.5 * (tail.max() - tail.min())),
                       spectral_entropy=spectral_entropy(tail), K01=zero_one_K(tail[-2000:] if tail.size > 2000 else tail))
        part2.append(rec)
    print("Part 2:", json.dumps(part2, indent=1))

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
GC = {"node-stable": "#d73027", "no-lock": "#999999", "saddle-stable ctrl": "#1a9850",
      "node-unstable ctrl": "#2166ac"}
for g_, c_ in GC.items():
    m = group == g_
    ax[0].scatter(match_k[m], aL[m], s=10, c=c_, label=g_)
    ax[1].scatter(pred[m], flo[m], s=10, c=c_)
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_xlabel("matched fixed-position k"); ax[0].set_ylabel(r"$a_L$ (matched branch)")
ax[0].legend(frameon=False, fontsize=5)
lim = np.nanmax(np.abs(np.concatenate([pred[np.isfinite(pred)], flo]))) if np.isfinite(pred).any() else 0.1
ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
ax[1].axhline(0, color="0.7", lw=0.5); ax[1].axvline(0, color="0.7", lw=0.5)
ax[1].set_xlabel("reduced-cubic rate"); ax[1].set_ylabel("Floquet rate (05b)")
if xs_long is not None:
    for n_ in range(xs_long.shape[1]):
        if np.isfinite(xs_long[-1, n_]):
            ax[2].plot(xs_long[-300:, n_] - np.nanmean(xs_long[-300:, n_]), lw=0.5)
    ax[2].set_xlabel("period (last 300)"); ax[2].set_ylabel(r"$x - \bar x$")
    ax[2].set_title("Surging orbits", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig01_exceptions_surging.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "fig01_exceptions_surging.svg"))
plt.close(fig)

with open(os.path.join(OUT_DIR, "summary_script05c.json"), "w") as f:
    json.dump(dict(part1_groups=part1, part1_rows=rows, part2=part2,
                   runtime_s=round(time.time() - t_start, 1)), f, indent=2)

print("\n================ VERDICT ================")
ns_ = part1["node-stable"]
if ns_:
    print(f"05b node-stable orbits: {ns_['n']}. Matched branch: {ns_['matched_branch']}; "
          f"a_L < 0 on matched branch: {ns_['aL_negative']}; theory agrees {ns_['theory_agrees']}/{ns_['theory_evaluable']}.")
    resolved = ns_["matched_branch"]["node"] == 0 or (ns_["aL_negative"] >= ns_["matched_branch"]["node"])
    print("RESOLVED: no stable orbit sits on the node branch with a_L > 0." if resolved else
          "UNRESOLVED: stable node-branch orbits with a_L > 0 remain; the theory needs revisiting.")
nl_ = part1["no-lock"]
if nl_:
    print(f"No-lock orbits: {nl_['n']}; on re-search, matched branch {nl_['matched_branch']}.")
for g_ in ("saddle-stable ctrl", "node-unstable ctrl"):
    if part1[g_]:
        print(f"{g_}: theory agrees {part1[g_]['theory_agrees']}/{part1[g_]['theory_evaluable']}, "
              f"matched {part1[g_]['matched_branch']}")
if part2:
    al = [p_ for p_ in part2 if p_["alive"]]
    chaotic = [p_ for p_ in al if p_["lyap"] > 0.005 and p_["K01"] > 0.8]
    print(f"Surging: {len(al)}/{len(part2)} survive {T_LONG} periods; chaotic (lambda > 0.005 and K > 0.8): "
          f"{len(chaotic)}; median K = {np.median([p_['K01'] for p_ in al]) if al else float('nan'):.2f}")
print(f"\nOutputs written to: {OUT_DIR}")
