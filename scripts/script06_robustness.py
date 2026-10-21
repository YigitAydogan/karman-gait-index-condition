# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 06: Robustness of the central result across CPG, body and wake
#            parameters.
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime: about 2 to 3 min per variant (FULL); 14 variants in total.
# Set VARIANT_SET = "A" or "B" to run half the variants per session, or "ALL".
# VARIANT_SET = "BODY" maps the body-tuning window: stiffness ratio 0.9 to 1.5
# at zeta 0.1 and 0.2 (12 variants, about 25 to 30 min).
#
# Central result (Scripts 04 to 05c): in the developed street, stable station
# holding never occurs on the fixed-position node branch with a_L > 0; it occurs
# on the saddle branch and where no fixed-position lock exists, at weak coupling.
#
# For each variant the 05b grid search is repeated:
#   - seeds: positional equilibria of the prescribed rhythm (dphi in {0, pi/2})
#   - eps rows 0.25 to 4.75 (step 0.5); the omega_c window follows the tilted
#     tongue: wide scan on the first row, then +/- 0.05 around the previous
#     row's orbits
#   - every station-holding orbit: Floquet stability, fixed-position k
# Reported per variant: stable counts by branch, Fisher p (node vs saddle),
# stable eps and omega_c ranges.
#
# Gate: robust if, in every variant with >= 20 node-branch orbits, the stable
# node fraction is < 2% and stable holding exists off the node branch.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
VARIANT_SET = "BODY"     # "A", "B", "ALL" or "BODY" (body-tuning window)
SEED = 9
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script06_robustness")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if MODE == "FULL":
    SPP = 100
    N_PHI_SEED, N_X_SEED = 12, 50
    EPS_ROWS = np.arange(0.25, 4.76, 0.5)
    WIDE = np.linspace(0.70, 1.50, 81)
    N_LOCAL, HALF_W = 21, 0.05
else:
    SPP = 80
    N_PHI_SEED, N_X_SEED = 6, 30
    EPS_ROWS = np.array([0.75, 1.75])
    WIDE = np.linspace(0.80, 1.30, 13)
    N_LOCAL, HALF_W = 9, 0.04
DPHI_SEEDS = [0.0, np.pi / 2]
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
VARIANTS = [
    ("baseline", {}),
    ("c = 1", dict(c=1.0)),
    ("mu x 0.5", dict(mu=np.pi)),
    ("mu x 2", dict(mu=4 * np.pi)),
    ("g = 12", dict(g=12.0)),
    ("g = 24", dict(g=24.0)),
    ("body ratio 0.8", dict(body_ratio=0.8)),
    ("body ratio 1.6", dict(body_ratio=1.6)),
    ("zeta 0.1", dict(zeta=0.1)),
    ("zeta 0.4", dict(zeta=0.4)),
    ("W0 0.25", dict(W0=0.25)),
    ("W0 0.45", dict(W0=0.45)),
    ("c_D 0.10", dict(c_D=0.10)),
    ("c_D 0.25", dict(c_D=0.25)),
]
BODY_VARIANTS = [(f"body {r:.1f}, zeta {z:.1f}", dict(body_ratio=r, zeta=z))
                 for r in (0.9, 1.0, 1.1, 1.3, 1.4, 1.5) for z in (0.1, 0.2)]
if VARIANT_SET == "A":
    VARIANTS = VARIANTS[:7]
elif VARIANT_SET == "B":
    VARIANTS = VARIANTS[7:]
elif VARIANT_SET == "BODY":
    VARIANTS = BODY_VARIANTS
if MODE == "QUICK":
    VARIANTS = VARIANTS[:2]

def apply_variant(over):
    """Set module-level model constants for a variant; returns (g, c)."""
    global MU, OM_B, ZE_B, WAKE
    cfg = dict(BASE); cfg.update({k: v for k, v in over.items() if k in BASE})
    MU = cfg["mu"]; OM_B = cfg["body_ratio"] * OMEGA_W; ZE_B = cfg["zeta"]
    WAKE = dict(BASE_WAKE); WAKE.update({k: v for k, v in over.items() if k in BASE_WAKE})
    return cfg["g"], cfg["c"]

def P_for(B, eps, ratio, dphi, g, c, freeze=False):
    return make_P(B, wc=OMEGA_W * np.asarray(ratio), eps=eps, g=g, c=c, dphi=dphi, freeze=freeze)

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
    return s, res < 1e-7, np.linalg.eigvals(J)

def fisher_two_sided(a, b, c, d):
    from math import comb
    n1, n2, m = a + b, c + d, a + c
    N = n1 + n2
    if N == 0 or m == 0 or m == N:
        return 1.0
    def p(x):
        return comb(n1, x) * comb(n2, m - x) / comb(N, m)
    p0 = p(a)
    lo, hi = max(0, m - n2), min(n1, m)
    return float(sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))

def run_variant(name, over):
    t0 = time.time()
    g, c = apply_variant(over)
    # seeds from the prescribed rhythm
    PHIS = np.linspace(0, 2 * np.pi, N_PHI_SEED, endpoint=False)
    XG = np.linspace(0.3, 2.2, N_X_SEED)
    combos = [(d_, ph) for d_ in DPHI_SEEDS for ph in PHIS]
    nc = len(combos)
    Xb = np.tile(XG, nc)
    Pp = make_P(Xb.size, g=g, presc=True, freeze=True,
                phi0=np.repeat([c_[1] for c_ in combos], XG.size),
                dphi=np.repeat([c_[0] for c_ in combos], XG.size))
    Rp = integrate(init_state(Xb), Pp, 40)
    Fp = integrate(Rp["s"], Pp, 10, t0=Rp["t"], mean_last=10)["F"].reshape(nc, XG.size)
    sx, sph, sdp = [], [], []
    for ci in range(nc):
        for j in range(XG.size - 1):
            f0, f1 = Fp[ci, j], Fp[ci, j + 1]
            if np.sign(f0) != np.sign(f1):
                sx.append(XG[j] - f0 * (XG[j + 1] - XG[j]) / (f1 - f0))
                sph.append(combos[ci][1]); sdp.append(combos[ci][0])
    if not sx:
        return dict(name=name, n_seeds=0, n_orbits=0, note="no prescribed equilibria")
    sx, sph, sdp = np.array(sx), np.array(sph), np.array(sdp)
    Pq = make_P(sx.size, g=g, presc=True, freeze=True, phi0=sph, dphi=sdp)
    Rq = integrate(init_state(sx, sph), Pq, 40)
    SEEDS = Rq["s"].copy()
    SEEDS[0] = sx; SEEDS[1] = 0.0; SEEDS[4] = np.cos(sph); SEEDS[5] = np.sin(sph)

    orbits = []
    window = WIDE
    for e in EPS_ROWS:
        nq = window.size
        S = np.concatenate([SEEDS] * nq, axis=1)
        D = np.tile(sdp, nq)
        R = np.repeat(window, sx.size)
        Pn = P_for(S.shape[1], e, R, D, g, c)
        sn, ok, mu = newton(S, Pn, list(range(6)))
        ok &= valid(sn)
        a = np.abs(mu); lead_ = mu[np.arange(len(mu)), np.argmax(a, 1)]
        stable = a.max(1) < 1 - 1e-6
        kept = []
        for j in np.where(ok)[0]:
            ph = np.arctan2(sn[5, j], sn[4, j])
            dup = any(abs(o["ratio"] - R[j]) < 1e-9 and o["dphi"] == D[j] and abs(o["x"] - sn[0, j]) < 0.01
                      and abs(np.angle(np.exp(1j * (o["ph"] - ph)))) < 0.1 for o in orbits if o["eps"] == e)
            if not dup:
                orbits.append(dict(eps=float(e), ratio=float(R[j]), dphi=float(D[j]), x=float(sn[0, j]),
                                   ph=float(ph), stable=bool(stable[j]), rate=float(np.log(a[j].max())),
                                   torus=bool(np.abs(lead_[j].imag) > 1e-6), state=sn[:, j].copy()))
                kept.append(len(orbits) - 1)
        if kept:
            ks = np.array(kept)
            St = np.array([orbits[i]["state"] for i in ks]).T
            Pk = P_for(ks.size, e, np.array([orbits[i]["ratio"] for i in ks]),
                       np.array([orbits[i]["dphi"] for i in ks]), g, c, freeze=True)
            St[1] = 0.0
            sf, okf, muf = newton(St, Pk, [2, 3, 4, 5], n_it=12, clip=0.2)
            kf = np.where(okf, -np.log(np.abs(muf).max(1)), np.nan)
            for n_, i in enumerate(ks):
                orbits[i]["k"] = float(kf[n_])
            centre = np.median([orbits[i]["ratio"] for i in ks])
            window = np.linspace(centre - HALF_W, centre + HALF_W, N_LOCAL)
    if not orbits:
        return dict(name=name, n_seeds=int(sx.size), n_orbits=0, runtime_s=round(time.time() - t0, 1))
    k_ = np.array([o.get("k", np.nan) for o in orbits]); st = np.array([o["stable"] for o in orbits])
    cls = np.where(np.isnan(k_), "none", np.where(k_ > 0, "node", "saddle"))
    cnt = {cl: dict(n=int(np.sum(cls == cl)), stable=int(np.sum(st & (cls == cl)))) for cl in ("node", "saddle", "none")}
    nd_, sd_ = cnt["node"], cnt["saddle"]
    fp = fisher_two_sided(nd_["stable"], nd_["n"] - nd_["stable"], sd_["stable"], sd_["n"] - sd_["stable"])
    se = [o["eps"] for o in orbits if o["stable"]]
    sr = [o["ratio"] for o in orbits if o["stable"]]
    res = dict(name=name, override={k: float(v) for k, v in over.items()}, n_seeds=int(sx.size),
               n_orbits=len(orbits), counts=cnt, fisher_p=fp,
               stable_eps_range=[min(se), max(se)] if se else None,
               stable_ratio_range=[min(sr), max(sr)] if sr else None,
               n_torus_unstable=int(sum(1 for o in orbits if (not o["stable"]) and o["torus"])),
               runtime_s=round(time.time() - t0, 1))
    res["_orbits"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items() if k != "state"}
                      for o in orbits]
    return res

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
t_start = time.time()
print(f"Mode {MODE}, variant set {VARIANT_SET}: {len(VARIANTS)} variants")
results = []
for name, over in VARIANTS:
    r = run_variant(name, over)
    results.append(r)
    c_ = r.get("counts")
    if c_:
        print(f"  {name:16s} orbits {r['n_orbits']:4d} | node {c_['node']['stable']:3d}/{c_['node']['n']:<4d} "
              f"saddle {c_['saddle']['stable']:3d}/{c_['saddle']['n']:<4d} none {c_['none']['stable']:3d}/{c_['none']['n']:<4d}"
              f" | Fisher p {r['fisher_p']:.2g} | {r['runtime_s']:.0f} s")
    else:
        print(f"  {name:16s} no orbits ({r.get('note', '')})")
apply_variant({})

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
tag = VARIANT_SET if MODE == "FULL" else "quick"
ok_r = [r for r in results if r.get("counts")]
if ok_r:
    names = [r["name"] for r in ok_r]
    pct = lambda r, cl: (100 * r["counts"][cl]["stable"] / r["counts"][cl]["n"]) if r["counts"][cl]["n"] else np.nan
    yy = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.0, 0.35 * len(names) + 1.2))
    ax.barh(yy - 0.25, [pct(r, "node") for r in ok_r], height=0.25, color="#2166ac", label="node (k>0)")
    ax.barh(yy, [pct(r, "saddle") for r in ok_r], height=0.25, color="#b2182b", label="saddle (k<0)")
    ax.barh(yy + 0.25, [pct(r, "none") for r in ok_r], height=0.25, color="#999999", label="no fixed lock")
    ax.set_yticks(yy); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("% of station-holding orbits that are stable")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig01_robustness_{tag}.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"fig01_robustness_{tag}.svg"))
    plt.close(fig)

# -----------------------------------------------------------------------------
# Save and gate
# -----------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, f"summary_script06_{tag}.json"), "w") as f:
    json.dump(dict(mode=MODE, variant_set=VARIANT_SET, eps_rows=EPS_ROWS.tolist(),
                   results=[{k: v for k, v in r.items() if k != "_orbits"} for r in results],
                   runtime_s=round(time.time() - t_start, 1)), f, indent=2)
with open(os.path.join(OUT_DIR, f"orbits_script06_{tag}.json"), "w") as f:
    json.dump({r["name"]: r.get("_orbits", []) for r in results}, f)

print("\n================ DECISION GATE (robustness) ================")
fails, weak = [], []
for r in results:
    c_ = r.get("counts")
    if not c_:
        weak.append((r["name"], "no orbits"))
        continue
    nd_ = c_["node"]
    off = c_["saddle"]["stable"] + c_["none"]["stable"]
    if nd_["n"] >= 20 and nd_["stable"] / nd_["n"] >= 0.02:
        fails.append((r["name"], f"node stable {nd_['stable']}/{nd_['n']}"))
    elif nd_["n"] < 20 or off == 0 or r["fisher_p"] >= 0.01:
        weak.append((r["name"], f"node n={nd_['n']}, off-node stable={off}, p={r['fisher_p']:.2g}"))
if not fails:
    print("ROBUST: no variant has >= 2% stable node-branch orbits.")
else:
    print("NOT ROBUST in:", fails)
    print("Check a_L on those node-branch orbits (Script 05c method) before concluding.")
if weak:
    print("Under-powered or no stable holding (report, do not over-interpret):")
    for w in weak:
        print("   ", w)
print(f"\nTotal runtime {time.time() - t_start:.0f} s. Outputs: {OUT_DIR}")
