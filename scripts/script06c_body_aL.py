# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 06c: a_L check of the node-branch stable orbits found in the body sweep
#             (Script 06, VARIANT_SET = "BODY").
#
# Loads orbits_script06_BODY.json. Colab ready. Runtime: about 5 to 10 min.
#
# Script 06 found 61/103 stable node-branch orbits at body ratio 1.1, zeta 0.1,
# and 3/52 at body ratio 1.0, zeta 0.2. The reduced theory forbids node-branch
# stability only when a_L > 0. This script:
#   1. rebuilds each orbit by Newton from its saved (x, CPG phase), keeping it
#      only if it returns to the same position with the same stability
#   2. applies the Script 05c analysis: all fixed-position locks at x* from 12
#      phases, branch matching by CPG phase, a_L on the matched branch (Newton
#      continuation), a_P, gamma, reduced-cubic prediction vs Floquet
# Groups per variant: all stable node-branch orbits, 20 unstable node-branch
# controls, 20 stable saddle-branch controls.
#
# Verdict: if the stable node-branch orbits have a_L < 0 (and the cubic agrees),
# the theory holds and the result is stated as the criterion a_L < 0 for
# entrained holding, met near body resonance at low damping.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
SEED = 10
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
IN_FILE = os.path.join(BASE_DIR, "script06_robustness", "orbits_script06_BODY.json")
OUT_DIR = os.path.join(BASE_DIR, "script06c_body_aL")
os.makedirs(OUT_DIR, exist_ok=True)

SPP = 100 if MODE == "FULL" else 80
N_CTRL = 20 if MODE == "FULL" else 4
MAX_NODE_STABLE = 70 if MODE == "FULL" else 6
G_HOLD = 16.0
TARGETS = [("body 1.1, zeta 0.1", 1.1, 0.1), ("body 1.0, zeta 0.2", 1.0, 0.2)]

OMEGA_W = 2.0 * np.pi
WAKE = dict(U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35,
            l_w=1.50, c_L=3.0, c_D=0.15, sigma=8.0)
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU = 2 * np.pi
G_LOCK = 8.0                    # default gain (make_P)
X_COLL, X_ESC = 0.15, 4.5

LOCK_TOL = 0.3          # max strobe excursion (rad) for a 1:1 lock


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

def analyse():
    global aL, match_k, pred, flo, gam, aP
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

    return part1, rows

def rebuild(orbs):
    """Newton from saved (x, phase): returns states (6, n) and a keep mask."""
    n = len(orbs)
    x = np.array([o["x"] for o in orbs]); ph = np.array([o["ph"] for o in orbs])
    dp = np.array([o["dphi"] for o in orbs])
    Pq = make_P(n, g=G_HOLD, presc=True, freeze=True, phi0=ph, dphi=dp)
    Rq = integrate(init_state(x, ph), Pq, 40)
    s = Rq["s"].copy()
    s[0] = x; s[1] = 0.0; s[4] = np.cos(ph); s[5] = np.sin(ph)
    P = P_for(n, np.array([o["eps"] for o in orbs]), np.array([o["ratio"] for o in orbs]), dp)
    s, ok, mu = newton(s, P, list(range(6)), n_it=12, clip=0.05)
    st = np.abs(mu).max(1) < 1 - 1e-6
    rate = np.log(np.abs(mu).max(1))
    keep = ok & (np.abs(s[0] - x) < 0.01) & (st == np.array([o["stable"] for o in orbs]))
    return s, keep, rate

t_start = time.time()
allorb = json.load(open(IN_FILE))
report = {}
for vname, ratio_b, zeta_b in TARGETS:
    if vname not in allorb or not allorb[vname]:
        print(f"{vname}: no saved orbits, skipped"); continue
    OM_B = ratio_b * OMEGA_W; ZE_B = zeta_b
    orbs = allorb[vname]
    k_s = np.array([o.get("k", np.nan) if o.get("k") is not None else np.nan for o in orbs], float)
    st_s = np.array([o["stable"] for o in orbs])
    i_ns = np.where(st_s & (k_s > 0))[0]
    if i_ns.size > MAX_NODE_STABLE:
        i_ns = rng.choice(i_ns, MAX_NODE_STABLE, replace=False)
    i_nu = np.where(~st_s & (k_s > 0))[0]; i_nu = rng.choice(i_nu, min(N_CTRL, i_nu.size), replace=False) if i_nu.size else i_nu
    i_ss = np.where(st_s & (k_s < 0))[0]; i_ss = rng.choice(i_ss, min(N_CTRL, i_ss.size), replace=False) if i_ss.size else i_ss
    pick = np.concatenate([i_ns, i_nu, i_ss]).astype(int)
    grp = np.array(["node-stable"] * i_ns.size + ["node-unstable ctrl"] * i_nu.size + ["saddle-stable ctrl"] * i_ss.size)
    sub_orbs = [orbs[i] for i in pick]
    S, keep, rate_new = rebuild(sub_orbs)
    print(f"{vname}: rebuilt {keep.sum()}/{keep.size} orbits ({time.time() - t_start:.0f} s)")
    # globals expected by the 05c analysis
    ST = S[:, keep].T
    eps_o = np.array([o["eps"] for o in sub_orbs])[keep]
    rat_o = np.array([o["ratio"] for o in sub_orbs])[keep]
    x_o = ST[:, 0]
    dphi_o = np.array([o["dphi"] for o in sub_orbs])[keep]
    rate_o = rate_new[keep]
    k_o = k_s[pick][keep]
    group = np.where(grp[keep] == "node-stable", "node-stable", grp[keep])
    sel = np.arange(keep.sum())
    NS = sel.size
    if NS == 0:
        continue
    part1, rows = analyse()
    ns_ = part1.get("node-stable")
    m = (group == "node-stable")
    report[vname] = dict(groups=part1,
                         node_stable_aL_quartiles=[float(v) for v in np.nanpercentile(aL[m], [0, 25, 50, 75, 100])] if m.any() else None,
                         node_stable_k_median=float(np.nanmedian(match_k[m])) if m.any() else None,
                         rows=rows)
    print(json.dumps(part1, indent=1))

with open(os.path.join(OUT_DIR, "summary_script06c.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n================ VERDICT ================")
for vname, r in report.items():
    g_ = r["groups"].get("node-stable")
    if not g_:
        print(f"{vname}: no node-stable orbits rebuilt"); continue
    mb = g_["matched_branch"]
    print(f"{vname}: node-stable {g_['n']}; matched node {mb['node']}, saddle {mb['saddle']}, none {mb['none']}; "
          f"a_L < 0: {g_['aL_negative']}; cubic agrees {g_['theory_agrees']}/{g_['theory_evaluable']}")
    print(f"   a_L quartiles (node-stable): {r['node_stable_aL_quartiles']}")
    ok_th = g_["theory_evaluable"] > 0 and g_["theory_agrees"] / g_["theory_evaluable"] > 0.9
    on_node = mb["node"]
    if on_node and g_["aL_negative"] >= 0.9 * on_node and ok_th:
        print("   THEORY HOLDS: stable entrained (node-branch) holding comes with a_L < 0.")
    elif ok_th:
        print("   THEORY HOLDS (cubic agrees) but some node-stable orbits have a_L >= 0: inspect rows.")
    else:
        print("   THEORY FAILS here: cubic disagrees with Floquet; inspect rows before writing.")
print(f"\nOutputs: {OUT_DIR}")
