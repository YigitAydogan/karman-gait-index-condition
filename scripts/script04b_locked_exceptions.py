# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 04b: Re-check the locked equilibria that Script 04 flagged as exceptions.
#
# Loads results_script04.npz. Colab ready. Runtime: about 3 to 6 min.
#
# The reduced theory forbids stable station holding when the relocked gradient
# a_L > 0 and the locking stiffness k > 0. Script 04 found 0% stable in that
# class, but its exceptions (a_L <= 0) came with |a_L| up to 13 and some k < 0,
# signs of a two-point gradient straddling a lock-branch jump or a lost lock.
#
# For every flagged locked equilibrium (a_L <= 0, k <= 0, |a_L| > 2, or Floquet
# stable) plus random controls:
#   1. local frozen landscape at 17 offsets in [-0.04, 0.04], long transients,
#      seeded with the locked phase (continuation) and with random phases
#      (multistability check)
#   2. a_L from a linear fit on the continuous locked branch (|offset| <= 0.02)
#   3. k from a converged 4D periodic orbit (Newton), with residual check
#   4. full 6D Floquet with more Newton iterations
#   5. "genuine" = locked at all central offsets, phase continuous, linear fit
#      R^2 > 0.9, k > 0 with converged orbit, full orbit converged
# Then re-test the theory on genuine cases only.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODE = "FULL"
SEED = 6
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
IN_FILE = os.path.join(BASE_DIR, "script04_free_position", "results_script04.npz")
OUT_DIR = os.path.join(BASE_DIR, "script04b_locked_exceptions")
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


SPP = 100 if MODE == "FULL" else 80
N_CTRL = 40 if MODE == "FULL" else 10
OFFS = np.linspace(-0.04, 0.04, 17)
CENTRAL = np.abs(OFFS) <= 0.02 + 1e-12
T_TR2, T_ST2 = 200, 40
N_RAND = 4                     # random-phase restarts at x* (multistability)

# -----------------------------------------------------------------------------
# Load Script 04 results and rebuild the neural delay of each pair
# -----------------------------------------------------------------------------
t_start = time.time()
d = np.load(IN_FILE, allow_pickle=True)
kind_p = d["kind"]; eps_p = d["pairs_eps"]; rat_p = d["pairs_ratio"]; g_p = d["pairs_g"]
if "pairs_dphi" in d.files:
    dphi_p = d["pairs_dphi"]
else:
    # Script 04 order: for each g in G_LOCK_LIST, for each eps, (g=16: for each dphi), ratios
    dphi_p = np.zeros(kind_p.size)
    idx16 = np.where((kind_p == "lock") & (g_p == 16.0))[0]
    per_eps = idx16.size // len(EPS_LIST)
    n_r = per_eps // len(DPHI_LIST)
    assert per_eps * len(EPS_LIST) == idx16.size and n_r * len(DPHI_LIST) == per_eps, "pair layout mismatch"
    for n, i in enumerate(idx16):
        dphi_p[i] = DPHI_LIST[(n % per_eps) // n_r]
eq_pair = d["eq_pair"]; eq_x = d["eq_x"]
a_L0 = d["a_L"]; a_P = d["a_P"]; k0 = d["k_lock"]; gam = d["gamma"]
flo0 = d["flo_rate"]; conv0 = d["flo_conv"]; phi_lock = d["phi_lock"]
ek = kind_p[eq_pair]
lockm = np.isin(ek, ["lock", "lock_c1"]) & conv0
flag = lockm & ((a_L0 <= 0) | (k0 <= 0) | (np.abs(a_L0) > 2) | (flo0 < 0))
ctrl_pool = np.where(lockm & ~flag)[0]
ctrl = rng.choice(ctrl_pool, size=min(N_CTRL, ctrl_pool.size), replace=False) if ctrl_pool.size else np.array([], int)
sel = np.concatenate([np.where(flag)[0], ctrl]).astype(int)
is_flag = np.isin(sel, np.where(flag)[0])
NS = sel.size
print(f"Locked equilibria: {lockm.sum()}, flagged: {flag.sum()}, controls: {ctrl.size}")

def params(idx, B_rep=1, **over):
    pi_ = eq_pair[idx]
    P = make_P(idx.size * B_rep,
               wc=np.repeat(OMEGA_W * rat_p[pi_], B_rep), eps=np.repeat(eps_p[pi_], B_rep),
               c=np.repeat(np.where(kind_p[pi_] == "lock_c1", 1.0, 0.0), B_rep),
               g=np.repeat(g_p[pi_], B_rep), dphi=np.repeat(dphi_p[pi_], B_rep), freeze=True)
    for k_, v_ in over.items():
        P[k_] = np.broadcast_to(np.asarray(v_), P[k_].shape).copy()
    return P

# -----------------------------------------------------------------------------
# 1. Local landscapes (continuation from locked phase) and random restarts
# -----------------------------------------------------------------------------
NO = OFFS.size
P1 = params(sel, NO)
X1 = (eq_x[sel][:, None] + OFFS[None, :]).ravel()
R1 = integrate(init_state(X1, np.repeat(phi_lock[sel], NO)), P1, T_TR2)
R1b = integrate(R1["s"], P1, T_ST2, t0=R1["t"], mean_last=20, strobe=True)
F1 = R1b["F"].reshape(NS, NO)
lock1 = (R1b["span"] < LOCK_TOL).reshape(NS, NO)
ph1 = np.mod(R1b["phase"], 2 * np.pi).reshape(NS, NO)

P2 = params(sel, N_RAND)
X2 = np.repeat(eq_x[sel], N_RAND)
R2 = integrate(init_state(X2), P2, T_TR2)
R2b = integrate(R2["s"], P2, T_ST2, t0=R2["t"], mean_last=20, strobe=True)
lock2 = (R2b["span"] < LOCK_TOL).reshape(NS, N_RAND)
ph2 = np.mod(R2b["phase"], 2 * np.pi).reshape(NS, N_RAND)
F2 = R2b["F"].reshape(NS, N_RAND)
print(f"1: local landscapes done ({time.time() - t_start:.0f} s)")

def circ_diff(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))

c0 = NO // 2
aL_fit = np.full(NS, np.nan); r2_fit = np.full(NS, np.nan)
cont = np.zeros(NS, bool); all_lock = np.zeros(NS, bool); multistable = np.zeros(NS, bool)
for n in range(NS):
    m = CENTRAL
    all_lock[n] = lock1[n, m].all()
    cont[n] = np.all(circ_diff(ph1[n, m], ph1[n, c0]) < 0.5)
    if all_lock[n]:
        A = np.vstack([OFFS[m], np.ones(m.sum())]).T
        coef, *_ = np.linalg.lstsq(A, F1[n, m], rcond=None)
        pred = A @ coef
        ss = np.sum((F1[n, m] - F1[n, m].mean()) ** 2)
        aL_fit[n] = coef[0]
        r2_fit[n] = 1 - np.sum((F1[n, m] - pred) ** 2) / ss if ss > 0 else 0.0
    lp = ph2[n][lock2[n]]
    multistable[n] = bool(lp.size and np.any(circ_diff(lp, ph1[n, c0]) > 0.5))

# -----------------------------------------------------------------------------
# 2. Converged 4D locked orbit at x*: locking stiffness k
# -----------------------------------------------------------------------------
def pmap(s, P, t0):
    return integrate(s, P, 1, t0=t0)["s"]
def jacobian(s, P, t0, dims, h=1e-6):
    s1 = pmap(s, P, t0)
    J = np.zeros((s.shape[1], len(dims), len(dims)))
    for a_, j in enumerate(dims):
        ds = np.zeros_like(s); ds[j] = h
        J[:, :, a_] = ((pmap(s + ds, P, t0) - s1)[dims] / h).T
    return J, s1
P3 = params(sel)
s3 = R1b["s"].reshape(6, NS, NO)[:, :, c0].copy()
t3 = 0.0
for _ in range(10):
    J, s1 = jacobian(s3, P3, t3, [2, 3, 4, 5])
    r = (s1 - s3)[2:]
    step = np.linalg.solve(J - np.eye(4)[None], -r.T[:, :, None])[:, :, 0].T
    s3[2:] += np.clip(np.nan_to_num(step), -0.2, 0.2)
J, s1 = jacobian(s3, P3, t3, [2, 3, 4, 5])
res3 = np.abs(s1 - s3)[2:].max(0)
mu4 = np.abs(np.linalg.eigvals(J)).max(1)
k_new = -np.log(mu4)

# -----------------------------------------------------------------------------
# 3. Full 6D Floquet at the refined equilibrium
# -----------------------------------------------------------------------------
x_star = eq_x[sel].copy()
for n in range(NS):          # refine x* on the local landscape (linear interpolation)
    Fi, ok_ = F1[n], lock1[n]
    for j in range(NO - 1):
        if ok_[j] and ok_[j + 1] and np.sign(Fi[j]) != np.sign(Fi[j + 1]):
            x_star[n] = eq_x[sel][n] + OFFS[j] - Fi[j] * (OFFS[j + 1] - OFFS[j]) / (Fi[j + 1] - Fi[j])
            break
P4 = params(sel, freeze=False)
s4 = s3.copy(); s4[0] = x_star; s4[1] = 0.0
for _ in range(20):
    J, s1 = jacobian(s4, P4, 0.0, list(range(6)))
    r = s1 - s4
    step = np.linalg.solve(J - np.eye(6)[None], -r.T[:, :, None])[:, :, 0].T
    s4 += np.clip(np.nan_to_num(step), -0.05, 0.05)
J, s1 = jacobian(s4, P4, 0.0, list(range(6)))
res4 = np.abs(s1 - s4).max(0)
rate_new = np.log(np.abs(np.linalg.eigvals(J)).max(1))
conv4 = (res4 < 1e-6) & (np.abs(s4[0] - x_star) < 0.05)

genuine = all_lock & cont & (r2_fit > 0.9) & (k_new > 0) & (res3 < 1e-6) & conv4

# reduced cubic with refit a_L and new k
pred_new = np.full(NS, np.nan)
for n in range(NS):
    if np.isfinite(aL_fit[n]):
        i = sel[n]
        roots = np.roots([1.0, k_new[n] + gam[i], gam[i] * k_new[n] - a_P[i], -aL_fit[n] * k_new[n]])
        pred_new[n] = roots.real.max()
print(f"3: Floquet done ({time.time() - t_start:.0f} s)")

# -----------------------------------------------------------------------------
# 4. Tests on genuine cases
# -----------------------------------------------------------------------------
G = genuine
pos = G & (aL_fit > 0)
neg = G & (aL_fit <= 0)
stab = rate_new < 0
res = dict(
    n_selected=int(NS), n_flagged=int(is_flag.sum()), n_genuine=int(G.sum()),
    flagged_genuine=int((G & is_flag).sum()),
    flagged_reasons_artefact=dict(
        lock_lost_near_xstar=int((is_flag & ~all_lock).sum()),
        phase_discontinuous=int((is_flag & all_lock & ~cont).sum()),
        poor_linear_fit=int((is_flag & all_lock & cont & ~(r2_fit > 0.9)).sum()),
        k_not_positive_or_unconverged=int((is_flag & ~((k_new > 0) & (res3 < 1e-6))).sum()),
        full_orbit_unconverged=int((is_flag & ~conv4).sum())),
    multistable_locked_branches=int(multistable.sum()),
    aL_fit_range_genuine=[float(np.nanmin(aL_fit[G])), float(np.nanmax(aL_fit[G]))] if G.any() else None,
    aL_two_point_range_flagged=[float(np.min(a_L0[sel][is_flag])), float(np.max(a_L0[sel][is_flag]))] if is_flag.any() else None,
    genuine_aL_positive_n=int(pos.sum()),
    genuine_aL_positive_stable=int((pos & stab).sum()),
    genuine_aL_nonpositive_n=int(neg.sum()),
    genuine_aL_nonpositive_stable=int((neg & stab).sum()),
    theory_agreement_genuine=float(np.mean((pred_new[G] < 0) == stab[G])) if G.any() else None,
    theory_rate_spearman_genuine=None,
    control_agreement_with_script04_rate=float(np.median(np.abs(rate_new[~is_flag & G] - flo0[sel][~is_flag & G])))
        if (~is_flag & G).any() else None,
)
if G.sum() > 2:
    ra = np.argsort(np.argsort(pred_new[G])); rb = np.argsort(np.argsort(rate_new[G]))
    res["theory_rate_spearman_genuine"] = float(np.corrcoef(ra, rb)[0, 1])
table = []
for n in np.where(is_flag)[0]:
    i = sel[n]
    table.append(dict(eq=int(i), g=float(g_p[eq_pair[i]]), eps=float(eps_p[eq_pair[i]]),
                      ratio=float(rat_p[eq_pair[i]]), dphi=float(dphi_p[eq_pair[i]]),
                      x=float(x_star[n]), aL_2pt=float(a_L0[i]), aL_fit=float(aL_fit[n]),
                      r2=float(r2_fit[n]), k_old=float(k0[i]), k_new=float(k_new[n]),
                      rate_old=float(flo0[i]), rate_new=float(rate_new[n]),
                      pred_new=float(pred_new[n]), all_lock=bool(all_lock[n]), cont=bool(cont[n]),
                      multistable=bool(multistable[n]), genuine=bool(G[n])))

# -----------------------------------------------------------------------------
# Figure: local landscapes of flagged cases
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
fl = np.where(is_flag)[0][:12]
if fl.size:
    nc = 4; nr = int(np.ceil(fl.size / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.6, 1.9 * nr), squeeze=False)
    for a_, n in zip(axs.ravel(), fl):
        a_.plot(OFFS, F1[n], "-", color="0.6", lw=0.8)
        a_.scatter(OFFS, F1[n], s=8, c=np.where(lock1[n], "#1a9850", "#d73027"))
        a_.axhline(0, color="0.7", lw=0.5); a_.axvline(0, color="0.7", lw=0.5)
        a_.set_title(f"aL fit {aL_fit[n]:.2f} | {'genuine' if G[n] else 'artefact'}", fontsize=7)
        a_.tick_params(labelsize=6)
    for a_ in axs.ravel()[fl.size:]:
        a_.axis("off")
    fig.supxlabel(r"offset from $x^*$ (green locked, red unlocked)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_flagged_landscapes.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig01_flagged_landscapes.svg"))
    plt.close(fig)

with open(os.path.join(OUT_DIR, "summary_script04b.json"), "w") as f:
    json.dump(dict(results=res, flagged_table=table, runtime_s=round(time.time() - t_start, 1)), f, indent=2)
print("\n================ SCRIPT 04b SUMMARY ================")
print(json.dumps(res, indent=2))
print("\nFlagged cases:")
for row in table:
    print("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
print("\n================ VERDICT ================")
if res["genuine_aL_positive_stable"] == 0:
    print(f"Theory holds on all {res['genuine_aL_positive_n']} genuine locked equilibria with a_L > 0 (none stable).")
else:
    print(f"THEORY VIOLATED: {res['genuine_aL_positive_stable']} genuine locked equilibria with a_L > 0 are stable.")
print(f"Genuine locked equilibria with a_L <= 0: {res['genuine_aL_nonpositive_n']} "
      f"(stable: {res['genuine_aL_nonpositive_stable']}). Theory allows these to be stable.")
if res["genuine_aL_nonpositive_n"] == 0:
    print("Every a_L <= 0 value in Script 04 was a finite-difference error (refit a_L > 0):")
    print("no locked fish holds station in the developed street.")
print(f"\nOutputs written to: {OUT_DIR}")
