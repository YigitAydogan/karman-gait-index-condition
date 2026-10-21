# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 02b: Inspect the passive exceptions found in Script 02.
#
# Loads results_script02.npz from Drive (Script 02 must have been run).
# Colab ready. Needs only numpy and matplotlib. Runtime: about 5 to 10 min.
#
# Questions:
#   Q1. Which wake/force parameters predict passive stability? (ranked AUC)
#   Q2. Floquet confusion matrix: averaged class vs full Floquet class, for
#       every averaged-stable and oscillatory equilibrium plus 300 saddles.
#       Which way do the disagreements go?
#   Q3. Robustness of stable equilibria: direct simulation with kicks of
#       0.01, 0.05, 0.1, 0.2 lambda in both directions (basin size).
#   Q4. Oscillatory (Hopf-type) cases: do they settle on bounded surging
#       limit cycles, or collide / escape?
#   Q5. Are the few downstream stable cases genuine or sitting at the peak?
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODE = "FULL"   # "FULL" or "QUICK"
SEED = 3
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
IN_FILE = os.path.join(BASE_DIR, "script02_passive_robustness", "results_script02.npz")
OUT_DIR = os.path.join(BASE_DIR, "script02b_passive_cases")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SPP = 200 if MODE == "FULL" else 120
N_SADDLE = 300 if MODE == "FULL" else 30
T_SIM = 300 if MODE == "FULL" else 120
KICKS = [0.01, 0.05, 0.1, 0.2]
OMEGA_W = 2.0 * np.pi
X_COLL, X_ESC = 0.15, 4.5
MARGINAL = 2e-3
PNAMES = ["U_inf", "delta0", "l_delta", "W0", "x_r", "l_w", "c_L", "c_D", "sigma"]
SHAPE_NAMES = {0: "exponential", 1: "plateau", 2: "algebraic", 3: "bump"}
BODY_NAMES = ["rigid", "soft", "resonant", "resonant-damped", "stiff", "very stiff"]

# -----------------------------------------------------------------------------
# Model (identical to Script 02, no heave)
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

def rhs_full(t, s, p):
    x, u, th, om = s
    V = U_loc(x, p) - u
    w = W_env(x, p) * np.sin(OMEGA_W * (x - t))
    beta = np.arctan2(w, np.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + np.tanh(V / 0.05))
    L = p["c_L"] * (V * V + w * w) * np.sin(alpha) * np.cos(alpha) * gate
    Fx = p["c_D"] * V * np.abs(V) - L * np.sin(beta)
    rig = p["rigid"]
    dom = np.where(rig, 0.0, -2 * p["ze"] * p["Om"] * om - p["Om"] ** 2 * th + p["sigma"] * L)
    dth = np.where(rig, 0.0, om)
    return np.array([u, Fx, dth, dom])

def rk4_period(s0, p, n=SPP):
    s = s0.copy()
    dt = 1.0 / n
    t = 0.0
    for _ in range(n):
        k1 = rhs_full(t, s, p)
        k2 = rhs_full(t + dt / 2, s + dt / 2 * k1, p)
        k3 = rhs_full(t + dt / 2, s + dt / 2 * k2, p)
        k4 = rhs_full(t + dt, s + dt * k3, p)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return s

def body_guess(x0, p, n_per=40):
    """Body state on the periodic response with x frozen (brute force)."""
    s = np.stack([x0, np.zeros_like(x0), np.zeros_like(x0), np.zeros_like(x0)])
    for _ in range(n_per):
        s1 = rk4_period(s, p)
        s1[0] = x0
        s1[1] = 0.0
        s = s1
    return s

def floquet(p, s_guess, n_newton=12, eps=1e-6):
    s = s_guess.copy()
    B = s.shape[1]
    rig = p["rigid"]
    for _ in range(n_newton):
        s1 = rk4_period(s, p)
        r = s1 - s
        r[2:, rig] = 0.0
        J = np.zeros((B, 4, 4))
        for j in range(4):
            ds = np.zeros_like(s); ds[j] = eps
            J[:, :, j] = ((rk4_period(s + ds, p) - s1) / eps).T
        A = J - np.eye(4)[None]
        A[rig, 2:, :] = 0.0; A[rig, :, 2:] = 0.0
        A[rig, 2, 2] = A[rig, 3, 3] = -1.0
        step = np.linalg.solve(A, -r.T[:, :, None])[:, :, 0].T
        s = s + np.clip(np.nan_to_num(step), -0.2, 0.2)
    s1 = rk4_period(s, p)
    res = np.abs(s1 - s); res[2:, rig] = 0.0; res = res.max(axis=0)
    J = np.zeros((B, 4, 4))
    for j in range(4):
        ds = np.zeros_like(s); ds[j] = eps
        J[:, :, j] = ((rk4_period(s + ds, p) - s1) / eps).T
    J[rig, 2:, :] = 0.0; J[rig, :, 2:] = 0.0
    mu = np.linalg.eigvals(J)
    a = np.abs(mu)
    k = np.argmax(a, axis=1)
    m = mu[np.arange(B), k]
    lam = np.log(a.max(1))
    cls = np.where(a.max(1) <= 1 + 1e-9, 0, np.where(np.abs(m.imag) > 1e-8, 2, 1))
    return s, cls, lam, res

def simulate(s0, p, T):
    """Return fate (0 hold, 1 collide, 2 escape), loss time, and x history."""
    s = s0.copy()
    B = s.shape[1]
    fate = np.zeros(B, int)
    tl = np.full(B, np.nan)
    alive = np.ones(B, bool)
    xh = np.full((T, B), np.nan)
    for k in range(T):
        s = rk4_period(s, p)
        xh[k] = np.where(alive, s[0], np.nan)
        c = alive & (s[0] < X_COLL); e = alive & (s[0] > X_ESC)
        fate[c] = 1; fate[e] = 2; tl[c | e] = k + 1
        alive &= ~(c | e)
        s[:, ~alive] = 1.0
        if not alive.any():
            break
    return fate, tl, xh

def auc(score, label):
    """Rank-based ROC AUC; 0.5 = no information, symmetric around 0.5."""
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = label.sum(); n0 = (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
t_start = time.time()
d = np.load(IN_FILE)
P_all = {k[5:]: d[k] for k in d.files if k.startswith("Plhs_")}
eq_set = d["lhs_eq_set"]; eq_x = d["lhs_eq_x"]; cls_avg = d["lhs_cls"]
F_x = d["lhs_F_x"]; F_u = d["lhs_F_u"]; xpk = d["lhs_xpeak"]
body_idx = eq_set % len(BODY_NAMES)
shape_eq = P_all["shape"][eq_set]
print(f"Loaded {len(eq_x)} LHS equilibria: stable {np.sum(cls_avg == 0)}, "
      f"saddle {np.sum(cls_avg == 1)}, osc {np.sum(cls_avg == 2)}")

# -----------------------------------------------------------------------------
# Q1: predictors of stability
# -----------------------------------------------------------------------------
stable = cls_avg == 0
feats = {n: P_all[n][eq_set] for n in PNAMES}
feats["x_star"] = eq_x
feats["x_star / x_peak"] = eq_x / xpk
feats["U_loc(x*)/U_inf"] = U_loc(eq_x, {k: v[eq_set] for k, v in P_all.items()}) / P_all["U_inf"][eq_set]
feats["x_peak"] = xpk
auc_tab = {k: float(auc(v, stable)) for k, v in feats.items()}
auc_rank = sorted(auc_tab.items(), key=lambda kv: -abs(kv[1] - 0.5))
by_body = {b: float(np.mean(stable[body_idx == i])) for i, b in enumerate(BODY_NAMES)}
by_shape = {SHAPE_NAMES[s]: float(np.mean(stable[shape_eq == s])) for s in SHAPE_NAMES}
print("Q1 AUC ranking (|AUC - 0.5|):")
for k, v in auc_rank:
    print(f"   {k:>18s}: {v:.3f}")

# -----------------------------------------------------------------------------
# Q2: Floquet confusion matrix
# -----------------------------------------------------------------------------
print("Q2: Floquet re-check ...")
i_st = np.where(cls_avg == 0)[0]
i_os = np.where(cls_avg == 2)[0]
i_sd = np.where(cls_avg == 1)[0]
i_sd = rng.choice(i_sd, size=min(N_SADDLE, len(i_sd)), replace=False)
chk = np.concatenate([i_st, i_os, i_sd]).astype(int)
p_chk = {k: v[eq_set[chk]] for k, v in P_all.items()}
sg = body_guess(eq_x[chk], p_chk)
s_orb, cls_f, lam_f, res_f = floquet(p_chk, sg)
conv = (res_f < 1e-6) & (np.abs(s_orb[0] - eq_x[chk]) < 0.1)
marg = conv & (np.abs(lam_f) < MARGINAL)
conf = np.zeros((3, 4), int)  # rows averaged class, cols Floquet class 0/1/2 or 3 = unconverged
for a_, f_, c_ in zip(cls_avg[chk], cls_f, conv):
    conf[a_, f_ if c_ else 3] += 1
print("   confusion (rows averaged stable/saddle/osc; cols Floquet stable/saddle/osc/unconv):")
print(conf)
print(f"   marginal (|ln mu| < {MARGINAL}): {marg.sum()}")

# -----------------------------------------------------------------------------
# Q3 and Q4: direct simulations with kicks from Floquet-confirmed orbits
# -----------------------------------------------------------------------------
print("Q3/Q4: direct simulations ...")
sel = np.where(conv & np.isin(cls_avg[chk], [0, 2]))[0]
rows = []
for kick in KICKS:
    for sign in (+1, -1):
        rows.append((kick, sign))
B = len(sel)
s0 = np.concatenate([s_orb[:, sel]] * len(rows), axis=1)
for j, (kick, sign) in enumerate(rows):
    s0[0, j * B:(j + 1) * B] += sign * kick
p_sim = {k: np.concatenate([v[sel]] * len(rows)) for k, v in p_chk.items()}
fate, tl, xh = simulate(s0, p_sim, T_SIM)
fate = fate.reshape(len(rows), B)
xh = xh.reshape(T_SIM, len(rows), B)
tail = xh[-50:]
surge_amp = 0.5 * (np.nanmax(tail, axis=0) - np.nanmin(tail, axis=0))   # (rows, B)
x0_sel = s_orb[0, sel]
# basin: largest kick size at which both directions hold
held = (fate == 0).reshape(len(KICKS), 2, B).all(axis=1)
basin = np.zeros(B)          # largest kick held in both directions (and all smaller)
still = np.ones(B, bool)
for kk, kick in enumerate(KICKS):
    still &= held[kk]
    basin = np.where(still, kick, basin)
fl_cls_sel = cls_f[sel]
is_st = fl_cls_sel == 0
is_os = fl_cls_sel == 2
st_basin = {str(k): int(np.sum(basin[is_st] >= k)) for k in KICKS}
os_fates = dict(
    n=int(is_os.sum()),
    bounded_surging_small_kick=int(np.sum((fate[0, is_os] == 0) & (surge_amp[0, is_os] > 1e-3))),
    collide_small_kick=int(np.sum(fate[0, is_os] == 1)),
    escape_small_kick=int(np.sum(fate[0, is_os] == 2)),
    median_surge_amp=float(np.nanmedian(surge_amp[0, is_os])) if is_os.any() else np.nan)
print(f"   Floquet-stable cases: {is_st.sum()}, holding for kicks >= "
      + ", ".join(f"{k}: {v}" for k, v in st_basin.items()))
print(f"   oscillatory cases: {os_fates}")

# -----------------------------------------------------------------------------
# Q5: downstream stable cases
# -----------------------------------------------------------------------------
down = conv & (cls_f == 0) & (eq_x[chk] >= xpk[chk])
q5 = dict(n_floquet_stable_downstream=int(down.sum()),
          x_minus_peak=[float(v) for v in (eq_x[chk] - xpk[chk])[down]],
          growth=[float(v) for v in lam_f[down]])
print(f"Q5: Floquet-stable equilibria downstream of the vortex peak: {down.sum()}")

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
names = [k for k, _ in auc_rank]
vals = [auc_tab[k] for k in names]
ax[0].barh(names[::-1], np.array(vals[::-1]) - 0.5, color="#4575b4")
ax[0].axvline(0, color="k", lw=0.6)
ax[0].set_xlabel("AUC - 0.5 (stable vs not)")
ax[0].set_title("Q1 predictors", fontsize=8)
ax[0].tick_params(axis="y", labelsize=7)
r = eq_x / xpk
bins = np.linspace(0, 2.5, 26)
cen = 0.5 * (bins[1:] + bins[:-1])
fr = [np.mean(stable[(r >= a) & (r < b)]) if np.any((r >= a) & (r < b)) else np.nan
      for a, b in zip(bins[:-1], bins[1:])]
ax[1].plot(cen, np.array(fr) * 100, "o-", ms=3, color="#1a9850")
ax[1].axvline(1, color="k", lw=0.6, ls="--")
ax[1].set_xlabel(r"$x^*/x_{peak}$")
ax[1].set_ylabel("% stable")
ax[1].set_title("Stability vs vortex peak", fontsize=8)
im = ax[2].imshow(conf, cmap="Blues")
for (i, j), v in np.ndenumerate(conf):
    ax[2].text(j, i, str(v), ha="center", va="center", fontsize=8)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["st", "sad", "osc", "unc"])
ax[2].set_yticks(range(3)); ax[2].set_yticklabels(["st", "sad", "osc"])
ax[2].set_xlabel("Floquet"); ax[2].set_ylabel("averaged")
ax[2].set_title("Q2 confusion", fontsize=8)
fig.tight_layout()
save(fig, "fig01_predictors_confusion")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for idx_set, col, lab in ((np.where(is_st)[0][:6], "#1a9850", "stable"),
                          (np.where(is_os)[0][:6], "#e08214", "oscillatory")):
    for n, b in enumerate(idx_set):
        ax[0].plot(xh[:, 0, b] - x0_sel[b], color=col, lw=0.7, label=lab if n == 0 else None)
ax[0].set_xlabel("time (periods)")
ax[0].set_ylabel(r"$x - x^*$")
ax[0].set_title("Kick +0.01: stable and oscillatory cases", fontsize=8)
ax[0].legend(frameon=False, fontsize=7)
ax[1].bar([str(k) for k in KICKS], [st_basin[str(k)] for k in KICKS], color="#1a9850")
ax[1].set_xlabel(r"kick size ($\lambda$)")
ax[1].set_ylabel("stable cases still holding")
ax[1].set_title(f"Q3 basin size (of {is_st.sum()})", fontsize=8)
fig.tight_layout()
save(fig, "fig02_basins_surging")

# -----------------------------------------------------------------------------
# Save and verdict
# -----------------------------------------------------------------------------
n_fs = int(np.sum(conv & (cls_f == 0)))
confirmed_frac = n_fs / max(1, int(np.sum(conv)))
summary = dict(
    mode=MODE, runtime_s=round(time.time() - t_start, 1),
    Q1_auc_ranked=auc_rank, Q1_stable_frac_by_body=by_body, Q1_stable_frac_by_shape=by_shape,
    Q2_confusion_rows_avg_cols_floquet=conf.tolist(),
    Q2_avg_stable_confirmed=int(conf[0, 0]), Q2_avg_stable_refuted=int(conf[0, 1] + conf[0, 2]),
    Q2_avg_saddle_actually_stable=int(conf[1, 0]), Q2_marginal=int(marg.sum()),
    Q3_floquet_stable_n=int(is_st.sum()), Q3_holding_for_kick_at_least=st_basin,
    Q4_oscillatory=os_fates, Q5=q5,
    corrected_stable_fraction_LHS=float(
        (conf[0, 0] + conf[1, 0] * (np.sum(cls_avg == 1) / max(1, len(i_sd))) + conf[2, 0])
        / len(eq_x)),
)
np.savez_compressed(os.path.join(OUT_DIR, "results_script02b.npz"),
                    chk=chk, cls_f=cls_f, lam_f=lam_f, conv=conv, s_orb=s_orb,
                    sel=sel, fate=fate, surge_amp=surge_amp, basin=basin)
with open(os.path.join(OUT_DIR, "summary_script02b.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n================ SCRIPT 02b SUMMARY ================")
print(json.dumps(summary, indent=2))
print(f"\nOutputs written to: {OUT_DIR}")
