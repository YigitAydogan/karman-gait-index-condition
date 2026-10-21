# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 05: Station-holding tongue by continuation in the full system.
#
# Colab ready. Needs only numpy and matplotlib. Runtime: about 10 to 20 min (FULL).
#
# Background (Scripts 04, 04b): inside a fixed-position 1:1 tongue there are two
# locked cycles, a stable node branch (k > 0) and a repelling saddle branch
# (k < 0). Reduced theory: with free position the characteristic polynomial is
#   s^3 + (k + gamma) s^2 + (gamma k - a_P) s - a_L k = 0,
# so with a_L > 0 holding is impossible on the node branch (k > 0) but possible
# on the saddle branch (k < 0) when |k| < gamma and the positional well is deep.
# Script 04 could only reach node-branch equilibria (through fixed-position
# locking). Here station-holding orbits are found and followed directly in the
# full 6D system.
#
# Parts:
#   A. Seeds from the free-rhythm limit: positional equilibria of the prescribed
#      rhythm at all phases, for neural delays dphi in {0, pi}.
#   B. Grid search over (omega_c / omega_w, eps): at each eps row, Newton in the
#      full 6D system from the prescribed seeds (first rows and every few rows)
#      and from the previous row's orbits. Every distinct station-holding orbit
#      is kept, stable or not. (The self-loop shifts the rhythm's frequency, so
#      the tongue is tilted; a grid avoids assuming where it sits.)
#   C. Fixed-position branch of every orbit: node (k > 0), saddle (k < 0), or
#      no fixed-position lock.
#   D. Loss of stability: type of the leading multiplier of unstable orbits that
#      neighbour stable ones (+1 fold, -1 flip, complex torus).
#   E. Direct simulations from unstable torus / flip neighbours: Lyapunov, fate.
#
# Gate: is stable holding confined to the saddle branch (k < 0) or to
# no-fixed-lock states, as the theory predicts?
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODE = "FULL"
SEED = 7
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script05_holding_tongue")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if MODE == "FULL":
    SPP = 100
    N_PHI_SEED, N_X_SEED = 16, 60
    EPS_ROWS = np.arange(1.0, 20.01, 1.0)
    RATIOS = np.linspace(0.90, 1.35, 61)
    SEED_EVERY = 4
    T_LYAP = 400
else:
    SPP = 80
    N_PHI_SEED, N_X_SEED = 8, 40
    EPS_ROWS = np.array([2.0, 4.0, 8.0])
    RATIOS = np.linspace(0.90, 1.35, 16)
    SEED_EVERY = 2
    T_LYAP = 150
MAX_PER_CELL = 4
G_HOLD = 16.0
DPHI_SEEDS = [0.0, np.pi]

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

# -----------------------------------------------------------------------------
# A. Seeds from the free-rhythm (prescribed) limit
# -----------------------------------------------------------------------------
t_start = time.time()
print(f"Mode {MODE}: g = {G_HOLD}, {EPS_ROWS.size} eps rows x {RATIOS.size} ratios")
PHIS = np.linspace(0, 2 * np.pi, N_PHI_SEED, endpoint=False)
XG = np.linspace(0.3, 2.2, N_X_SEED)
combos = [(d_, ph) for d_ in DPHI_SEEDS for ph in PHIS]
nc = len(combos)
Xb = np.tile(XG, nc)
Pp = make_P(Xb.size, g=G_HOLD, presc=True, freeze=True,
            phi0=np.repeat([c_[1] for c_ in combos], XG.size),
            dphi=np.repeat([c_[0] for c_ in combos], XG.size))
Rp = integrate(init_state(Xb), Pp, 40)
Rp2 = integrate(Rp["s"], Pp, 10, t0=Rp["t"], mean_last=10)
Fp = Rp2["F"].reshape(nc, XG.size)
sx, sph, sdp = [], [], []
for c in range(nc):
    for j in range(XG.size - 1):
        f0, f1 = Fp[c, j], Fp[c, j + 1]
        if np.sign(f0) != np.sign(f1):
            sx.append(XG[j] - f0 * (XG[j + 1] - XG[j]) / (f1 - f0))
            sph.append(combos[c][1]); sdp.append(combos[c][0])
sx, sph, sdp = np.array(sx), np.array(sph), np.array(sdp)
Pq = make_P(sx.size, g=G_HOLD, presc=True, freeze=True, phi0=sph, dphi=sdp)
Rq = integrate(init_state(sx, sph), Pq, 40)
SEEDS = Rq["s"].copy()
SEEDS[0] = sx; SEEDS[1] = 0.0; SEEDS[4] = np.cos(sph); SEEDS[5] = np.sin(sph)
SEED_DPHI = sdp
print(f"A: {sx.size} prescribed seeds ({time.time() - t_start:.0f} s)")

# -----------------------------------------------------------------------------
# B. Grid search over (ratio, eps)
# -----------------------------------------------------------------------------
def classify(mu):
    m, a = lead(mu)
    stable = a < 1 - 1e-6
    typ = np.where(stable, "stable", np.where(np.abs(m.imag) > 1e-6, "torus",
                   np.where(m.real < 0, "flip", "fold")))
    return stable, typ, np.log(a)

orbits = []      # dicts: eps, ratio, dphi, state, rate, type, k
prev = {}        # (ratio index) -> list of (state, dphi)
for ri, e in enumerate(EPS_ROWS):
    t0 = time.time()
    cand_s, cand_d, cand_r = [], [], []
    use_seeds = (ri % SEED_EVERY == 0)
    for qi, r in enumerate(RATIOS):
        if use_seeds:
            cand_s.append(SEEDS); cand_d.append(SEED_DPHI); cand_r.append(np.full(SEED_DPHI.size, qi))
        for st_, dp_ in prev.get(qi, []):
            cand_s.append(st_[:, None]); cand_d.append(np.array([dp_])); cand_r.append(np.array([qi]))
    if not cand_s:
        continue
    S = np.concatenate(cand_s, axis=1); D = np.concatenate(cand_d); Q = np.concatenate(cand_r)
    Pn = P_for(S.shape[1], e, RATIOS[Q], D)
    sn, ok, mu = newton(S, Pn, list(range(6)), n_it=10, clip=0.05)
    ok &= valid(sn)
    st, ty, ra = classify(mu)
    new_prev = {}
    kept = []
    for j in np.where(ok)[0]:
        qi = int(Q[j])
        lst = new_prev.setdefault(qi, [])
        ph = np.arctan2(sn[5, j], sn[4, j])
        dup = any(abs(o[0][0] - sn[0, j]) < 0.01 and
                  abs(np.angle(np.exp(1j * (np.arctan2(o[0][5], o[0][4]) - ph)))) < 0.1 and
                  o[1] == D[j] for o in lst)
        if not dup and len(lst) < MAX_PER_CELL:
            lst.append((sn[:, j].copy(), D[j]))
            kept.append(j)
    prev = new_prev
    if kept:
        kept = np.array(kept)
        kk = fixed_k(sn[:, kept], P_for(kept.size, e, RATIOS[Q[kept]], D[kept]))
        for n_, j in enumerate(kept):
            orbits.append(dict(eps=float(e), qi=int(Q[j]), ratio=float(RATIOS[Q[j]]), dphi=float(D[j]),
                               x=float(sn[0, j]), rate=float(ra[j]), type=str(ty[j]), k=float(kk[n_]),
                               lead_re=float(lead(mu[j:j + 1])[0][0].real),
                               lead_im=float(lead(mu[j:j + 1])[0][0].imag), state=sn[:, j].copy()))
    print(f"   eps {e:5.2f}: candidates {S.shape[1]}, orbits {len(kept)}, "
          f"stable {sum(1 for o in orbits if o['eps'] == e and o['type'] == 'stable')}, "
          f"{time.time() - t0:.0f} s")
NO_ = len(orbits)
print(f"B: {NO_} distinct station-holding orbits ({time.time() - t_start:.0f} s)")
if NO_ == 0:
    raise SystemExit("No station-holding orbits found.")

# -----------------------------------------------------------------------------
# C. Theory test by fixed-position branch
# -----------------------------------------------------------------------------
eps_o = np.array([o["eps"] for o in orbits]); rat_o = np.array([o["ratio"] for o in orbits])
x_o = np.array([o["x"] for o in orbits]); rate_o = np.array([o["rate"] for o in orbits])
typ_o = np.array([o["type"] for o in orbits]); k_o = np.array([o["k"] for o in orbits])
dphi_o = np.array([o["dphi"] for o in orbits]); qi_o = np.array([o["qi"] for o in orbits])
stable_o = typ_o == "stable"
cls_o = np.where(np.isnan(k_o), "no fixed lock", np.where(k_o > 0, "node (k>0)", "saddle (k<0)"))
theory = {c: dict(n=int(np.sum(cls_o == c)), stable=int(np.sum(stable_o & (cls_o == c))))
          for c in ("node (k>0)", "saddle (k<0)", "no fixed lock")}
theory_by_dphi = {f"{d_:.2f}": {c: dict(n=int(np.sum((cls_o == c) & (dphi_o == d_))),
                                        stable=int(np.sum(stable_o & (cls_o == c) & (dphi_o == d_))))
                                for c in ("node (k>0)", "saddle (k<0)", "no fixed lock")}
                  for d_ in DPHI_SEEDS}
print(f"C: {theory}")

# -----------------------------------------------------------------------------
# D. Loss of stability: unstable orbits adjacent to stable ones
# -----------------------------------------------------------------------------
row_of = {e: i for i, e in enumerate(EPS_ROWS)}
ri_o = np.array([row_of[e] for e in eps_o])
border = np.zeros(NO_, bool)
st_idx = np.where(stable_o)[0]
for j in np.where(~stable_o)[0]:
    near = st_idx[(np.abs(ri_o[st_idx] - ri_o[j]) + np.abs(qi_o[st_idx] - qi_o[j]) == 1)
                  & (np.abs(x_o[st_idx] - x_o[j]) < 0.05) & (dphi_o[st_idx] == dphi_o[j])]
    border[j] = near.size > 0
loss_types = {t_: int(np.sum(border & (typ_o == t_))) for t_ in ("fold", "flip", "torus")}
print(f"D: loss-of-stability neighbours {loss_types}")

# -----------------------------------------------------------------------------
# E. Beyond torus / flip borders: Lyapunov exponent and fate
# -----------------------------------------------------------------------------
lyap_res = []
cand = np.where(border & np.isin(typ_o, ["torus", "flip"]))[0][:40]
if cand.size:
    print(f"E: {cand.size} torus/flip border orbits: direct simulation ...")
    s = np.array([orbits[j]["state"] for j in cand]).T
    Pf_ = P_for(cand.size, eps_o[cand], rat_o[cand], dphi_o[cand])
    s[0] += 1e-3
    d0 = 1e-8
    dv = rng.normal(size=s.shape); dv *= d0 / np.linalg.norm(dv, axis=0)
    sp = s + dv
    lsum = np.zeros(cand.size); alive = np.ones(cand.size, bool); xs = []
    for n in range(T_LYAP):
        s = pmap(s, Pf_); sp = pmap(sp, Pf_)
        alive &= (s[0] > X_COLL) & (s[0] < X_ESC)
        s[:, ~alive] = np.nan_to_num(s[:, ~alive], nan=1.0)
        dd = sp - s; dn = np.maximum(np.linalg.norm(dd, axis=0), 1e-300)
        if n >= T_LYAP // 4:
            lsum += np.where(alive, np.log(dn / d0), 0.0)
        sp = s + dd * (d0 / dn)
        xs.append(np.where(alive, s[0], np.nan))
    lam = lsum / (T_LYAP - T_LYAP // 4)
    xs = np.array(xs)
    for n_, j in enumerate(cand):
        tail = xs[-100:, n_]
        lyap_res.append(dict(eps=float(eps_o[j]), ratio=float(rat_o[j]), dphi=float(dphi_o[j]),
                             type=str(typ_o[j]), alive=bool(alive[n_]),
                             lyap=float(lam[n_]) if alive[n_] else None,
                             surge_amp=float(0.5 * (np.nanmax(tail) - np.nanmin(tail))) if alive[n_] else None))
    xs_E = xs
else:
    xs_E = None

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)
TCOL = {"stable": "#1a9850", "fold": "#d73027", "flip": "#4575b4", "torus": "#fdae61"}
KCOL = {"node (k>0)": "#2166ac", "saddle (k<0)": "#b2182b", "no fixed lock": "#999999"}
dr = RATIOS[1] - RATIOS[0] if RATIOS.size > 1 else 0.01
fig, axs = plt.subplots(2, len(DPHI_SEEDS), figsize=(7.4, 5.8), squeeze=False)
for c_, d_ in enumerate(DPHI_SEEDS):
    m = dphi_o == d_
    jit = rng.uniform(-0.3, 0.3, m.sum()) * dr
    axs[0, c_].scatter(rat_o[m] + jit, eps_o[m], s=7, c=[TCOL[t_] for t_ in typ_o[m]])
    axs[0, c_].set_title(rf"$\Delta\phi$={d_:.2f}: green stable, red fold, orange torus, blue flip", fontsize=7)
    axs[1, c_].scatter(rat_o[m] + jit, eps_o[m], s=7, c=[KCOL[t_] for t_ in cls_o[m]],
                       edgecolors=np.where(stable_o[m], "k", "none"), linewidths=0.6)
    axs[1, c_].set_title("Fixed-position branch: blue node, red saddle, grey none\n(black ring = stable)", fontsize=7)
    for a_ in axs[:, c_]:
        a_.set_xlabel(r"$\omega_c/\omega_w$"); a_.set_ylabel(r"$\varepsilon$")
fig.tight_layout(); save(fig, "fig01_holding_tongue")

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
for c in KCOL:
    m = cls_o == c
    ax[0].scatter(k_o[m] if c != "no fixed lock" else np.zeros(m.sum()), rate_o[m], s=5, c=KCOL[c], label=c)
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_xlabel("fixed-position k"); ax[0].set_ylabel(r"Floquet $\ln|\mu|_{max}$")
ax[0].legend(frameon=False, fontsize=6); ax[0].set_title("Stability vs fixed-position branch", fontsize=8)
for c in KCOL:
    m = (cls_o == c) & stable_o
    ax[1].scatter(rat_o[m], x_o[m], s=6, c=KCOL[c])
ax[1].set_xlabel(r"$\omega_c/\omega_w$"); ax[1].set_ylabel(r"$x^*/\lambda$ (stable only)")
ax[1].set_title("Where stable holding sits", fontsize=8)
if xs_E is not None:
    for n_ in range(min(6, xs_E.shape[1])):
        ax[2].plot(xs_E[:, n_] - np.nanmean(xs_E[:50, n_]), lw=0.6)
    ax[2].set_xlabel("time (periods)"); ax[2].set_ylabel(r"$x - \bar x$")
    ax[2].set_title("E: beyond torus/flip borders", fontsize=8)
fig.tight_layout(); save(fig, "fig02_theory_and_borders")

# -----------------------------------------------------------------------------
# Save and gate
# -----------------------------------------------------------------------------
summary = dict(mode=MODE, runtime_s=round(time.time() - t_start, 1), g=G_HOLD, spp=SPP,
               eps_rows=EPS_ROWS.tolist(), ratio_range=[float(RATIOS[0]), float(RATIOS[-1])],
               n_orbits=NO_, n_stable=int(stable_o.sum()),
               type_counts={t_: int(np.sum(typ_o == t_)) for t_ in ("stable", "fold", "flip", "torus")},
               theory_classes=theory, theory_by_dphi=theory_by_dphi, loss_neighbour_types=loss_types,
               stable_ratio_range_by_eps={f"{e:.1f}": [float(rat_o[stable_o & (eps_o == e)].min()),
                                                        float(rat_o[stable_o & (eps_o == e)].max())]
                                          for e in EPS_ROWS if np.any(stable_o & (eps_o == e))},
               post_border=lyap_res)
np.savez_compressed(os.path.join(OUT_DIR, "results_script05.npz"),
                    eps=eps_o, ratio=rat_o, x=x_o, rate=rate_o, type=typ_o, k=k_o, dphi=dphi_o,
                    lead_re=np.array([o["lead_re"] for o in orbits]),
                    lead_im=np.array([o["lead_im"] for o in orbits]),
                    states=np.array([o["state"] for o in orbits]), border=border)
with open(os.path.join(OUT_DIR, "summary_script05.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n================ SCRIPT 05 SUMMARY ================")
print(json.dumps(summary, indent=2))
print("\n================ DECISION GATE ================")
nd, ns, nn = theory["node (k>0)"], theory["saddle (k<0)"], theory["no fixed lock"]
print(f"Node branch (k>0): {nd['stable']}/{nd['n']} stable  (theory: 0 when a_L > 0)")
print(f"Saddle branch (k<0): {ns['stable']}/{ns['n']} stable  (theory: possible)")
print(f"No fixed-position lock: {nn['stable']}/{nn['n']} stable")
if nd["n"] >= 10 and nd["stable"] / nd["n"] < 0.02 and (ns["stable"] + nn["stable"]) > 0:
    print("CONFIRMED: stable station holding lives off the node branch. The fish holds station")
    print("only where its rhythm, held at that position, would not lock stably by itself.")
elif nd["n"] >= 10 and nd["stable"] / nd["n"] >= 0.02:
    print("CHECK: stable node-branch orbits exist; compute a_L for them (Script 04b method).")
else:
    print("INCONCLUSIVE: too few orbits in one class; widen the ratio range or eps rows.")
print(f"\nOutputs written to: {OUT_DIR}")
