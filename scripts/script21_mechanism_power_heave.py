# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 21: Which part of the wake holds the fish in place, what it costs,
#            and how lateral motion interacts with holding.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX; runs on the GPU when one is attached (L4: roughly 30 to 60 min FULL),
# falls back to the CPU (much slower). Every part is checkpointed to Drive, so
# a disconnected run resumes where it stopped. No settings need to be changed.
# Needs on Drive: script10_continuation (points_script10.npz and
# checkpoints_full/stageC_branches.npz, stageD2_index.npz) and
# script17_biology_questions/checkpoints_full/C1_orbits.npz.
#
# Parts:
#   A. Wake ablation. The prescribed wake gives the fish three positional cues:
#      the phase of the vortices, omega_w (x - t); the envelope of their
#      strength, W(x); and the mean velocity deficit, U_loc(x). Each cue is
#      frozen at the station x0 of a stable orbit, so that it no longer changes
#      when the fish moves, while the forcing at x0 itself is unchanged.
#      Conditions: full wake; phase frozen; envelope frozen; deficit frozen;
#      phase only (envelope and deficit frozen); envelope only; deficit only;
#      all frozen (no positional information).
#      For every stable orbit (baseline: phase lag 0 saddle branch, phase lag
#      pi/2 saddle and node branches; reactive-thrust variant) and condition:
#        A1  Newton search for the station-holding orbit and its Floquet
#            multipliers
#        A2  relocked force at fixed position: zero x_L, gradient a_L,
#            det(I - M_int), and the sign predicted by the index condition
#        A3  response to a positional kick of 0.05 wavelengths
#        A4  confinement under noise (wake phase, vortex amplitude and CPG phase
#            noise, D = 0.01): time to leave bands of +/- 0.1 and +/- 0.25
#            wavelengths, and loss from the wake region
#   B. Mechanical power. Mean muscle power g * drive * dtheta/dt (positive:
#      the muscle does work on the body), mean absolute muscle power, muscle
#      torque, and the power delivered by the hydrodynamic lift, per unit
#      moment of inertia, on every stable orbit of Part A; and for the
#      reactive-thrust fish in uniform flow at the gain g* at which thrust
#      balances drag (found here by bisection), to compare with the same fish
#      in the wake.
#   C. Heave sweep. A lateral body mode is added to the baseline orbits, with
#      natural frequency Omega_y / omega_w from 0.2 to 3 and damping ratios
#      0.1, 0.3 and 0.6. For each, the orbit is continued directly by Newton
#      iterations and, separately, relaxed by simulation; the outcome is stable
#      period-1 holding, bounded motion without a period-1 orbit, or loss.
#      Lateral excursion of the stable orbits is reported.
#
# Gates (reported):
#   G1  the model of this script reproduces Script 10 stable orbits
#       (one-period residual below 1e-6 from the stored states)
#   G2  in the full wake, Newton returns the stored orbit (|x - x0| < 1e-4)
#       and finds it stable for at least 95% of baseline orbits
#   G3  the sign of a_L in the full wake agrees with Script 10 for at least
#       90% of baseline orbits
#   G3b the fixed-position branch reached by the homotopy agrees with
#       Script 10 for the baseline orbits (reported)
#   G4  with all cues frozen no isolated station-holding orbit exists. With
#       the phase as the only cue the model is invariant under a shift of x
#       and t by the same amount, so station-holding orbits form a family
#       with a neutral direction and are not isolated either
#   G5  heave at Omega_y/omega_w = 1, damping 0.3 (the resonant variant of
#       Script 09) keeps fewer stable orbits than 0.25 and 2
#
# Outputs: script21_mechanism_power_heave/summary_script21.json, tables (CSV),
#          checkpoints, draft figures.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

MODE = os.environ.get("S21_MODE", "FULL")
SEED = 21
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath(os.environ.get("S21_BASE", "./karman_gait_project"))
OUT_DIR = os.path.join(BASE_DIR, "script21_mechanism_power_heave")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
for d_ in (FIG_DIR, TAB_DIR, CKPT_DIR):
    os.makedirs(d_, exist_ok=True)

def P(*a):
    return os.path.join(BASE_DIR, *a)

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

SPP = 100
if MODE == "FULL":
    N_PER_GROUP = 40
    N_REACTIVE = 64
    T_RELAX = 200
    T_KICK = 200
    N_REP, T_NOISE = 16, 400
    T_SCAN, N_BISECT = 300, 16
    OMY_GRID = np.unique(np.round(np.concatenate([np.geomspace(0.2, 3.0, 21), [0.25, 1.0, 2.0]]), 4))
    ZEY_GRID = [0.1, 0.3, 0.6]
    N_HEAVE_ORB = 60
    T_HEAVE = 300
    CHUNK = 4096 if ON_GPU else 256
    NCHUNK = 1024 if ON_GPU else 64
else:
    N_PER_GROUP = 2
    N_REACTIVE = 2
    T_RELAX = 40
    T_KICK = 60
    N_REP, T_NOISE = 3, 60
    T_SCAN, N_BISECT = 120, 10
    OMY_GRID = np.array([0.25, 1.0, 2.0])
    ZEY_GRID = [0.3]
    N_HEAVE_ORB = 3
    T_HEAVE = 60
    CHUNK = 256
    NCHUNK = 64
DT = 1.0 / SPP
X_COLL, X_ESC = 0.15, 4.5
TAU_A = 1.0
OMEGA_W = 2.0 * np.pi
KICK = 0.05
D_NOISE = 0.01
H_AL = 0.005
CT_VAR = 0.1
TROUT = dict(wavelength_m=0.208, body_length_m=0.178)
BL_PER_LAMBDA = TROUT["wavelength_m"] / TROUT["body_length_m"]

# -----------------------------------------------------------------------------
# 1. Model in JAX: baseline of Scripts 10, 11 and 17, with reactive thrust,
#    an optional heave mode (Script 09) and frozen positional cues.
#    State: 0 x, 1 u, 2 theta, 3 dtheta/dt, 4 y, 5 dy/dt, 6 z_r, 7 z_i
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA, LT = 3.0, 0.15, 8.0, 0.5
MU = 2 * np.pi
NS = 8
KEYS = ["ratio", "eps", "dphi", "g", "omr", "ze", "cT", "bnd", "frz", "xref", "kph", "kenv", "kdef",
        "hv", "omy", "zey", "wf", "df", "eta"]
NPAR = len(KEYS)
BASE_PAR = dict(ratio=1.0, eps=0.0, dphi=0.0, g=16.0, omr=1.2, ze=0.2, cT=0.0, bnd=1.0, frz=0.0, xref=1.0,
                kph=0.0, kenv=0.0, kdef=0.0, hv=0.0, omy=1.0, zey=0.3, wf=1.0, df=1.0, eta=1.0)

def par(n, **kw):
    b = dict(BASE_PAR); b.update(kw)
    return np.stack([np.broadcast_to(np.asarray(b[k], float), (n,)) for k in KEYS], 1)

def W_env(x):
    return W0 * (1.0 - np.exp(-(x / X_R) ** 2)) * np.exp(-x / L_W)

def rhs_full(t, s, p, psi, aW):
    x, u, th, om, y, vy, zr, zi = s
    (ratio, eps, dphi, g, omr, ze, cT, bnd, frz, xref, kph, kenv, kdef, hv, omy, zey, wf, df, eta) = p
    xp = x + kph * (xref - x)
    xe = x + kenv * (xref - x)
    xd = x + kdef * (xref - x)
    V = U_INF * (1.0 - df * DELTA0 * jnp.exp(-xd / L_DELTA)) - u
    w = wf * (1.0 + aW) * W0 * (1.0 - jnp.exp(-(xe / X_R) ** 2)) * jnp.exp(-xe / L_W) * \
        jnp.sin(OMEGA_W * (xp - t) + psi)
    wr = w - hv * vy
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    beta = jnp.arctan2(wr, jnp.abs(V))
    alpha = beta - th
    L = C_L * (V * V + wr * wr) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    w_te = wr - jnp.abs(V) * th - LT * om
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta) - cT * w_te * w_te * gate
    Om = omr * OMEGA_W
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    tq = g * drive
    dom = -2 * ze * Om * om - Om ** 2 * th + SIGMA * L + tq
    Omy = omy * OMEGA_W
    dvy = hv * (-2 * zey * Omy * vy - Omy ** 2 * y + L * jnp.cos(beta))
    r2 = zr * zr + zi * zi
    a = MU * (1.0 - r2)
    b = ratio * OMEGA_W
    ds = jnp.stack([eta * u * (1.0 - frz), eta * Fx * (1.0 - frz), om, dom, hv * vy, dvy,
                    a * zr - b * zi + eps * th, a * zi + b * zr])
    aux = jnp.stack([Fx, tq * om, jnp.abs(tq * om), tq * tq, SIGMA * L * om])
    return ds, aux

def rk4(t, s, p, psi, aW):
    k1, a1 = rhs_full(t, s, p, psi, aW)
    k2, _ = rhs_full(t + DT / 2, s + DT / 2 * k1, p, psi, aW)
    k3, _ = rhs_full(t + DT / 2, s + DT / 2 * k2, p, psi, aW)
    k4, _ = rhs_full(t + DT, s + DT * k3, p, psi, aW)
    return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), a1

def pmap(s, p):
    """One shedding period, deterministic, from t = 0."""
    def step(i, s_):
        return rk4(i * DT, s_, p, 0.0, 0.0)[0]
    return jax.lax.fori_loop(0, SPP, step, s)

# Newton iterations on the period map over a mask of active dimensions
NEWTON_FNS = {}
def get_newton(n_it, clip):
    key = (n_it, clip)
    if key in NEWTON_FNS:
        return NEWTON_FNS[key]
    def one(s0, p, mask):
        I = jnp.eye(NS)
        MM = jnp.outer(mask, mask)
        f = lambda z: (pmap(z, p), pmap(z, p))
        def body(k, s):
            J, s1 = jax.jacfwd(f, has_aux=True)(s)
            r = (s1 - s) * mask
            A = (J - I) * MM + jnp.diag(1.0 - mask)
            st = jnp.linalg.solve(A, -r)
            st = jnp.clip(jnp.nan_to_num(st), -clip, clip) * mask
            return s + st
        s = jax.lax.fori_loop(0, n_it, body, s0)
        J, s1 = jax.jacfwd(f, has_aux=True)(s)
        res = jnp.max(jnp.abs((s1 - s) * mask))
        return s, res, J * MM
    fn = jax.jit(jax.vmap(one))
    NEWTON_FNS[key] = fn
    return fn

def newton(S0, PP, MASK, n_it=12, clip=0.05):
    fn = get_newton(n_it, clip)
    outs = []
    for i0 in range(0, S0.shape[0], NCHUNK):
        sl = slice(i0, min(S0.shape[0], i0 + NCHUNK))
        o = fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]), jnp.asarray(MASK[sl]))
        outs.append([np.asarray(v) for v in o])
    S = np.concatenate([o[0] for o in outs]); R = np.concatenate([o[1] for o in outs])
    J = np.concatenate([o[2] for o in outs])
    return S, R, J

def multipliers(J):
    mu = np.linalg.eigvals(J)
    a = np.abs(mu)
    lead = mu[np.arange(mu.shape[0]), np.argmax(a, 1)]
    dI = np.real(np.prod(1.0 - mu, axis=1))
    return a.max(1), lead, dI

# Simulation: strobed x, loss, and averages over the last n_avg periods
SIMS = {}
def get_sim(n_per, n_avg, noisy):
    key = (n_per, n_avg, noisy)
    if key in SIMS:
        return SIMS[key]
    def sim(s0, p, nz, k0):
        Dpsi, sW, Dz = nz[0], nz[1], nz[2]
        def one_period(carry, k):
            s, psi, aW, alive, tloss, acc, ymin, ymax = carry
            kp = jax.random.fold_in(k0, k)
            def step(i, c):
                s_, psi_, aW_, al_, tl_, ac_, ymn, ymx = c
                t = k + i * DT
                sn, aux = rk4(t, s_, p, psi_, aW_)
                if noisy:
                    nrm = jax.random.normal(jax.random.fold_in(kp, i), (3,))
                    ea = jnp.exp(-DT / TAU_A)
                    psi_n = psi_ + jnp.sqrt(2 * Dpsi * DT) * nrm[0]
                    aW_n = aW_ * ea + sW * jnp.sqrt(1 - ea * ea) * nrm[1]
                    ang = jnp.sqrt(2 * Dz * DT) * nrm[2]
                    zr2 = sn[6] * jnp.cos(ang) - sn[7] * jnp.sin(ang)
                    zi2 = sn[6] * jnp.sin(ang) + sn[7] * jnp.cos(ang)
                    sn = sn.at[6].set(zr2).at[7].set(zi2)
                else:
                    psi_n, aW_n = psi_, aW_
                lost = (((sn[0] < X_COLL) | (sn[0] > X_ESC)) & (p[7] > 0.5)) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + DT, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                return sn, psi_n, aW_n, al_n, tl_n, ac_ + aux * DT, jnp.minimum(ymn, sn[4]), jnp.maximum(ymx, sn[4])
            s2, psi2, aW2, al2, tl2, acp, ymn, ymx = jax.lax.fori_loop(
                0, SPP, step, (s, psi, aW, alive, tloss, jnp.zeros(5), jnp.inf, -jnp.inf))
            use = k >= n_per - n_avg
            acc = acc + jnp.where(use, acp, 0.0)
            ymin = jnp.where(use, jnp.minimum(ymin, ymn), ymin)
            ymax = jnp.where(use, jnp.maximum(ymax, ymx), ymax)
            return (s2, psi2, aW2, al2, tl2, acc, ymin, ymax), s2[0]
        c0 = (s0, 0.0, 0.0, True, jnp.nan, jnp.zeros(5), jnp.inf, -jnp.inf)
        (s, psi, aW, alive, tloss, acc, ymin, ymax), xs = jax.lax.scan(one_period, c0, jnp.arange(n_per))
        return s, alive, tloss, acc / max(n_avg, 1), xs, ymax - ymin
    fn = jax.jit(jax.vmap(sim))
    SIMS[key] = fn
    return fn

def run(n_per, S0, PP, n_avg=1, NZ=None, noisy=False, seed=0, label=None):
    n = S0.shape[0]
    if NZ is None:
        NZ = np.zeros((n, 3))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    fn = get_sim(n_per, n_avg, noisy)
    outs = []
    tb = time.time()
    for c_, i0 in enumerate(range(0, n, CHUNK)):
        sl = slice(i0, min(n, i0 + CHUNK))
        o = fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]), jnp.asarray(NZ[sl]), keys[sl])
        outs.append([np.asarray(v) for v in o])
        if label and c_ % 5 == 0:
            print(f"      {label}: {min(n, i0 + CHUNK)}/{n} runs ({time.time() - tb:.0f} s)", flush=True)
    return [np.concatenate([o[j] for o in outs]) for j in range(6)]

MASK_POS = np.array([1, 1, 1, 1, 0, 0, 1, 1], float)        # free position, no heave
MASK_POS_H = np.ones(NS)                                    # free position with heave
MASK_INT = np.array([0, 0, 1, 1, 0, 0, 1, 1], float)        # fixed position, no heave

def valid(S, x0=None, dx=0.5):
    r = np.hypot(S[:, 6], S[:, 7])
    ok = np.isfinite(S).all(1) & (S[:, 0] > X_COLL) & (S[:, 0] < X_ESC) & (r > 0.3) & (r < 2.5)
    if x0 is not None:
        ok &= np.abs(S[:, 0] - x0) < dx
    return ok

def to8(S6):
    S6 = np.atleast_2d(S6)
    S = np.zeros((S6.shape[0], NS))
    S[:, :4] = S6[:, :4]; S[:, 6:] = S6[:, 4:6]
    return S

def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    return o

def save_csv(name, header, rows):
    with open(os.path.join(TAB_DIR, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r) + "\n")

def dist(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), min=float(v.min()), q10=float(np.quantile(v, 0.1)), median=float(np.median(v)),
                q90=float(np.quantile(v, 0.9)), max=float(v.max()))

def km_median(t_event, censored, t_max):
    tt = np.where(censored, t_max, t_event)
    order = np.argsort(tt)
    surv, n_at = 1.0, tt.size
    for t_, c_ in zip(tt[order], censored[order]):
        if not c_:
            surv *= (n_at - 1) / n_at
            if surv <= 0.5:
                return float(t_)
        n_at -= 1
    return np.nan

SUMMARY = dict(mode=MODE, spp=SPP)
G = {}
t_start = time.time()

# -----------------------------------------------------------------------------
# 2. Stable orbits
# -----------------------------------------------------------------------------
print("\n=== Orbits ===")
S10_DIR = P("script10_continuation")
S10_CK = os.path.join(S10_DIR, "checkpoints_full")
PTS = np.load(os.path.join(S10_DIR, "points_script10.npz"))["points"]
BR = np.load(os.path.join(S10_CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
CL10 = dict(np.load(os.path.join(S10_CK, "stageD2_index.npz")))
SEL, FT, AL10 = CL10["sel"].astype(int), CL10["ftype"].astype(str), CL10["aL"]
GROUPS = [("phase lag 0, saddle", 0.0, "saddle-real"), ("phase lag pi/2, saddle", np.pi / 2, "saddle-real"),
          ("phase lag pi/2, node", np.pi / 2, "node")]
rows_o = []
for gi, (gname, dp, ft) in enumerate(GROUPS):
    idx = np.where((np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft))[0]
    pick = np.sort(rng.choice(idx, min(N_PER_GROUP, idx.size), replace=False))
    for j in pick:
        i = SEL[j]
        s6 = BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6]
        rows_o.append(dict(group=gi, ratio=PTS[i, 4], eps=PTS[i, 2], dphi=PTS[i, 3], cT=0.0, s=to8(s6)[0],
                           aL10=AL10[j]))
fC1 = P("script17_biology_questions", "checkpoints_full", "C1_orbits.npz")
NG = len(GROUPS)
if os.path.exists(fC1):
    dC1 = dict(np.load(fC1))
    pick = np.sort(rng.choice(dC1["S"].shape[0], min(N_REACTIVE, dC1["S"].shape[0]), replace=False))
    for j in pick:
        rows_o.append(dict(group=NG, ratio=dC1["ratio"][j], eps=dC1["eps"][j], dphi=dC1["dphi"][j], cT=CT_VAR,
                           s=to8(dC1["S"][j])[0], aL10=np.nan))
    GROUP_NAMES = [g_[0] for g_ in GROUPS] + ["reactive thrust"]
else:
    print("  Script 17 C1 orbits not found; the reactive-thrust group and the uniform-flow power comparison are skipped")
    GROUP_NAMES = [g_[0] for g_ in GROUPS]
NO = len(rows_o)
OG = np.array([r["group"] for r in rows_o]); ORAT = np.array([r["ratio"] for r in rows_o])
OEPS = np.array([r["eps"] for r in rows_o]); ODPH = np.array([r["dphi"] for r in rows_o])
OCT = np.array([r["cT"] for r in rows_o]); OS = np.array([r["s"] for r in rows_o])
OAL10 = np.array([r["aL10"] for r in rows_o]); OX0 = OS[:, 0].copy()
for g_ in range(len(GROUP_NAMES)):
    print(f"  {GROUP_NAMES[g_]}: {int((OG == g_).sum())} orbits, x {OX0[OG == g_].min():.2f} to {OX0[OG == g_].max():.2f}")

print("\n=== G1. Model check ===")
P0 = par(NO, ratio=ORAT, eps=OEPS, dphi=ODPH, cT=OCT, xref=OX0)
s1 = run(1, OS, P0)[0]
res = np.max(np.abs(s1 - OS), 1)
G["G1_max_residual"] = float(res.max())
G["G1_pass"] = bool(res.max() < 1e-6)
print(f"  one-period residual from {NO} stored orbits: max {res.max():.2e} ({'pass' if G['G1_pass'] else 'FAIL'})")

# =============================================================================
# A. Wake ablation
# =============================================================================
CONDS = [("full wake", 0, 0, 0), ("phase frozen", 1, 0, 0), ("envelope frozen", 0, 1, 0),
         ("deficit frozen", 0, 0, 1), ("phase only", 0, 1, 1), ("envelope only", 1, 0, 1),
         ("deficit only", 1, 1, 0), ("all frozen", 1, 1, 1)]
NC = len(CONDS)
NOISE_CONDS = [0, 1, 2, 3, 4]
oi = np.repeat(np.arange(NO), NC)
ci = np.tile(np.arange(NC), NO)
KPH = np.array([c_[1] for c_ in CONDS], float)[ci]
KEN = np.array([c_[2] for c_ in CONDS], float)[ci]
KDE = np.array([c_[3] for c_ in CONDS], float)[ci]
def PA(**kw):
    return par(oi.size, ratio=ORAT[oi], eps=OEPS[oi], dphi=ODPH[oi], cT=OCT[oi], xref=OX0[oi],
               kph=KPH, kenv=KEN, kdef=KDE, **kw)

print("\n=== A1. Station-holding orbits under each ablation ===")
fA1 = ckpt("A1_orbits.npz")
if os.path.exists(fA1):
    dA1 = dict(np.load(fA1)); print("  loaded checkpoint")
else:
    t0 = time.time()
    PP = PA()
    MK = np.tile(MASK_POS, (oi.size, 1))
    S, R, J = newton(OS[oi], PP, MK)
    found = (R < 1e-7) & valid(S, OX0[oi])
    # second attempt from the relaxed state
    redo = np.where(~found)[0]
    via_relax = np.zeros(oi.size, bool)
    if redo.size:
        Sr, alive_r = run(T_RELAX, OS[oi[redo]], PP[redo], label="relax")[:2]
        S2, R2, J2 = newton(Sr, PP[redo], MK[redo])
        ok2 = alive_r & (R2 < 1e-7) & valid(S2, OX0[oi[redo]])
        S[redo[ok2]], R[redo[ok2]], J[redo[ok2]] = S2[ok2], R2[ok2], J2[ok2]
        found[redo[ok2]] = True
        via_relax[redo[ok2]] = True
    amax, lead, dI = multipliers(J)
    dA1 = dict(S=S, R=R, found=found, via_relax=via_relax, amax=amax, lead_re=lead.real, lead_im=lead.imag, dI=dI)
    np.savez(fA1, **dA1)
    print(f"  {oi.size} Newton searches in {time.time() - t0:.0f} s")
FOUND = dA1["found"]; STAB = FOUND & (dA1["amax"] < 1 - 1e-6)
m0 = (ci == 0) & (OG[oi] < NG)
same = m0 & FOUND & (np.abs(dA1["S"][:, 0] - OX0[oi]) < 1e-4)
G["G2_returned_and_stable_frac"] = float(np.mean(same[m0] & STAB[m0]))
G["G2_pass"] = bool(G["G2_returned_and_stable_frac"] >= 0.95)
print(f"  full wake: stored orbit returned and stable for {G['G2_returned_and_stable_frac']:.1%} of baseline orbits "
      f"({'pass' if G['G2_pass'] else 'FAIL'})")
m7 = ci == NC - 1
G["G4_all_frozen_found_frac"] = float(FOUND[m7].mean())
G["G4_pass"] = bool(FOUND[m7].mean() == 0.0)
print(f"  all cues frozen: isolated orbit found for {FOUND[m7].mean():.1%} ({'pass' if G['G4_pass'] else 'FAIL'})")

print("\n=== A2. Fixed-position branch and relocked force ===")
fA2 = ckpt("A2_relock.npz")
ETAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
def relocked_force(X, base_S, PPf):
    """Lock the internal variables at fixed position X; return mean force, det(I - M_int), lock flag, state."""
    S0 = base_S.copy(); S0[:, 0] = X; S0[:, 1] = 0.0; S0[:, 4:6] = 0.0
    MK = np.tile(MASK_INT, (X.size, 1))
    S, R, J = newton(S0, PPf, MK, n_it=12, clip=0.2)
    ok = (R < 1e-7) & valid(S)
    F = run(1, S, PPf)[3][:, 0]
    amax, lead, dI = multipliers(J)
    return np.where(ok, F, np.nan), dI, ok, S, amax, lead
if os.path.exists(fA2):
    dA2 = dict(np.load(fA2)); print("  loaded checkpoint")
else:
    t0 = time.time()
    # homotopy x' = eta u, u' = eta F_x from eta = 1 to 0.01 on the full-wake orbit (Scripts 08 to 10)
    Pfull = par(NO, ratio=ORAT, eps=OEPS, dphi=ODPH, cT=OCT, xref=OX0)
    s_h = OS.copy(); alive_h = np.ones(NO, bool); fail_eta = np.full(NO, np.nan)
    for eta in ETAS[1:]:
        idx = np.where(alive_h)[0]
        if idx.size == 0:
            break
        Pe = Pfull[idx].copy(); Pe[:, KEYS.index("eta")] = eta
        Sn, Rn, _ = newton(s_h[idx], Pe, np.tile(MASK_POS, (idx.size, 1)), n_it=8, clip=0.05)
        ok = (Rn < 1e-7) & valid(Sn)
        s_h[idx[ok]] = Sn[ok]
        fail_eta[idx[~ok]] = eta; alive_h[idx[~ok]] = False
    # fixed-position lock at the end of the homotopy: branch type
    PPf0 = par(NO, ratio=ORAT, eps=OEPS, dphi=ODPH, cT=OCT, xref=OX0, frz=1.0)
    F_e, dI_e, ok_e, S_int, amax_e, lead_e = relocked_force(s_h[:, 0], s_h, PPf0)
    ok_e &= alive_h
    btype = np.where(~ok_e, "none", np.where(amax_e < 1, "node",
                     np.where(np.abs(lead_e.imag) < 1e-6, "saddle-real", "unstable-complex")))
    # relocked force for every condition near x_L, the position at the end of the homotopy, continued
    # outwards in small steps from the full-wake lock so that the same locked branch is followed
    PPf = PA(frz=1.0)
    xL0 = s_h[oi, 0]
    okb = ok_e[oi]
    def lock_at(X, Sb):
        S0 = Sb.copy(); S0[:, 0] = X; S0[:, 1] = 0.0; S0[:, 4:6] = 0.0
        S, R, J = newton(S0, PPf, np.tile(MASK_INT, (X.size, 1)), n_it=15, clip=0.05)
        ok = (R < 1e-7) & valid(S)
        F = run(1, S, PPf)[3][:, 0]
        return np.where(ok, F, np.nan), S, ok, J
    def lock_at_sub(X, Sb, idx):
        S0 = Sb.copy(); S0[:, 0] = X; S0[:, 1] = 0.0; S0[:, 4:6] = 0.0
        S, R, J = newton(S0, PPf[idx], np.tile(MASK_INT, (X.size, 1)), n_it=20, clip=0.05)
        ok = (R < 1e-7) & valid(S)
        F = run(1, S, PPf[idx])[3][:, 0]
        return np.where(ok, F, np.nan), S, ok
    F_c, S_c, ok_c, J_c = lock_at(xL0, S_int[oi])
    dIint = np.where(ok_c & okb, multipliers(J_c)[2], np.nan)
    aL = np.full(oi.size, np.nan); r2 = np.full(oi.size, np.nan); FxL0 = np.where(ok_c & okb, F_c, np.nan)
    h_used = np.full(oi.size, np.nan)
    todo = ok_c & okb
    for scale in (1.0, 0.2, 0.05):
        idx = np.where(todo & ~np.isfinite(aL))[0]
        if idx.size == 0:
            break
        hh = H_AL / 2 * scale
        offs = [0.0]; Fs = [F_c[idx]]
        for sgn in (1.0, -1.0):
            Sprev, okp = S_c[idx], ok_c[idx]
            for k in (1, 2):
                F_k, S_k, ok_k = lock_at_sub(xL0[idx] + sgn * k * hh, Sprev, idx)
                ok_k &= okp
                offs.append(sgn * k * hh); Fs.append(np.where(ok_k, F_k, np.nan))
                Sprev, okp = S_k, ok_k
        offs = np.array(offs); Fs = np.array(Fs).T
        allok = np.isfinite(Fs).all(1)
        if allok.any():
            X_ = offs - offs.mean()
            Fa_ = Fs[allok]
            sl = (Fa_ - Fa_.mean(1, keepdims=True)) @ X_ / (X_ @ X_)
            fit = Fa_.mean(1, keepdims=True) + sl[:, None] * X_[None, :]
            ss_res = ((Fa_ - fit) ** 2).sum(1); ss_tot = ((Fa_ - Fa_.mean(1, keepdims=True)) ** 2).sum(1)
            aL[idx[allok]] = sl
            r2[idx[allok]] = np.where(ss_tot > 1e-24, 1 - ss_res / np.maximum(ss_tot, 1e-30), 1.0)
            h_used[idx[allok]] = hh
    allok = np.isfinite(aL)
    FxL = np.where(allok, FxL0, np.nan)
    xl = np.where(allok & (np.abs(aL) > 1e-12), xL0 - FxL / np.where(np.abs(aL) > 1e-12, aL, 1.0), np.nan)
    dA2_extra = dict(aL_r2=r2, aL_h=h_used)
    dA2 = dict(xL=xl, aL=aL, dIint=dIint, FxL=FxL, btype=btype, fail_eta=fail_eta, homotopy_ok=alive_h, **dA2_extra)
    np.savez(fA2, **dA2)
    print(f"  done in {time.time() - t0:.0f} s")
BTYPE = dA2["btype"].astype(str)
mb = OG < NG
FT_O = np.array([GROUPS[g_][2] if g_ < NG else "" for g_ in OG])
G["G3b_branch_agreement"] = float(np.mean(BTYPE[mb] == FT_O[mb]))
print(f"  homotopy reached the fixed position for {dA2['homotopy_ok'].mean():.1%} of orbits; branch type agrees "
      f"with Script 10 for {G['G3b_branch_agreement']:.1%} of baseline orbits")
if (OG == NG).any():
    vals, cnt = np.unique(BTYPE[OG == NG], return_counts=True)
    print("  reactive-thrust orbits by branch: " + ", ".join(f"{v} {c}" for v, c in zip(vals, cnt)))
aL, dIint = dA2["aL"], dA2["dIint"]
AL_ZERO = 1e-6                    # |a_L| below this is zero: the relocked force does not depend on position
aL_sign = np.where(np.isfinite(aL), np.where(np.abs(aL) < AL_ZERO, 0.0, np.sign(aL)), np.nan)
idx_pred = np.sign(dIint) * (-aL_sign)
g3 = m0 & np.isfinite(aL) & np.isfinite(OAL10[oi])
G["G3_aL_sign_agreement"] = float(np.mean(aL_sign[g3] == np.sign(OAL10[oi][g3]))) if g3.any() else np.nan
G["G3_aL_defined_frac"] = float(np.mean(np.isfinite(aL[m0])))
G["G3_pass"] = bool(np.isfinite(G["G3_aL_sign_agreement"]) and G["G3_aL_sign_agreement"] >= 0.9)
print(f"  full wake: a_L obtained for {G['G3_aL_defined_frac']:.1%} of baseline orbits; its sign agrees with Script 10 "
      f"for {G['G3_aL_sign_agreement']:.1%} of them "
      f"({'pass' if G['G3_pass'] else 'FAIL'})")

print("\n=== A3. Positional kick ===")
fA3 = ckpt("A3_kick.npz")
S_START = np.where(FOUND[:, None], dA1["S"], OS[oi])
if os.path.exists(fA3):
    dA3 = dict(np.load(fA3)); print("  loaded checkpoint")
else:
    t0 = time.time()
    PP = PA()
    Sk = S_START.copy(); Sk[:, 0] += KICK
    o = run(T_KICK, np.concatenate([S_START, Sk]), np.concatenate([PP, PP]), label="kick")
    n = oi.size
    xs, alive = o[4], o[1]
    diff = xs[n:] - xs[:n]
    dA3 = dict(diff=diff.astype(np.float32), alive_u=alive[:n], alive_k=alive[n:],
               drift=(xs[:n, -1] - xs[:n, -1 - min(100, T_KICK // 2)]) / min(100, T_KICK // 2))
    np.savez(fA3, **dA3)
    print(f"  done in {time.time() - t0:.0f} s")
i50 = min(49, T_KICK - 1)
REM50 = np.where(dA3["alive_u"] & dA3["alive_k"], np.abs(dA3["diff"][:, i50]) / KICK, np.nan)
REMEND = np.where(dA3["alive_u"] & dA3["alive_k"], np.abs(dA3["diff"][:, -1]) / KICK, np.nan)

print("\n=== A4. Confinement under noise ===")
fA4 = ckpt("A4_noise.npz")
if os.path.exists(fA4):
    dA4 = dict(np.load(fA4)); print("  loaded checkpoint")
else:
    t0 = time.time()
    sel = np.where(np.isin(ci, NOISE_CONDS))[0]
    rr = np.repeat(sel, N_REP)
    PP = PA()[rr]
    NZ = np.tile([D_NOISE, np.sqrt(D_NOISE), D_NOISE], (rr.size, 1))
    o = run(T_NOISE, S_START[rr], PP, NZ=NZ, noisy=True, seed=SEED, label="noise")
    dA4 = dict(rr=rr, alive=o[1], tloss=o[2], x0=S_START[rr, 0], xs=o[4].astype(np.float32))
    np.savez(fA4, **dA4)
    print(f"  done in {time.time() - t0:.0f} s")

# ---- tables for Part A
A = {}
rowsA = []
for g_ in range(len(GROUP_NAMES)):
    A[GROUP_NAMES[g_]] = {}
    for c_ in range(NC):
        m = (OG[oi] == g_) & (ci == c_)
        n = int(m.sum())
        rec = dict(n=n, found=float(FOUND[m].mean()), stable=float(STAB[m].mean()),
                   torus_lead=float(np.mean(STAB[m] | (np.abs(dA1["lead_im"][m]) < 1e-6))) if n else np.nan,
                   x_shift=dist(np.abs(dA1["S"][m, 0] - OX0[oi][m])[FOUND[m]]),
                   aL=dist(aL[m]), aL_positive_frac=float(np.mean(aL_sign[m][np.isfinite(aL_sign[m])] > 0)) if np.isfinite(aL_sign[m]).any() else np.nan,
                   aL_zero_frac=float(np.mean(aL_sign[m][np.isfinite(aL_sign[m])] == 0)) if np.isfinite(aL_sign[m]).any() else np.nan,
                   aL_defined_frac=float(np.mean(np.isfinite(aL[m]))) if n else np.nan,
                   index_positive_frac=float(np.mean(idx_pred[m][np.isfinite(idx_pred[m])] > 0)) if np.isfinite(idx_pred[m]).any() else np.nan,
                   kick_remaining_50=dist(REM50[m]), kick_remaining_end=dist(REMEND[m]),
                   kick_lost=float(np.mean(~(dA3["alive_u"][m] & dA3["alive_k"][m]))),
                   drift=dist(np.abs(dA3["drift"][m])))
        # consistency of the index condition with the stability of the ablated orbit
        mm = m & STAB & np.isfinite(idx_pred)
        rec["stable_with_negative_index"] = int(np.sum(idx_pred[mm] < 0))
        if c_ in NOISE_CONDS:
            rsel = np.isin(dA4["rr"], np.where(m)[0])
            dev = np.abs(dA4["xs"][rsel].astype(float) - dA4["x0"][rsel][:, None])
            tl = dA4["tloss"][rsel]
            lost = np.isfinite(tl)
            for band in (0.1, 0.25):
                out = dev > band
                first = np.where(out.any(1), out.argmax(1) + 1.0, np.inf)
                first = np.where(lost, np.minimum(first, tl), first)
                cens = ~np.isfinite(first)
                rec[f"noise_leave_{band}_km_median"] = km_median(first, cens, T_NOISE)
                rec[f"noise_leave_{band}_censored"] = float(cens.mean())
            rec["noise_lost_frac"] = float(lost.mean())
        rec["branch_counts"] = {b_: int(np.sum(BTYPE[oi][m] == b_)) for b_ in ("node", "saddle-real", "unstable-complex", "none")}
        A[GROUP_NAMES[g_]][CONDS[c_][0]] = rec
        rowsA.append([GROUP_NAMES[g_], CONDS[c_][0], n, rec["found"], rec["stable"],
                      rec["aL"].get("median", np.nan), rec["aL_positive_frac"], rec["index_positive_frac"],
                      rec["stable_with_negative_index"], rec["kick_remaining_50"].get("median", np.nan),
                      rec["kick_lost"], rec.get("noise_leave_0.1_km_median", np.nan),
                      rec.get("noise_leave_0.25_km_median", np.nan), rec.get("noise_lost_frac", np.nan)])
save_csv("A_ablation.csv", ["group", "condition", "n", "found", "stable", "aL_median", "aL_positive_frac",
                            "index_positive_frac", "stable_with_negative_index", "kick_remaining_50_median",
                            "kick_lost_frac", "noise_leave_0.1_median", "noise_leave_0.25_median",
                            "noise_lost_frac"], rowsA)
print("\n  group / condition: found, stable, a_L > 0, index > 0, stable with index < 0, kick left after 50, "
      "noise median to leave +/-0.25")
for r in rowsA:
    print(f"  {r[0]:>24s} / {r[1]:<16s}: {r[3]:.2f}, {r[4]:.2f}, {r[6]:.2f}, {r[7]:.2f}, {r[8]:d}, "
          f"{r[9]:.3f}, {r[12]}")
SUMMARY["A_ablation"] = A

# =============================================================================
# B. Mechanical power
# =============================================================================
print("\n=== B. Mechanical power ===")
fB = ckpt("B_power.npz")
B = {}
if os.path.exists(fB):
    dB = dict(np.load(fB)); print("  loaded checkpoint")
else:
    t0 = time.time()
    m = (ci == 0) & STAB
    PPw = PA()[m]
    acc_w = run(1, dA1["S"][m], PPw)[3]
    dB = dict(orb=oi[m], acc_w=acc_w)
    mr = np.where(OG == NG)[0]
    if mr.size:
        # uniform flow: bisection on the gain at which thrust balances drag
        nr = mr.size
        lo, hi = np.zeros(nr), np.full(nr, 60.0)
        def drift_at(gv):
            S0 = np.zeros((nr, NS)); S0[:, 0] = 1.0; S0[:, 6] = 1.0
            PPu = par(nr, ratio=ORAT[mr], eps=OEPS[mr], dphi=ODPH[mr], cT=CT_VAR, g=gv, wf=0.0, df=0.0, bnd=0.0)
            xs = run(T_SCAN, S0, PPu)[4]
            nl = T_SCAN // 3
            return (xs[:, -1] - xs[:, -1 - nl]) / nl
        dlo, dhi = drift_at(lo), drift_at(hi)
        ext = ~(dhi < 0)
        if ext.any():
            hi = np.where(ext, 90.0, hi)
            dhi = np.where(ext, drift_at(hi), dhi)
        okb = (dlo > 0) & (dhi < 0)
        for it in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            dm = drift_at(mid)
            lo = np.where(dm > 0, mid, lo); hi = np.where(dm > 0, hi, mid)
        gstar = np.where(okb, 0.5 * (lo + hi), np.nan)
        S0 = np.zeros((nr, NS)); S0[:, 0] = 1.0; S0[:, 6] = 1.0
        PPu = par(nr, ratio=ORAT[mr], eps=OEPS[mr], dphi=ODPH[mr], cT=CT_VAR, g=np.nan_to_num(gstar, nan=16.0),
                  wf=0.0, df=0.0, bnd=0.0)
        o = run(T_SCAN, S0, PPu, n_avg=T_SCAN // 3)
        dB.update(r_orb=mr, gstar=gstar, acc_u=o[3], drift_u=(o[4][:, -1] - o[4][:, -1 - T_SCAN // 3]) / (T_SCAN // 3))
    np.savez(fB, **dB)
    print(f"  done in {time.time() - t0:.0f} s")
NAMES_P = ["mean force", "mean muscle power", "mean absolute muscle power", "mean squared torque",
           "mean lift power"]
orb_w = dB["orb"]
rowsB = []
for g_ in range(len(GROUP_NAMES)):
    mg = OG[orb_w] == g_
    if not mg.any():
        continue
    B[GROUP_NAMES[g_]] = {NAMES_P[k]: dist(dB["acc_w"][mg, k]) for k in range(1, 5)}
    B[GROUP_NAMES[g_]]["rms torque"] = dist(np.sqrt(dB["acc_w"][mg, 3]))
    rowsB.append([GROUP_NAMES[g_], "wake", int(mg.sum()), np.median(dB["acc_w"][mg, 1]), np.median(dB["acc_w"][mg, 2]),
                  np.median(np.sqrt(dB["acc_w"][mg, 3])), np.median(dB["acc_w"][mg, 4])])
if "gstar" in dB:
    gstar = dB["gstar"]; ok = np.isfinite(gstar)
    # pair each uniform-flow fish with the same fish in the wake (if stable there)
    pos_w = {int(o): k for k, o in enumerate(orb_w)}
    pair = np.array([pos_w.get(int(o), -1) for o in dB["r_orb"]])
    okp = ok & (pair >= 0)
    au, aw = dB["acc_u"][okp], dB["acc_w"][pair[okp]]
    B["uniform flow"] = dict(gstar=dist(gstar), drift_at_gstar=dist(np.abs(dB["drift_u"][ok])),
                             **{NAMES_P[k]: dist(dB["acc_u"][ok, k]) for k in range(1, 5)},
                             rms_torque=dist(np.sqrt(dB["acc_u"][ok, 3])),
                             ratio_uniform_to_wake={NAMES_P[k]: dist(au[:, k] / aw[:, k]) for k in (2, 3)},
                             ratio_rms_torque=dist(np.sqrt(au[:, 3] / aw[:, 3])), n_pairs=int(okp.sum()))
    G["G5b_uniform_drift_at_gstar_max"] = float(np.nanmax(np.abs(dB["drift_u"][ok]))) if ok.any() else np.nan
    rowsB.append(["reactive thrust", "uniform flow", int(ok.sum()), np.median(dB["acc_u"][ok, 1]),
                  np.median(dB["acc_u"][ok, 2]), np.median(np.sqrt(dB["acc_u"][ok, 3])), np.median(dB["acc_u"][ok, 4])])
    print(f"  uniform flow: g* median {np.nanmedian(gstar):.1f} (wake gain 16), residual drift up to "
          f"{G['G5b_uniform_drift_at_gstar_max']:.1e} wavelengths per period")
    if okp.any():
        print(f"  same fish, uniform over wake: mean absolute muscle power x{np.median(au[:, 2] / aw[:, 2]):.2f}, "
              f"rms torque x{np.median(np.sqrt(au[:, 3] / aw[:, 3])):.2f} (medians over {okp.sum()} fish)")
save_csv("B_power.csv", ["group", "condition", "n", "mean_muscle_power", "mean_abs_muscle_power", "rms_torque",
                         "mean_lift_power"], rowsB)
for r in rowsB:
    print(f"  {r[0]:>24s}, {r[1]:<12s}: n {r[2]}, muscle power {r[3]:+.3f}, absolute {r[4]:.3f}, "
          f"rms torque {r[5]:.3f}, lift power {r[6]:+.3f}")
SUMMARY["B_power"] = B

# =============================================================================
# C. Heave sweep
# =============================================================================
print("\n=== C. Heave sweep ===")
fC = ckpt("C_heave.npz")
base_idx = np.where(OG < NG)[0]
hpick = np.sort(rng.choice(base_idx, min(N_HEAVE_ORB, base_idx.size), replace=False))
nh, nO, nZ = hpick.size, OMY_GRID.size, len(ZEY_GRID)
hi_o = np.repeat(hpick, nO * nZ)
hi_w = np.tile(np.repeat(np.arange(nO), nZ), nh)
hi_z = np.tile(np.arange(nZ), nh * nO)
if os.path.exists(fC):
    dC = dict(np.load(fC)); print("  loaded checkpoint")
else:
    t0 = time.time()
    PPh = par(hi_o.size, ratio=ORAT[hi_o], eps=OEPS[hi_o], dphi=ODPH[hi_o], xref=OX0[hi_o], hv=1.0,
              omy=OMY_GRID[hi_w], zey=np.array(ZEY_GRID)[hi_z])
    MK = np.tile(MASK_POS_H, (hi_o.size, 1))
    S1, R1, J1 = newton(OS[hi_o], PPh, MK, n_it=15)
    f1 = (R1 < 1e-7) & valid(S1, OX0[hi_o])
    a1 = multipliers(J1)[0]
    Sr, alive_r = run(T_HEAVE, OS[hi_o], PPh, label="heave relax")[:2]
    S2, R2, J2 = newton(Sr, PPh, MK, n_it=12)
    f2 = alive_r & (R2 < 1e-7) & valid(S2, OX0[hi_o])
    a2 = multipliers(J2)[0]
    st1, st2 = f1 & (a1 < 1 - 1e-6), f2 & (a2 < 1 - 1e-6)
    Sst = np.where(st1[:, None], S1, S2)
    yamp = np.full(hi_o.size, np.nan)
    anyst = st1 | st2
    if anyst.any():
        yamp[anyst] = run(1, Sst[anyst], PPh[anyst])[5]
    dC = dict(f1=f1, a1=a1, f2=f2, a2=a2, alive_r=alive_r, st1=st1, st2=st2, yamp=yamp, x=Sst[:, 0])
    np.savez(fC, **dC)
    print(f"  {hi_o.size} cases in {time.time() - t0:.0f} s")
stable_any = dC["st1"] | dC["st2"]
bounded = ~stable_any & dC["alive_r"]
Cres = dict(omy_grid=OMY_GRID, zey_grid=ZEY_GRID, n_orbits=nh, rows=[])
rowsC = []
for w_ in range(nO):
    for z_ in range(nZ):
        m = (hi_w == w_) & (hi_z == z_)
        rec = [float(OMY_GRID[w_]), float(ZEY_GRID[z_]), float(dC["st1"][m].mean()), float(stable_any[m].mean()),
               float(bounded[m].mean()), float(np.mean(~dC["alive_r"][m])),
               float(np.nanmedian(dC["yamp"][m])) if np.isfinite(dC["yamp"][m]).any() else np.nan]
        rowsC.append(rec)
save_csv("C_heave.csv", ["Omega_y_over_omega_w", "zeta_y", "continued_stable", "stable_any", "bounded_no_orbit",
                         "lost", "lateral_excursion_median"], rowsC)
Cres["rows"] = rowsC
def frac_at(w, z):
    m = (np.abs(OMY_GRID[hi_w] - w) < 1e-9) & (np.abs(np.array(ZEY_GRID)[hi_z] - z) < 1e-9)
    return float(stable_any[m].mean()) if m.any() else np.nan
fr = [frac_at(0.25, 0.3), frac_at(1.0, 0.3), frac_at(2.0, 0.3)]
G["G5_stable_frac_0.25_1_2"] = fr
G["G5_pass"] = bool(np.all(np.isfinite(fr)) and fr[1] < fr[0] and fr[1] < fr[2])
print(f"  stable fraction at damping 0.3: 0.25 -> {fr[0]:.2f}, 1 -> {fr[1]:.2f}, 2 -> {fr[2]:.2f} "
      f"({'pass' if G['G5_pass'] else 'FAIL'})")
for z_ in range(nZ):
    line = ", ".join(f"{OMY_GRID[w_]:.2f}:{rowsC[w_ * nZ + z_][3]:.2f}" for w_ in range(nO))
    print(f"  damping {ZEY_GRID[z_]}: stable fraction by Omega_y/omega_w  {line}")
SUMMARY["C_heave"] = Cres
SUMMARY["gates"] = G

# -----------------------------------------------------------------------------
# Draft figures (final versions are made with the figure script)
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
try:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    wbar = 0.8 / len(GROUP_NAMES)
    for g_ in range(len(GROUP_NAMES)):
        st = [A[GROUP_NAMES[g_]][c_[0]]["stable"] for c_ in CONDS]
        kk = [A[GROUP_NAMES[g_]][c_[0]]["kick_remaining_50"].get("median", np.nan) for c_ in CONDS]
        ax[0].bar(np.arange(NC) + g_ * wbar, st, wbar, label=GROUP_NAMES[g_])
        ax[1].bar(np.arange(NC) + g_ * wbar, kk, wbar)
    for a_ in ax:
        a_.set_xticks(np.arange(NC) + 0.4 - wbar / 2)
        a_.set_xticklabels([c_[0] for c_ in CONDS], rotation=40, ha="right")
    ax[0].set_ylabel("fraction stable"); ax[0].set_ylim(0, 1)
    ax[1].set_ylabel("kick remaining after 50 periods"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=9, framealpha=1, edgecolor="k")
    save(fig, "S21_A_ablation")
except Exception as e:
    print(f"  figure A skipped: {e}")
try:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for z_ in range(nZ):
        ax.plot(OMY_GRID, [rowsC[w_ * nZ + z_][3] for w_ in range(nO)], "o-", label=f"$\\zeta_y={ZEY_GRID[z_]}$")
    ax.set_xscale("log"); ax.set_xlabel(r"$\Omega_y/\omega_w$"); ax.set_ylabel("fraction with stable holding")
    ax.set_ylim(0, 1); ax.legend(framealpha=1, edgecolor="k")
    save(fig, "S21_C_heave")
except Exception as e:
    print(f"  figure C skipped: {e}")

SUMMARY["runtime_s"] = time.time() - t_start
with open(os.path.join(OUT_DIR, "summary_script21.json"), "w") as f:
    json.dump(to_py(SUMMARY), f, indent=2)

print("\n================ SCRIPT 21 SUMMARY ================")
print("Gates: " + ", ".join(f"{k}: {v}" for k, v in G.items() if k.endswith("pass")))
print(f"Runtime {SUMMARY['runtime_s'] / 60:.1f} min. Outputs: {OUT_DIR}")