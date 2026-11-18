# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 23: Can an adaptive rhythm keep station holding across flow speeds?
#
# Colab ready. Paste into one cell and run, with Google Drive already mounted.
# Uses JAX; runs on the GPU when one is attached (L4: roughly 20 to 40 min),
# falls back to the CPU (much slower). Every part is checkpointed to Drive, so
# a disconnected run resumes where it stopped. No settings need to be changed.
# Needs on Drive the outputs of Script 10 (points_script10.npz and
# checkpoints_full/stageC_branches.npz, stageD2_index.npz).
#
# Background. With the fish's internal rates fixed in physical time, a change of
# flow speed s = U/U_ref changes the shedding frequency, and station holding
# survives only a change of a fraction of a percent (Scripts 11 and 22).
# Imposing a CPG frequency that follows the shedding widens this to several
# percent. Here the CPG adapts its own frequency to its sensory input:
#     d omega_c / dt = -K * (eps * theta) * (z_i cos ph0 - z_r sin ph0) / |z|.
# With ph0 = 0 this is the adaptive-frequency Hopf oscillator of Righetti,
# Buchli and Ijspeert (2006), which drifts towards the frequency of its input.
# Because the input here is the body angle, which the CPG itself drives, the
# script also uses a phase-locked loop: ph0 is the phase between the CPG and
# its input on the reference orbit, so that the CPG retunes its frequency to
# keep the phase relation it had at the reference flow speed. The sign of K
# that makes the adapted orbit stable is chosen per orbit at s = 1.
#
# Schemes (all internal rates are fixed in physical time, so in shedding units
# they scale by 1/s, the gain by 1/s^2, unless stated as retuned):
#   fixed                          K = 0, nothing retuned (Script 22)
#   imposed rhythm                 K = 0, CPG frequency set to follow the shedding
#   input-frequency rule, K = 1   adaptive-frequency oscillator, ph0 = 0
#   phase-locked, K = 0.03, 0.1, 0.3
#                                  phase-locked loop, ph0 from the reference orbit
#                                  (adaptation must be slow: larger K destabilizes)
#   phase-locked K = 0.1, gain retuned
#                                  and the muscle gain follows the flow
#   phase-locked K = 0.1, gain and stiffness retuned
#                                  and the body natural frequency follows too
#   all retuned                    self-similar control
#
# For every scheme and stable orbit (20 per group: phase lag 0 saddle, phase lag
# pi/2 saddle and node, Script 10):
#   1. at s = 1 the orbit of the scheme (for adaptive schemes the adapted orbit,
#      found by relaxation and Newton iterations with omega_c as a state) and
#      its Floquet stability
#   2. continuation in s from 0.7 to 1.3: the band of s over which the orbit
#      stays stable
#   3. direct simulation on a coarse grid of s from the s = 1 orbit: holding
#      (no collision, no escape) and locking (one CPG cycle per shedding
#      period), and the adapted CPG frequency
#   4. time series after a step change of flow speed, for illustration
#
# Gates (reported):
#   G1  with K = 0 the model reproduces the Script 10 orbits
#   G2  the fixed scheme reproduces Script 22 (median band about 0.15%)
#   G3  the all-retuned control holds over the whole range of s
#
# Outputs: script23_adaptive_rhythm/summary_script23.json, tables (CSV),
#          checkpoints, draft figure.
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

MODE = os.environ.get("S23_MODE", "FULL")
SEED = 23
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Drive is expected to be mounted already, at /content/drive
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
    if not os.path.isdir(BASE_DIR):
        raise SystemExit(f"{BASE_DIR} not found: mount Google Drive first.")
else:
    BASE_DIR = os.path.abspath(os.environ.get("S23_BASE", "./karman_gait_project"))
OUT_DIR = os.path.join(BASE_DIR, "script23_adaptive_rhythm")
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
    N_PER_GROUP = 20
    T_ADAPT = 400
    S_STEPS = np.concatenate([np.arange(0.001, 0.0501, 0.001), np.arange(0.055, 0.3001, 0.005)])
    S_COARSE = np.round(np.arange(0.70, 1.3001, 0.025), 4)
    T_FLOW = 600
    N_STEP_EX, T_STEP_EX, S_STEP_EX = 6, 400, [0.95, 1.05]
    CHUNK = 4096 if ON_GPU else 256
    NCHUNK = 1024 if ON_GPU else 64
else:
    N_PER_GROUP = 1
    T_ADAPT = 80
    S_STEPS = np.array([0.002, 0.01, 0.03])
    S_COARSE = np.array([0.95, 1.0, 1.05])
    T_FLOW = 60
    N_STEP_EX, T_STEP_EX, S_STEP_EX = 1, 40, [1.1]
    CHUNK = 256
    NCHUNK = 64
DT = 1.0 / SPP
X_COLL, X_ESC = 0.15, 4.5
OMEGA_W = 2.0 * np.pi
U_REF_BL = (1.0 / 0.87) * 0.208 * 2.74 / 0.178

# -----------------------------------------------------------------------------
# 1. Model: baseline of Scripts 10 and 11, with the CPG frequency as a state.
#    State: 0 x, 1 u, 2 theta, 3 dtheta, 4 z_r, 5 z_i, 6 omega_c / omega_w
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA, LT = 3.0, 0.15, 8.0, 0.5
NS = 7
KEYS = ["eps", "dphi", "g", "omr", "ze", "mu", "K", "bnd", "frz", "ph0"]
NPAR = len(KEYS)
BASE_PAR = dict(eps=0.0, dphi=0.0, g=16.0, omr=1.2, ze=0.2, mu=2 * np.pi, K=0.0, bnd=1.0, frz=0.0, ph0=0.0)

def par(n, **kw):
    b = dict(BASE_PAR); b.update(kw)
    return np.stack([np.broadcast_to(np.asarray(b[k], float), (n,)) for k in KEYS], 1)

def rhs(t, s, p):
    x, u, th, om, zr, zi, wc = s
    eps, dphi, g, omr, ze, mu, K, bnd, frz, ph0 = p
    V = U_INF * (1.0 - DELTA0 * jnp.exp(-x / L_DELTA)) - u
    w = W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * jnp.sin(OMEGA_W * (x - t))
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
    Om = omr * OMEGA_W
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    dom = -2 * ze * Om * om - Om ** 2 * th + SIGMA * L + g * drive
    r2 = zr * zr + zi * zi
    a = mu * (1.0 - r2)
    b = wc * OMEGA_W
    inp = eps * th
    # adaptation of the CPG frequency: input-frequency rule for ph0 = 0 (Righetti et al.), otherwise a
    # phase-locked loop that keeps the preferred phase ph0 between the CPG and its proprioceptive input
    dwc = -K * inp * (zi * jnp.cos(ph0) - zr * jnp.sin(ph0)) / jnp.sqrt(r2 + 1e-12) / OMEGA_W
    return jnp.stack([u * (1.0 - frz), Fx * (1.0 - frz), om, dom, a * zr - b * zi + inp, a * zi + b * zr, dwc])

def rk4(t, s, p):
    k1 = rhs(t, s, p)
    k2 = rhs(t + DT / 2, s + DT / 2 * k1, p)
    k3 = rhs(t + DT / 2, s + DT / 2 * k2, p)
    k4 = rhs(t + DT, s + DT * k3, p)
    return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def pmap(s, p):
    return jax.lax.fori_loop(0, SPP, lambda i, s_: rk4(i * DT, s_, p), s)

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
            st = jnp.clip(jnp.nan_to_num(jnp.linalg.solve(A, -r)), -clip, clip) * mask
            return s + st
        s = jax.lax.fori_loop(0, n_it, body, s0)
        J, s1 = jax.jacfwd(f, has_aux=True)(s)
        return s, jnp.max(jnp.abs((s1 - s) * mask)), J * MM
    fn = jax.jit(jax.vmap(one))
    NEWTON_FNS[key] = fn
    return fn

def newton(S0, PP, n_it=12, clip=0.05):
    if S0.shape[0] == 0:
        return S0.copy(), np.zeros(0), np.zeros((0, NS, NS))
    MASK = np.ones((S0.shape[0], NS))
    MASK[:, 6] = (PP[:, KEYS.index("K")] != 0).astype(float)
    fn = get_newton(n_it, clip)
    outs = []
    for i0 in range(0, S0.shape[0], NCHUNK):
        sl = slice(i0, min(S0.shape[0], i0 + NCHUNK))
        o = fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]), jnp.asarray(MASK[sl]))
        outs.append([np.asarray(v) for v in o])
    return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]),
            np.concatenate([o[2] for o in outs]))

def max_mult(J):
    if J.shape[0] == 0:
        return np.zeros(0)
    return np.abs(np.linalg.eigvals(J)).max(1)

SIMS = {}
def get_sim(n_per, n_avg):
    key = (n_per, n_avg)
    if key in SIMS:
        return SIMS[key]
    def sim(s0, p):
        def one_period(carry, k):
            s, alive, tloss, rot, nacc = carry
            def step(i, c):
                s_, al_, tl_, dph_ = c
                t = k + i * DT
                sn = rk4(t, s_, p)
                lost = (((sn[0] < X_COLL) | (sn[0] > X_ESC)) & (p[7] > 0.5)) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + DT, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                dph = jnp.angle((sn[4] + 1j * sn[5]) * jnp.conj(s_[4] + 1j * s_[5]))
                return sn, al_n, tl_n, dph_ + dph
            s2, al2, tl2, dph = jax.lax.fori_loop(0, SPP, step, (s, alive, tloss, 0.0))
            use = (k >= n_per - n_avg) & al2
            rot = rot + jnp.where(use, dph / (2 * jnp.pi), 0.0)
            nacc = nacc + jnp.where(use, 1.0, 0.0)
            return (s2, al2, tl2, rot, nacc), jnp.stack([s2[0], s2[6]])
        (s, alive, tloss, rot, nacc), xw = jax.lax.scan(one_period, (s0, True, jnp.nan, 0.0, 0.0), jnp.arange(n_per))
        return s, alive, tloss, rot / jnp.maximum(nacc, 1.0), xw
    fn = jax.jit(jax.vmap(sim))
    SIMS[key] = fn
    return fn

def run(n_per, S0, PP, n_avg=1, label=None):
    n = S0.shape[0]
    fn = get_sim(n_per, n_avg)
    outs = []
    tb = time.time()
    for c_, i0 in enumerate(range(0, n, CHUNK)):
        sl = slice(i0, min(n, i0 + CHUNK))
        outs.append([np.asarray(v) for v in fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]))])
        if label and c_ % 5 == 0:
            print(f"      {label}: {min(n, i0 + CHUNK)}/{n} runs ({time.time() - tb:.0f} s)", flush=True)
    return [np.concatenate([o[j] for o in outs]) for j in range(5)]

def valid(S, x0=None, dx=0.5):
    r = np.hypot(S[:, 4], S[:, 5])
    ok = np.isfinite(S).all(1) & (S[:, 0] > X_COLL) & (S[:, 0] < X_ESC) & (r > 0.3) & (r < 2.5) & (S[:, 6] > 0.2)
    if x0 is not None:
        ok &= np.abs(S[:, 0] - x0) < dx
    return ok

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

SUMMARY = dict(mode=MODE, spp=SPP, U_ref_body_lengths_per_s=U_REF_BL)
G = {}
t_start = time.time()

# -----------------------------------------------------------------------------
# 2. Stable orbits of Script 10
# -----------------------------------------------------------------------------
print("\n=== Orbits ===")
S10_DIR = P("script10_continuation"); S10_CK = os.path.join(S10_DIR, "checkpoints_full")
PTS = np.load(os.path.join(S10_DIR, "points_script10.npz"))["points"]
BR = np.load(os.path.join(S10_CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
CL10 = dict(np.load(os.path.join(S10_CK, "stageD2_index.npz")))
SEL, FT = CL10["sel"].astype(int), CL10["ftype"].astype(str)
GROUPS = [("phase lag 0, saddle", 0.0, "saddle-real"), ("phase lag pi/2, saddle", np.pi / 2, "saddle-real"),
          ("phase lag pi/2, node", np.pi / 2, "node")]
rows = []
for gi, (gname, dp, ft) in enumerate(GROUPS):
    idx = np.where((np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft))[0]
    for j in np.sort(rng.choice(idx, min(N_PER_GROUP, idx.size), replace=False)):
        rows.append((gi, j))
NO = len(rows)
OG = np.array([g_ for g_, _ in rows])
JJ = np.array([j for _, j in rows])
OR, OE, OD = PTS[SEL[JJ], 4], PTS[SEL[JJ], 2], PTS[SEL[JJ], 3]
OS = np.zeros((NO, NS))
for k, j in enumerate(JJ):
    i = SEL[j]
    z = BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6]
    OS[k, :6] = z; OS[k, 6] = OR[k]
print(f"  {NO} orbits: " + ", ".join(f"{GROUPS[g_][0]} {int((OG == g_).sum())}" for g_ in range(len(GROUPS))))

print("\n=== G1. Model check ===")
res = np.max(np.abs(run(1, OS, par(NO, eps=OE, dphi=OD))[0] - OS), 1)
G["G1_max_residual"] = float(res.max()); G["G1_pass"] = bool(res.max() < 1e-6)
print(f"  one-period residual: max {res.max():.2e} ({'pass' if G['G1_pass'] else 'FAIL'})")

# -----------------------------------------------------------------------------
# 3. Schemes
#    flags: rhythm (CPG frequency set to follow the shedding), gain, stiffness, rates
#    (CPG relaxation rate, coupling and adaptation rate) retuned
# -----------------------------------------------------------------------------
SCHEMES = [("fixed", 0.0, dict(rhythm=0, gain=0, stiff=0, rates=0, pll=0)),
           ("imposed rhythm", 0.0, dict(rhythm=1, gain=0, stiff=0, rates=0, pll=0)),
           ("input-frequency rule K=1", 1.0, dict(rhythm=0, gain=0, stiff=0, rates=0, pll=0)),
           ("phase-locked K=0.03", 0.03, dict(rhythm=0, gain=0, stiff=0, rates=0, pll=1)),
           ("phase-locked K=0.1", 0.1, dict(rhythm=0, gain=0, stiff=0, rates=0, pll=1)),
           ("phase-locked K=0.3", 0.3, dict(rhythm=0, gain=0, stiff=0, rates=0, pll=1)),
           ("phase-locked K=0.1, gain retuned", 0.1, dict(rhythm=0, gain=1, stiff=0, rates=0, pll=1)),
           ("phase-locked K=0.1, gain and stiffness retuned", 0.1, dict(rhythm=0, gain=1, stiff=1, rates=0, pll=1)),
           ("all retuned", 0.0, dict(rhythm=1, gain=1, stiff=1, rates=1, pll=0))]
NSCH = len(SCHEMES)

PH0 = np.zeros(NO)          # preferred phase of the phase-locked loop, from the reference orbit
KSIGN = np.ones(NO)         # sign of K that makes the adapted orbit stable, chosen at s = 1

def scheme_par(idx, s, K, fl):
    idx = np.atleast_1d(np.asarray(idx, int))
    ts = 1.0 / s
    f = lambda keep: 1.0 if keep else ts
    Ke = K * f(fl["rates"]) * (KSIGN[idx] if fl.get("pll", 0) else 1.0)
    return par(idx.size, eps=OE[idx] * f(fl["rates"]), dphi=OD[idx], g=16.0 * f(fl["gain"]) ** 2,
               omr=1.2 * f(fl["stiff"]), mu=2 * np.pi * f(fl["rates"]), K=Ke,
               ph0=PH0[idx] if fl.get("pll", 0) else 0.0)

# preferred phase: <theta (z_i cos ph0 - z_r sin ph0)> = 0 over the reference orbit
@jax.jit
@jax.vmap
def theta_z_avg(s0, p):
    def step(i, c):
        s_, a = c
        sn = rk4(i * DT, s_, p)
        return sn, a + jnp.stack([sn[2] * sn[4], sn[2] * sn[5]]) * DT
    return jax.lax.fori_loop(0, SPP, step, (s0, jnp.zeros(2)))[1]
TZ = np.asarray(theta_z_avg(jnp.asarray(OS), jnp.asarray(par(NO, eps=OE, dphi=OD))))
PH0[:] = np.arctan2(TZ[:, 1], TZ[:, 0])

def set_rhythm(S, idx, s, K, fl, wc_ref):
    """CPG frequency at flow speed s: imposed (follows the shedding), fixed in physical time, or for adaptive
    schemes the current state, which starts from the reference value converted to the new flow speed."""
    S = S.copy()
    if fl["rhythm"]:
        S[:, 6] = wc_ref[idx]
    elif K == 0:
        S[:, 6] = wc_ref[idx] / s
    return S

# -----------------------------------------------------------------------------
# 4. Orbits of each scheme at s = 1
# -----------------------------------------------------------------------------
print("\n=== 1. Orbits at the reference flow speed ===")
f1 = ckpt("S1_reference.npz")
if os.path.exists(f1):
    d1 = dict(np.load(f1)); print("  loaded checkpoint")
else:
    t0 = time.time()
    REF_S = np.zeros((NSCH, NO, NS)); REF_OK = np.zeros((NSCH, NO), bool); REF_MU = np.full((NSCH, NO), np.nan)
    KSIGN_SAVE = np.ones((NSCH, NO))
    for k, (name, K, fl) in enumerate(SCHEMES):
        if fl.get("pll", 0):
            # the reference orbit is (close to) a periodic orbit of the loop for either sign of K; keep the stable one
            cand = []
            for sg in (1.0, -1.0):
                KSIGN[:] = sg
                PPk = scheme_par(np.arange(NO), 1.0, K, fl)
                Sn, Rn, Jn = newton(OS.copy(), PPk)
                okc = (Rn < 1e-7) & valid(Sn, OS[:, 0]); mmc = max_mult(Jn)
                cand.append((Sn, okc, mmc))
            st_p = cand[0][1] & (cand[0][2] < 1 - 1e-6); st_m = cand[1][1] & (cand[1][2] < 1 - 1e-6)
            use_m = ~st_p & st_m
            KSIGN[:] = np.where(use_m, -1.0, 1.0)
            Sn = np.where(use_m[:, None], cand[1][0], cand[0][0])
            ok = np.where(use_m, cand[1][1], cand[0][1]); mm = np.where(use_m, cand[1][2], cand[0][2])
            KSIGN_SAVE[k] = KSIGN
        else:
            PPk = scheme_par(np.arange(NO), 1.0, K, fl)
            S0 = OS.copy()
            if K > 0:
                S0 = run(T_ADAPT, S0, PPk, label=f"adapt {name}")[0]
            Sn, Rn, Jn = newton(S0, PPk)
            ok = (Rn < 1e-7) & valid(Sn, OS[:, 0])
            mm = max_mult(Jn)
        REF_S[k] = np.where(ok[:, None], Sn, OS); REF_OK[k] = ok & (mm < 1 - 1e-6); REF_MU[k] = np.where(ok, mm, np.nan)
        print(f"  {name}: stable orbit at s = 1 for {REF_OK[k].mean():.2f} of orbits; adapted omega_c/omega_w "
              f"median {np.median(Sn[ok, 6]) if ok.any() else np.nan:.3f} (original {np.median(OR):.3f})")
    d1 = dict(S=REF_S, ok=REF_OK, mu=REF_MU, ksign=KSIGN_SAVE, ph0=PH0)
    np.savez(f1, **d1)
    print(f"  done in {time.time() - t0:.0f} s")

# -----------------------------------------------------------------------------
# 5. Continuation in flow speed
# -----------------------------------------------------------------------------
print("\n=== 2. Continuation in flow speed ===")
f2 = ckpt("S2_continuation.npz")
if os.path.exists(f2):
    d2 = dict(np.load(f2)); print("  loaded checkpoint")
else:
    t0 = time.time()
    band = np.zeros((NSCH, NO, 2)); wc_edge = np.full((NSCH, NO, 2), np.nan)
    for k, (name, K, fl) in enumerate(SCHEMES):
        KSIGN[:] = d1["ksign"][k]
        wc_ref = d1["S"][k, :, 6]
        for d_, sgn in enumerate((-1.0, 1.0)):
            S = d1["S"][k].copy(); alive = d1["ok"][k].copy(); last = np.zeros(NO)
            last_wc = d1["S"][k, :, 6].copy()
            for ds in S_STEPS:
                idx = np.where(alive)[0]
                if idx.size == 0:
                    break
                s_val = 1.0 + sgn * ds
                PPk = scheme_par(idx, s_val, K, fl)
                S0 = set_rhythm(S[idx], idx, s_val, K, fl, wc_ref)
                Sn, Rn, Jn = newton(S0, PPk, n_it=10)
                st = (Rn < 1e-7) & valid(Sn) & (max_mult(Jn) < 1 - 1e-6)
                S[idx[st]] = Sn[st]; last[idx[st]] = ds; last_wc[idx[st]] = Sn[st, 6]
                alive[idx[~st]] = False
            band[k, :, d_] = np.where(d1["ok"][k], last, np.nan)
            wc_edge[k, :, d_] = np.where(d1["ok"][k], last_wc, np.nan)
        print(f"  {name}: median band {np.nanmedian(100 * band[k].sum(1)):.2f}% ({time.time() - t0:.0f} s)")
    d2 = dict(band=band, wc_edge=wc_edge)
    np.savez(f2, **d2)

# -----------------------------------------------------------------------------
# 6. Direct simulation on a coarse grid of flow speeds
# -----------------------------------------------------------------------------
print("\n=== 3. Direct simulation on a coarse grid of flow speeds ===")
f3 = ckpt("S3_coarse.npz")
if os.path.exists(f3):
    d3 = dict(np.load(f3)); print("  loaded checkpoint")
else:
    t0 = time.time()
    nS = S_COARSE.size
    hold = np.zeros((NSCH, NO, nS), bool); lock = np.zeros_like(hold); wc_end = np.full((NSCH, NO, nS), np.nan)
    for k, (name, K, fl) in enumerate(SCHEMES):
        KSIGN[:] = d1["ksign"][k]
        ii = np.repeat(np.arange(NO), nS); ss = np.tile(S_COARSE, NO)
        wc_ref = d1["S"][k, :, 6]
        PPk = np.concatenate([scheme_par(np.array([i]), s_, K, fl) for i, s_ in zip(ii, ss)])
        S0 = np.concatenate([set_rhythm(d1["S"][k, [i]], np.array([i]), s_, K, fl, wc_ref) for i, s_ in zip(ii, ss)])
        S0[:, 6] = np.where((K > 0) & (fl["rhythm"] == 0), d1["S"][k, ii, 6] / ss, S0[:, 6])   # adaptive: start from fixed physical value
        o = run(T_FLOW, S0, PPk, n_avg=T_FLOW // 4, label=name)
        okref = d1["ok"][k][ii]
        hold[k] = (o[1] & okref).reshape(NO, -1)
        lock[k] = (o[1] & okref & (np.abs(o[3] - 1.0) < 0.01)).reshape(NO, -1)
        wc_end[k] = np.where(o[1] & okref, o[0][:, 6], np.nan).reshape(NO, -1)
        print(f"  {name}: done ({time.time() - t0:.0f} s)")
    d3 = dict(hold=hold, lock=lock, wc_end=wc_end)
    np.savez(f3, **d3)

# -----------------------------------------------------------------------------
# 7. Time series after a step change of flow speed (illustration)
# -----------------------------------------------------------------------------
print("\n=== 4. Step change of flow speed ===")
f4 = ckpt("S4_step.npz")
if os.path.exists(f4):
    d4 = dict(np.load(f4)); print("  loaded checkpoint")
else:
    ex_k = [i for i, sc in enumerate(SCHEMES) if sc[0] in ("fixed", "phase-locked K=0.1", "phase-locked K=0.1, gain retuned")]
    k_ref = [i for i, sc in enumerate(SCHEMES) if sc[0] == "phase-locked K=0.1"][0]
    ex_o = np.array([i for g_ in range(len(GROUPS))
                     for i in np.where(d1["ok"][k_ref] & (OG == g_))[0][:max(1, N_STEP_EX // len(GROUPS))]], dtype=int)
    xs_all, wc_all, tag = [], [], []
    for k in ex_k:
        name, K, fl = SCHEMES[k]
        KSIGN[:] = d1["ksign"][k]
        for s_ in S_STEP_EX:
            PPk = scheme_par(ex_o, s_, K, fl)
            S0 = d1["S"][k, ex_o].copy(); S0[:, 6] = S0[:, 6] / s_ if not fl["rhythm"] else S0[:, 6]
            o = run(T_STEP_EX, S0, PPk)
            xs_all.append(o[4][:, :, 0]); wc_all.append(o[4][:, :, 1])
            tag.append(np.column_stack([np.full(ex_o.size, k), np.full(ex_o.size, s_), ex_o]))
    d4 = dict(x=np.concatenate(xs_all), wc=np.concatenate(wc_all), tag=np.concatenate(tag))
    np.savez(f4, **d4)
    print("  done")

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
R = {}
rows_csv = []
for k, (name, K, fl) in enumerate(SCHEMES):
    ok = d1["ok"][k]
    width = 100 * d2["band"][k].sum(1)
    rec = dict(K=K, flags=fl, stable_at_reference=float(ok.mean()),
               negative_K_fraction=float(np.mean(d1["ksign"][k][ok] < 0)) if ok.any() else np.nan,
               adapted_ratio_at_reference=dist(d1["S"][k, ok, 6]),
               band_width_percent=dist(width), band_lower=dist(d2["band"][k, :, 0]), band_upper=dist(d2["band"][k, :, 1]),
               hold_fraction_by_s={f"{s_:.3f}": float(d3["hold"][k, :, i].mean()) for i, s_ in enumerate(S_COARSE)},
               lock_fraction_by_s={f"{s_:.3f}": float(d3["lock"][k, :, i].mean()) for i, s_ in enumerate(S_COARSE)},
               adapted_ratio_by_s={f"{s_:.3f}": float(np.nanmedian(d3["wc_end"][k, :, i]))
                                   if np.isfinite(d3["wc_end"][k, :, i]).any() else np.nan for i, s_ in enumerate(S_COARSE)})
    for gi, (gname, _, _) in enumerate(GROUPS):
        rec[f"band_width_percent {gname}"] = dist(width[OG == gi])
        rec[f"stable_at_reference {gname}"] = float(ok[OG == gi].mean())
    # contiguous range of s around 1 in which at least half of the fish hold with a locked rhythm
    lf = d3["lock"][k].mean(0); i1 = int(np.argmin(np.abs(S_COARSE - 1.0)))
    lo = i1
    while lo > 0 and lf[lo - 1] >= 0.5:
        lo -= 1
    hi = i1
    while hi < S_COARSE.size - 1 and lf[hi + 1] >= 0.5:
        hi += 1
    rec["half_locked_s_range"] = [float(S_COARSE[lo]), float(S_COARSE[hi])] if lf[i1] >= 0.5 else None
    R[name] = rec
    rows_csv.append([name, K, rec["stable_at_reference"], np.nanmedian(width), np.nanmin(width), np.nanmax(width),
                     rec["half_locked_s_range"][0] if rec["half_locked_s_range"] else np.nan,
                     rec["half_locked_s_range"][1] if rec["half_locked_s_range"] else np.nan])
save_csv("flow_speed_schemes.csv", ["scheme", "K", "stable_at_reference", "band_width_median_percent",
                                    "band_width_min", "band_width_max", "half_locked_s_low", "half_locked_s_high"], rows_csv)
print("\n  scheme: stable at s = 1, band width median (min to max), range of s with half of the fish locked")
for r in rows_csv:
    print(f"  {r[0]:>46s}: {r[2]:.2f}, {r[3]:.2f}% ({r[4]:.2f} to {r[5]:.2f}%), s {r[6]} to {r[7]}")
    print("      locked fraction by s: " + ", ".join(f"{s_:.3f}:{R[r[0]]['lock_fraction_by_s'][f'{s_:.3f}']:.2f}" for s_ in S_COARSE))
G["G2_fixed_band_median_percent"] = R["fixed"]["band_width_percent"].get("median", np.nan)
G["G2_pass"] = bool(np.isfinite(G["G2_fixed_band_median_percent"]) and G["G2_fixed_band_median_percent"] < 1.0)
G["G3_pass"] = bool(np.all(d2["band"][NSCH - 1][d1["ok"][NSCH - 1]] >= S_STEPS[-1] - 1e-9))
print(f"\n  G2: fixed scheme band median {G['G2_fixed_band_median_percent']:.2f}% "
      f"({'pass' if G['G2_pass'] else 'FAIL'}: Script 22 gives 0.15%)")
print(f"  G3: all-retuned control holds over the whole range ({'pass' if G['G3_pass'] else 'FAIL'})")
SUMMARY["schemes"] = R
SUMMARY["gates"] = G
SUMMARY["s_coarse"] = S_COARSE

# -----------------------------------------------------------------------------
# Draft figure (final version is made with the figure script)
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
try:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, (name, K, fl) in enumerate(SCHEMES):
        ax[0].plot(S_COARSE, d3["lock"][k].mean(0), "o-", ms=3, label=name)
        if K > 0:
            ax[1].plot(S_COARSE, [R[name]["adapted_ratio_by_s"][f"{s_:.3f}"] for s_ in S_COARSE], "o-", ms=3, label=name)
    ax[0].set_xlabel(r"$U/U_{\mathrm{ref}}$"); ax[0].set_ylabel("fraction locked and holding"); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8, framealpha=1, edgecolor="k")
    ax[1].plot(S_COARSE, np.median(OR) / S_COARSE, "k--", label="fixed in physical time")
    ax[1].set_xlabel(r"$U/U_{\mathrm{ref}}$"); ax[1].set_ylabel(r"adapted $\omega_c/\omega_w$")
    ax[1].legend(fontsize=8, framealpha=1, edgecolor="k")
    for j in range(min(6, d4["wc"].shape[0])):
        ax[2].plot(np.arange(1, d4["wc"].shape[1] + 1), d4["wc"][j], lw=1)
    ax[2].set_xlabel("shedding periods after the step"); ax[2].set_ylabel(r"$\omega_c/\omega_w$")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"S23_adaptive.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"  figure skipped: {e}")

SUMMARY["runtime_s"] = time.time() - t_start
with open(os.path.join(OUT_DIR, "summary_script23.json"), "w") as f:
    json.dump(to_py(SUMMARY), f, indent=2)
print("\n================ SCRIPT 23 SUMMARY ================")
print("Gates: " + ", ".join(f"{k}: {v}" for k, v in G.items() if k.endswith("pass")))
print(f"Runtime {SUMMARY['runtime_s'] / 60:.1f} min. Outputs: {OUT_DIR}")