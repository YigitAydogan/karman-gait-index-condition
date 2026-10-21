# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 11: Bounded holding, noise, and testable predictions.
#
# Colab ready. Uses JAX (preinstalled on Colab); runs on the GPU when one is
# attached (L4: about 20 to 40 min FULL). Every part is checkpointed to
# Drive; if Colab disconnects, run again and finished parts are loaded.
# Needs the Script 10 outputs (points, branches, refined index) on Drive.
#
# Gaps closed (manuscript plan): 5 (bounded surging as holding), 6 (turbulence-
# like noise), 7 (lateral line), 13 (trout parameter mapping), 15 (predictions
# that can be checked against data).
#
# Model: the baseline of Scripts 05b to 10, plus
#   - wake phase noise:      psi' = sqrt(2 D) xi,  w = (1 + a) W(x) sin(2 pi (x - t) + psi)
#   - wake amplitude noise:  a is Ornstein-Uhlenbeck, correlation time 1 T, std sqrt(D)
#   - CPG phase noise:       z rotated by sqrt(2 D dt) xi each step
#   - lateral-line input:    CPG input eps (theta + e_LL w), w sensed at the fish
#   - flow speed s = U/U_ref: the fish's internal rates are fixed in physical
#     time, so in shedding units Omega, mu, omega_c, eps scale by 1/s and g by 1/s^2
# Noise level D is one number (rad^2 per shedding period for the phases).
# A run "holds" while 0.15 < x < 4.5 (no collision with the cylinder, no
# escape downstream); bounded surging counts as holding. Initial conditions:
# x uniform in [0.3, 1.5] wavelengths (the region where holding exists), at
# rest, random CPG phase. Holding probability is therefore a basin fraction:
# about 0.1 to 0.2 inside the stable region, since holding position and
# rhythm phase must match.
#
# Parts:
#   A. Deterministic holding-probability maps over (omega_c/omega_w, eps), all
#      4 neural delays, random initial conditions; period-1 vs surging.
#   B. The same maps under noise (D = 0.003), delays 0 and pi/2.
#   C. Bout durations: survival of holding from Script 10 stable orbits, by
#      fixed-position branch (saddle, node), at 4 joint noise levels and with
#      each noise source alone (D = 0.01) to show which source ends holding.
#   D. Tail-beat phase relative to vortex arrival at the fish, per branch,
#      deterministic (all Script 10 classified points) and under noise.
#   E. Lateral-line removal: hold with lateral line + proprioception, then
#      switch the lateral line off; holding retained vs kinematics changed.
#   F. Flow-speed path: holding probability and tail-beat frequency locking
#      as flow speed changes with the fish's own parameters fixed.
#   G. Trout mapping: dimensional values from Beal et al. (2006) trout.
#
# Gates (reported):
#   G1  >= 90% of Script 10 stable cells hold in direct simulation (map, then a
#       recheck of missed cells with 512 random starts and 32 small kicks)
#   G2  at delay pi/2 the tail-beat phase differs between node and saddle
#       branches by more than 3 combined circular standard errors
#   G3  the G2 difference keeps its sign under noise D = 0.003
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

MODE = "FULL"
SEED = 15
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script11_noise_predictions")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
for d_ in (FIG_DIR, CKPT_DIR):
    os.makedirs(d_, exist_ok=True)
S10_DIR = os.path.join(BASE_DIR, "script10_continuation")
S10_CK = os.path.join(S10_DIR, f"checkpoints_{MODE.lower()}")

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
if MODE == "FULL":
    SPP = 100
    MAP_RATIOS = np.round(np.arange(0.95, 1.2501, 0.005), 6)
    MAP_EPS = np.round(np.arange(0.25, 6.001, 0.25), 6)
    DPHI_ALL = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    DPHI_HOLD = [0.0, np.pi / 2]
    N_IC = 32
    T_MAP = 400
    NOISE_LEVELS = [0.003, 0.01, 0.03, 0.1]
    N_BOUT_PTS, N_REP, T_BOUT = 20, 64, 2000
    T_LL = 400
    FLOW_S = np.round(np.linspace(0.7, 1.4, 29), 6)
    FLOW_EPS = [0.5, 1.0, 2.0, 3.0]
    CHUNK = 16384 if ON_GPU else 1024
else:
    SPP = 80
    MAP_RATIOS = np.round(np.arange(1.00, 1.0601, 0.02), 6)
    MAP_EPS = np.array([1.0, 2.0])
    DPHI_ALL = [0.0, np.pi / 2]
    DPHI_HOLD = [0.0, np.pi / 2]
    N_IC = 4
    T_MAP = 60
    NOISE_LEVELS = [0.003, 0.1]
    N_BOUT_PTS, N_REP, T_BOUT = 3, 4, 80
    T_LL = 60
    FLOW_S = np.array([0.8, 1.0, 1.2])
    FLOW_EPS = [1.0]
    CHUNK = 512
NOISE_MAP = 0.003
N_IC_A2 = 512 if MODE == "FULL" else 16      # re-test of stable cells with no holding in the map
N_KICK = 32 if MODE == "FULL" else 4
FLOW_S_FINE = np.round(np.arange(0.95, 1.0501, 0.0025), 6) if MODE == "FULL" else np.array([0.99, 1.0, 1.01])
N_IC_F2 = 64 if MODE == "FULL" else 4
NOISE_SRC_D = 0.01               # single-source runs: which noise source ends holding
X_COLL, X_ESC = 0.15, 4.5
TAU_A = 1.0
SURGE_MIN = 0.01                 # strobe x span over the last 50 periods above which holding counts as surging
OMEGA_W = 2.0 * np.pi

# Beal et al. (2006) trout in a cylinder wake: wake wavelength, shedding frequency, body length
TROUT = dict(wavelength_m=0.208, f_shed_Hz=2.74, body_length_m=0.178)

# -----------------------------------------------------------------------------
# 1. Model in JAX
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA = 3.0, 0.15, 8.0
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU, G_HOLD = 2 * np.pi, 16.0
DT = 1.0 / SPP

def rhs(t, s, p, psi, amp):
    """s = (x, u, theta, omega, zr, zi); p = (ratio, eps, dphi, e_LL, w_prop, tscale)."""
    x, u, th, om, zr, zi = s
    ratio, eps, dphi, eLL, wp, ts = p
    V = U_INF * (1.0 - DELTA0 * jnp.exp(-x / L_DELTA)) - u
    w = (1.0 + amp) * W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * \
        jnp.sin(OMEGA_W * (x - t) + psi)
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
    Om = OM_B * ts
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    dom = -2 * ZE_B * Om * om - Om ** 2 * th + SIGMA * L + G_HOLD * ts ** 2 * drive
    r2 = zr * zr + zi * zi
    a = MU * ts * (1.0 - r2)
    b = ratio * ts * OMEGA_W
    inp = eps * ts * (wp * th + eLL * w)
    return jnp.stack([u, Fx, om, dom, a * zr - b * zi + inp, a * zi + b * zr]), w

def make_sim(n_per, record, noisy, acc_from=None):
    """Simulate n_per shedding periods. Returns final state, alive flag, loss time,
    per-period strobe x (if record), cross-spectrum of tail angle vs local wake
    velocity (second half, alive periods), mean tail-beat amplitude, CPG
    rotation per period (second half)."""
    def sim(s0, p, nz, key):
        Dpsi, sig_a, Dz = nz[0], nz[1], nz[2]
        def one_period(carry, k):
            s, psi, amp, alive, tloss, X, A, rot, nacc = carry
            kp = jax.random.fold_in(key, k)
            def step(i, c):
                s_, psi_, amp_, al_, tl_, Th_, Wf_, dphs_ = c
                t = k + i * DT
                k1, w1 = rhs(t, s_, p, psi_, amp_)
                k2, _ = rhs(t + DT / 2, s_ + DT / 2 * k1, p, psi_, amp_)
                k3, _ = rhs(t + DT / 2, s_ + DT / 2 * k2, p, psi_, amp_)
                k4, _ = rhs(t + DT, s_ + DT * k3, p, psi_, amp_)
                sn = s_ + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if noisy:
                    nrm = jax.random.normal(jax.random.fold_in(kp, i), (3,))
                    psi_n = psi_ + jnp.sqrt(2 * Dpsi * DT) * nrm[0]
                    ea = jnp.exp(-DT / TAU_A)
                    amp_n = amp_ * ea + sig_a * jnp.sqrt(1 - ea * ea) * nrm[1]
                    ang = jnp.sqrt(2 * Dz * DT) * nrm[2]
                    zr2 = sn[4] * jnp.cos(ang) - sn[5] * jnp.sin(ang)
                    zi2 = sn[4] * jnp.sin(ang) + sn[5] * jnp.cos(ang)
                    sn = sn.at[4].set(zr2).at[5].set(zi2)
                else:
                    psi_n, amp_n = psi_, amp_
                lost = (sn[0] < X_COLL) | (sn[0] > X_ESC) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + DT, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                e = jnp.exp(-1j * OMEGA_W * (i * DT))
                Th_n = Th_ + s_[2] * e * DT
                Wf_n = Wf_ + w1 * e * DT
                dph = jnp.angle((sn[4] + 1j * sn[5]) * jnp.conj(s_[4] + 1j * s_[5]))
                return sn, psi_n, amp_n, al_n, tl_n, Th_n, Wf_n, dphs_ + dph
            init = (s, psi, amp, alive, tloss, 0j, 0j, 0.0)
            s, psi, amp, alive2, tloss, Th, Wf, dphs = jax.lax.fori_loop(0, SPP, step, init)
            use = alive2 & (k >= (n_per // 2 if acc_from is None else acc_from))
            X = X + jnp.where(use, Th * jnp.conj(Wf), 0j)
            A = A + jnp.where(use, 2 * jnp.abs(Th), 0.0)
            rot = rot + jnp.where(use, dphs / OMEGA_W, 0.0)
            nacc = nacc + jnp.where(use, 1.0, 0.0)
            out = s[0] if record else None
            return (s, psi, amp, alive2, tloss, X, A, rot, nacc), out
        carry0 = (s0, 0.0, 0.0, True, jnp.nan, 0j, 0.0, 0.0, 0.0)
        (s, psi, amp, alive, tloss, X, A, rot, nacc), xs = jax.lax.scan(one_period, carry0, jnp.arange(n_per))
        res = (s, alive, tloss, jnp.real(X), jnp.imag(X), A / jnp.maximum(nacc, 1), rot / jnp.maximum(nacc, 1), nacc)
        return res + ((xs,) if record else ())
    return jax.jit(jax.vmap(sim))

def batched(fn, *arrays, label=None):
    n = arrays[0].shape[0]
    outs = []
    tb = time.time()
    n_ch = -(-n // CHUNK)
    for c_, i0 in enumerate(range(0, n, CHUNK)):
        if label and (c_ % 5 == 0):
            print(f"     {label}: chunk {c_ + 1}/{n_ch} ({time.time() - tb:.0f} s)")
        sl = [a[i0:i0 + CHUNK] for a in arrays]
        m = sl[0].shape[0]
        if m < CHUNK:
            sl = [np.concatenate([a, np.repeat(a[-1:], CHUNK - m, axis=0)]) for a in sl]
        o = fn(*[jnp.asarray(a) for a in sl])
        outs.append([np.asarray(v)[:m] for v in o])
    return tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))

def keys_for(n, salt):
    return np.asarray(jax.random.split(jax.random.PRNGKey(SEED * 1000 + salt), n))

def random_ics(n):
    s = np.zeros((n, 6))
    s[:, 0] = rng.uniform(0.3, 1.5, n)
    s[:, 2] = rng.normal(0, 0.1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s[:, 4], s[:, 5] = np.cos(ph), np.sin(ph)
    return s

def params(ratio, eps, dphi, eLL=0.0, wp=1.0, ts=1.0):
    b = np.broadcast_arrays(ratio, eps, dphi, eLL, wp, ts)
    return np.stack([np.asarray(v, float) for v in b], 1)

def noise_arr(n, D):
    return np.tile([D, np.sqrt(D), D], (n, 1)).astype(float)

# noise conditions for bouts: joint levels, then each source alone at NOISE_SRC_D
NOISE_CONDS = [(f"joint D={D}", np.array([D, np.sqrt(D), D])) for D in NOISE_LEVELS] + [
    (f"wake phase only D={NOISE_SRC_D}", np.array([NOISE_SRC_D, 0.0, 0.0])),
    (f"wake amplitude only D={NOISE_SRC_D}", np.array([0.0, np.sqrt(NOISE_SRC_D), 0.0])),
    (f"CPG phase only D={NOISE_SRC_D}", np.array([0.0, 0.0, NOISE_SRC_D]))]

def circ_stats(ang, wts=None):
    ang = np.asarray(ang)
    if ang.size == 0:
        return dict(n=0, mean=None, R=None, se=None)
    z = np.exp(1j * ang)
    m = z.mean()
    R = np.abs(m)
    se = np.sqrt(max(1 - R ** 2, 1e-12) / (ang.size * max(R ** 2, 1e-12)))
    return dict(n=int(ang.size), mean=float(np.angle(m)), R=float(R), se=float(se))

def cdiff(a, b):
    return float(np.angle(np.exp(1j * (a - b))))

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

t_start = time.time()
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

# Script 10 outputs
PTS = np.load(os.path.join(S10_DIR, "points_script10.npz" if MODE == "FULL" else "points_script10_quick.npz"))["points"]
BR = np.load(os.path.join(S10_CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
CL10 = dict(np.load(os.path.join(S10_CK, "stageD2_index.npz")))
SEL, FT = CL10["sel"].astype(int), CL10["ftype"].astype(str)
S_CL = np.array([BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6] for i in SEL])
print(f"Script 10: {SEL.size} classified stable points loaded")

SIM_MAP = make_sim(T_MAP, True, False)
SIM_MAPN = make_sim(T_MAP, False, True)
SIM_BOUT = make_sim(T_BOUT, False, True, acc_from=5)     # phase measured over the bout itself
SIM_ONE = make_sim(20, False, False)
SIM_LL = make_sim(T_LL, False, False)

# -----------------------------------------------------------------------------
# A. Deterministic holding-probability maps
# -----------------------------------------------------------------------------
def holding_map(dphis, noisy, label, eLL=0.0):
    R_, E_, D_ = np.meshgrid(MAP_RATIOS, MAP_EPS, dphis, indexing="ij")
    R_, E_, D_ = R_.ravel(), E_.ravel(), D_.ravel()
    n = R_.size * N_IC
    S0 = random_ics(n)
    P = params(np.repeat(R_, N_IC), np.repeat(E_, N_IC), np.repeat(D_, N_IC), eLL)
    if noisy:
        out = batched(SIM_MAPN, S0, P, noise_arr(n, NOISE_MAP), keys_for(n, 2), label=label)
        xs = None
    else:
        out = batched(SIM_MAP, S0, P, np.zeros((n, 3)), keys_for(n, 1), label=label)
        xs = out[8]
    alive = out[1]
    shape = (MAP_RATIOS.size, MAP_EPS.size, len(dphis), N_IC)
    res = dict(alive=alive.reshape(shape), x_end=out[0][:, 0].reshape(shape),
               rot=out[6].reshape(shape))
    if xs is not None:
        tail = xs[:, -50:]
        span = tail.max(1) - tail.min(1)
        res["p1"] = (alive & (span < SURGE_MIN)).reshape(shape)
        res["surge"] = (alive & (span >= SURGE_MIN)).reshape(shape)
        res["surge_amp"] = np.where(alive, span, np.nan).reshape(shape)
    return res

if os.path.exists(ckpt("partA_map.npz")):
    MA = dict(np.load(ckpt("partA_map.npz")))
    print("A: loaded from checkpoint")
else:
    t0 = time.time()
    MA = holding_map(DPHI_ALL, False, "A map")
    np.savez_compressed(ckpt("partA_map.npz"), **MA)
    print(f"A: deterministic map done ({time.time() - t0:.0f} s)")
PH_A = MA["alive"].mean(-1)                         # (ratio, eps, dphi)

# G1: Script 10 stable cells inside the map's holding region
g1_in, g1_n = 0, 0
g1_cells_missed = []
for dq, dp in enumerate(DPHI_ALL):
    m = (PTS[:, 7] > 0) & (np.abs(PTS[:, 3] - dp) < 1e-6)
    cells = set()
    for r_, e_ in zip(PTS[m, 4], PTS[m, 2]):
        ei = np.where(np.abs(MAP_EPS - e_) < 1e-6)[0]
        if ei.size == 0:
            continue
        ri = int(np.argmin(np.abs(MAP_RATIOS - r_)))
        if abs(MAP_RATIOS[ri] - r_) <= 0.0026:
            cells.add((ri, int(ei[0])))
    for ri, ei in cells:
        g1_n += 1
        g1_in += int(PH_A[ri, ei, dq] > 0)
        if PH_A[ri, ei, dq] == 0:
            g1_cells_missed.append((dq, ri, ei))

# A2: stable cells where the map found no holding: many more random starts, and
# small kicks (x +/- 0.05, CPG phase +/- 0.3) from the continued stable orbit
def nearest_orbit(dp, r_, e_):
    m = (PTS[:, 7] > 0) & (np.abs(PTS[:, 3] - dp) < 1e-6) & (np.abs(PTS[:, 2] - e_) < 1e-6)
    i = np.where(m)[0][int(np.argmin(np.abs(PTS[m, 4] - r_)))]
    return BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1])]          # (state, ratio)

if os.path.exists(ckpt("partA2_recheck.npz")):
    A2 = dict(np.load(ckpt("partA2_recheck.npz")))
    print("A2: loaded from checkpoint")
elif g1_cells_missed:
    t0 = time.time()
    rows_S, rows_P, rows_c, rows_k = [], [], [], []
    for c_, (dq, ri, ei) in enumerate(g1_cells_missed):
        dp, r_, e_ = DPHI_ALL[dq], MAP_RATIOS[ri], MAP_EPS[ei]
        z = nearest_orbit(dp, r_, e_)
        S_r = random_ics(N_IC_A2)
        S_k = np.repeat(z[None, :6], N_KICK, 0)
        S_k[:, 0] += rng.normal(0, 0.05, N_KICK)
        ph = np.arctan2(S_k[:, 5], S_k[:, 4]) + rng.normal(0, 0.3, N_KICK)
        S_k[:, 4], S_k[:, 5] = np.cos(ph), np.sin(ph)
        for S_, kind in ((S_r, 0), (S_k, 1)):
            rows_S.append(S_); rows_P.append(params(np.full(len(S_), z[6]), e_, dp))
            rows_c.append(np.full(len(S_), c_)); rows_k.append(np.full(len(S_), kind))
    S_ = np.concatenate(rows_S); P_ = np.concatenate(rows_P)
    o = batched(SIM_MAPN, S_, P_, np.zeros((len(S_), 3)), keys_for(len(S_), 8), label="A2 recheck")
    A2 = dict(cell=np.concatenate(rows_c), kind=np.concatenate(rows_k), alive=o[1], n_cells=len(g1_cells_missed))
    np.savez_compressed(ckpt("partA2_recheck.npz"), **A2)
    print(f"A2: {len(g1_cells_missed)} cells rechecked ({time.time() - t0:.0f} s)")
else:
    A2 = None
a2_sum = None
if A2 is not None and int(A2["n_cells"]) > 0:
    nc_ = int(A2["n_cells"])
    rnd = [A2["alive"][(A2["cell"] == c_) & (A2["kind"] == 0)].mean() for c_ in range(nc_)]
    kck = [A2["alive"][(A2["cell"] == c_) & (A2["kind"] == 1)].mean() for c_ in range(nc_)]
    a2_sum = dict(cells=nc_, found_by_random_starts=int(sum(v > 0 for v in rnd)),
                  basin_fraction_median=float(np.median(rnd)),
                  held_after_kicks=int(sum(v > 0 for v in kck)), kick_survival_median=float(np.median(kck)))
    g1_in_final = g1_in + sum(1 for a_, b_ in zip(rnd, kck) if a_ > 0 or b_ > 0)
else:
    g1_in_final = g1_in

# -----------------------------------------------------------------------------
# B. Holding-probability maps under noise
# -----------------------------------------------------------------------------
if os.path.exists(ckpt("partB_map_noise.npz")):
    MB = dict(np.load(ckpt("partB_map_noise.npz")))
    print("B: loaded from checkpoint")
else:
    t0 = time.time()
    MB = holding_map(DPHI_HOLD, True, f"B map, noise {NOISE_MAP}")
    np.savez_compressed(ckpt("partB_map_noise.npz"), **MB)
    print(f"B: noisy map done ({time.time() - t0:.0f} s)")
PH_B = MB["alive"].mean(-1)

# -----------------------------------------------------------------------------
# C + D. Bout durations and tail-beat phase signature
# -----------------------------------------------------------------------------
def sample_points(dp, ftype, k):
    m = (np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ftype)
    idx = np.where(m)[0]
    if idx.size == 0:
        return idx
    return np.sort(rng.choice(idx, min(k, idx.size), replace=False))

GROUPS = [(dp, ft_) for dp in DPHI_HOLD for ft_ in ("saddle-real", "node")]
if os.path.exists(ckpt("partCD2.npz")):
    CD = dict(np.load(ckpt("partCD2.npz"), allow_pickle=True))
    print("C+D: loaded from checkpoint")
else:
    t0 = time.time()
    # D, deterministic: phase of tail angle relative to local wake velocity on every classified point
    n = SEL.size
    P_all = params(PTS[SEL, 4], PTS[SEL, 2], PTS[SEL, 3])
    o = batched(SIM_ONE, S_CL, P_all, np.zeros((n, 3)), keys_for(n, 3), label="D deterministic")
    phase_det = np.angle(o[3] + 1j * o[4])
    amp_det = o[5]
    # C: bouts from sampled points of each group, at each noise level
    rows = []
    for g_, (dp, ft_) in enumerate(GROUPS):
        pick = sample_points(dp, ft_, N_BOUT_PTS)
        for lv in range(len(NOISE_CONDS)):
            for q in pick:
                rows.append((g_, lv, q))
    rows = np.array(rows, int).reshape(-1, 3)
    m_ = rows.shape[0] * N_REP
    gi = np.repeat(rows[:, 0], N_REP); li = np.repeat(rows[:, 1], N_REP); qi = np.repeat(rows[:, 2], N_REP)
    S0 = S_CL[qi]
    P0 = params(PTS[SEL[qi], 4], PTS[SEL[qi], 2], PTS[SEL[qi], 3])
    NZ = np.stack([NOISE_CONDS[l_][1] for l_ in li]) if m_ else np.zeros((0, 3))
    ob = batched(SIM_BOUT, S0, P0, NZ, keys_for(m_, 4), label="C bouts") if m_ else None
    CD = dict(phase_det=phase_det, amp_det=amp_det,
              b_group=gi, b_level=li, b_point=qi,
              b_alive=ob[1] if m_ else np.zeros(0, bool), b_tloss=ob[2] if m_ else np.zeros(0),
              b_phase=np.angle(ob[3] + 1j * ob[4]) if m_ else np.zeros(0),
              b_nacc=ob[7] if m_ else np.zeros(0))
    np.savez_compressed(ckpt("partCD2.npz"), **CD)
    print(f"C+D: done ({time.time() - t0:.0f} s)")

phase_det = CD["phase_det"]
sig_det = {}
for dp in DPHI_HOLD:
    for ft_ in ("saddle-real", "node", "none"):
        m = (np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft_)
        sig_det[f"{dp:.3f}/{ft_}"] = circ_stats(phase_det[m])
bouts = {}
sig_noise = {}
for g_, (dp, ft_) in enumerate(GROUPS):
    for lv, (lab, _) in enumerate(NOISE_CONDS):
        m = (CD["b_group"] == g_) & (CD["b_level"] == lv)
        if not m.any():
            continue
        tl = np.where(CD["b_alive"][m], np.inf, CD["b_tloss"][m])
        bouts[f"{dp:.3f}/{ft_}/{lab}"] = dict(
            runs=int(m.sum()), surviving_fraction=float(np.isinf(tl).mean()),
            median_bout_T=float(np.median(tl)) if np.isfinite(np.median(tl)) else None,
            q25_bout_T=float(np.quantile(tl, 0.25)) if np.isfinite(np.quantile(tl, 0.25)) else None)
        ok = m & (CD["b_nacc"] > 0)
        sig_noise[f"{dp:.3f}/{ft_}/{lab}"] = circ_stats(CD["b_phase"][ok])

# median bout vs joint noise level: power-law exponent per group
bout_scaling = {}
for g_, (dp, ft_) in enumerate(GROUPS):
    Ds, med = [], []
    for D in NOISE_LEVELS:
        v_ = bouts.get(f"{dp:.3f}/{ft_}/joint D={D}")
        if v_ and v_["median_bout_T"]:
            Ds.append(D); med.append(v_["median_bout_T"])
    if len(Ds) >= 3:
        sl = np.polyfit(np.log(Ds), np.log(med), 1)[0]
        bout_scaling[f"{dp:.3f}/{ft_}"] = dict(exponent=float(sl), D=Ds, median_T=med)

# G2, G3 at delay pi/2
def g2_test(sn, ss):
    if sn["n"] < 3 or ss["n"] < 3:
        return None
    d = cdiff(sn["mean"], ss["mean"])
    return dict(diff=d, combined_se=sn["se"] + ss["se"], separated=bool(abs(d) > 3 * (sn["se"] + ss["se"])))
dpk = f"{np.pi / 2:.3f}"
G2 = g2_test(sig_det.get(f"{dpk}/node", dict(n=0)), sig_det.get(f"{dpk}/saddle-real", dict(n=0)))
G3 = None
kn, ks = f"{dpk}/node/joint D={NOISE_MAP}", f"{dpk}/saddle-real/joint D={NOISE_MAP}"
if G2 is not None and kn in sig_noise and ks in sig_noise:
    t_ = g2_test(sig_noise[kn], sig_noise[ks])
    if t_ is not None:
        G3 = dict(t_, same_sign=bool(np.sign(t_["diff"]) == np.sign(G2["diff"])))

# -----------------------------------------------------------------------------
# E. Lateral-line removal
# -----------------------------------------------------------------------------
if os.path.exists(ckpt("partE_ll.npz")):
    LL = dict(np.load(ckpt("partE_ll.npz")))
    print("E: loaded from checkpoint")
else:
    t0 = time.time()
    R_, E_, D_ = np.meshgrid(MAP_RATIOS, MAP_EPS, DPHI_HOLD, indexing="ij")
    R_, E_, D_ = R_.ravel(), E_.ravel(), D_.ravel()
    n = R_.size * N_IC
    S0 = random_ics(n)
    Pll = params(np.repeat(R_, N_IC), np.repeat(E_, N_IC), np.repeat(D_, N_IC), 1.0)
    o1 = batched(SIM_LL, S0, Pll, np.zeros((n, 3)), keys_for(n, 5), label="E with lateral line")
    Poff = Pll.copy(); Poff[:, 3] = 0.0
    o2 = batched(SIM_LL, o1[0], Poff, np.zeros((n, 3)), keys_for(n, 6), label="E lateral line off")
    LL = dict(held_on=o1[1], held_after=o1[1] & o2[1],
              amp_on=o1[5], amp_off=o2[5], ph_on=np.angle(o1[3] + 1j * o1[4]), ph_off=np.angle(o2[3] + 1j * o2[4]),
              rot_on=o1[6], rot_off=o2[6], dphi=np.repeat(D_, N_IC))
    np.savez_compressed(ckpt("partE_ll.npz"), **LL)
    print(f"E: done ({time.time() - t0:.0f} s)")
both = LL["held_after"].astype(bool)
ll_sum = dict(
    runs=int(LL["held_on"].size), holding_with_lateral_line=int(LL["held_on"].sum()),
    retained_after_removal=int(both.sum()),
    retention_fraction=float(both.sum() / max(LL["held_on"].sum(), 1)),
    tailbeat_amp_ratio_median=float(np.median(LL["amp_off"][both] / LL["amp_on"][both])) if both.any() else None,
    tailbeat_phase_shift_median_rad=float(np.median(np.abs(np.angle(np.exp(1j * (LL["ph_off"][both] - LL["ph_on"][both])))))) if both.any() else None,
    frequency_locked_before=float(np.mean(np.abs(LL["rot_on"][LL["held_on"].astype(bool)] - 1) < 0.01)) if LL["held_on"].any() else None,
    frequency_locked_after=float(np.mean(np.abs(LL["rot_off"][both] - 1) < 0.01)) if both.any() else None)

# -----------------------------------------------------------------------------
# F. Flow-speed path
# -----------------------------------------------------------------------------
def ref_ratio(dp, e_):
    m = (PTS[:, 7] > 0) & (np.abs(PTS[:, 3] - dp) < 1e-6) & (np.abs(PTS[:, 2] - e_) < 1e-6)
    return float(np.median(PTS[m, 4])) if m.any() else None

if os.path.exists(ckpt("partF_flow.npz")):
    FL = dict(np.load(ckpt("partF_flow.npz"), allow_pickle=True))
    print("F: loaded from checkpoint")
else:
    t0 = time.time()
    combos = [(dp, e_, ref_ratio(dp, e_)) for dp in DPHI_HOLD for e_ in FLOW_EPS]
    combos = [c_ for c_ in combos if c_[2] is not None]
    rows = [(ci, si, lv) for ci in range(len(combos)) for si in range(FLOW_S.size) for lv in (0, 1)]
    rows = np.array(rows, int).reshape(-1, 3)
    n = rows.shape[0] * N_IC
    ci = np.repeat(rows[:, 0], N_IC); si = np.repeat(rows[:, 1], N_IC); lv = np.repeat(rows[:, 2], N_IC)
    cmb = np.array([combos[c_] for c_ in ci]) if n else np.zeros((0, 3))
    P = params(cmb[:, 2], cmb[:, 1], cmb[:, 0], 0.0, 1.0, 1.0 / FLOW_S[si]) if n else np.zeros((0, 6))
    NZ = np.where(lv[:, None] == 1, noise_arr(n, NOISE_MAP), 0.0) if n else np.zeros((0, 3))
    o = batched(SIM_MAPN, random_ics(n), P, NZ, keys_for(n, 7), label="F flow path") if n else None
    FL = dict(combos=np.array(combos, dtype=float), ci=ci, si=si, lv=lv,
              alive=o[1] if n else np.zeros(0, bool), rot=o[6] if n else np.zeros(0))
    np.savez_compressed(ckpt("partF_flow.npz"), **FL)
    print(f"F: done ({time.time() - t0:.0f} s)")
flow = []
for c_, (dp, e_, r0) in enumerate(FL["combos"]):
    for lv in (0, 1):
        ph = [float(FL["alive"][(FL["ci"] == c_) & (FL["si"] == s_) & (FL["lv"] == lv)].mean()) for s_ in range(FLOW_S.size)]
        lk = []
        for s_ in range(FLOW_S.size):
            m = (FL["ci"] == c_) & (FL["si"] == s_) & (FL["lv"] == lv) & FL["alive"].astype(bool)
            lk.append(float(np.mean(np.abs(FL["rot"][m] - 1) < 0.01)) if m.any() else None)
        pk = max(ph) if ph else 0.0
        held = [FLOW_S[s_] for s_ in range(FLOW_S.size) if pk > 0 and ph[s_] >= 0.5 * pk]
        flow.append(dict(dphi=float(dp), eps=float(e_), ratio_ref=float(r0), noise=NOISE_MAP if lv else 0.0,
                         p_hold=ph, locked_fraction=lk,
                         p_hold_peak=float(pk), band_s=[float(min(held)), float(max(held))] if held else None))

# F2: fine flow-speed path near the reference speed (deterministic, more starts)
if os.path.exists(ckpt("partF2_flow_fine.npz")):
    F2 = dict(np.load(ckpt("partF2_flow_fine.npz"), allow_pickle=True))
    print("F2: loaded from checkpoint")
else:
    t0 = time.time()
    combos = [tuple(c_) for c_ in FL["combos"]]
    rows = np.array([(ci, si) for ci in range(len(combos)) for si in range(FLOW_S_FINE.size)], int).reshape(-1, 2)
    n = rows.shape[0] * N_IC_F2
    ci = np.repeat(rows[:, 0], N_IC_F2); si = np.repeat(rows[:, 1], N_IC_F2)
    cmb = np.array([combos[c_] for c_ in ci]) if n else np.zeros((0, 3))
    P = params(cmb[:, 2], cmb[:, 1], cmb[:, 0], 0.0, 1.0, 1.0 / FLOW_S_FINE[si]) if n else np.zeros((0, 6))
    o = batched(SIM_MAPN, random_ics(n), P, np.zeros((n, 3)), keys_for(n, 9), label="F2 fine flow path") if n else None
    F2 = dict(combos=np.array(combos, dtype=float), ci=ci, si=si,
              alive=o[1] if n else np.zeros(0, bool), rot=o[6] if n else np.zeros(0))
    np.savez_compressed(ckpt("partF2_flow_fine.npz"), **F2)
    print(f"F2: done ({time.time() - t0:.0f} s)")
flow_fine = []
for c_, (dp, e_, r0) in enumerate(F2["combos"]):
    ph = [float(F2["alive"][(F2["ci"] == c_) & (F2["si"] == s_)].mean()) for s_ in range(FLOW_S_FINE.size)]
    pk = max(ph) if ph else 0.0
    held = [FLOW_S_FINE[s_] for s_ in range(FLOW_S_FINE.size) if ph[s_] > 0]
    flow_fine.append(dict(dphi=float(dp), eps=float(e_), ratio_ref=float(r0), p_hold=ph, p_hold_peak=float(pk),
                          band_s_any_holding=[float(min(held)), float(max(held))] if held else None,
                          band_width_percent=float(100 * (max(held) - min(held))) if held else None))

# -----------------------------------------------------------------------------
# G. Trout mapping
# -----------------------------------------------------------------------------
lam, fs, Lb = TROUT["wavelength_m"], TROUT["f_shed_Hz"], TROUT["body_length_m"]
Tsh = 1.0 / fs
stable_x = PTS[PTS[:, 7] > 0, 5]
trout = dict(
    shedding_period_s=Tsh, vortex_convection_speed_m_s=lam * fs, free_stream_m_s=U_INF * lam * fs,
    free_stream_body_lengths_s=U_INF * lam * fs / Lb,
    body_natural_frequency_Hz=1.2 * fs, body_damping_ratio=ZE_B,
    holding_position_range_m=[float(np.min(stable_x) * lam), float(np.max(stable_x) * lam)],
    holding_position_range_body_lengths=[float(np.min(stable_x) * lam / Lb), float(np.max(stable_x) * lam / Lb)],
    holding_position_median_body_lengths=float(np.median(stable_x) * lam / Lb),
    noise_D_0p01_wake_phase_coherence_s=float(Tsh / 0.01),
    bout_median_s={k_: (v_["median_bout_T"] * Tsh if v_["median_bout_T"] is not None else None) for k_, v_ in bouts.items()},
    flow_band_m_s={f"dphi {f_['dphi']:.2f}, eps {f_['eps']}, noise {f_['noise']}":
                   ([f_["band_s"][0] * U_INF * lam * fs, f_["band_s"][1] * U_INF * lam * fs] if f_["band_s"] else None)
                   for f_ in flow},
    flow_band_fine_m_s={f"dphi {f_['dphi']:.2f}, eps {f_['eps']}":
                        ([f_["band_s_any_holding"][0] * U_INF * lam * fs, f_["band_s_any_holding"][1] * U_INF * lam * fs]
                         if f_["band_s_any_holding"] else None) for f_ in flow_fine})

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
tag = "" if MODE == "FULL" else "_quick"
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}{tag}.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"{name}{tag}.svg"))
    plt.close(fig)

def edges(v):
    d = np.diff(v).min() if v.size > 1 else 0.01
    return np.concatenate([v - d / 2, [v[-1] + d / 2]])

fig, axs = plt.subplots(1, len(DPHI_ALL), figsize=(3.0 * len(DPHI_ALL), 3.0), squeeze=False)
for c_, dp in enumerate(DPHI_ALL):
    ax = axs[0, c_]
    pc = ax.pcolormesh(edges(MAP_RATIOS), edges(MAP_EPS), PH_A[:, :, c_].T, cmap="Greens", vmin=0, vmax=1, rasterized=True)
    m = (PTS[:, 7] > 0) & (np.abs(PTS[:, 3] - dp) < 1e-6)
    ax.scatter(PTS[m, 4], PTS[m, 2], s=0.5, c="k", lw=0)
    ax.set_title(rf"$\Delta\phi$ = {dp:.2f}", fontsize=8); ax.set_xlabel(r"$\omega_c/\omega_w$")
    if c_ == 0:
        ax.set_ylabel(r"$\varepsilon$")
fig.colorbar(pc, ax=axs[0, -1], label="holding probability")
fig.suptitle("Direct simulation, no noise (black: stable period-1 holding from continuation)", fontsize=8)
save(fig, "fig01_holding_probability")

fig, axs = plt.subplots(1, len(DPHI_HOLD), figsize=(3.0 * len(DPHI_HOLD), 3.0), squeeze=False)
for c_, dp in enumerate(DPHI_HOLD):
    ax = axs[0, c_]
    pc = ax.pcolormesh(edges(MAP_RATIOS), edges(MAP_EPS), PH_B[:, :, c_].T, cmap="Greens", vmin=0, vmax=1, rasterized=True)
    ax.set_title(rf"$\Delta\phi$ = {dp:.2f}, noise D = {NOISE_MAP}", fontsize=8); ax.set_xlabel(r"$\omega_c/\omega_w$")
    if c_ == 0:
        ax.set_ylabel(r"$\varepsilon$")
fig.colorbar(pc, ax=axs[0, -1], label="holding probability")
save(fig, "fig02_holding_probability_noise")

fig, axs = plt.subplots(1, len(NOISE_CONDS), figsize=(2.4 * len(NOISE_CONDS), 2.6), squeeze=False, sharey=True)
COLG = {("saddle-real", 0): "#b2182b", ("saddle-real", 1): "#ef8a62", ("node", 0): "#2166ac", ("node", 1): "#67a9cf"}
for lv, (lab, _) in enumerate(NOISE_CONDS):
    ax = axs[0, lv]
    for g_, (dp, ft_) in enumerate(GROUPS):
        m = (CD["b_group"] == g_) & (CD["b_level"] == lv)
        if not m.any():
            continue
        tl = np.sort(np.where(CD["b_alive"][m], T_BOUT, CD["b_tloss"][m]))
        surv = 1 - np.arange(1, tl.size + 1) / tl.size
        ax.step(np.concatenate([[0], tl]), np.concatenate([[1], surv]), where="post",
                color=COLG[(ft_, DPHI_HOLD.index(dp))], lw=1, label=f"{ft_.split('-')[0]}, $\\Delta\\phi$={dp:.2f}")
    ax.set_xscale("symlog", linthresh=10); ax.set_title(lab, fontsize=7); ax.set_xlabel("time (shedding periods)")
    if lv == 0:
        ax.set_ylabel("still holding"); ax.legend(frameon=False, fontsize=6)
save(fig, "fig03_bout_survival")

fig, axs = plt.subplots(1, len(DPHI_HOLD), figsize=(3.2 * len(DPHI_HOLD), 2.8), squeeze=False)
bins = np.linspace(-np.pi, np.pi, 37)
for c_, dp in enumerate(DPHI_HOLD):
    ax = axs[0, c_]
    for ft_, col in (("saddle-real", "#b2182b"), ("node", "#2166ac"), ("none", "#999999")):
        m = (np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft_)
        if m.any():
            ax.hist(phase_det[m], bins=bins, color=col, alpha=0.6, label=ft_.split("-")[0])
    ax.set_xlabel("tail angle phase minus local wake phase (rad)")
    ax.set_title(rf"$\Delta\phi$ = {dp:.2f}", fontsize=8)
    if c_ == 0:
        ax.set_ylabel("stable holding points"); ax.legend(frameon=False, fontsize=7)
save(fig, "fig04_phase_signature")

if LL["held_on"].any():
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ho = LL["held_on"].astype(bool)
    ax[0].bar(["holding with\nlateral line", "still holding\nafter removal"], [ho.sum(), both.sum()], color=["#1a9850", "#999999"])
    ax[0].set_ylabel("runs")
    if both.any():
        ax[1].scatter(LL["amp_on"][both], LL["amp_off"][both], s=3, c="k")
        lim = [0, max(LL["amp_on"][both].max(), LL["amp_off"][both].max()) * 1.05]
        ax[1].plot(lim, lim, "k:", lw=0.6)
        ax[1].set_xlabel("tail-beat amplitude, lateral line on"); ax[1].set_ylabel("lateral line off")
    save(fig, "fig05_lateral_line_removal")

if flow:
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    Uf = FLOW_S * U_INF * lam * fs / Lb
    for f_ in flow:
        ls = "-" if f_["noise"] == 0 else ":"
        lab = rf"$\Delta\phi$={f_['dphi']:.2f}, $\varepsilon$={f_['eps']}" if f_["noise"] == 0 else None
        ax[0].plot(Uf, f_["p_hold"], ls, lw=1, label=lab)
        ax[1].plot(Uf, [np.nan if v is None else v for v in f_["locked_fraction"]], ls, lw=1)
    ax[0].set_xlabel("flow speed (body lengths per s, trout scaling)"); ax[0].set_ylabel("holding probability")
    ax[1].set_xlabel("flow speed (body lengths per s)"); ax[1].set_ylabel("tail beat locked to shedding")
    ax[0].legend(frameon=False, fontsize=6)
    save(fig, "fig06_flow_speed_path")

# -----------------------------------------------------------------------------
# Save and gates
# -----------------------------------------------------------------------------
area_cell = np.diff(MAP_RATIOS).min() * np.diff(MAP_EPS).min() if MAP_RATIOS.size > 1 and MAP_EPS.size > 1 else 0.0
map_sum = {}
for c_, dp in enumerate(DPHI_ALL):
    k_ = f"{dp:.3f}"
    map_sum[k_] = dict(area_p_hold_gt_0=float((PH_A[:, :, c_] > 0).sum() * area_cell),
                       area_p_hold_ge_0p1=float((PH_A[:, :, c_] >= 0.1).sum() * area_cell),
                       p_hold_max=float(PH_A[:, :, c_].max()),
                       holding_runs_period1=int(MA["p1"][:, :, c_].sum()), holding_runs_surging=int(MA["surge"][:, :, c_].sum()),
                       surge_amp_median=float(np.nanmedian(MA["surge_amp"][:, :, c_][MA["surge"][:, :, c_].astype(bool)]))
                       if MA["surge"][:, :, c_].any() else None)
    if dp in DPHI_HOLD:
        cb = DPHI_HOLD.index(dp)
        map_sum[k_]["noise_area_p_hold_gt_0"] = float((PH_B[:, :, cb] > 0).sum() * area_cell)
        map_sum[k_]["noise_area_p_hold_ge_0p1"] = float((PH_B[:, :, cb] >= 0.1).sum() * area_cell)
        map_sum[k_]["noise_p_hold_max"] = float(PH_B[:, :, cb].max())
summary = dict(
    mode=MODE, device=[d.platform for d in jax.devices()], runtime_this_session_s=round(time.time() - t_start, 1),
    noise_map_D=NOISE_MAP, noise_levels=NOISE_LEVELS, noise_conditions=[c_[0] for c_ in NOISE_CONDS],
    holding_maps=map_sum,
    gate_G1=dict(cells=g1_n, with_holding_in_map=g1_in, with_holding_after_recheck=g1_in_final,
                 fraction=g1_in_final / g1_n if g1_n else None, recheck=a2_sum),
    phase_signature_deterministic=sig_det, phase_signature_noise=sig_noise,
    gate_G2=G2, gate_G3=G3, bouts=bouts, bout_scaling=bout_scaling, lateral_line=ll_sum, flow_speed=flow,
    flow_speed_fine=flow_fine, trout=trout)
with open(os.path.join(OUT_DIR, f"summary_script11{tag}.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n================ SCRIPT 11 SUMMARY ================")
print("Holding maps (area in omega_c/omega_w x eps):")
for k_, v_ in map_sum.items():
    print(f"  dphi {k_}: P>0 {v_['area_p_hold_gt_0']:.4f}, P>=0.1 {v_['area_p_hold_ge_0p1']:.4f}, max P {v_['p_hold_max']:.2f}"
          + (f" | noise: P>0 {v_['noise_area_p_hold_gt_0']:.4f}, P>=0.1 {v_['noise_area_p_hold_ge_0p1']:.4f}, max P {v_['noise_p_hold_max']:.2f}"
             if "noise_area_p_hold_gt_0" in v_ else "")
          + f" | runs holding: period-1 or converging {v_['holding_runs_period1']}, surging {v_['holding_runs_surging']}"
          + (f" (median strobe span {v_['surge_amp_median']:.3f})" if v_["surge_amp_median"] is not None else ""))
print("\nTail-beat phase relative to local wake velocity (deterministic):")
for k_, v_ in sig_det.items():
    if v_["n"]:
        print(f"  {k_:22s} n {v_['n']:4d}, mean {v_['mean']:+.3f} rad, R {v_['R']:.3f}")
print("\nUnder noise:")
for k_, v_ in sig_noise.items():
    if v_["n"]:
        print(f"  {k_:30s} n {v_['n']:4d}, mean {v_['mean']:+.3f} rad, R {v_['R']:.3f}")
print("\nBouts (fraction still holding after T_BOUT, median bout in shedding periods):")
for k_, v_ in bouts.items():
    print(f"  {k_:30s} surviving {v_['surviving_fraction']:.2f}, median {v_['median_bout_T']}")
print("\nMedian bout vs joint noise level (power-law exponent):")
for k_, v_ in bout_scaling.items():
    print(f"  {k_:20s} exponent {v_['exponent']:+.2f}, medians {[round(m_, 1) for m_ in v_['median_T']]} at D {v_['D']}")
print("\nLateral line:", json.dumps(ll_sum, indent=2))
print("\nFlow-speed path (band of s = U/U_ref where holding probability is at least half its peak):")
for f_ in flow:
    print(f"  dphi {f_['dphi']:.2f}, eps {f_['eps']}, ratio_ref {f_['ratio_ref']:.3f}, noise {f_['noise']}: "
          f"peak P {f_['p_hold_peak']:.2f}, band {f_['band_s']}")
print("\nFine flow-speed path (any holding, s = U/U_ref):")
for f_ in flow_fine:
    print(f"  dphi {f_['dphi']:.2f}, eps {f_['eps']}: peak P {f_['p_hold_peak']:.2f}, band {f_['band_s_any_holding']}, "
          f"width {f_['band_width_percent']} %")
print("\nTrout mapping:", json.dumps({k_: v_ for k_, v_ in trout.items() if k_ not in ("bout_median_s", "flow_band_m_s")}, indent=2))
print("\n================ GATES ================")
g1f = summary["gate_G1"]["fraction"]
print(f"G1 Script 10 stable cells with holding in direct simulation: {g1_in}/{g1_n} in the map, "
      f"{g1_in_final}/{g1_n} after recheck {a2_sum} -> {'PASS' if g1f is not None and g1f >= 0.9 else 'FAIL'}")
print(f"G2 phase signature separates node and saddle at delay pi/2: {G2} -> "
      f"{'PASS' if G2 and G2['separated'] else 'FAIL or n/a'}")
print(f"G3 signature keeps its sign under noise D = {NOISE_MAP}: {G3} -> "
      f"{'PASS' if G3 and G3['same_sign'] else 'FAIL or n/a'}")
print(f"\nOutputs: {OUT_DIR}")
