# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 22: Sensing under noise, the phase signature across phase lags and
#            model variants, flow speed with a retuned rhythm, and passive
#            bodies under noise.
#
# Colab ready. Paste into one cell and run, with Google Drive already mounted.
# Uses JAX; runs on the GPU when one is attached (L4: roughly 20 to 40 min
# FULL), falls back to the CPU (much slower). Every part is checkpointed to
# Drive, so a disconnected run resumes where it stopped. No settings need to be
# changed. Needs on Drive the outputs of Scripts 02, 02b, 09, 10 and 14.
#
# Parts:
#   A. Sensing under noise. Stable orbits of three sensing configurations from
#      the same grid search (Script 09): proprioception only (baseline),
#      lateral line only, and both. Each is recovered here, then simulated
#      with wake phase, vortex amplitude and CPG phase noise at D = 0.003 and
#      0.01 (the lateral line senses the noisy wake). A fourth condition takes
#      the fish that uses both senses and removes its lateral line. Measures:
#      residence in the wake region (median time to loss, censored), time to
#      leave bands of +/- 0.1 and +/- 0.25 wavelengths, and loss fraction.
#   B. Phase signature of the branch. Phase of the tail beat relative to the
#      transverse wake velocity at the fish (as in Script 11) on deterministic
#      stable orbits of known branch:
#      B1  baseline at all 16 neural phase lags (Script 14 sweep, plus the
#          Script 10 orbits at phase lag pi/2); per lag mean phase of each
#          branch, separation where both branches exist, and discrimination
#          with one threshold pooled over phase lags
#      B2  model variants that have both branches (stiff heave, two-mode body,
#          plateau deficit wake), orbits recovered from Script 09 records and
#          classified by homotopy
#   C. Flow speed. Stable orbits are continued in s = U/U_ref from 0.7 to 1.3.
#      With the fish's internal rates fixed in physical time, the CPG
#      frequency, CPG relaxation rate, coupling and body frequency scale by
#      1/s in shedding units and the gain by 1/s^2 (Script 11). Schemes:
#        fixed               nothing retuned (Script 11)
#        rhythm retuned      CPG frequency follows the shedding frequency
#        rhythm and gain     CPG frequency and muscle gain retuned
#        all retuned         all rates follow the flow (self-similar control)
#      For each: the band of s over which the continued orbit stays stable,
#      and direct simulation on a coarse grid of s (holding and locking).
#   D. Passive bodies under noise.
#      D1  baseline wake: zeros of the period-averaged force at fixed position
#          over the body grid of Script 17 (no stable passive equilibrium)
#      D2  the Floquet-stable passive equilibria of the varied wakes (Scripts
#          02 and 02b) are reproduced and simulated with wake phase and
#          amplitude noise, and compared with active fish under the same noise
#      D3  fate of passive bodies released at rest along the baseline wake,
#          with and without noise, for four bodies
#
# Gates (reported):
#   G1  the model of this script reproduces Script 10 stable orbits
#   G2  Script 09 variant orbits are recovered (fraction per variant)
#   G3  the deterministic phase at phase lag pi/2 reproduces Script 11
#       (node about 1.72 rad, saddle about 1.21 rad)
#   G4  the all-retuned control holds over the whole range of s
#   G5  no restoring zero of the passive force in the baseline wake (Script 17)
#
# Outputs: script22_sensing_phase_speed_passive/summary_script22.json,
#          tables (CSV), checkpoints, draft figures.
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

MODE = os.environ.get("S22_MODE", "FULL")
SEED = 22
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Drive is expected to be mounted already, at /content/drive
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
    if not os.path.isdir(BASE_DIR):
        raise SystemExit(f"{BASE_DIR} not found: mount Google Drive first.")
else:
    BASE_DIR = os.path.abspath(os.environ.get("S22_BASE", "./karman_gait_project"))
OUT_DIR = os.path.join(BASE_DIR, "script22_sensing_phase_speed_passive")
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
    N_SEED_PH = 8
    T_RECOVER = 300
    N_REP_A, T_NOISE_A = 64, 1000
    NOISE_A = [0.003, 0.01]
    N_PER_BRANCH_PI2 = 100000                 # all classified Script 10 orbits at phase lag pi/2
    N_FLOW_PER_GROUP = 20
    S_STEPS = np.concatenate([np.arange(0.001, 0.0501, 0.001), np.arange(0.055, 0.3001, 0.005)])
    S_COARSE = np.round(np.arange(0.70, 1.3001, 0.025), 4)
    T_FLOW = 300
    T_FORCE = 40
    XF_GRID = np.round(np.linspace(0.16, 3.6, 87), 6)
    OMR_GRID = np.round(np.geomspace(0.3, 4.0, 22), 6)
    ZE_GRID = [0.05, 0.1, 0.3]
    N_REP_D, T_NOISE_D = 64, 1000
    NOISE_D = [0.003, 0.01]
    N_ACTIVE_D = 20
    X0_FATE = np.round(np.linspace(0.2, 3.0, 29), 4)
    N_REP_FATE, T_FATE = 32, 300
    CHUNK = 4096 if ON_GPU else 256
    NCHUNK = 1024 if ON_GPU else 64
else:
    N_SEED_PH = 4
    T_RECOVER = 80
    N_REP_A, T_NOISE_A = 3, 60
    NOISE_A = [0.01]
    N_PER_BRANCH_PI2 = 3
    N_FLOW_PER_GROUP = 1
    S_STEPS = np.array([0.002, 0.005, 0.01, 0.02])
    S_COARSE = np.array([0.95, 1.0, 1.05])
    T_FLOW = 60
    T_FORCE = 20
    XF_GRID = np.round(np.linspace(0.16, 3.6, 18), 6)
    OMR_GRID = np.array([0.5, 1.2])
    ZE_GRID = [0.1]
    N_REP_D, T_NOISE_D = 3, 60
    NOISE_D = [0.01]
    N_ACTIVE_D = 1
    X0_FATE = np.array([0.4, 1.0, 2.0])
    N_REP_FATE, T_FATE = 2, 60
    CHUNK = 256
    NCHUNK = 64
DT = 1.0 / SPP
X_COLL, X_ESC = 0.15, 4.5
TAU_A = 1.0
OMEGA_W = 2.0 * np.pi
TROUT = dict(wavelength_m=0.208, body_length_m=0.178, f_shed_Hz=2.74)
U_REF_BL = (1.0 / 0.87) * TROUT["wavelength_m"] * TROUT["f_shed_Hz"] / TROUT["body_length_m"]

# -----------------------------------------------------------------------------
# 1. Model in JAX: baseline of Scripts 10 and 11 with the structural options of
#    Script 09 used here (lateral line, heave, second body mode, plateau deficit).
#    State: 0 x, 1 u, 2 theta, 3 dtheta, 4 y, 5 dy, 6 theta2, 7 dtheta2, 8 z_r, 9 z_i
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA, LT = 3.0, 0.15, 8.0, 0.5
NS = 10
KEYS = ["ratio", "eps", "dphi", "g", "omr", "ze", "cT", "bnd", "frz", "hv", "omy", "zey", "wf", "df", "eta",
        "m2", "shape", "eLL", "wp", "mu", "rigid", "Uinf", "d0", "ld", "W0p", "xr", "lw", "cL", "cD", "sig"]
NPAR = len(KEYS)
BASE_PAR = dict(ratio=1.0, eps=0.0, dphi=0.0, g=16.0, omr=1.2, ze=0.2, cT=0.0, bnd=1.0, frz=0.0, hv=0.0, omy=2.0,
                zey=0.3, wf=1.0, df=1.0, eta=1.0, m2=0.0, shape=0.0, eLL=0.0, wp=1.0, mu=2 * np.pi, rigid=0.0,
                Uinf=1.0 / 0.87, d0=1.30, ld=0.90, W0p=0.35, xr=0.35, lw=1.50, cL=3.0, cD=0.15, sig=8.0)
OM2_RATIO, W2 = 2.2, 0.5

def par(n, **kw):
    b = dict(BASE_PAR); b.update(kw)
    return np.stack([np.broadcast_to(np.asarray(b[k], float), (n,)) for k in KEYS], 1)

def rhs_full(t, s, p, psi, aW):
    x, u, th, om, y, vy, th2, om2, zr, zi = s
    (ratio, eps, dphi, g, omr, ze, cT, bnd, frz, hv, omy, zey, wf, df, eta, m2, shape, eLL, wp, mu, rigid,
     Uinf, d0, ld, W0p, xr, lw, cL, cD, sig) = p
    # wake families of Scripts 02 and 09: shape 0 exponential, 1 plateau deficit, 2 algebraic, 3 bump envelope
    dfc = jnp.where(shape == 1, jnp.exp(-(x / ld) ** 2),
                    jnp.where(shape == 2, 1.0 / jnp.sqrt(1.0 + (x / ld) ** 2), jnp.exp(-x / ld)))
    rise = 1.0 - jnp.exp(-(x / xr) ** 2)
    env = jnp.where(shape == 3, W0p * jnp.exp(-((x - xr) / (0.5 * lw)) ** 2),
                    jnp.where(shape == 2, W0p * rise / (1.0 + x / lw), W0p * rise * jnp.exp(-x / lw)))
    V = Uinf * (1.0 - df * d0 * dfc) - u
    w = wf * (1.0 + aW) * env * jnp.sin(OMEGA_W * (x - t) + psi)
    wr = w - hv * vy
    th_e = th + m2 * W2 * th2
    om_e = om + m2 * W2 * om2
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    beta = jnp.arctan2(wr, jnp.abs(V))
    alpha = beta - th_e
    L = cL * (V * V + wr * wr) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    w_te = wr - jnp.abs(V) * th_e - LT * om_e
    Fx = cD * V * jnp.abs(V) - L * jnp.sin(beta) - cT * w_te * w_te * gate
    Om = omr * OMEGA_W
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    tq = g * drive
    dom = (-2 * ze * Om * om - Om ** 2 * th + sig * L + tq) * (1.0 - rigid)
    Om2 = OM2_RATIO * Om
    dom2 = m2 * (-2 * ze * Om2 * om2 - Om2 ** 2 * th2 + sig * L + tq)
    Omy = omy * OMEGA_W
    dvy = hv * (-2 * zey * Omy * vy - Omy ** 2 * y + L * jnp.cos(beta))
    r2 = zr * zr + zi * zi
    a = mu * (1.0 - r2)
    b = ratio * OMEGA_W
    inp = eps * (wp * th_e + eLL * w)
    ds = jnp.stack([eta * u * (1.0 - frz), eta * Fx * (1.0 - frz), om * (1.0 - rigid), dom, hv * vy, dvy,
                    m2 * om2, dom2, a * zr - b * zi + inp, a * zi + b * zr])
    c, sn_ = jnp.cos(OMEGA_W * t), jnp.sin(OMEGA_W * t)
    aux = jnp.stack([Fx, th_e * c, -th_e * sn_, w * c, -w * sn_])
    return ds, aux

def rk4(t, s, p, psi, aW):
    k1, a1 = rhs_full(t, s, p, psi, aW)
    k2, _ = rhs_full(t + DT / 2, s + DT / 2 * k1, p, psi, aW)
    k3, _ = rhs_full(t + DT / 2, s + DT / 2 * k2, p, psi, aW)
    k4, _ = rhs_full(t + DT, s + DT * k3, p, psi, aW)
    return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), a1

def pmap(s, p):
    def step(i, s_):
        return rk4(i * DT, s_, p, 0.0, 0.0)[0]
    return jax.lax.fori_loop(0, SPP, step, s)

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
    if S0.shape[0] == 0:
        return S0.copy(), np.zeros(0), np.zeros((0, NS, NS))
    fn = get_newton(n_it, clip)
    outs = []
    for i0 in range(0, S0.shape[0], NCHUNK):
        sl = slice(i0, min(S0.shape[0], i0 + NCHUNK))
        o = fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]), jnp.asarray(MASK[sl]))
        outs.append([np.asarray(v) for v in o])
    return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]),
            np.concatenate([o[2] for o in outs]))

def multipliers(J):
    if J.shape[0] == 0:
        return np.zeros(0), np.zeros(0, complex), np.zeros(0)
    mu = np.linalg.eigvals(J)
    a = np.abs(mu)
    lead = mu[np.arange(mu.shape[0]), np.argmax(a, 1)]
    dI = np.real(np.prod(1.0 - mu, axis=1))
    return a.max(1), lead, dI

def mask_of(PP, fixed=False):
    """Active dimensions from the parameters: position (unless fixed), pitch (unless rigid),
    heave, second mode, CPG (only when coupled to the body)."""
    n = PP.shape[0]
    M = np.zeros((n, NS))
    if not fixed:
        M[:, 0:2] = 1.0
    M[:, 2:4] = (PP[:, KEYS.index("rigid")] < 0.5)[:, None]
    M[:, 4:6] = (PP[:, KEYS.index("hv")] > 0.5)[:, None]
    M[:, 6:8] = (PP[:, KEYS.index("m2")] > 0.5)[:, None]
    M[:, 8:10] = ((PP[:, KEYS.index("g")] != 0) | (PP[:, KEYS.index("eps")] != 0))[:, None]
    return M

SIMS = {}
def get_sim(n_per, n_avg, noisy):
    key = (n_per, n_avg, noisy)
    if key in SIMS:
        return SIMS[key]
    def sim(s0, p, nz, k0):
        Dpsi, sW, Dz = nz[0], nz[1], nz[2]
        def one_period(carry, k):
            s, psi, aW, alive, tloss, acc, X, rot, nacc = carry
            kp = jax.random.fold_in(k0, k)
            def step(i, c):
                s_, psi_, aW_, al_, tl_, ac_, dph_ = c
                t = k + i * DT
                sn, aux = rk4(t, s_, p, psi_, aW_)
                if noisy:
                    nrm = jax.random.normal(jax.random.fold_in(kp, i), (3,))
                    ea = jnp.exp(-DT / TAU_A)
                    psi_n = psi_ + jnp.sqrt(2 * Dpsi * DT) * nrm[0]
                    aW_n = aW_ * ea + sW * jnp.sqrt(1 - ea * ea) * nrm[1]
                    ang = jnp.sqrt(2 * Dz * DT) * nrm[2]
                    zr2 = sn[8] * jnp.cos(ang) - sn[9] * jnp.sin(ang)
                    zi2 = sn[8] * jnp.sin(ang) + sn[9] * jnp.cos(ang)
                    sn = sn.at[8].set(zr2).at[9].set(zi2)
                else:
                    psi_n, aW_n = psi_, aW_
                lost = (((sn[0] < X_COLL) | (sn[0] > X_ESC)) & (p[7] > 0.5)) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + DT, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                dph = jnp.angle((sn[8] + 1j * sn[9]) * jnp.conj(s_[8] + 1j * s_[9]))
                return sn, psi_n, aW_n, al_n, tl_n, ac_ + aux * DT, dph_ + dph
            s2, psi2, aW2, al2, tl2, acp, dph = jax.lax.fori_loop(
                0, SPP, step, (s, psi, aW, alive, tloss, jnp.zeros(5), 0.0))
            use = (k >= n_per - n_avg) & al2
            Th = acp[1] + 1j * acp[2]; Wf = acp[3] + 1j * acp[4]
            acc = acc + jnp.where(use, acp, 0.0)
            X = X + jnp.where(use, Th * jnp.conj(Wf), 0j)
            rot = rot + jnp.where(use, dph / (2 * jnp.pi), 0.0)
            nacc = nacc + jnp.where(use, 1.0, 0.0)
            return (s2, psi2, aW2, al2, tl2, acc, X, rot, nacc), s2[0]
        c0 = (s0, 0.0, 0.0, True, jnp.nan, jnp.zeros(5), 0j, 0.0, 0.0)
        (s, psi, aW, alive, tloss, acc, X, rot, nacc), xs = jax.lax.scan(one_period, c0, jnp.arange(n_per))
        nn = jnp.maximum(nacc, 1.0)
        return s, alive, tloss, acc / nn, xs, jnp.angle(X), rot / nn, nacc
    fn = jax.jit(jax.vmap(sim))
    SIMS[key] = fn
    return fn

def run(n_per, S0, PP, n_avg=1, NZ=None, noisy=False, seed=0, label=None):
    n = S0.shape[0]
    if n == 0:
        return [np.zeros((0, NS)), np.zeros(0, bool), np.zeros(0), np.zeros((0, 5)), np.zeros((0, n_per)),
                np.zeros(0), np.zeros(0), np.zeros(0)]
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
    return [np.concatenate([o[j] for o in outs]) for j in range(8)]

def valid(S, x0=None, dx=0.5, cpg=True):
    ok = np.isfinite(S).all(1) & (S[:, 0] > X_COLL) & (S[:, 0] < X_ESC)
    if cpg:
        r = np.hypot(S[:, 8], S[:, 9])
        ok &= (r > 0.3) & (r < 2.5)
    if x0 is not None:
        ok &= np.abs(S[:, 0] - x0) < dx
    return ok

def to10(S6):
    S6 = np.atleast_2d(S6)
    S = np.zeros((S6.shape[0], NS))
    S[:, :4] = S6[:, :4]; S[:, 8:] = S6[:, 4:6]
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

def residence_metrics(xs, x0, tloss, T):
    """Median time to loss and to leave bands (Kaplan-Meier, censored at T), loss fraction."""
    lost = np.isfinite(tloss)
    out = dict(n=int(xs.shape[0]), lost_frac=float(lost.mean()) if lost.size else np.nan,
               residence_km_median=km_median(np.where(lost, tloss, np.inf), ~lost, T))
    dev = np.abs(xs.astype(float) - x0[:, None])
    for band in (0.1, 0.25):
        o_ = dev > band
        first = np.where(o_.any(1), o_.argmax(1) + 1.0, np.inf)
        first = np.where(lost, np.minimum(first, tloss), first)
        cens = ~np.isfinite(first)
        out[f"leave_{band}_km_median"] = km_median(first, cens, T)
        out[f"leave_{band}_censored"] = float(cens.mean())
    return out

def circ_mean(a):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(a))))) if np.size(a) else np.nan

def auc(a, b):
    """Probability that a value from a exceeds one from b (ties count one half)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    d = a[:, None] - b[None, :]
    return float(np.mean(d > 0) + 0.5 * np.mean(d == 0))

ETAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
def classify(S, PP):
    """Homotopy x' = eta u, u' = eta F_x to eta = 0.01, then the locked rhythm at fixed position."""
    n = S.shape[0]
    if n == 0:
        return np.zeros(0, dtype="<U16")
    s_h = S.copy(); alive = np.ones(n, bool)
    MK = mask_of(PP)
    for eta in ETAS[1:]:
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        Pe = PP[idx].copy(); Pe[:, KEYS.index("eta")] = eta
        Sn, Rn, _ = newton(s_h[idx], Pe, MK[idx], n_it=8, clip=0.05)
        ok = (Rn < 1e-7) & valid(Sn)
        s_h[idx[ok]] = Sn[ok]; alive[idx[~ok]] = False
    Pf = PP.copy(); Pf[:, KEYS.index("frz")] = 1.0
    S0 = s_h.copy(); S0[:, 1] = 0.0
    Sf, Rf, Jf = newton(S0, Pf, mask_of(Pf, fixed=True), n_it=15, clip=0.2)
    ok = alive & (Rf < 1e-7) & valid(Sf)
    amax, lead, _ = multipliers(Jf)
    return np.where(~ok, "none", np.where(amax < 1, "node",
                    np.where(np.abs(lead.imag) < 1e-6, "saddle-real", "unstable-complex")))

def recover(recs, over):
    """Recover stable orbits of a Script 09 variant from their records (x, ratio, eps, dphi) by
    relaxation from several CPG phases, then Newton; keep the one matching the recorded position."""
    nr = len(recs)
    if nr == 0:
        return np.zeros((0, NS)), np.zeros((0, NPAR)), np.zeros(0, int)
    ph = np.linspace(0, 2 * np.pi, N_SEED_PH, endpoint=False)
    ri = np.repeat(np.arange(nr), N_SEED_PH); pp = np.tile(ph, nr)
    S0 = np.zeros((ri.size, NS))
    S0[:, 0] = [recs[i]["x"] for i in ri]
    S0[:, 8], S0[:, 9] = np.cos(pp), np.sin(pp)
    PP = par(ri.size, ratio=[recs[i]["ratio"] for i in ri], eps=[recs[i]["eps"] for i in ri],
             dphi=[recs[i]["dphi"] for i in ri], **over)
    Sr, alive = run(T_RECOVER, S0, PP)[:2]
    Sn, Rn, Jn = newton(Sr, PP, mask_of(PP))
    amax = multipliers(Jn)[0]
    xr = np.array([recs[i]["x"] for i in ri])
    ok = alive & (Rn < 1e-7) & valid(Sn) & (np.abs(Sn[:, 0] - xr) < 0.01) & (amax < 1 - 1e-6)
    keepS, keepP, keepR = [], [], []
    for i in range(nr):
        j = np.where(ok & (ri == i))[0]
        if j.size:
            keepS.append(Sn[j[0]]); keepP.append(PP[j[0]]); keepR.append(i)
    return (np.array(keepS).reshape(-1, NS), np.array(keepP).reshape(-1, NPAR), np.array(keepR, int))

SUMMARY = dict(mode=MODE, spp=SPP, trout=dict(U_ref_body_lengths_per_s=U_REF_BL))
G = {}
t_start = time.time()

# -----------------------------------------------------------------------------
# 2. Inputs
# -----------------------------------------------------------------------------
print("\n=== Inputs ===")
S10_DIR = P("script10_continuation"); S10_CK = os.path.join(S10_DIR, "checkpoints_full")
PTS = np.load(os.path.join(S10_DIR, "points_script10.npz"))["points"]
BR = np.load(os.path.join(S10_CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
CL10 = dict(np.load(os.path.join(S10_CK, "stageD2_index.npz")))
SEL, FT = CL10["sel"].astype(int), CL10["ftype"].astype(str)
def s10_state(j):
    i = SEL[j]
    return to10(BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6])[0]
GROUPS = [("phase lag 0, saddle", 0.0, "saddle-real"), ("phase lag pi/2, saddle", np.pi / 2, "saddle-real"),
          ("phase lag pi/2, node", np.pi / 2, "node")]
R9 = json.load(open(P("script09_model_structure", "records_script09_ALL.json")))
def recs_of(v):
    return [r_ for r_ in R9 if r_["variant"] == v and r_.get("stable")]

print("\n=== G1. Model check ===")
pick = np.sort(rng.choice(SEL.size, min(64, SEL.size), replace=False))
S1 = np.array([s10_state(j) for j in pick])
P1 = par(pick.size, ratio=PTS[SEL[pick], 4], eps=PTS[SEL[pick], 2], dphi=PTS[SEL[pick], 3])
res = np.max(np.abs(run(1, S1, P1)[0] - S1), 1)
G["G1_max_residual"] = float(res.max()); G["G1_pass"] = bool(res.max() < 1e-6)
print(f"  one-period residual from {pick.size} stored orbits: max {res.max():.2e} ({'pass' if G['G1_pass'] else 'FAIL'})")

# =============================================================================
# A. Sensing under noise
# =============================================================================
print("\n=== A. Sensing configurations under noise ===")
SENSE = [("proprioception only", "baseline", dict()),
         ("lateral line only", "lateral line only", dict(eLL=1.0, wp=0.0)),
         ("both senses", "lateral line + proprio", dict(eLL=1.0))]
fA0 = ckpt("A0_orbits.npz")
if os.path.exists(fA0):
    dA0 = dict(np.load(fA0)); print("  loaded recovered orbits")
else:
    t0 = time.time()
    SS, PPs, tags, nrec = [], [], [], []
    for k, (name, var, over) in enumerate(SENSE):
        recs = recs_of(var)
        S_, P_, r_ = recover(recs, over)
        SS.append(S_); PPs.append(P_); tags.append(np.full(S_.shape[0], k)); nrec.append(len(recs))
        print(f"  {name}: recovered {S_.shape[0]} of {len(recs)} stable orbits")
    dA0 = dict(S=np.concatenate(SS), P=np.concatenate(PPs), tag=np.concatenate(tags), nrec=np.array(nrec))
    np.savez(fA0, **dA0)
    print(f"  recovery in {time.time() - t0:.0f} s")
G["G2_recovered_sensing"] = {SENSE[k][0]: [int((dA0["tag"] == k).sum()), int(dA0["nrec"][k])] for k in range(len(SENSE))}
fA = ckpt("A_noise.npz")
if os.path.exists(fA):
    dA = dict(np.load(fA)); print("  loaded noise runs")
else:
    t0 = time.time()
    S_all, P_all, NZ_all, tg = [], [], [], []
    conds = [(k, k) for k in range(len(SENSE))] + [(3, 2)]     # condition 3: both senses, lateral line removed
    for lv, D in enumerate(NOISE_A):
        for c_, k in conds:
            m = np.where(dA0["tag"] == k)[0]
            if m.size == 0:
                continue
            r_ = np.repeat(m, N_REP_A)
            P_ = dA0["P"][r_].copy()
            if c_ == 3:
                P_[:, KEYS.index("eLL")] = 0.0
            S_all.append(dA0["S"][r_]); P_all.append(P_)
            NZ_all.append(np.tile([D, np.sqrt(D), D], (r_.size, 1)))
            tg.append(np.column_stack([np.full(r_.size, lv), np.full(r_.size, c_), r_]))
    S_all, P_all, NZ_all, tg = map(np.concatenate, (S_all, P_all, NZ_all, tg))
    o = run(T_NOISE_A, S_all, P_all, NZ=NZ_all, noisy=True, seed=SEED, label="sensing noise")
    dA = dict(tags=tg, alive=o[1], tloss=o[2], x0=S_all[:, 0], xs=o[4].astype(np.float32))
    # deterministic lateral-line removal for comparison
    m = np.where(dA0["tag"] == 2)[0]
    if m.size:
        P_ = dA0["P"][m].copy(); P_[:, KEYS.index("eLL")] = 0.0
        od = run(400, dA0["S"][m], P_)
        dA["det_ll_removed_alive"] = od[1]
    np.savez(fA, **dA)
    print(f"  noise runs in {time.time() - t0:.0f} s")
COND_A = [s_[0] for s_ in SENSE] + ["both senses, lateral line removed"]
Ares = {}
rowsA = []
for lv, D in enumerate(NOISE_A):
    for c_ in range(len(COND_A)):
        m = (dA["tags"][:, 0] == lv) & (dA["tags"][:, 1] == c_)
        if not m.any():
            continue
        r = residence_metrics(dA["xs"][m], dA["x0"][m], dA["tloss"][m], T_NOISE_A)
        r["n_orbits"] = int(np.unique(dA["tags"][m, 2]).size)
        Ares[f"D={D} {COND_A[c_]}"] = r
        rowsA.append([D, COND_A[c_], r["n_orbits"], r["n"], r["residence_km_median"], r["leave_0.1_km_median"],
                      r["leave_0.25_km_median"], r["lost_frac"]])
        print(f"  D = {D}, {COND_A[c_]:>34s}: {r['n_orbits']} orbits, residence median {r['residence_km_median']}, "
              f"leave +/-0.1 {r['leave_0.1_km_median']}, +/-0.25 {r['leave_0.25_km_median']}, lost {r['lost_frac']:.2f}")
if "det_ll_removed_alive" in dA:
    Ares["deterministic_ll_removed_still_holding_frac"] = float(dA["det_ll_removed_alive"].mean())
    print(f"  deterministic: lateral line removed, still in the wake after 400 periods: "
          f"{dA['det_ll_removed_alive'].mean():.2f}")
save_csv("A_sensing_noise.csv", ["D", "condition", "orbits", "runs", "residence_median", "leave_0.1_median",
                                 "leave_0.25_median", "lost_frac"], rowsA)
SUMMARY["A_sensing"] = Ares

# =============================================================================
# B. Phase signature
# =============================================================================
print("\n=== B1. Phase signature across neural phase lags ===")
fB1 = ckpt("B1_phase_lags.npz")
if os.path.exists(fB1):
    dB1 = dict(np.load(fB1)); print("  loaded checkpoint")
else:
    t0 = time.time()
    SS, PPs, FTs, LAG = [], [], [], []
    S14 = P("script14_referee_analyses", "checkpoints_full")
    for li in range(16):
        f_ = os.path.join(S14, f"partI_lag{li:02d}.npz")
        if not os.path.exists(f_):
            continue
        d_ = dict(np.load(f_, allow_pickle=True))
        if "cls_idx" not in d_ or d_["cls_idx"].size == 0:
            continue
        ii = d_["cls_idx"].astype(int)
        S_ = to10(d_["S"][ii]); Pm = d_["P"][ii]
        SS.append(S_); PPs.append(par(ii.size, ratio=Pm[:, 0], eps=Pm[:, 1], dphi=Pm[:, 2]))
        FTs.append(d_["cls_ftype"].astype(str)); LAG.append(np.full(ii.size, li))
    # Script 10 orbits at phase lag pi/2 (larger sample of both branches)
    for ft in ("saddle-real", "node"):
        idx = np.where((np.abs(PTS[SEL, 3] - np.pi / 2) < 1e-6) & (FT == ft))[0]
        pk = np.sort(rng.choice(idx, min(N_PER_BRANCH_PI2, idx.size), replace=False))
        SS.append(np.array([s10_state(j) for j in pk]))
        PPs.append(par(pk.size, ratio=PTS[SEL[pk], 4], eps=PTS[SEL[pk], 2], dphi=PTS[SEL[pk], 3]))
        FTs.append(np.full(pk.size, ft)); LAG.append(np.full(pk.size, 4))
    S_, P_ = np.concatenate(SS), np.concatenate(PPs)
    FT_, LAG_ = np.concatenate(FTs), np.concatenate(LAG)
    o = run(1, S_, P_)
    dB1 = dict(phase=o[5], amp=np.hypot(o[3][:, 1], o[3][:, 2]) * 2, ftype=FT_, lag=LAG_,
               src=np.concatenate([np.zeros(sum(len(f) for f in FTs[:-2])), np.ones(len(FTs[-2]) + len(FTs[-1]))]))
    np.savez(fB1, **dB1)
    print(f"  done in {time.time() - t0:.0f} s")
ph, ftb, lagb = dB1["phase"], dB1["ftype"].astype(str), dB1["lag"].astype(int)
B1 = dict(per_lag={})
rowsB = []
for li in range(16):
    rec = {}
    for ft in ("saddle-real", "node"):
        m = (lagb == li) & (ftb == ft)
        if m.any():
            rec[ft] = dict(n=int(m.sum()), mean=circ_mean(ph[m]), sd=float(np.std(np.angle(np.exp(1j * (ph[m] - circ_mean(ph[m])))))))
    ms, mn = (lagb == li) & (ftb == "saddle-real"), (lagb == li) & (ftb == "node")
    if ms.any() and mn.any():
        rec["node_minus_saddle"] = float(np.angle(np.exp(1j * (circ_mean(ph[mn]) - circ_mean(ph[ms])))))
        rec["auc_node_gt_saddle"] = auc(ph[mn], ph[ms])
    if rec:
        B1["per_lag"][f"{li}pi/8"] = rec
        rowsB.append([li, rec.get("saddle-real", {}).get("n", 0), rec.get("saddle-real", {}).get("mean", np.nan),
                      rec.get("node", {}).get("n", 0), rec.get("node", {}).get("mean", np.nan),
                      rec.get("node_minus_saddle", np.nan), rec.get("auc_node_gt_saddle", np.nan)])
m4s = (lagb == 4) & (ftb == "saddle-real") & (dB1["src"] > 0.5)
m4n = (lagb == 4) & (ftb == "node") & (dB1["src"] > 0.5)
med_s, med_n = circ_mean(ph[m4s]), circ_mean(ph[m4n])
G["G3_phase_pi2_saddle_node"] = [med_s, med_n]
G["G3_pass"] = bool(abs(med_s - 1.207) < 0.02 and abs(med_n - 1.715) < 0.02)
print(f"  phase lag pi/2 (Script 10 orbits): saddle mean {med_s:.3f}, node mean {med_n:.3f} rad "
      f"({'pass' if G['G3_pass'] else 'FAIL'}: Script 11 gives 1.207 and 1.715)")
# pooled over all lags with both branches: one threshold
lags_both = [li for li in range(16) if ((lagb == li) & (ftb == "node")).any() and ((lagb == li) & (ftb == "saddle-real")).any()]
mpool = np.isin(lagb, lags_both)
B1["lags_with_both_branches"] = lags_both
B1["pooled_auc_node_gt_saddle"] = auc(ph[mpool & (ftb == "node")], ph[mpool & (ftb == "saddle-real")])
B1["all_lags_saddle_phase_range"] = [float(np.nanmin([r[2] for r in rowsB])), float(np.nanmax([r[2] for r in rowsB]))]
B1["all_lags_auc_node_vs_all_saddle"] = auc(ph[ftb == "node"], ph[ftb == "saddle-real"])
for r in rowsB:
    print(f"  phase lag {r[0]:2d} pi/8: saddle n {r[1]:3d} mean {r[2]:+.2f}; node n {r[3]:3d} mean {r[4]:+.2f}; "
          f"difference {r[5]:+.2f}, AUC {r[6]:.3f}")
print(f"  pooled over phase lags with both branches {lags_both}: AUC {B1['pooled_auc_node_gt_saddle']:.3f}; "
      f"node orbits against saddle orbits of all phase lags: AUC {B1['all_lags_auc_node_vs_all_saddle']:.3f}")
save_csv("B1_phase_by_lag.csv", ["lag_pi_over_8", "n_saddle", "saddle_mean_phase", "n_node", "node_mean_phase",
                                 "node_minus_saddle", "auc"], rowsB)
SUMMARY["B1_phase_lags"] = B1

print("\n=== B2. Phase signature in model variants ===")
VARS_B2 = [("pitch and stiff heave", "pitch + stiff heave", dict(hv=1.0, omy=2.0, zey=0.3)),
           ("two-mode body", "two-mode body", dict(m2=1.0)),
           ("plateau deficit wake", "plateau deficit", dict(shape=1.0)),
           ("baseline", "baseline", dict())]
fB2 = ckpt("B2_variants.npz")
if os.path.exists(fB2):
    dB2 = dict(np.load(fB2)); print("  loaded checkpoint")
else:
    t0 = time.time()
    out = dict(phase=[], ftype=[], ftype09=[], var=[], nrec=[], nrecov=[])
    for k, (name, v, over) in enumerate(VARS_B2):
        recs = recs_of(v)
        S_, P_, r_ = recover(recs, over)
        ft_ = classify(S_, P_)
        phs = run(1, S_, P_)[5]
        out["phase"].append(phs); out["ftype"].append(ft_)
        out["ftype09"].append(np.array([recs[i]["fixed_type"] for i in r_], dtype="<U16"))
        out["var"].append(np.full(S_.shape[0], k)); out["nrec"].append(len(recs)); out["nrecov"].append(S_.shape[0])
        print(f"  {name}: recovered {S_.shape[0]} of {len(recs)}")
    dB2 = dict(phase=np.concatenate(out["phase"]), ftype=np.concatenate(out["ftype"]),
               ftype09=np.concatenate(out["ftype09"]), var=np.concatenate(out["var"]),
               nrec=np.array(out["nrec"]), nrecov=np.array(out["nrecov"]))
    np.savez(fB2, **dB2)
    print(f"  done in {time.time() - t0:.0f} s")
B2 = {}
for k, (name, v, over) in enumerate(VARS_B2):
    m = dB2["var"] == k
    ft_ = dB2["ftype"][m].astype(str)
    rec = dict(recovered=int(dB2["nrecov"][k]), of=int(dB2["nrec"][k]),
               agree_with_script09=float(np.mean(ft_ == dB2["ftype09"][m].astype(str))) if m.any() else np.nan)
    for ft in ("saddle-real", "node"):
        mm = ft_ == ft
        if mm.any():
            rec[ft] = dict(n=int(mm.sum()), mean=circ_mean(dB2["phase"][m][mm]), values=dB2["phase"][m][mm])
    if "saddle-real" in rec and "node" in rec:
        rec["node_minus_saddle"] = float(np.angle(np.exp(1j * (rec["node"]["mean"] - rec["saddle-real"]["mean"]))))
        rec["auc_node_gt_saddle"] = auc(rec["node"]["values"], rec["saddle-real"]["values"])
    B2[name] = rec
    print(f"  {name}: recovered {rec['recovered']} of {rec['of']}, branch agreement {rec['agree_with_script09']:.2f}; "
          + "; ".join(f"{ft} n {rec[ft]['n']} mean {rec[ft]['mean']:+.2f}" for ft in ("saddle-real", "node") if ft in rec)
          + (f"; difference {rec['node_minus_saddle']:+.2f}, AUC {rec['auc_node_gt_saddle']:.2f}" if "node_minus_saddle" in rec else ""))
G["G2_recovered_variants"] = {k: [v["recovered"], v["of"]] for k, v in B2.items()}
SUMMARY["B2_phase_variants"] = B2

# =============================================================================
# C. Flow speed
# =============================================================================
print("\n=== C. Flow speed with fixed and retuned rhythms ===")
SCHEMES = [("fixed", dict(c=1, b=1, m=1, e=1, g=1)), ("rhythm retuned", dict(c=0, b=1, m=1, e=1, g=1)),
           ("rhythm and gain retuned", dict(c=0, b=1, m=1, e=1, g=0)), ("all retuned", dict(c=0, b=0, m=0, e=0, g=0))]
fl_rows = []
for gi, (gname, dp, ft) in enumerate(GROUPS):
    idx = np.where((np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft))[0]
    for j in np.sort(rng.choice(idx, min(N_FLOW_PER_GROUP, idx.size), replace=False)):
        fl_rows.append((gi, j))
NF = len(fl_rows)
FS0 = np.array([s10_state(j) for _, j in fl_rows])
FG = np.array([g_ for g_, _ in fl_rows])
FR, FE, FD = PTS[SEL[[j for _, j in fl_rows]], 4], PTS[SEL[[j for _, j in fl_rows]], 2], PTS[SEL[[j for _, j in fl_rows]], 3]
def flow_par(idx, s, sc):
    ts = 1.0 / s
    f = lambda flag: ts if flag else 1.0
    return par(idx.size, ratio=FR[idx] * f(sc["c"]), eps=FE[idx] * f(sc["e"]), dphi=FD[idx],
               omr=1.2 * f(sc["b"]), mu=2 * np.pi * f(sc["m"]), g=16.0 * f(sc["g"]) ** 2)
fC = ckpt("C_flow.npz")
if os.path.exists(fC):
    dC = dict(np.load(fC)); print("  loaded checkpoint")
else:
    t0 = time.time()
    band = np.zeros((len(SCHEMES), NF, 2))
    for k, (sname, sc) in enumerate(SCHEMES):
        for d_, sgn in enumerate((-1.0, 1.0)):
            S = FS0.copy(); alive = np.ones(NF, bool); last = np.zeros(NF)
            for ds in S_STEPS:
                idx = np.where(alive)[0]
                if idx.size == 0:
                    break
                s_val = 1.0 + sgn * ds
                PPk = np.concatenate([flow_par(np.array([i]), s_val, sc) for i in idx])
                Sn, Rn, Jn = newton(S[idx], PPk, mask_of(PPk), n_it=10)
                st = (Rn < 1e-7) & valid(Sn) & (multipliers(Jn)[0] < 1 - 1e-6)
                S[idx[st]] = Sn[st]; last[idx[st]] = ds; alive[idx[~st]] = False
            band[k, :, d_] = last
        print(f"  {sname}: continuation done ({time.time() - t0:.0f} s)")
    # direct simulation on a coarse grid of s from the reference orbit
    hold = np.zeros((len(SCHEMES), NF, S_COARSE.size), bool); lock = np.zeros_like(hold)
    for k, (sname, sc) in enumerate(SCHEMES):
        ii = np.repeat(np.arange(NF), S_COARSE.size); ss = np.tile(S_COARSE, NF)
        PPk = np.concatenate([flow_par(np.array([i]), s_, sc) for i, s_ in zip(ii, ss)])
        o = run(T_FLOW, FS0[ii], PPk, n_avg=T_FLOW // 3)
        hold[k] = o[1].reshape(NF, -1)
        cpg_ratio = PPk[:, KEYS.index("ratio")]
        lock[k] = (o[1] & (np.abs(o[6] - 1.0) < 0.01)).reshape(NF, -1)
    dC = dict(band=band, hold=hold, lock=lock)
    np.savez(fC, **dC)
    print(f"  done in {time.time() - t0:.0f} s")
Cres = {}
rowsC = []
for k, (sname, sc) in enumerate(SCHEMES):
    width = dC["band"][k].sum(1)
    rec = dict(band_lower=dist(dC["band"][k, :, 0]), band_upper=dist(dC["band"][k, :, 1]),
               band_width_percent=dist(100 * width),
               hold_fraction_by_s={f"{s_:.3f}": float(dC["hold"][k, :, i].mean()) for i, s_ in enumerate(S_COARSE)},
               lock_fraction_by_s={f"{s_:.3f}": float(dC["lock"][k, :, i].mean()) for i, s_ in enumerate(S_COARSE)})
    for gi, (gname, _, _) in enumerate(GROUPS):
        rec[f"band_width_percent {gname}"] = dist(100 * width[FG == gi])
    Cres[sname] = rec
    rowsC.append([sname, np.median(100 * width), np.min(100 * width), np.max(100 * width),
                  np.median(dC["band"][k, :, 0]), np.median(dC["band"][k, :, 1])])
    print(f"  {sname:>24s}: stable band width median {np.median(100 * width):.2f}% (range {np.min(100 * width):.2f} "
          f"to {np.max(100 * width):.2f}%); locked holding by s: "
          + ", ".join(f"{s_:.3f}:{dC['lock'][k, :, i].mean():.2f}" for i, s_ in enumerate(S_COARSE)))
G["G4_all_retuned_full_band"] = bool(np.all(dC["band"][3] >= S_STEPS[-1] - 1e-9))
G["G4_pass"] = G["G4_all_retuned_full_band"]
Cres["s_range_tested"] = [1 - float(S_STEPS[-1]), 1 + float(S_STEPS[-1])]
Cres["U_ref_body_lengths_per_s"] = U_REF_BL
save_csv("C_flow_speed.csv", ["scheme", "band_width_median_percent", "band_width_min", "band_width_max",
                              "lower_extent_median", "upper_extent_median"], rowsC)
SUMMARY["C_flow_speed"] = Cres

# =============================================================================
# D. Passive bodies under noise
# =============================================================================
print("\n=== D. Passive bodies under noise ===")
# D1. Baseline wake: period-averaged force at fixed position for every body (as Script 17 A1)
BODIES = [(float(r), float(z), 0.0) for z in ZE_GRID for r in OMR_GRID] + [(1.0, 0.1, 1.0)]
NB = len(BODIES)
BA = np.array(BODIES)
fD0 = ckpt("D0_baseline_force.npz")
if os.path.exists(fD0):
    dD0 = dict(np.load(fD0)); print("  loaded baseline force curves")
else:
    t0 = time.time()
    bb = np.repeat(np.arange(NB), XF_GRID.size); xx = np.tile(XF_GRID, NB)
    S0 = np.zeros((bb.size, NS)); S0[:, 0] = xx
    PPf = par(bb.size, g=0.0, eps=0.0, omr=BA[bb, 0], ze=BA[bb, 1], rigid=BA[bb, 2], frz=1.0)
    Sr = run(T_FORCE, S0, PPf)[0]
    F = run(10, Sr, PPf, n_avg=10)[3][:, 0].reshape(NB, -1)
    dD0 = dict(F=F)
    np.savez(fD0, **dD0)
    print(f"  baseline force curves in {time.time() - t0:.0f} s")
F = dD0["F"]
n_restoring = sum(int(np.sum((F[b, :-1] > 0) & (F[b, 1:] <= 0))) for b in range(NB))
n_antirest = sum(int(np.sum((F[b, :-1] < 0) & (F[b, 1:] >= 0))) for b in range(NB))
G["G5_baseline_restoring_zero_crossings"] = n_restoring
G["G5_pass"] = bool(n_restoring == 0)
print(f"  baseline wake: {n_restoring} restoring and {n_antirest} anti-restoring zeros of the force over {NB} bodies "
      f"({'pass' if G['G5_pass'] else 'FAIL'}: Script 17 finds no stable passive equilibrium in this wake)")

# D2. Stable passive equilibria of the varied wakes (Scripts 02 and 02b, Floquet confirmed)
f2 = P("script02_passive_robustness", "results_script02.npz"); f2b = P("script02b_passive_cases", "results_script02b.npz")
fD1 = ckpt("D1_passive_equilibria.npz")
if os.path.exists(fD1):
    dD1 = dict(np.load(fD1)); print("  loaded passive equilibria")
elif os.path.exists(f2) and os.path.exists(f2b):
    t0 = time.time()
    d2 = np.load(f2, allow_pickle=True); b2 = np.load(f2b)
    ok = b2["conv"].astype(bool) & (b2["cls_f"] == 0)
    chk = b2["chk"][ok]; so = b2["s_orb"][:, ok]
    st = d2["lhs_eq_set"][chk]
    pk = lambda k: d2[k][st].astype(float)
    PPp = par(chk.size, g=0.0, eps=0.0, omr=pk("Plhs_Om") / OMEGA_W, ze=pk("Plhs_ze"), rigid=pk("Plhs_rigid"),
              shape=pk("Plhs_shape"), Uinf=pk("Plhs_U_inf"), d0=pk("Plhs_delta0"), ld=pk("Plhs_l_delta"),
              W0p=pk("Plhs_W0"), xr=pk("Plhs_x_r"), lw=pk("Plhs_l_w"), cL=pk("Plhs_c_L"), cD=pk("Plhs_c_D"),
              sig=pk("Plhs_sigma"))
    S0 = np.zeros((chk.size, NS)); S0[:, :4] = so.T
    Sn, Rn, Jn = newton(S0, PPp, mask_of(PPp))
    amax = multipliers(Jn)[0]
    okp = (Rn < 1e-7) & valid(Sn, S0[:, 0], 0.05, cpg=False) & (amax < 1 - 1e-6)
    xpk = d2["lhs_xpeak"][chk]
    dD1 = dict(S=Sn[okp], P=PPp[okp], xpeak=xpk[okp], n_input=chk.size)
    np.savez(fD1, **dD1)
    print(f"  {int(okp.sum())} of {chk.size} Floquet-stable passive equilibria reproduced ({time.time() - t0:.0f} s)")
else:
    dD1 = dict(S=np.zeros((0, NS)), P=np.zeros((0, NPAR)), xpeak=np.zeros(0), n_input=0)
    print("  Script 02 outputs not found; passive equilibria skipped")
fD = ckpt("D_noise.npz")
if os.path.exists(fD):
    dD = dict(np.load(fD)); print("  loaded noise runs")
else:
    t0 = time.time()
    act = []
    for gi, (gname, dp, ft) in enumerate(GROUPS):
        idx = np.where((np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft))[0]
        act += [(gi, j) for j in np.sort(rng.choice(idx, min(N_ACTIVE_D, idx.size), replace=False))]
    SA = np.array([s10_state(j) for _, j in act]).reshape(-1, NS)
    PA_ = par(len(act), ratio=PTS[SEL[[j for _, j in act]], 4], eps=PTS[SEL[[j for _, j in act]], 2],
              dphi=PTS[SEL[[j for _, j in act]], 3])
    S_all, P_all, NZ_all, tg = [], [], [], []
    for lv, D in enumerate(NOISE_D):
        for kind, S_, P_ in ((0, dD1["S"], dD1["P"]), (1, SA, PA_)):
            if S_.shape[0] == 0:
                continue
            r_ = np.repeat(np.arange(S_.shape[0]), N_REP_D)
            S_all.append(S_[r_]); P_all.append(P_[r_])
            NZ_all.append(np.tile([D, np.sqrt(D), 0.0], (r_.size, 1)))      # wake phase and amplitude noise only
            tg.append(np.column_stack([np.full(r_.size, lv), np.full(r_.size, kind), r_]))
    S_all, P_all, NZ_all, tg = map(np.concatenate, (S_all, P_all, NZ_all, tg))
    o = run(T_NOISE_D, S_all, P_all, NZ=NZ_all, noisy=True, seed=SEED + 1, label="passive and active noise")
    dD = dict(tags=tg, alive=o[1], tloss=o[2], x0=S_all[:, 0], xs=o[4].astype(np.float32), xend=o[0][:, 0])
    # D3. Fate of passive bodies released at rest in the baseline wake, with and without noise
    FB = [(0.5, 0.1, 0.0), (0.93, 0.1, 0.0), (3.09, 0.1, 0.0), (1.0, 0.1, 1.0)]
    fb = np.repeat(np.arange(len(FB)), X0_FATE.size * N_REP_FATE)
    fx = np.tile(np.repeat(X0_FATE, N_REP_FATE), len(FB))
    S0 = np.zeros((fb.size, NS)); S0[:, 0] = fx
    FBa = np.array(FB)
    PPb = par(fb.size, g=0.0, eps=0.0, omr=FBa[fb, 0], ze=FBa[fb, 1], rigid=FBa[fb, 2])
    of = run(T_FATE, S0, PPb, NZ=np.tile([0.01, 0.1, 0.0], (fb.size, 1)), noisy=True, seed=SEED + 2, label="noisy fates")
    od = run(T_FATE, S0[::N_REP_FATE], PPb[::N_REP_FATE])
    dD.update(fate_body=fb, fate_x0=fx, fate_alive=of[1], fate_xend=of[0][:, 0], fate_tloss=of[2],
              det_alive=od[1], det_xend=od[0][:, 0], det_tloss=od[2], fate_bodies=FBa)
    np.savez(fD, **dD)
    print(f"  done in {time.time() - t0:.0f} s")
Dres = dict(baseline_restoring_zeros=n_restoring, baseline_antirestoring_zeros=n_antirest,
            varied_wakes_stable_equilibria=int(dD1["S"].shape[0]), of_floquet_stable=int(dD1["n_input"]),
            equilibria_x=dist(dD1["S"][:, 0]) if dD1["S"].shape[0] else dict(n=0),
            equilibria_x_over_peak=dist(dD1["S"][:, 0] / dD1["xpeak"]) if dD1["S"].shape[0] else dict(n=0))
rowsD = []
for lv, D in enumerate(NOISE_D):
    for kind, kname in ((0, "passive equilibria"), (1, "active orbits")):
        m = (dD["tags"][:, 0] == lv) & (dD["tags"][:, 1] == kind)
        if not m.any():
            continue
        r = residence_metrics(dD["xs"][m], dD["x0"][m], dD["tloss"][m], T_NOISE_D)
        lost = np.isfinite(dD["tloss"][m])
        r["lost_to_cylinder_frac"] = float(np.mean(lost & (dD["xend"][m] < 1.0)))
        r["lost_downstream_frac"] = float(np.mean(lost & (dD["xend"][m] >= 1.0)))
        Dres[f"D={D} {kname}"] = r
        rowsD.append([D, kname, r["n"], r["residence_km_median"], r["leave_0.1_km_median"], r["leave_0.25_km_median"],
                      r["lost_frac"], r["lost_to_cylinder_frac"], r["lost_downstream_frac"]])
        print(f"  D = {D}, {kname:>18s}: residence median {r['residence_km_median']}, leave +/-0.1 "
              f"{r['leave_0.1_km_median']}, +/-0.25 {r['leave_0.25_km_median']}, lost {r['lost_frac']:.2f} "
              f"(cylinder {r['lost_to_cylinder_frac']:.2f}, downstream {r['lost_downstream_frac']:.2f})")
save_csv("D_passive_noise.csv", ["D", "set", "runs", "residence_median", "leave_0.1_median", "leave_0.25_median",
                                 "lost_frac", "lost_to_cylinder", "lost_downstream"], rowsD)
FBa = dD["fate_bodies"]
fates = {}
rowsF = []
for b in range(FBa.shape[0]):
    m = dD["fate_body"] == b
    al, xe = dD["fate_alive"][m], dD["fate_xend"][m]
    up = ~al & (xe < 1.0); down = ~al & (xe >= 1.0)
    md = np.repeat(np.arange(FBa.shape[0]), X0_FATE.size) == b
    dal, dxe = dD["det_alive"][md], dD["det_xend"][md]
    # release position that separates upstream and downstream fates
    x0n = dD["fate_x0"][m]
    pup = np.array([np.mean(up[x0n == x_]) for x_ in X0_FATE])
    sep_noisy = np.nan
    for j in range(X0_FATE.size - 1):
        if pup[j] >= 0.5 > pup[j + 1]:
            sep_noisy = float(X0_FATE[j] + (pup[j] - 0.5) * (X0_FATE[j + 1] - X0_FATE[j]) / (pup[j] - pup[j + 1]))
            break
    dup = (~dal & (dxe < 1.0)).astype(float)
    sep_det = float(X0_FATE[np.argmax(dup < 0.5)]) if dup.any() and not dup.all() else np.nan
    name = "rigid" if FBa[b, 2] > 0.5 else f"Omega/omega_w {FBa[b, 0]}"
    fates[name] = dict(noisy=dict(upstream=float(up.mean()), downstream=float(down.mean()), stays=float(al.mean()),
                                  median_time_to_fate=float(np.nanmedian(dD["fate_tloss"][m])),
                                  release_x_with_half_upstream=sep_noisy),
                       deterministic=dict(upstream=float(dup.mean()), downstream=float(np.mean(~dal & (dxe >= 1.0))),
                                          stays=float(dal.mean()), first_release_x_downstream=sep_det,
                                          median_time_to_fate=float(np.nanmedian(dD["det_tloss"][md]))))
    rowsF.append([name, up.mean(), down.mean(), al.mean(), sep_noisy, fates[name]["deterministic"]["upstream"],
                  fates[name]["deterministic"]["downstream"], fates[name]["deterministic"]["stays"], sep_det])
    print(f"  released at rest, {name}: with noise upstream {up.mean():.2f}, downstream {down.mean():.2f}, "
          f"stays {al.mean():.2f}, half upstream from x0 = {sep_noisy:.2f}; without noise upstream "
          f"{fates[name]['deterministic']['upstream']:.2f}, downstream from x0 = {sep_det:.2f}")
Dres["release_fates"] = fates
save_csv("D_release_fates.csv", ["body", "noisy_upstream", "noisy_downstream", "noisy_stays", "noisy_x0_half_upstream",
                                 "det_upstream", "det_downstream", "det_stays", "det_first_x0_downstream"], rowsF)
SUMMARY["D_passive"] = Dres
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
    for ft, c_ in (("saddle-real", "#1f77b4"), ("node", "#d62728")):
        m = ftb == ft
        ax[0].plot(lagb[m] + rng.uniform(-0.15, 0.15, m.sum()), ph[m], ".", color=c_, ms=4, label=ft)
    ax[0].set_xlabel(r"neural phase lag ($\pi/8$)"); ax[0].set_ylabel("tail-beat phase (rad)")
    ax[0].legend(framealpha=1, edgecolor="k")
    for k, (sname, _) in enumerate(SCHEMES):
        ax[1].plot(S_COARSE, dC["lock"][k].mean(0), "o-", label=sname)
    ax[1].set_xlabel(r"flow speed $s=U/U_{\mathrm{ref}}$"); ax[1].set_ylabel("fraction locked and holding")
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=9, framealpha=1, edgecolor="k")
    save(fig, "S22_phase_and_flow")
except Exception as e:
    print(f"  figure skipped: {e}")

SUMMARY["runtime_s"] = time.time() - t_start
with open(os.path.join(OUT_DIR, "summary_script22.json"), "w") as f:
    json.dump(to_py(SUMMARY), f, indent=2)

print("\n================ SCRIPT 22 SUMMARY ================")
print("Gates: " + ", ".join(f"{k}: {v}" for k, v in G.items() if k.endswith("pass")))
print(f"Runtime {SUMMARY['runtime_s'] / 60:.1f} min. Outputs: {OUT_DIR}")