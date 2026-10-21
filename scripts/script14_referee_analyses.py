# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 14: Additional analyses: exclusive stability areas, orbit accounting,
#            passive-body bounds, residence-bout statistics, phase-signature
#            discrimination, hypothesis margins, phase-lag sweep, confinement
#            and stochastic time step.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX on the GPU (L4) for the new simulations; every part is checkpointed
# to Drive, so a disconnected run resumes where it stopped. No settings need to
# be changed. Parts whose input files are missing are skipped with a message.
#
# Reanalysis of existing outputs (no new simulation):
#   A. Table III rebuilt: exclusive area of every combination of fixed-position
#      branches (saddle, node, no lock, all pairwise and triple overlaps,
#      unclassified), checked against the Script 10 summary.            [R2]
#   B. Orbit accounting: found, tested, homotopy fold, unlocked limit, node,
#      saddle, unstable-complex, classified, violations, recomputation
#      re-found, for Scripts 08, 09, 10 and 12; distinct parameter settings.
#                                                                          [R3]
#   E. Passive bodies: Wilson upper bound for 0 stable of N downstream
#      equilibria (overall and per wake family); Fig. 2c slope with fit
#      interval, R^2 and bootstrap confidence interval.               [R4, m5]
#   F. Bout statistics: Kaplan-Meier survival with censoring at 2000 periods,
#      medians with cluster-bootstrap intervals over sampled points, and
#      bootstrap intervals for the noise exponents.                       [R8]
#   D. Phase signature: distribution overlap, area under the receiver
#      operating characteristic curve, parameter-disjoint classification
#      (leave one coupling row out), cluster-bootstrap intervals over coupling
#      rows; the same under noise, clustered by sampled point.           [R6]
#
# New computations (JAX):
#   C. Hypotheses of the index condition on every classified Script 10 point:
#      homotopy reached, hyperbolicity margin of M_int, distance of the
#      leading multiplier from +1, |a_L|; the points with the smallest margins
#      are reclassified at 200 and 400 steps per period; the single branch change in
#      the convergence table is identified by Script 16.                                  [R1, m9]
#   I. Continuous phase-lag sweep: 16 neural phase lags in steps of pi/8;
#      grid search, Floquet stability, stable area per lag, and the index test
#      (homotopy, M_int, a_L) on the stable orbits at every lag.         [R7]
#   G. Confinement: strobed position recorded during bouts and after random
#      starts; time until the fish leaves a band of +/- 0.1 and +/- 0.25
#      wavelengths around its station, compared with the residence bout.  [R8]
#   H. Stochastic time step: bouts at 100 and 200 steps per period with the
#      same noise intensity; medians and phase means compared.            [R8]
#
# Gates (reported, not blocking):
#   GA  exclusive areas sum to the total; total and single-branch areas match
#       the Script 10 summary
#   GC  no fixed-position branch changes between 100, 200 and 400 steps per
#       period among the lowest-margin points, or every change explained by
#       a margin below 1e-2
#   GI  0 non-marginal index violations at every phase lag of the sweep
#   GH  median bout lengths at 100 and 200 steps per period within 15%
# =============================================================================

import os, sys, json, time, glob, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

MODE = "FULL"
SEED = 16
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script14_referee_analyses")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
for d_ in (FIG_DIR, TAB_DIR, CKPT_DIR):
    os.makedirs(d_, exist_ok=True)
CK = f"checkpoints_{MODE.lower()}"
Q = "" if MODE == "FULL" else "_quick"

def P(*parts):
    return os.path.join(BASE_DIR, *parts)

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
if MODE == "FULL":
    SPP = 100
    CHUNK = 8192 if ON_GPU else 1024
    # Script 10 settings (needed to rebuild its area masks exactly)
    S10_DPHI = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    S10_EPS = np.round(np.arange(0.125, 6.001, 0.125), 6)
    # C
    N_MARGIN_RECHECK = 60
    # I: phase-lag sweep
    SWEEP_DPHI = [k * np.pi / 8 for k in range(16)]
    SWEEP_RATIOS = np.round(np.arange(0.95, 1.2501, 0.005), 6)
    SWEEP_EPS = np.round(np.arange(0.25, 6.001, 0.25), 6)
    N_PHI_SEED, N_X_SEED = 16, 60
    MAX_CLASSIFY_LAG = 400
    # G, H
    N_PTS_G, N_REP_G, T_G = 20, 32, 2000
    N_IC_G, T_IC_G = 32, 400
    G_NOISE = [0.0, 0.003, 0.01]
    N_PTS_H, N_REP_H, T_H, D_H = 20, 64, 2000, 0.01
    N_BOOT = 2000
else:
    SPP = 60
    CHUNK = 256
    S10_DPHI = [0.0]
    S10_EPS = np.array([1.0, 2.0])
    N_MARGIN_RECHECK = 4
    SWEEP_DPHI = [0.0, np.pi / 2]
    SWEEP_RATIOS = np.round(np.arange(1.02, 1.0601, 0.02), 6)
    SWEEP_EPS = np.array([1.0, 2.0])
    N_PHI_SEED, N_X_SEED = 6, 20
    MAX_CLASSIFY_LAG = 6
    N_PTS_G, N_REP_G, T_G = 2, 3, 60
    N_IC_G, T_IC_G = 4, 40
    G_NOISE = [0.0, 0.01]
    N_PTS_H, N_REP_H, T_H, D_H = 2, 4, 60, 0.01
    N_BOOT = 200
RATIO_LIM = (0.90, 1.35)
AREA_BIN = 0.0025
AL_SCALES = [0.005, 0.0015, 0.02]
AL_MARGIN = 0.02
ETAS_FINE = list(np.geomspace(1.0, 0.005, 25))
X_COLL, X_ESC = 0.15, 4.5
TAU_A = 1.0
CONF_BANDS = [0.1, 0.25]
OMEGA_W = 2.0 * np.pi
TROUT_T = 1.0 / 2.74

# -----------------------------------------------------------------------------
# 1. Baseline model (identical to Scripts 10 and 11)
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA = 3.0, 0.15, 8.0
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU, C_SL, G_HOLD = 2 * np.pi, 0.0, 16.0

def wake(t, x, psi=0.0, amp=0.0):
    V0 = U_INF * (1.0 - DELTA0 * jnp.exp(-x / L_DELTA))
    w = (1.0 + amp) * W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * jnp.sin(OMEGA_W * (x - t) + psi)
    return V0, w

def forces(t, x, u, th, psi=0.0, amp=0.0):
    V0, w = wake(t, x, psi, amp)
    V = V0 - u
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
    return L, Fx, w

def rhs(t, s, p, psi=0.0, amp=0.0):
    """p = (ratio, eps, dphi, eta); eta scales the positional dynamics (0 = frozen)."""
    x, u, th, om, zr, zi = s
    ratio, eps, dphi, eta = p[0], p[1], p[2], p[3]
    L, Fx, w = forces(t, x, u, th, psi, amp)
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    dom = -2 * ZE_B * OM_B * om - OM_B ** 2 * th + SIGMA * L + G_HOLD * drive
    r2 = zr * zr + zi * zi
    a = MU * (1.0 - r2)
    b = C_SL * MU * (1.0 - r2) + ratio * OMEGA_W
    return jnp.stack([eta * u, eta * Fx, om, dom, a * zr - b * zi + eps * th, a * zi + b * zr]), Fx, w

def period_map(s, p, spp=SPP):
    dt = 1.0 / spp
    def step(i, c):
        s_, F_ = c
        t = i * dt
        k1, f1, _ = rhs(t, s_, p)
        k2, f2, _ = rhs(t + dt / 2, s_ + dt / 2 * k1, p)
        k3, f3, _ = rhs(t + dt / 2, s_ + dt / 2 * k2, p)
        k4, f4, _ = rhs(t + dt, s_ + dt * k3, p)
        return s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F_ + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return jax.lax.fori_loop(0, spp, step, (s, 0.0))

def newton(s, p, mask, n_it, clip, spp=SPP):
    I6 = jnp.eye(6)
    def body(i, c):
        s_, _ = c
        s1, _ = period_map(s_, p, spp)
        M = jax.jacfwd(lambda y: period_map(y, p, spp)[0])(s_)
        r = (s1 - s_) * mask
        A = (M - I6) * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask)
        d = jnp.clip(jnp.nan_to_num(jnp.linalg.solve(A, -r)), -clip, clip) * mask
        return s_ + d, jnp.max(jnp.abs(r))
    s, _ = jax.lax.fori_loop(0, n_it, body, (s, 1.0))
    s1, F = period_map(s, p, spp)
    M = jax.jacfwd(lambda y: period_map(y, p, spp)[0])(s)
    return s, jnp.max(jnp.abs((s1 - s) * mask)), M, F

MASK_ALL = jnp.ones(6)
MASK_INT = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

def presc_run(s, q, n_per, n_mean, spp=SPP):
    """Body driven by a prescribed rhythm at frozen position. q = (phi0, dphi)."""
    dt = 1.0 / spp
    def f(t, s_):
        x, u, th, om = s_
        L, Fx, _ = forces(t, x, u, th)
        drive = jnp.cos(OMEGA_W * t + q[0] - q[1])
        return jnp.stack([0.0 * u, 0.0 * Fx, om, -2 * ZE_B * OM_B * om - OM_B ** 2 * th + SIGMA * L + G_HOLD * drive]), Fx
    def step(i, c):
        s_, F_ = c
        t = (i % spp) * dt
        k1, f1 = f(t, s_)
        k2, f2 = f(t + dt / 2, s_ + dt / 2 * k1)
        k3, f3 = f(t + dt / 2, s_ + dt / 2 * k2)
        k4, f4 = f(t + dt, s_ + dt * k3)
        w_ = jnp.where(i >= (n_per - n_mean) * spp, 1.0, 0.0)
        return s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F_ + w_ * dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    s1, F = jax.lax.fori_loop(0, n_per * spp, step, (s, 0.0))
    return s1, F / n_mean

_JIT = {}
def J(name, spp=SPP):
    """Compiled, vectorized kernels, cached per time step."""
    key = (name, spp)
    if key not in _JIT:
        if name == "full":
            f = lambda s, p: newton(s, p, MASK_ALL, 10, 0.05, spp)
        elif name == "hom":
            f = lambda s, p: newton(s, p, MASK_ALL, 8, 0.05, spp)
        elif name == "int":
            f = lambda s, p: newton(s, p, MASK_INT, 12, 0.2, spp)
        elif name == "presc":
            f = lambda s, q: presc_run(s, q, 50, 10, spp)
        elif name == "presc_settle":
            f = lambda s, q: presc_run(s, q, 40, 1, spp)
        else:
            raise KeyError(name)
        _JIT[key] = jax.jit(jax.vmap(f))
    return _JIT[key]

def batched(fn, *arrays, label=None):
    n = arrays[0].shape[0]
    if n == 0:
        return None
    outs = []
    tb = time.time()
    n_ch = -(-n // CHUNK)
    for c_, i0 in enumerate(range(0, n, CHUNK)):
        if label and c_ % 10 == 0:
            print(f"     {label}: chunk {c_ + 1}/{n_ch} ({time.time() - tb:.0f} s)")
        sl = [a[i0:i0 + CHUNK] for a in arrays]
        m = sl[0].shape[0]
        if m < CHUNK:
            sl = [np.concatenate([a, np.repeat(a[-1:], CHUNK - m, axis=0)]) for a in sl]
        o = fn(*[jnp.asarray(a) for a in sl])
        o = o if isinstance(o, tuple) else (o,)
        outs.append([np.asarray(v)[:m] for v in o])
    res = tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))
    return res if len(res) > 1 else res[0]

def valid(s):
    return ((s[:, 0] > 0.2) & (s[:, 0] < 3.5) & (np.abs(np.hypot(s[:, 4], s[:, 5]) - 1) < 0.5)
            & np.isfinite(s).all(1))

def homotopy(S, Pm, etas, spp=SPP):
    n = S.shape[0]
    s = S.copy()
    reached = np.ones(n, bool)
    fail_eta = np.full(n, np.nan)
    for eta in etas[1:]:
        idx = np.where(reached)[0]
        if idx.size == 0:
            break
        Pe = Pm[idx].copy(); Pe[:, 3] = eta
        sn, res, _, _ = batched(J("hom", spp), s[idx], Pe)
        okh = (res < 1e-7) & valid(sn)
        s[idx[okh]] = sn[okh]
        reached[idx[~okh]] = False
        fail_eta[idx[~okh]] = eta
    return s, reached, fail_eta

def fixed_lock(s_end, Pm, spp=SPP):
    """Locked rhythm at frozen position; returns state, success, det(I - M_int), type, M_int multipliers."""
    Pf = Pm.copy(); Pf[:, 3] = 0.0
    sf0 = s_end.copy(); sf0[:, 1] = 0.0
    sf, resf, Mf, _ = batched(J("int", spp), sf0, Pf)
    ok = resf < 1e-7
    mu = np.linalg.eigvals(Mf[:, 2:, 2:])
    dI = np.real(np.prod(1.0 - mu, axis=1))
    am = np.abs(mu)
    lead = mu[np.arange(len(mu)), np.argmax(am, 1)]
    ftype = np.where(~ok, "none", np.where(am.max(1) < 1, "node",
                     np.where(np.abs(lead.imag) < 1e-6, "saddle-real", "unstable-complex")))
    return sf, ok, dI, ftype.astype(str), mu

def relocked_gradient(sf, Pm, half, spp=SPP):
    n = sf.shape[0]
    off = np.linspace(-half, half, 7)
    Pf = Pm.copy(); Pf[:, 3] = 0.0
    Fo = np.full((n, off.size), np.nan)
    for side in (+1, -1):
        cur = sf.copy()
        for o in sorted([o for o in off if (o >= 0 if side > 0 else o <= 0)], key=abs):
            cur[:, 0] = sf[:, 0] + o
            cn, rq, _, F = batched(J("int", spp), cur, Pf)
            cur = np.where((rq < 1e-7)[:, None], cn, cur)
            Fo[:, int(np.argmin(np.abs(off - o)))] = np.where(rq < 1e-7, F, np.nan)
    aL = np.full(n, np.nan)
    for i in range(n):
        m = np.isfinite(Fo[i])
        if m.sum() >= 5:
            aL[i] = np.polyfit(off[m], Fo[i, m], 1)[0]
    return aL

def classify(S, Pm, spp=SPP):
    """Homotopy to the frozen limit, fixed-position lock, a_L (narrowest window that fits), index."""
    n = S.shape[0]
    s_end, reached, fail_eta = homotopy(S, Pm, ETAS_FINE, spp)
    sf, okf, dI, ftype, mu = fixed_lock(s_end, Pm, spp)
    okf &= reached
    ftype = np.where(okf, ftype, "none")
    aL = np.full(n, np.nan)
    for half in AL_SCALES:
        todo = np.where(okf & np.isnan(aL))[0]
        if todo.size:
            aL[todo] = relocked_gradient(sf[todo], Pm[todo], half, spp)
    index = np.sign(dI) * np.sign(-aL)
    return dict(reached=reached, fail_eta=fail_eta, ok=okf, ftype=ftype, dI=np.where(okf, dI, np.nan),
                aL=aL, index=np.where(okf, index, np.nan), x_L=sf[:, 0], mu=mu)

# -----------------------------------------------------------------------------
# 2. Noisy simulation with strobed position (Script 11 model, eLL = 0, ts = 1)
# -----------------------------------------------------------------------------
_SIMS = {}
def make_sim(n_per, spp, record):
    key = (n_per, spp, record)
    if key in _SIMS:
        return _SIMS[key]
    dt = 1.0 / spp
    def sim(s0, p, nz, k0):
        Dpsi, sig_a, Dz = nz[0], nz[1], nz[2]
        def one_period(carry, k):
            s, psi, amp, alive, tloss, X, nacc = carry
            kp = jax.random.fold_in(k0, k)
            def step(i, c):
                s_, psi_, amp_, al_, tl_, Th_, Wf_ = c
                t = k + i * dt
                k1, _, w1 = rhs(t, s_, p, psi_, amp_)
                k2, _, _ = rhs(t + dt / 2, s_ + dt / 2 * k1, p, psi_, amp_)
                k3, _, _ = rhs(t + dt / 2, s_ + dt / 2 * k2, p, psi_, amp_)
                k4, _, _ = rhs(t + dt, s_ + dt * k3, p, psi_, amp_)
                sn = s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                nrm = jax.random.normal(jax.random.fold_in(kp, i), (3,))
                psi_n = psi_ + jnp.sqrt(2 * Dpsi * dt) * nrm[0]
                ea = jnp.exp(-dt / TAU_A)
                amp_n = amp_ * ea + sig_a * jnp.sqrt(1 - ea * ea) * nrm[1]
                ang = jnp.sqrt(2 * Dz * dt) * nrm[2]
                zr2 = sn[4] * jnp.cos(ang) - sn[5] * jnp.sin(ang)
                zi2 = sn[4] * jnp.sin(ang) + sn[5] * jnp.cos(ang)
                sn = sn.at[4].set(zr2).at[5].set(zi2)
                lost = (sn[0] < X_COLL) | (sn[0] > X_ESC) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + dt, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                e = jnp.exp(-1j * OMEGA_W * (i * dt))
                return sn, psi_n, amp_n, al_n, tl_n, Th_ + s_[2] * e * dt, Wf_ + w1 * e * dt
            init = (s, psi, amp, alive, tloss, 0j, 0j)
            s, psi, amp, alive2, tloss, Th, Wf = jax.lax.fori_loop(0, spp, step, init)
            use = alive2 & (k >= 5)
            X = X + jnp.where(use, Th * jnp.conj(Wf), 0j)
            nacc = nacc + jnp.where(use, 1.0, 0.0)
            xrec = jnp.where(alive2, s[0], jnp.nan) if record else 0.0
            return (s, psi, amp, alive2, tloss, X, nacc), xrec
        carry0 = (s0, 0.0, 0.0, True, jnp.nan, 0j, 0.0)
        (s, psi, amp, alive, tloss, X, nacc), xs = jax.lax.scan(one_period, carry0, jnp.arange(n_per))
        return s, alive, tloss, jnp.angle(X), nacc, xs
    _SIMS[key] = jax.jit(jax.vmap(sim))
    return _SIMS[key]

def keys_for(n, salt):
    return np.asarray(jax.random.split(jax.random.PRNGKey(SEED * 1000 + salt), n))

def noise_vec(D):
    return np.array([D, np.sqrt(D), D], float)

# -----------------------------------------------------------------------------
# 3. Statistics helpers
# -----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)

def km(times, event, horizon):
    """Kaplan-Meier survival. times: loss time or horizon if censored; event: True if lost."""
    t = np.asarray(times, float); e = np.asarray(event, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = t.size
    surv, ts, S = [], [], 1.0
    at_risk = n
    i = 0
    while i < n:
        j = i
        d = 0
        while j < n and t[j] == t[i]:
            d += int(e[j]); j += 1
        if d:
            S *= 1 - d / at_risk
            ts.append(t[i]); surv.append(S)
        at_risk -= (j - i)
        i = j
    ts, surv = np.array(ts), np.array(surv)
    below = np.where(surv <= 0.5)[0]
    med = float(ts[below[0]]) if below.size else np.inf
    return med, ts, surv

def cluster_boot(groups, stat, n_boot=N_BOOT, seed=0):
    """Bootstrap over clusters. groups: array of cluster labels; stat(idx) -> float."""
    r = np.random.default_rng(seed)
    labs = np.unique(groups)
    if labs.size == 0:
        return (np.nan, np.nan)
    members = {g: np.where(groups == g)[0] for g in labs}
    vals = []
    for _ in range(n_boot):
        pick = r.choice(labs, labs.size, replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def cmean(a):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(a))))) if len(a) else np.nan

def cdiff(a, b):
    return float(np.angle(np.exp(1j * (a - b))))

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if pos.size == 0 or neg.size == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))

def overlap_coef(a, b, bins=36):
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    pa, _ = np.histogram(a, edges); pb, _ = np.histogram(b, edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return np.nan
    return float(np.minimum(pa / pa.sum(), pb / pb.sum()).sum())

def write_table(name, header, rows, caption):
    with open(os.path.join(TAB_DIR, f"{name}.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for r_ in rows:
            f.write(",".join("" if v is None else str(v) for v in r_) + "\n")
    with open(os.path.join(TAB_DIR, f"{name}.tex"), "w") as f:
        f.write("\\begin{table}\n\\caption{" + caption + "}\n\\label{tab:" + name + "}\n\\begin{ruledtabular}\n")
        f.write("\\begin{tabular}{l" + "c" * (len(header) - 1) + "}\n")
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for r_ in rows:
            f.write(" & ".join("--" if v is None else str(v) for v in r_) + " \\\\\n")
        f.write("\\end{tabular}\n\\end{ruledtabular}\n\\end{table}\n")

def fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return f"{v:.{d}f}" if isinstance(v, float) else v

SUMMARY = dict(mode=MODE, device=[d.platform for d in jax.devices()])
t_start = time.time()
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

# -----------------------------------------------------------------------------
# Script 10 data (used by A, C, D, G, H)
# -----------------------------------------------------------------------------
F10_PTS = P("script10_continuation", f"points_script10{Q}.npz")
F10_BR = P("script10_continuation", CK, "stageC_branches.npz")
F10_CL = P("script10_continuation", CK, "stageD2_index.npz")
HAVE10 = all(os.path.exists(f_) for f_ in (F10_PTS, F10_BR, F10_CL))
if HAVE10:
    PTS = np.load(F10_PTS)["points"]
    BR = np.load(F10_BR, allow_pickle=True)["branches"].tolist()
    CL = {k: v for k, v in np.load(F10_CL).items()}
    SEL, FT = CL["sel"].astype(int), CL["ftype"].astype(str)
    S_CL = np.array([BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6] for i in SEL])
    print(f"Script 10: {SEL.size} classified stable points, {len(BR)} half-branches")
else:
    print("Script 10 outputs not found: parts A, C and D are skipped; G and H use orbits from part I")

# =============================================================================
# A. Table III rebuilt: exclusive areas
# =============================================================================
print("\n=== A. Exclusive stable-holding areas ===")
if HAVE10:
    def stable_keys(rows):
        return np.stack([rows[:, 2], rows[:, 3], np.round(rows[:, 4] / 0.001),
                         np.round(rows[:, 5] / 0.01), np.round(rows[:, 6] / 0.1)], 1)
    BINS = np.arange(RATIO_LIM[0], RATIO_LIM[1] + 1e-9, AREA_BIN)
    d_eps = float(np.diff(S10_EPS).min()) if S10_EPS.size > 1 else 0.5
    key_cls = {tuple(k_): str(t_) for k_, t_ in zip(stable_keys(PTS[SEL]), FT)}
    cls_of_point = {}
    stab_all = np.where(PTS[:, 7] > 0)[0]
    for p_, k_ in zip(stab_all, stable_keys(PTS[stab_all])):
        c_ = key_cls.get(tuple(k_))
        if c_ is not None:
            cls_of_point[int(p_)] = c_
    occ = {(dp, cl): np.zeros((S10_EPS.size, BINS.size - 1), bool)
           for dp in S10_DPHI for cl in ("any", "node", "saddle-real", "none", "unstable-complex")}
    pt_index = {(int(r_[0]), int(r_[1])): k_ for k_, r_ in enumerate(PTS)}
    for bi, b in enumerate(BR):
        Z = b["Z"]
        if len(Z) < 2:
            continue
        ei = int(np.argmin(np.abs(S10_EPS - b["eps"])))
        dp = min(S10_DPHI, key=lambda v: abs(v - b["dphi"]))
        for j in range(len(Z) - 1):
            k1, k2 = pt_index.get((bi, j)), pt_index.get((bi, j + 1))
            if k1 is None or k2 is None or not (PTS[k1, 7] and PTS[k2, 7]):
                continue
            lo, hi = sorted((Z[j, 6], Z[j + 1, 6]))
            cells = np.where((BINS[1:] > lo) & (BINS[:-1] < hi))[0]
            if cells.size == 0:
                cells = np.array([np.clip(np.searchsorted(BINS, lo) - 1, 0, BINS.size - 2)])
            occ[(dp, "any")][ei, cells] = True
            c_ = cls_of_point.get(k1, cls_of_point.get(k2))
            if c_ in ("node", "saddle-real", "none", "unstable-complex"):
                occ[(dp, c_)][ei, cells] = True
    cell_A = AREA_BIN * d_eps
    s10sum = jload(P("script10_continuation", f"summary_script10{Q}.json"))
    s10area = (s10sum or {}).get("stable_area_by_delay", {})
    A_rows, A_out = [], {}
    for dp in S10_DPHI:
        N_, S_, O_ = occ[(dp, "node")], occ[(dp, "saddle-real")], occ[(dp, "none")]
        U_ = occ[(dp, "unstable-complex")]
        anyc = occ[(dp, "any")]
        cls_any = N_ | S_ | O_ | U_
        reg = dict(
            saddle_only=S_ & ~N_ & ~O_, node_only=N_ & ~S_ & ~O_, nolock_only=O_ & ~N_ & ~S_,
            node_saddle=N_ & S_ & ~O_, node_nolock=N_ & O_ & ~S_, saddle_nolock=S_ & O_ & ~N_,
            all_three=N_ & S_ & O_, unstable_complex_only=U_ & ~N_ & ~S_ & ~O_,
            unclassified=anyc & ~cls_any)
        a_ = {k_: float(v_.sum() * cell_A) for k_, v_ in reg.items()}
        a_["total"] = float(anyc.sum() * cell_A)
        a_["sum_of_exclusive"] = float(sum(v_ for k_, v_ in a_.items() if k_ != "total"))
        ref = s10area.get(f"{dp:.3f}")
        a_["matches_script10"] = (None if ref is None else bool(
            abs(ref["any"] - a_["total"]) < 1e-9 and abs(ref["saddle_only"] - a_["saddle_only"]) < 1e-9
            and abs(ref["node_only"] - a_["node_only"]) < 1e-9))
        A_out[f"{dp:.3f}"] = a_
        A_rows.append([f"{dp:.3f}", fmt(a_["total"]), fmt(a_["saddle_only"]), fmt(a_["node_only"]),
                       fmt(a_["nolock_only"]), fmt(a_["node_saddle"]), fmt(a_["node_nolock"]),
                       fmt(a_["saddle_nolock"]), fmt(a_["all_three"]), fmt(a_["unclassified"] + a_["unstable_complex_only"])])
        print(f"  dphi {dp:.3f}: total {a_['total']:.4f} = sum of exclusive {a_['sum_of_exclusive']:.4f} | "
              + ", ".join(f"{k_} {v_:.4f}" for k_, v_ in a_.items() if k_ not in ("total", "sum_of_exclusive", "matches_script10"))
              + f" | matches Script 10 summary: {a_['matches_script10']}")
    write_table("T3_areas_exclusive",
                ["phase lag", "total", "saddle only", "node only", "no lock only", "node and saddle",
                 "node and no lock", "saddle and no lock", "all three", "unclassified"],
                A_rows, "Mutually exclusive areas of stable station holding in the (omega_c/omega_w, eps) plane by "
                "fixed-position branch; the columns after the total sum to the total.")
    GA = all(abs(v_["total"] - v_["sum_of_exclusive"]) < 1e-9 and v_["matches_script10"] is not False for v_ in A_out.values())
    SUMMARY["A_areas_exclusive"] = A_out
    SUMMARY["gate_GA"] = GA
else:
    GA = None

# =============================================================================
# B. Orbit accounting
# =============================================================================
print("\n=== B. Orbit accounting ===")
B_rows, B_out = [], {}

def flow_counts(recs, stable_found=None):
    st = [r_ for r_ in recs if r_.get("stable") and r_.get("kept", True)]
    fixed = [r_ for r_ in st if r_.get("status") == "fixed"]
    node = [r_ for r_ in fixed if r_.get("fixed_type") == "node"]
    sad = [r_ for r_ in fixed if r_.get("fixed_type") == "saddle-real"]
    cplx = [r_ for r_ in fixed if r_.get("fixed_type") == "unstable-complex"]
    ev = [r_ for r_ in node + sad if r_.get("index") is not None]
    viol = [r_ for r_ in ev if r_["index"] < 0]
    settings = {(r_.get("dataset", r_.get("group")), r_.get("variant"), round(r_["eps"], 6), round(r_["dphi"], 6)) for r_ in ev}
    return dict(stable_found=stable_found, stable_tested=sum(1 for r_ in recs if r_.get("stable")),
                not_reconverged=sum(1 for r_ in recs if r_.get("stable") and not r_.get("kept", True)),
                homotopy_fold=sum(r_.get("status") == "homotopy-fold" for r_ in st),
                unlocked_limit=sum(r_.get("status") == "no-fixed-lock" for r_ in st),
                node=len(node), saddle=len(sad), unstable_complex=len(cplx),
                fixed_but_no_aL=sum(1 for r_ in node + sad if r_.get("index") is None),
                classified=len(ev), violations=len(viol), distinct_settings=len(settings))

# Script 08: stable orbits found in its inputs (05b, 06) and its records
rec08 = jload(P("script08_homotopy", "records_script08.json"))
if rec08 is not None:
    found08 = 0
    f5 = next((P("script05b_holding_dense", n_) for n_ in ("results_script05b.npz", "results_script05.npz")
               if os.path.exists(P("script05b_holding_dense", n_))), None)
    if f5:
        found08 += int(sum(str(t) == "stable" for t in np.load(f5, allow_pickle=True)["type"]))
    for tag_ in ("ALL", "BODY"):
        o6 = jload(P("script06_robustness", f"orbits_script06_{tag_}.json"))
        if o6:
            found08 += int(sum(o["stable"] for v in o6.values() for o in v))
    B_out["parameter_sets_08"] = flow_counts(rec08, found08 or None)
    by_var = {}
    for r_ in rec08:
        by_var.setdefault(f"{r_.get('dataset')}/{r_.get('variant')}", []).append(r_)
    B_out["parameter_sets_08_by_variant"] = {k_: flow_counts(v_) for k_, v_ in by_var.items()}
else:
    print("  Script 08 records not found")

# Script 09: per-variant checkpoints (results and records)
c09 = sorted(glob.glob(P("script09_model_structure", CK, "*.json")))
if c09:
    tot_recs, per_var = [], {}
    for f_ in c09:
        d_ = json.load(open(f_))
        r9, recs = d_["res"], d_["recs"]
        for x_ in recs:
            x_.setdefault("variant", r9["name"])
        tot_recs += recs
        per_var[r9["name"]] = dict(group=r9.get("group"), **flow_counts(recs, int(r9.get("n_stable", 0))))
    found09 = sum(v_["stable_found"] for v_ in per_var.values())
    B_out["model_structures_09"] = flow_counts(tot_recs, found09)
    B_out["model_structures_09_by_variant"] = per_var
    B_out["model_structures_09_capped"] = int(sum(max(v_["stable_found"] - v_["stable_tested"], 0) for v_ in per_var.values()))
else:
    print("  Script 09 checkpoints not found")

# Script 10: classified stable continuation points
if HAVE10:
    ft10, ix10 = FT, CL["index"]
    reached10 = CL["reached"].astype(bool)
    st_pts = int((PTS[:, 7] > 0).sum())
    B_out["continued_families_10"] = dict(
        stable_points=st_pts, distinct_stable_points=int(SEL.size), homotopy_fold=int((~reached10).sum()),
        unlocked_limit=int((reached10 & (ft10 == "none")).sum()), node=int((ft10 == "node").sum()),
        saddle=int((ft10 == "saddle-real").sum()), unstable_complex=int((ft10 == "unstable-complex").sum()),
        fixed_but_no_aL=int(((ft10 == "node") | (ft10 == "saddle-real")).sum() - np.isfinite(ix10[(ft10 == "node") | (ft10 == "saddle-real")]).sum()),
        classified=int(np.isfinite(ix10).sum()), violations=int((np.isfinite(ix10) & (ix10 < 0)).sum()),
        distinct_settings=int(len({(round(PTS[i, 2], 6), round(PTS[i, 3], 6)) for i, v in zip(SEL, ix10) if np.isfinite(v)})),
        distinct_branches=int(len({int(PTS[i, 0]) for i, v in zip(SEL, ix10) if np.isfinite(v)})))

# Script 12: recomputation
for scr in ("08", "09"):
    rows12 = jload(P("script12_verification", CK, f"partB_{scr}.json"))
    if rows12 is None:
        continue
    st12 = [r_ for r_ in rows12 if r_["stable"]]
    rf = [r_ for r_ in st12 if r_["refound"]]
    B_out[f"recomputation_12_{scr}"] = dict(
        stable_records=len(st12), refound=len(rf), not_refound=len(st12) - len(rf),
        classified=sum(1 for r_ in rf if r_.get("index") is not None),
        violations=sum(1 for r_ in rf if r_.get("index") is not None and r_["index"] < 0),
        type_changed=sum(1 for r_ in rf if r_.get("type") and r_.get("orig_type") and r_["type"] != r_["orig_type"]))

lab = {"parameter_sets_08": "Parameter sets", "model_structures_09": "Model variants and samples",
       "continued_families_10": "Continued families"}
for k_, nm in lab.items():
    v_ = B_out.get(k_)
    if v_ is None:
        continue
    B_rows.append([nm, v_.get("stable_found", v_.get("stable_points")), v_.get("stable_tested", v_.get("distinct_stable_points")),
                   v_["homotopy_fold"], v_["unlocked_limit"], v_["unstable_complex"], v_["fixed_but_no_aL"],
                   v_["node"], v_["saddle"], v_["classified"], v_["violations"], v_["distinct_settings"]])
for scr in ("08", "09"):
    v_ = B_out.get(f"recomputation_12_{scr}")
    if v_:
        B_rows.append([f"Recomputation of {lab['parameter_sets_08' if scr == '08' else 'model_structures_09'].lower()}",
                       v_["stable_records"], v_["refound"], None, None, None, None, None, None, v_["classified"],
                       v_["violations"], None])
if B_rows:
    write_table("T_accounting",
                ["source", "stable found", "tested", "fold in eta", "unlocked limit", "unstable complex",
                 "no a_L fit", "node", "saddle", "classified", "violations", "distinct settings"],
                B_rows, "Flow of stable orbits from discovery to the index test.")
    for r_ in B_rows:
        print("  " + " | ".join(str(v) for v in r_))
SUMMARY["B_accounting"] = B_out

# =============================================================================
# E. Passive bodies: Wilson bound and Fig. 2c slope
# =============================================================================
print("\n=== E. Passive bodies ===")
E_out = {}
f2 = P("script02_passive_robustness", "results_script02.npz")
if os.path.exists(f2):
    d2 = np.load(f2, allow_pickle=True)
    r = d2["lhs_eq_x"] / d2["lhs_xpeak"]
    stable = d2["lhs_cls"] == 0
    f2b = P("script02b_passive_cases", "results_script02b.npz")
    if os.path.exists(f2b):
        d2b = np.load(f2b)
        chk, conv, cf = d2b["chk"].astype(int), d2b["conv"].astype(bool), d2b["cls_f"]
        stable = stable.copy(); stable[chk[conv]] = cf[conv] == 0
    down = r > 1
    k_, n_ = int(stable[down].sum()), int(down.sum())
    p_, lo_, hi_ = wilson(k_, n_)
    E_out["downstream"] = dict(stable=k_, n=n_, wilson_upper_95=hi_)
    shape_key = next((k for k in ("lhs_shape", "shape", "lhs_family") if k in d2.files), None)
    if shape_key:
        fam = d2[shape_key]
        E_out["downstream_by_family"] = {str(f): dict(stable=int(stable[down & (fam == f)].sum()), n=int((down & (fam == f)).sum()),
                                                      wilson_upper_95=wilson(int(stable[down & (fam == f)].sum()), int((down & (fam == f)).sum()))[2])
                                         for f in np.unique(fam)}
    up = r <= 1
    E_out["upstream"] = dict(stable=int(stable[up].sum()), n=int(up.sum()))
    print(f"  downstream of the vortex peak: {k_} stable of {n_}, 95% Wilson upper bound {100 * hi_:.3f}%")
f1 = P("script01_passive_baseline", "results_script01.npz")
if os.path.exists(f1):
    d1 = np.load(f1)
    lam = d1["lam_max"]; tl = np.nanmin(np.stack([d1["tloss_plus"], d1["tloss_minus"]]), 0)
    m = (lam > 1e-3) & np.isfinite(tl) & d1["converged"].astype(bool)
    x_, y_ = np.log(1 / lam[m]), np.log(tl[m])
    c_ = np.polyfit(x_, y_, 1)
    r2 = 1 - np.sum((y_ - np.polyval(c_, x_)) ** 2) / np.sum((y_ - y_.mean()) ** 2)
    rb = np.random.default_rng(3)
    bs = [np.polyfit(x_[i], y_[i], 1)[0] for i in (rb.integers(0, x_.size, x_.size) for _ in range(N_BOOT))]
    E_out["fig2c_slope"] = dict(n=int(m.sum()), slope=float(c_[0]), r2=float(r2),
                                ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                growth_time_range=[float(np.exp(x_.min())), float(np.exp(x_.max()))])
    print(f"  Fig. 2c: slope {c_[0]:.3f} (95% CI {E_out['fig2c_slope']['ci95'][0]:.3f} to {E_out['fig2c_slope']['ci95'][1]:.3f}), "
          f"R^2 {r2:.3f}, n {int(m.sum())}, growth time {np.exp(x_.min()):.1f} to {np.exp(x_.max()):.1f} periods")
SUMMARY["E_passive"] = E_out

# =============================================================================
# F. Bout statistics from Script 11 (Kaplan-Meier, censoring, bootstrap)
# =============================================================================
print("\n=== F. Bout statistics ===")
F_out = {}
f11 = P("script11_noise_predictions", CK, "partCD2.npz")
NOISE_LEVELS11 = [0.003, 0.01, 0.03, 0.1]
COND11 = [f"joint D={D}" for D in NOISE_LEVELS11] + ["wake phase only D=0.01", "wake amplitude only D=0.01", "CPG phase only D=0.01"]
GROUPS11 = [(0.0, "saddle-real"), (0.0, "node"), (np.pi / 2, "saddle-real"), (np.pi / 2, "node")]
T_BOUT11 = 2000
if os.path.exists(f11):
    CD = dict(np.load(f11, allow_pickle=True))
    F_rows = []
    for g_, (dp, ft_) in enumerate(GROUPS11):
        for lv, lab_ in enumerate(COND11):
            m = (CD["b_group"] == g_) & (CD["b_level"] == lv)
            if not m.any():
                continue
            ev_ = ~CD["b_alive"][m].astype(bool)
            tt = np.where(ev_, CD["b_tloss"][m], T_BOUT11)
            med, _, _ = km(tt, ev_, T_BOUT11)
            pts = CD["b_point"][m]
            ci = cluster_boot(pts, lambda idx: (lambda v: v if np.isfinite(v) else np.nan)(km(tt[idx], ev_[idx], T_BOUT11)[0]), seed=g_ * 10 + lv)
            F_out[f"{dp:.3f}/{ft_}/{lab_}"] = dict(runs=int(m.sum()), points=int(np.unique(pts).size),
                                                  censored_fraction=float((~ev_).mean()), km_median=None if not np.isfinite(med) else med,
                                                  km_median_ci95=ci)
            F_rows.append([f"{dp:.3f}", ft_.split("-")[0], lab_, int(m.sum()), f"{(~ev_).mean():.2f}",
                           ">2000" if not np.isfinite(med) else f"{med:.0f}",
                           None if not np.isfinite(ci[0]) else f"{ci[0]:.0f} to {ci[1]:.0f}"])
        # exponent with bootstrap over sampled points (same points at every level)
        mj = [(CD["b_group"] == g_) & (CD["b_level"] == lv) for lv in range(len(NOISE_LEVELS11))]
        if all(x.any() for x in mj):
            pts_all = np.unique(np.concatenate([CD["b_point"][x] for x in mj]))
            def expo(pidx):
                meds = []
                for x in mj:
                    sel = x & np.isin(CD["b_point"], pidx)
                    ev_ = ~CD["b_alive"][sel].astype(bool)
                    meds.append(km(np.where(ev_, CD["b_tloss"][sel], T_BOUT11), ev_, T_BOUT11)[0])
                meds = np.array(meds)
                if not np.all(np.isfinite(meds)):
                    return np.nan
                return float(np.polyfit(np.log(NOISE_LEVELS11), np.log(meds), 1)[0])
            e0 = expo(pts_all)
            rb = np.random.default_rng(g_)
            bs = [expo(rb.choice(pts_all, pts_all.size, replace=True)) for _ in range(min(N_BOOT, 500))]
            bs = [b for b in bs if np.isfinite(b)]
            F_out[f"{dp:.3f}/{ft_}/exponent"] = dict(exponent=e0, ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else None)
            print(f"  {dp:.3f} {ft_:12s} exponent {e0:+.2f}" + (f" (95% CI {np.quantile(bs, 0.025):+.2f} to {np.quantile(bs, 0.975):+.2f})" if bs else ""))
    write_table("T6_bouts_km", ["phase lag", "branch", "noise", "runs", "censored", "KM median", "95% CI"], F_rows,
                "Station-holding bouts: Kaplan-Meier median with censoring at 2000 periods; intervals from a bootstrap over sampled points.")
else:
    print("  Script 11 partCD2 not found")
SUMMARY["F_bouts"] = F_out

# =============================================================================
# D. Phase signature: discrimination with parameter-disjoint evaluation
# =============================================================================
print("\n=== D. Phase signature ===")
D_out = {}
HAVE_D = False
if HAVE10 and os.path.exists(f11):
    ph = CD["phase_det"]
    dp = np.pi / 2
    mN = (np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == "node")
    mS = (np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == "saddle-real")
    HAVE_D = bool(mN.any() and mS.any())
if HAVE_D:
    rows_eps = PTS[SEL, 2]
    def score(a, centre):
        return np.angle(np.exp(1j * (a - centre)))
    def disc_stats(pn, ps, centre=None):
        mn_, ms_ = cmean(pn), cmean(ps)
        if centre is None:
            centre = np.angle(np.exp(1j * mn_) + np.exp(1j * ms_))
        sgn = np.sign(cdiff(mn_, ms_)) or 1.0
        return dict(n_node=int(len(pn)), n_saddle=int(len(ps)), mean_node=mn_, mean_saddle=ms_,
                    diff=cdiff(mn_, ms_), auc=auc(sgn * score(pn, centre), sgn * score(ps, centre)),
                    overlap=overlap_coef(pn, ps), centre=float(centre), sign=float(sgn))
    det = disc_stats(ph[mN], ph[mS])
    # parameter-disjoint: leave one coupling row out
    m_ = mN | mS
    rows_u = np.unique(rows_eps[m_])
    correct, total, acc_rows = 0, 0, []
    for e_ in rows_u:
        tr = m_ & (rows_eps != e_); te = m_ & (rows_eps == e_)
        if not (tr & mN).any() or not (tr & mS).any() or not te.any():
            continue
        st_ = disc_stats(ph[tr & mN], ph[tr & mS])
        pred_node = st_["sign"] * score(ph[te], st_["centre"]) > 0
        truth = mN[te]
        correct += int((pred_node == truth).sum()); total += int(te.sum())
        acc_rows.append(float(np.mean(pred_node == truth)))
    # balanced accuracy of the leave-one-row-out classifier
    tp = tn = fn = fp = 0
    for e_ in rows_u:
        tr = m_ & (rows_eps != e_); te = m_ & (rows_eps == e_)
        if not (tr & mN).any() or not (tr & mS).any() or not te.any():
            continue
        st_ = disc_stats(ph[tr & mN], ph[tr & mS])
        pred = st_["sign"] * score(ph[te], st_["centre"]) > 0
        tru = mN[te]
        tp += int((pred & tru).sum()); tn += int((~pred & ~tru).sum()); fn += int((~pred & tru).sum()); fp += int((pred & ~tru).sum())
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    # cluster bootstrap over coupling rows
    idx_all = np.where(m_)[0]
    grp = rows_eps[idx_all]
    ci_diff = cluster_boot(grp, lambda i: cdiff(cmean(ph[idx_all[i]][mN[idx_all[i]]]), cmean(ph[idx_all[i]][mS[idx_all[i]]]))
                           if mN[idx_all[i]].any() and mS[idx_all[i]].any() else np.nan, seed=11)
    ci_auc = cluster_boot(grp, lambda i: disc_stats(ph[idx_all[i]][mN[idx_all[i]]], ph[idx_all[i]][mS[idx_all[i]]])["auc"]
                          if mN[idx_all[i]].any() and mS[idx_all[i]].any() else np.nan, seed=12)
    D_out["deterministic"] = dict(det, coupling_rows=int(rows_u.size), loro_accuracy=correct / max(total, 1),
                                  loro_balanced_accuracy=bal, diff_ci95_rows=ci_diff, auc_ci95_rows=ci_auc)
    print(f"  deterministic, phase lag pi/2: diff {det['diff']:.3f} rad (95% CI over coupling rows {ci_diff[0]:.3f} to {ci_diff[1]:.3f}), "
          f"AUC {det['auc']:.3f} ({ci_auc[0]:.3f} to {ci_auc[1]:.3f}), overlap {det['overlap']:.3f}, "
          f"leave-one-row-out balanced accuracy {bal:.3f} over {rows_u.size} rows")
    # under noise: one phase per bout, clustered by sampled point
    gN = [g for g, v in enumerate(GROUPS11) if abs(v[0] - dp) < 1e-6 and v[1] == "node"]
    gS = [g for g, v in enumerate(GROUPS11) if abs(v[0] - dp) < 1e-6 and v[1] == "saddle-real"]
    D_rows = [["deterministic", det["n_node"], det["n_saddle"], f"{det['diff']:.3f}", f"{ci_diff[0]:.3f} to {ci_diff[1]:.3f}",
               f"{det['auc']:.3f}", f"{ci_auc[0]:.3f} to {ci_auc[1]:.3f}", f"{det['overlap']:.3f}", f"{bal:.3f}"]]
    for lv, D in enumerate(NOISE_LEVELS11[:2]):
        if not gN or not gS:
            break
        mm = (CD["b_level"] == lv) & (CD["b_nacc"] > 0)
        bN = mm & (CD["b_group"] == gN[0]); bS = mm & (CD["b_group"] == gS[0])
        if not bN.any() or not bS.any():
            continue
        st_ = disc_stats(CD["b_phase"][bN], CD["b_phase"][bS], centre=det["centre"])
        # single-bout classification with the deterministic threshold
        predN = det["sign"] * score(CD["b_phase"][bN], det["centre"]) > 0
        predS = det["sign"] * score(CD["b_phase"][bS], det["centre"]) > 0
        bal_n = 0.5 * (predN.mean() + (~predS).mean())
        ii = np.where(bN | bS)[0]
        pts_ = CD["b_point"][ii] + 100000 * CD["b_group"][ii]
        ci_d = cluster_boot(pts_, lambda i: cdiff(cmean(CD["b_phase"][ii[i]][bN[ii[i]]]), cmean(CD["b_phase"][ii[i]][bS[ii[i]]]))
                            if bN[ii[i]].any() and bS[ii[i]].any() else np.nan, seed=20 + lv)
        ci_a = cluster_boot(pts_, lambda i: disc_stats(CD["b_phase"][ii[i]][bN[ii[i]]], CD["b_phase"][ii[i]][bS[ii[i]]], det["centre"])["auc"]
                            if bN[ii[i]].any() and bS[ii[i]].any() else np.nan, seed=30 + lv)
        D_out[f"noise_D={D}"] = dict(st_, single_bout_balanced_accuracy=float(bal_n), diff_ci95_points=ci_d, auc_ci95_points=ci_a)
        D_rows.append([f"noise D={D}", st_["n_node"], st_["n_saddle"], f"{st_['diff']:.3f}", f"{ci_d[0]:.3f} to {ci_d[1]:.3f}",
                       f"{st_['auc']:.3f}", f"{ci_a[0]:.3f} to {ci_a[1]:.3f}", f"{st_['overlap']:.3f}", f"{bal_n:.3f}"])
        print(f"  noise D={D}: diff {st_['diff']:.3f} rad ({ci_d[0]:.3f} to {ci_d[1]:.3f}), AUC {st_['auc']:.3f} "
              f"({ci_a[0]:.3f} to {ci_a[1]:.3f}), overlap {st_['overlap']:.3f}, single-bout balanced accuracy {bal_n:.3f}")
    write_table("T_phase_discrimination",
                ["condition", "node n", "saddle n", "mean difference (rad)", "95% CI", "AUC", "95% CI", "overlap", "balanced accuracy"],
                D_rows, "Discrimination of node and saddle branches by tail-beat phase at phase lag pi/2. Deterministic intervals "
                "from a bootstrap over coupling rows and accuracy from leave-one-row-out classification; noisy intervals from a "
                "bootstrap over sampled points and accuracy of single bouts with the deterministic threshold.")
else:
    print("  needs Script 10 and Script 11 outputs with node and saddle points at phase lag pi/2: skipped")
SUMMARY["D_phase"] = D_out

# =============================================================================
# C. Hypotheses of the index condition, and resolution check at low margins
# =============================================================================
print("\n=== C. Hypothesis margins ===")
C_out = {}
if HAVE10:
    if os.path.exists(ckpt("partC_margins.npz")):
        C_ = dict(np.load(ckpt("partC_margins.npz"), allow_pickle=True))
        print("C: loaded from checkpoint")
    else:
        t0 = time.time()
        Pm = np.stack([PTS[SEL, 4], PTS[SEL, 2], PTS[SEL, 3], np.ones(SEL.size)], 1)
        cls = classify(S_CL, Pm, SPP)
        mu = cls["mu"]
        am = np.abs(mu)
        hyp = np.min(np.abs(am - 1.0), axis=1)
        d1 = np.min(np.abs(mu - 1.0), axis=1)
        C_ = dict(ftype=cls["ftype"], reached=cls["reached"], dI=cls["dI"], aL=cls["aL"], hyp=hyp, d1=d1,
                  index=cls["index"])
        # lowest-margin points (fixed lock found) rechecked at 200 and 400 steps per period
        cand = np.where(cls["ok"])[0]
        marg = np.minimum(hyp[cand], np.abs(cls["aL"][cand]) / 0.1)
        low = cand[np.argsort(marg)[:N_MARGIN_RECHECK]]
        C_["low"] = low
        for spp2 in (2 * SPP, 4 * SPP):
            if low.size == 0:
                C_[f"ftype_{spp2}"] = np.zeros(0, str); C_[f"index_{spp2}"] = np.zeros(0); C_[f"hyp_{spp2}"] = np.zeros(0)
                continue
            S2, r2, _, _ = batched(J("full", spp2), S_CL[low], Pm[low])
            c2 = classify(S2, Pm[low], spp2)
            C_[f"ftype_{spp2}"] = c2["ftype"]; C_[f"index_{spp2}"] = c2["index"]
            C_[f"hyp_{spp2}"] = np.min(np.abs(np.abs(c2["mu"]) - 1), axis=1)
        np.savez_compressed(ckpt("partC_margins.npz"), **C_)
        print(f"C: {SEL.size} points reclassified, {low.size} low-margin points rechecked ({time.time() - t0:.0f} s)")
    ftc = C_["ftype"].astype(str)
    agree = float(np.mean(ftc == FT))
    okm = (ftc == "node") | (ftc == "saddle-real")
    q = lambda v: [float(np.nanquantile(v, a)) for a in (0.0, 0.01, 0.05, 0.5)]
    low = C_["low"].astype(int)
    chg = {}
    for spp2 in (2 * SPP, 4 * SPP):
        ch = np.where(C_[f"ftype_{spp2}"].astype(str) != ftc[low])[0]
        chg[str(spp2)] = [dict(point=int(low[i]), eps=float(PTS[SEL[low[i]], 2]), dphi=float(PTS[SEL[low[i]], 3]),
                               ratio=float(PTS[SEL[low[i]], 4]), type_100=str(ftc[low[i]]),
                               type_new=str(C_[f"ftype_{spp2}"][i]), hyperbolicity_margin=float(C_["hyp"][low[i]]),
                               aL=None if not np.isfinite(C_["aL"][low[i]]) else float(C_["aL"][low[i]])) for i in ch]
    C_out = dict(points=int(SEL.size), type_agreement_with_script10=agree,
                 homotopy_reached=int(C_["reached"].sum()), homotopy_fold=int((~C_["reached"].astype(bool)).sum()),
                 hyperbolicity_margin_quantiles_min_1_5_50=q(C_["hyp"][okm]),
                 distance_to_plus1_quantiles=q(C_["d1"][okm]),
                 abs_aL_quantiles=q(np.abs(C_["aL"][okm])),
                 abs_det_quantiles=q(np.abs(C_["dI"][okm])),
                 below_1e2_hyperbolicity=int((C_["hyp"][okm] < 1e-2).sum()),
                 below_aL_margin=int((np.abs(C_["aL"][okm]) < AL_MARGIN).sum()),
                 low_margin_rechecked=int(low.size), branch_changes=chg)
    GC = all(all(c_["hyperbolicity_margin"] < 1e-2 for c_ in v) for v in chg.values())
    C_out["gate_GC"] = GC
    print(f"  type agreement with Script 10: {agree:.4f}; hyperbolicity margin min/1%/5%/median "
          f"{C_out['hyperbolicity_margin_quantiles_min_1_5_50']}; |a_L| quantiles {C_out['abs_aL_quantiles']}")
    print(f"  branch changes among {low.size} lowest-margin points: "
          + ", ".join(f"{k_} steps: {len(v_)}" for k_, v_ in chg.items()))
else:
    GC = None
SUMMARY["C_hypotheses"] = C_out

# =============================================================================
# I. Continuous phase-lag sweep
# =============================================================================
print("\n=== I. Phase-lag sweep ===")
def seeds_for(dphi):
    PHIS = np.linspace(0, 2 * np.pi, N_PHI_SEED, endpoint=False)
    XG = np.linspace(0.3, 2.2, N_X_SEED)
    S0 = np.zeros((PHIS.size * XG.size, 4)); S0[:, 0] = np.tile(XG, PHIS.size)
    Q0 = np.stack([np.repeat(PHIS, XG.size), np.full(PHIS.size * XG.size, dphi)], 1)
    _, Fp = batched(J("presc"), S0, Q0)
    Fp = Fp.reshape(PHIS.size, XG.size)
    sx, sph = [], []
    for c in range(PHIS.size):
        for j in range(XG.size - 1):
            f0, f1 = Fp[c, j], Fp[c, j + 1]
            if np.isfinite(f0) and np.isfinite(f1) and np.sign(f0) != np.sign(f1):
                sx.append(XG[j] - f0 * (XG[j + 1] - XG[j]) / (f1 - f0)); sph.append(PHIS[c])
    sx, sph = np.array(sx), np.array(sph)
    if sx.size == 0:
        return np.zeros((0, 6))
    Ss = np.zeros((sx.size, 4)); Ss[:, 0] = sx
    Ss, _ = batched(J("presc_settle"), Ss, np.stack([sph, np.full(sx.size, dphi)], 1))
    seeds = np.concatenate([Ss, np.cos(sph)[:, None], np.sin(sph)[:, None]], 1)
    seeds[:, 0] = sx; seeds[:, 1] = 0.0
    return seeds

I_out, I_rows, SWEEP_STABLE = {}, [], {}
cellA = (np.diff(SWEEP_RATIOS).min() if SWEEP_RATIOS.size > 1 else 0.005) * (np.diff(SWEEP_EPS).min() if SWEEP_EPS.size > 1 else 0.25)
for li, dphi in enumerate(SWEEP_DPHI):
    cp = ckpt(f"partI_lag{li:02d}.npz")
    if os.path.exists(cp):
        d_ = dict(np.load(cp, allow_pickle=True))
    else:
        t0 = time.time()
        seeds = seeds_for(dphi)
        R_, E_ = np.meshgrid(SWEEP_RATIOS, SWEEP_EPS, indexing="ij")
        R_, E_ = R_.ravel(), E_.ravel()
        ns = seeds.shape[0]
        if ns == 0:
            d_ = dict(S=np.zeros((0, 6)), P=np.zeros((0, 4)), stable=np.zeros(0, bool), n_seeds=0)
        else:
            S = np.tile(seeds, (R_.size, 1))
            Pm = np.stack([np.repeat(R_, ns), np.repeat(E_, ns), np.full(R_.size * ns, dphi), np.ones(R_.size * ns)], 1)
            Sn, res, M, _ = batched(J("full"), S, Pm, label=f"lag {dphi:.3f} grid Newton")
            ok = (res < 1e-7) & valid(Sn)
            Sn, Pm, M = Sn[ok], Pm[ok], M[ok]
            phz = np.arctan2(Sn[:, 5], Sn[:, 4])
            key = np.stack([np.round(Pm[:, 0], 6), np.round(Pm[:, 1], 6), np.round(Sn[:, 0] / 0.01), np.round(phz / 0.1)], 1)
            _, first = np.unique(key, axis=0, return_index=True)
            Sn, Pm, M = Sn[first], Pm[first], M[first]
            amax = np.abs(np.linalg.eigvals(M)).max(1)
            stable = amax < 1 - 1e-6
            d_ = dict(S=Sn, P=Pm, stable=stable, amax=amax, n_seeds=ns)
            ist = np.where(stable)[0]
            if ist.size > MAX_CLASSIFY_LAG:
                ist = np.sort(rng.choice(ist, MAX_CLASSIFY_LAG, replace=False))
            if ist.size:
                cls = classify(Sn[ist], Pm[ist])
                d_.update(cls_idx=ist, cls_ftype=cls["ftype"], cls_index=cls["index"], cls_aL=cls["aL"],
                          cls_dI=cls["dI"], cls_reached=cls["reached"])
        np.savez_compressed(cp, **d_)
        print(f"   lag {dphi:.3f}: {int(d_['n_seeds'])} seeds, {d_['S'].shape[0]} orbits, {int(np.sum(d_['stable']))} stable "
              f"({time.time() - t0:.0f} s)")
    st = d_["stable"].astype(bool)
    Pm = d_["P"]
    cells = {(round(a, 6), round(b, 6)) for a, b in zip(Pm[st, 0], Pm[st, 1])} if st.any() else set()
    rec = dict(dphi=float(dphi), n_orbits=int(st.size), n_stable=int(st.sum()), stable_cells=len(cells),
               stable_area=float(len(cells) * cellA))
    if "cls_ftype" in d_:
        ftl = d_["cls_ftype"].astype(str); ixl = d_["cls_index"]
        ev = np.isfinite(ixl)
        aLl = d_["cls_aL"]
        rec.update(classified=int(ev.sum()), node=int((ftl == "node").sum()), saddle=int((ftl == "saddle-real").sum()),
                   unlocked=int((d_["cls_reached"].astype(bool) & (ftl == "none")).sum()),
                   fold=int((~d_["cls_reached"].astype(bool)).sum()),
                   violations=int((ev & (ixl < 0)).sum()),
                   violations_nonmarginal=int((ev & (ixl < 0) & (np.abs(aLl) >= AL_MARGIN)).sum()),
                   x_range=[float(d_["S"][st, 0].min()), float(d_["S"][st, 0].max())])
        SWEEP_STABLE[float(dphi)] = (d_["S"][d_["cls_idx"].astype(int)], Pm[d_["cls_idx"].astype(int)], ftl)
    I_out[f"{dphi:.4f}"] = rec
    I_rows.append([f"{dphi / np.pi:.3f} pi", rec["n_stable"], rec["stable_cells"], fmt(rec["stable_area"]),
                   rec.get("classified"), rec.get("node"), rec.get("saddle"), rec.get("unlocked"), rec.get("fold"),
                   rec.get("violations_nonmarginal")])
    print(f"  lag {dphi / np.pi:.3f} pi: stable orbits {rec['n_stable']}, cells {rec['stable_cells']}, area {rec['stable_area']:.4f}"
          + (f", classified {rec['classified']} (node {rec['node']}, saddle {rec['saddle']}), violations {rec['violations']}"
             if "classified" in rec else ""))
write_table("T_phase_lag_sweep", ["phase lag", "stable orbits", "stable cells", "area", "classified", "node", "saddle",
                                  "unlocked", "fold in eta", "violations"], I_rows,
            "Stable station holding and the index test across neural phase lags; grid of omega_c/omega_w in steps of 0.005 "
            "and eps in steps of 0.25.")
GI = all(v_.get("violations_nonmarginal", 0) == 0 for v_ in I_out.values())
SUMMARY["I_phase_lag_sweep"] = I_out
SUMMARY["gate_GI"] = GI
if I_out:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    lags = np.array([v_["dphi"] for v_ in I_out.values()])
    ar = np.array([v_["stable_area"] for v_ in I_out.values()])
    ax.plot(lags / np.pi, ar, "o-", color="#252525", ms=4, lw=1)
    ax.set_xlabel(r"$\Delta\phi/\pi$"); ax.set_ylabel("stable holding area")
    ax.set_xlim(0, 2); ax.set_ylim(0, max(ar.max() * 1.1, 1e-3))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(True)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"phase_lag_sweep{Q}.png"), dpi=300); plt.close(fig)

# =============================================================================
# Starting orbits for G and H: Script 10 classified points, else part I orbits
# =============================================================================
def start_groups():
    out = []
    for dp, ft_ in ((0.0, "saddle-real"), (np.pi / 2, "saddle-real"), (np.pi / 2, "node")):
        if HAVE10:
            m = np.where((np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft_))[0]
            if m.size:
                S_ = S_CL[m]; P_ = np.stack([PTS[SEL[m], 4], PTS[SEL[m], 2], PTS[SEL[m], 3], np.ones(m.size)], 1)
                out.append((dp, ft_, S_, P_)); continue
        best = min(SWEEP_STABLE, key=lambda v: abs(v - dp)) if SWEEP_STABLE else None
        if best is not None and abs(best - dp) < 1e-6:
            S_, P_, ftl = SWEEP_STABLE[best]
            m = np.where(ftl == ft_)[0]
            if m.size:
                out.append((dp, ft_, S_[m], P_[m]))
    return out
GROUPS_GH = start_groups()

# =============================================================================
# G. Confinement
# =============================================================================
print("\n=== G. Confinement ===")
G_out, G_rows = {}, []
if GROUPS_GH:
    if os.path.exists(ckpt("partG_confinement.npz")):
        G_ = dict(np.load(ckpt("partG_confinement.npz"), allow_pickle=True))
        print("G: loaded from checkpoint")
    else:
        t0 = time.time()
        rowsS, rowsP, rowsNZ, gi, li, pi_, xref = [], [], [], [], [], [], []
        for g_, (dp, ft_, S_, P_) in enumerate(GROUPS_GH):
            pick = np.sort(rng.choice(S_.shape[0], min(N_PTS_G, S_.shape[0]), replace=False))
            for lv, D in enumerate(G_NOISE):
                for q_ in pick:
                    rowsS.append(np.repeat(S_[q_][None], N_REP_G, 0)); rowsP.append(np.repeat(P_[q_][None], N_REP_G, 0))
                    rowsNZ.append(np.repeat(noise_vec(D)[None], N_REP_G, 0))
                    gi += [g_] * N_REP_G; li += [lv] * N_REP_G; pi_ += [int(q_)] * N_REP_G; xref += [float(S_[q_][0])] * N_REP_G
        S0 = np.concatenate(rowsS); P0 = np.concatenate(rowsP); NZ = np.concatenate(rowsNZ)
        o = batched(make_sim(T_G, SPP, True), S0, P0, NZ, keys_for(S0.shape[0], 41), label="G bouts with strobe")
        # random starts at the same points, deterministic
        rS, rP, rg, rq, rx = [], [], [], [], []
        for g_, (dp, ft_, S_, P_) in enumerate(GROUPS_GH):
            pick = np.sort(rng.choice(S_.shape[0], min(N_PTS_G, S_.shape[0]), replace=False))
            for q_ in pick:
                s0 = np.zeros((N_IC_G, 6)); s0[:, 0] = rng.uniform(0.3, 1.5, N_IC_G); s0[:, 2] = rng.normal(0, 0.1, N_IC_G)
                ph0 = rng.uniform(0, 2 * np.pi, N_IC_G); s0[:, 4], s0[:, 5] = np.cos(ph0), np.sin(ph0)
                rS.append(s0); rP.append(np.repeat(P_[q_][None], N_IC_G, 0)); rg += [g_] * N_IC_G; rq += [int(q_)] * N_IC_G
                rx += [float(S_[q_][0])] * N_IC_G
        R0 = np.concatenate(rS); RP = np.concatenate(rP)
        o2 = batched(make_sim(T_IC_G, SPP, True), R0, RP, np.zeros((R0.shape[0], 3)), keys_for(R0.shape[0], 42), label="G random starts")
        G_ = dict(g=np.array(gi), lv=np.array(li), q=np.array(pi_), xref=np.array(xref), alive=o[1], tloss=o[2], xs=o[5].astype(np.float32),
                  r_g=np.array(rg), r_q=np.array(rq), r_xref=np.array(rx), r_alive=o2[1], r_xs=o2[5].astype(np.float32))
        np.savez_compressed(ckpt("partG_confinement.npz"), **G_)
        print(f"G: done ({time.time() - t0:.0f} s)")
    xs = G_["xs"].astype(float)
    for g_, (dp, ft_, _, _) in enumerate(GROUPS_GH):
        for lv, D in enumerate(G_NOISE):
            m = (G_["g"] == g_) & (G_["lv"] == lv)
            if not m.any():
                continue
            ev_ = ~G_["alive"][m].astype(bool)
            tt = np.where(ev_, G_["tloss"][m], T_G)
            res_med = km(tt, ev_, T_G)[0]
            dev = np.abs(xs[m] - G_["xref"][m][:, None])
            rec = dict(runs=int(m.sum()), residence_km_median=None if not np.isfinite(res_med) else res_med)
            row = [f"{dp:.3f}", ft_.split("-")[0], D, int(m.sum()), ">%d" % T_G if not np.isfinite(res_med) else f"{res_med:.0f}"]
            for band in CONF_BANDS:
                out_ = np.where(np.nan_to_num(dev, nan=np.inf) > band, 1, 0)
                first = np.where(out_.any(1), out_.argmax(1), T_G).astype(float)
                evb = out_.any(1)
                medb = km(first, evb, T_G)[0]
                rec[f"confined_{band}_km_median"] = None if not np.isfinite(medb) else medb
                row.append(">%d" % T_G if not np.isfinite(medb) else f"{medb:.0f}")
            alive_dev = np.where(np.isfinite(xs[m]), dev, np.nan)
            rec["rms_deviation_while_holding"] = float(np.sqrt(np.nanmean(alive_dev ** 2)))
            rec["p95_deviation_while_holding"] = float(np.nanquantile(alive_dev, 0.95))
            row += [f"{rec['rms_deviation_while_holding']:.3f}", f"{rec['p95_deviation_while_holding']:.3f}"]
            G_out[f"{dp:.3f}/{ft_}/D={D}"] = rec
            G_rows.append(row)
        # random starts: of runs that hold, how many end localized near the station orbit
        mr = G_["r_g"] == g_
        if mr.any():
            al = G_["r_alive"][mr].astype(bool)
            tail = G_["r_xs"][mr][:, -100:].astype(float)
            span = np.nanmax(tail, 1) - np.nanmin(tail, 1)
            near = np.abs(np.nanmean(tail, 1) - G_["r_xref"][mr]) < 0.1
            G_out[f"{dp:.3f}/{ft_}/random_starts"] = dict(
                runs=int(mr.sum()), holding=int(al.sum()),
                holding_span_below_0p05=int((al & (span < 0.05)).sum()),
                holding_near_station=int((al & near).sum()),
                holding_span_median=float(np.nanmedian(span[al])) if al.any() else None)
    hdr = ["phase lag", "branch", "D", "runs", "residence median"] + [f"confined +/-{b} median" for b in CONF_BANDS] + \
          ["RMS deviation", "95th pct deviation"]
    write_table("T_confinement", hdr, G_rows, "Residence bouts compared with confinement bouts, defined as the time until the "
                "strobed position first leaves a band around the station-holding orbit; deviations in wake wavelengths while holding.")
    for r_ in G_rows:
        print("  " + " | ".join(str(v) for v in r_))
    for k_, v_ in G_out.items():
        if k_.endswith("random_starts"):
            print(f"  {k_}: {v_}")
else:
    print("  no starting orbits available: skipped")
SUMMARY["G_confinement"] = G_out

# =============================================================================
# H. Stochastic time-step check
# =============================================================================
print("\n=== H. Stochastic time step ===")
H_out = {}
if GROUPS_GH:
    if os.path.exists(ckpt("partH_timestep.npz")):
        H_ = dict(np.load(ckpt("partH_timestep.npz"), allow_pickle=True))
        print("H: loaded from checkpoint")
    else:
        t0 = time.time()
        rowsS, rowsP, gi = [], [], []
        for g_, (dp, ft_, S_, P_) in enumerate(GROUPS_GH):
            pick = np.sort(rng.choice(S_.shape[0], min(N_PTS_H, S_.shape[0]), replace=False))
            for q_ in pick:
                rowsS.append(np.repeat(S_[q_][None], N_REP_H, 0)); rowsP.append(np.repeat(P_[q_][None], N_REP_H, 0))
                gi += [g_] * N_REP_H
        S0 = np.concatenate(rowsS); P0 = np.concatenate(rowsP); NZ = np.repeat(noise_vec(D_H)[None], S0.shape[0], 0)
        H_ = dict(g=np.array(gi))
        for spp2 in (SPP, 2 * SPP):
            o = batched(make_sim(T_H, spp2, False), S0, P0, NZ, keys_for(S0.shape[0], 50 + spp2), label=f"H {spp2} steps")
            H_[f"alive_{spp2}"] = o[1]; H_[f"tloss_{spp2}"] = o[2]; H_[f"phase_{spp2}"] = o[3]; H_[f"nacc_{spp2}"] = o[4]
        np.savez_compressed(ckpt("partH_timestep.npz"), **H_)
        print(f"H: done ({time.time() - t0:.0f} s)")
    H_rows = []
    GH = True
    for g_, (dp, ft_, _, _) in enumerate(GROUPS_GH):
        m = H_["g"] == g_
        rec = {}
        for spp2 in (SPP, 2 * SPP):
            ev_ = ~H_[f"alive_{spp2}"][m].astype(bool)
            med = km(np.where(ev_, H_[f"tloss_{spp2}"][m], T_H), ev_, T_H)[0]
            okp = H_[f"nacc_{spp2}"][m] > 0
            rec[str(spp2)] = dict(km_median=None if not np.isfinite(med) else med, censored=float((~ev_).mean()),
                                  phase_mean=cmean(H_[f"phase_{spp2}"][m][okp]))
        a_, b_ = rec[str(SPP)]["km_median"], rec[str(2 * SPP)]["km_median"]
        if a_ is None and b_ is None:
            rel = 0.0          # both medians beyond the horizon: no measurable change
        else:
            rel = None if (a_ is None or b_ is None) else abs(b_ - a_) / a_
        rec["relative_median_change"] = rel
        rec["phase_mean_change"] = cdiff(rec[str(2 * SPP)]["phase_mean"], rec[str(SPP)]["phase_mean"])
        GH &= (rel is not None and rel <= 0.15)
        H_out[f"{dp:.3f}/{ft_}"] = rec
        H_rows.append([f"{dp:.3f}", ft_.split("-")[0], fmt(a_, 0) if a_ else ">%d" % T_H, fmt(b_, 0) if b_ else ">%d" % T_H,
                       None if rel is None else f"{100 * rel:.1f}%", f"{rec['phase_mean_change']:.3f}"])
        print(f"  {dp:.3f} {ft_:12s} median bout {a_} ({SPP} steps) vs {b_} ({2 * SPP} steps), "
              f"change {None if rel is None else f'{100 * rel:.1f}%'}, phase mean change {rec['phase_mean_change']:.3f} rad")
    write_table("T_stochastic_timestep", ["phase lag", "branch", f"median, {SPP} steps", f"median, {2 * SPP} steps",
                                          "relative change", "phase mean change (rad)"], H_rows,
                f"Stochastic time-step check with all three noise sources at D = {D_H}.")
    SUMMARY["gate_GH"] = GH
else:
    GH = None
SUMMARY["H_timestep"] = H_out

# =============================================================================
# Save and gates
# =============================================================================
SUMMARY["runtime_this_session_s"] = round(time.time() - t_start, 1)
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
with open(os.path.join(OUT_DIR, f"summary_script14{Q}.json"), "w") as f:
    json.dump(clean(SUMMARY), f, indent=2)

print("\n================ SCRIPT 14 GATES ================")
print(f"GA exclusive areas sum to the total and match Script 10: {GA}")
print(f"GC low-margin reclassification changes only below margin 1e-2: {GC}")
print(f"GI no non-marginal index violations at any phase lag: {GI}")
print(f"GH median bouts at {SPP} and {2 * SPP} steps per period within 15%: {GH}")
print(f"\nOutputs: {OUT_DIR} (summary_script14{Q}.json, tables/, figures/)")
print(f"Runtime this session: {time.time() - t_start:.0f} s")
