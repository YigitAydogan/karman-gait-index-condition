# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 10: Continuation and bifurcations of station holding (baseline model).
#
# Colab ready. Uses JAX (preinstalled on Colab) and runs on the GPU when one is
# attached (L4: about 20 to 40 min FULL); falls back to CPU (much slower).
# Every stage is checkpointed to Drive; if Colab disconnects, run again and
# finished stages are loaded instead of recomputed. No settings to change.
#
# Why JAX: Scripts 01 to 09 integrate thousands of orbits in parallel with
# numpy, which uses one CPU core and leaves the GPU idle. Here the period map
# is compiled once (jit), vectorised over orbits (vmap) and differentiated
# exactly (jacfwd): monodromy matrices are exact up to RK4 error instead of
# finite differences, and the whole batch runs on the GPU.
#
# Gaps closed (manuscript plan): 4 (region areas instead of orbit fractions),
# 9 (continuation and bifurcation analysis), 11 (where no-lock orbits sit).
#
# Stages:
#   A. Seeds: positional equilibria of the prescribed rhythm (4 neural delays).
#   B. Grid: Newton in the full 6D system on a dense (omega_c/omega_w, eps)
#      grid from every seed; all station-holding orbits with exact multipliers.
#   C. Pseudo-arclength continuation in omega_c/omega_w at every eps row
#      (step 0.125) from the stable grid orbits, both directions, adaptive
#      step; repeated passes until every stable grid orbit lies on a continued
#      branch. Branches stop at the parameter range, x outside [0.25, 2.5],
#      or on closing into a loop.
#      Along every branch: Floquet multipliers, folds (d ratio / ds = 0),
#      loss of stability by fold (+1), flip (-1) or torus (complex pair).
#   D. Branch identity of every stable continuation point by homotopy (25 eta
#      values from 1 to 0.005), then det(I - M_int), a_L and the index. a_L is
#      fitted over x_L +/- 0.005 (fallbacks 0.0015, 0.02): it is a local
#      derivative, and wide offsets leave the fixed-position tongue near its
#      edge. Suspect points (index < 0, or a_L sign depending on the offset
#      width) are recomputed at twice the steps per period.
#   E. Two-parameter picture: stable region in (ratio, eps) per delay, its
#      area split by fixed-position branch (node, saddle, no lock), fold and
#      torus points bounding it.
#
# Gates:
#   G1  >= 95% of Script 05b stable orbits lie inside a continued stable
#       segment at the same eps and delay (if 05b results are on Drive)
#   G2  >= 90% of stable-segment ends are bifurcations (fold, flip, torus),
#       not numerical terminations
#   G3  0 non-marginal index violations on continued stable points that
#       survive the 2x-steps recheck
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
SEED = 14
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script10_continuation")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
for d_ in (FIG_DIR, CKPT_DIR):
    os.makedirs(d_, exist_ok=True)
F05B = os.path.join(BASE_DIR, "script05b_holding_dense", "results_script05.npz")

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
if MODE == "FULL":
    SPP = 100
    N_PHI_SEED, N_X_SEED = 16, 60
    DPHI_SET = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    EPS_ROWS = np.round(np.arange(0.125, 6.001, 0.125), 6)
    GRID_RATIOS = np.round(np.linspace(0.95, 1.25, 121), 6)   # step 0.0025
    START_EVERY = 10              # first pass starts from every 10th grid ratio
    N_STEPS = 400
    CHUNK = 16384 if ON_GPU else 1024
else:
    SPP = 60
    N_PHI_SEED, N_X_SEED = 6, 24
    DPHI_SET = [0.0]
    EPS_ROWS = np.array([1.0, 2.0])
    GRID_RATIOS = np.round(np.linspace(1.00, 1.10, 6), 6)
    START_EVERY = 3
    N_STEPS = 60
    CHUNK = 256
RATIO_LIM = (0.90, 1.35)
X_LIM = (0.25, 2.5)
H0, HMIN, HMAX = 0.02, 2e-4, 0.15
N_PASSES = 4
COVER_TOL = (0.004, 0.03, 0.15)          # ratio, x, CPG phase
N_CORR = 6
ETAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
OFF = np.linspace(-0.02, 0.02, 7)
AL_SCALES = [0.005, 0.0015, 0.02]          # half-widths of the x offsets for a_L (first that works is used)
ETAS_FINE = list(np.geomspace(1.0, 0.005, 25))
AL_MARGIN = 0.02
MAX_CLASSIFY = 60000 if MODE == "FULL" else 200
AREA_BIN = 0.0025
OMEGA_W = 2.0 * np.pi
# continuation scaling: (x, u, theta, omega, zr, zi, lambda)
WSCALE = np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 10.0])

# -----------------------------------------------------------------------------
# 1. Baseline model (Scripts 05b to 08) in JAX
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA = 3.0, 0.15, 8.0
OM_B, ZE_B = 1.2 * OMEGA_W, 0.2
MU, C_SL, G_HOLD = 2 * np.pi, 0.0, 16.0
DT = 1.0 / SPP

def forces(t, x, u, th):
    V = U_INF * (1.0 - DELTA0 * jnp.exp(-x / L_DELTA)) - u
    w = W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * jnp.sin(OMEGA_W * (x - t))
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
    return L, Fx

def rhs(t, s, p):
    """p = (ratio, eps, dphi, eta). eta scales positional dynamics (0 = frozen)."""
    x, u, th, om, zr, zi = s
    ratio, eps, dphi, eta = p
    L, Fx = forces(t, x, u, th)
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    dom = -2 * ZE_B * OM_B * om - OM_B ** 2 * th + SIGMA * L + G_HOLD * drive
    r2 = zr * zr + zi * zi
    a = MU * (1.0 - r2)
    b = C_SL * MU * (1.0 - r2) + ratio * OMEGA_W
    return jnp.stack([eta * u, eta * Fx, om, dom, a * zr - b * zi + eps * th, a * zi + b * zr]), Fx

def period_map(s, p, spp=SPP):
    dt = 1.0 / spp
    def step(i, c):
        s_, F_ = c
        t = i * dt
        k1, f1 = rhs(t, s_, p)
        k2, f2 = rhs(t + dt / 2, s_ + dt / 2 * k1, p)
        k3, f3 = rhs(t + dt / 2, s_ + dt / 2 * k2, p)
        k4, f4 = rhs(t + dt, s_ + dt * k3, p)
        return s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F_ + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return jax.lax.fori_loop(0, spp, step, (s, 0.0))

def presc_rhs(t, s, q):
    """Prescribed rhythm at frozen position. q = (phi0, dphi)."""
    x, u, th, om = s
    L, Fx = forces(t, x, u, th)
    drive = jnp.cos(OMEGA_W * t + q[0] - q[1])
    return jnp.stack([0.0 * u, 0.0 * Fx, om, -2 * ZE_B * OM_B * om - OM_B ** 2 * th + SIGMA * L + G_HOLD * drive]), Fx

def presc_run(s, q, n_per, n_mean):
    def step(i, c):
        s_, F_ = c
        t = (i % SPP) * DT
        k1, f1 = presc_rhs(t, s_, q)
        k2, f2 = presc_rhs(t + DT / 2, s_ + DT / 2 * k1, q)
        k3, f3 = presc_rhs(t + DT / 2, s_ + DT / 2 * k2, q)
        k4, f4 = presc_rhs(t + DT, s_ + DT * k3, q)
        w = jnp.where(i >= (n_per - n_mean) * SPP, 1.0, 0.0)
        return s_ + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F_ + w * DT / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    s1, F = jax.lax.fori_loop(0, n_per * SPP, step, (s, 0.0))
    return s1, F / n_mean

def newton(s, p, mask, n_it, clip, spp=SPP):
    """Newton for P(s) = s on the dims where mask = 1 (others held fixed)."""
    I6 = jnp.eye(6)
    def body(i, c):
        s_, _ = c
        s1, _ = period_map(s_, p, spp)
        M = jax.jacfwd(lambda y: period_map(y, p, spp)[0])(s_)
        r = (s1 - s_) * mask
        A = (M - I6) * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask)
        d = jnp.linalg.solve(A, -r)
        d = jnp.clip(jnp.nan_to_num(d), -clip, clip) * mask
        return s_ + d, jnp.max(jnp.abs(r))
    s, _ = jax.lax.fori_loop(0, n_it, body, (s, 1.0))
    s1, F = period_map(s, p, spp)
    M = jax.jacfwd(lambda y: period_map(y, p, spp)[0])(s)
    res = jnp.max(jnp.abs((s1 - s) * mask))
    return s, res, M, F

MASK_ALL = jnp.ones(6)
MASK_INT = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

def _newton_full(s, p): return newton(s, p, MASK_ALL, 10, 0.05)
def _newton_hom(s, p): return newton(s, p, MASK_ALL, 8, 0.05)
def _newton_int(s, p): return newton(s, p, MASK_INT, 12, 0.2)

def G_of(y, p, k):
    """Continuation residual in scaled coordinates y = z * WSCALE, z = (s, lambda)."""
    z = y / WSCALE
    p2 = p.at[k].set(z[6])
    s1, _ = period_map(z[:6], p2)
    return s1 - z[:6]

def tangent(y, p, k, t_old):
    J = jax.jacfwd(lambda yy: G_of(yy, p, k))(y)          # 6 x 7
    A = jnp.vstack([J, t_old[None, :]])
    t = jnp.linalg.solve(A, jnp.eye(7)[6])
    t = t / jnp.linalg.norm(t)
    return t * jnp.sign(jnp.dot(t, t_old) + 1e-300)

def corrector(y0, t0, h, p, k):
    yp = y0 + h * t0
    def body(i, c):
        y, n_ok, done = c
        F = G_of(y, p, k)
        J = jax.jacfwd(lambda yy: G_of(yy, p, k))(y)
        R = jnp.concatenate([F, jnp.array([jnp.dot(t0, y - yp)])])
        A = jnp.vstack([J, t0[None, :]])
        d = jnp.clip(jnp.nan_to_num(jnp.linalg.solve(A, -R)), -0.05, 0.05)
        conv = jnp.max(jnp.abs(F)) < 1e-9
        y = jnp.where(done, y, y + d)
        n_ok = jnp.where(done, n_ok, n_ok + 1)
        return y, n_ok, done | conv
    y, n_it, done = jax.lax.fori_loop(0, N_CORR, body, (yp, 0, False))
    res = jnp.max(jnp.abs(G_of(y, p, k)))
    t1 = tangent(y, p, k, t0)
    z = y / WSCALE
    M = jax.jacfwd(lambda s: period_map(s, p.at[k].set(z[6]))[0])(z[:6])
    return y, t1, res, n_it, M

def _init_tangent(y, p, k, sgn):
    t_guess = jnp.zeros(7).at[6].set(sgn)
    return tangent(y, p, k, t_guess)

# jit + vmap, padded to fixed chunk sizes (one compilation per function)
J_PRESC = jax.jit(jax.vmap(lambda s, q: presc_run(s, q, 50, 10)))
J_PRESC_SETTLE = jax.jit(jax.vmap(lambda s, q: presc_run(s, q, 40, 1)))
J_NFULL = jax.jit(jax.vmap(_newton_full))
J_NHOM = jax.jit(jax.vmap(_newton_hom))
J_NINT = jax.jit(jax.vmap(_newton_int))
J_CORR = jax.jit(jax.vmap(lambda y, t, h, p: corrector(y, t, h, p, 0)))
J_TAN0 = jax.jit(jax.vmap(lambda y, p, sg: _init_tangent(y, p, 0, sg)))
SPP_CHECK = 2 * SPP              # time-step convergence check of the index on suspect points
J_NFULL2 = jax.jit(jax.vmap(lambda s, p: newton(s, p, MASK_ALL, 10, 0.05, SPP_CHECK)))
J_NHOM2 = jax.jit(jax.vmap(lambda s, p: newton(s, p, MASK_ALL, 8, 0.05, SPP_CHECK)))
J_NINT2 = jax.jit(jax.vmap(lambda s, p: newton(s, p, MASK_INT, 12, 0.2, SPP_CHECK)))

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

def lead_info(M):
    mu = np.linalg.eigvals(M)
    a = np.abs(mu)
    i = np.argmax(a, axis=1)
    lead = mu[np.arange(len(mu)), i]
    typ = np.where(a.max(1) < 1 - 1e-6, "stable",
                   np.where(np.abs(lead.imag) > 1e-6, "torus", np.where(lead.real < 0, "flip", "fold")))
    return a.max(1), lead, typ

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

t_start = time.time()
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

# -----------------------------------------------------------------------------
# A + B. Seeds and grid
# -----------------------------------------------------------------------------
if os.path.exists(ckpt("stageB_grid.npz")):
    d = np.load(ckpt("stageB_grid.npz"))
    GS, GP, GSTAB = d["S"], d["P"], d["stable"]
    print(f"A+B: loaded {GS.shape[0]} grid orbits from checkpoint")
else:
    t0 = time.time()
    PHIS = np.linspace(0, 2 * np.pi, N_PHI_SEED, endpoint=False)
    XG = np.linspace(0.3, 2.2, N_X_SEED)
    combos = np.array([(d_, ph) for d_ in DPHI_SET for ph in PHIS])
    nc = len(combos)
    S0 = np.zeros((nc * XG.size, 4)); S0[:, 0] = np.tile(XG, nc)
    Q0 = np.stack([np.repeat(combos[:, 1], XG.size), np.repeat(combos[:, 0], XG.size)], 1)
    _, Fp = batched(J_PRESC, S0, Q0)
    Fp = Fp.reshape(nc, XG.size)
    sx, sph, sdp = [], [], []
    for c in range(nc):
        for j in range(XG.size - 1):
            f0, f1 = Fp[c, j], Fp[c, j + 1]
            if np.isfinite(f0) and np.isfinite(f1) and np.sign(f0) != np.sign(f1):
                sx.append(XG[j] - f0 * (XG[j + 1] - XG[j]) / (f1 - f0))
                sph.append(combos[c, 1]); sdp.append(combos[c, 0])
    sx, sph, sdp = np.array(sx), np.array(sph), np.array(sdp)
    Ss = np.zeros((sx.size, 4)); Ss[:, 0] = sx
    Ss, _ = batched(J_PRESC_SETTLE, Ss, np.stack([sph, sdp], 1))
    SEEDS = np.concatenate([Ss, np.cos(sph)[:, None], np.sin(sph)[:, None]], 1)
    SEEDS[:, 0] = sx; SEEDS[:, 1] = 0.0
    print(f"A: {sx.size} prescribed seeds ({time.time() - t0:.0f} s)")
    # grid: every seed x every ratio x every eps row
    ns = SEEDS.shape[0]
    R_, E_ = np.meshgrid(GRID_RATIOS, EPS_ROWS, indexing="ij")
    R_, E_ = R_.ravel(), E_.ravel()
    S = np.tile(SEEDS, (R_.size, 1))
    P = np.stack([np.repeat(R_, ns), np.repeat(E_, ns), np.tile(sdp, R_.size), np.ones(R_.size * ns)], 1)
    t0 = time.time()
    Sn, res, M, _ = batched(J_NFULL, S, P, label="grid Newton")
    ok = (res < 1e-7) & valid(Sn)
    Sn, P, M = Sn[ok], P[ok], M[ok]
    # deduplicate per (ratio, eps, dphi)
    ph = np.arctan2(Sn[:, 5], Sn[:, 4])
    key = np.stack([np.round(P[:, 0], 6), np.round(P[:, 1], 6), np.round(P[:, 2], 6),
                    np.round(Sn[:, 0] / 0.01), np.round(ph / 0.1)], 1)
    _, first = np.unique(key, axis=0, return_index=True)
    GS, GP, GM = Sn[first], P[first], M[first]
    amax, _, typ = lead_info(GM)
    GSTAB = typ == "stable"
    np.savez_compressed(ckpt("stageB_grid.npz"), S=GS, P=GP, stable=GSTAB, amax=amax, typ=typ)
    print(f"B: {S.shape[0]} Newton problems -> {GS.shape[0]} distinct orbits, {GSTAB.sum()} stable "
          f"({time.time() - t0:.0f} s)")

# -----------------------------------------------------------------------------
# C. Pseudo-arclength continuation in omega_c / omega_w
# -----------------------------------------------------------------------------
def continue_branches(S0, P0):
    """Pseudo-arclength continuation in ratio from each (S0, P0), both directions."""
    n0 = S0.shape[0]
    S0 = np.concatenate([S0, S0]); P0 = np.concatenate([P0, P0])
    sgn = np.concatenate([np.ones(n0), -np.ones(n0)])
    nb = S0.shape[0]
    Y = np.concatenate([S0, P0[:, :1]], 1) * WSCALE
    T = batched(J_TAN0, Y, P0, sgn)
    H = np.full(nb, H0)
    alive = np.ones(nb, bool)
    end_reason = np.array(["max-steps"] * nb, dtype=object)
    Mfirst = batched(J_NFULL, S0, P0)[2]
    a0, l0, _ = lead_info(Mfirst)
    traces = [[(Y[i] / WSCALE, T[i][6], a0[i], l0[i])] for i in range(nb)]
    start_pts = Y.copy()
    t0 = time.time()
    for step in range(N_STEPS):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        Yn, Tn, res, nit, M = batched(J_CORR, Y[idx], T[idx], H[idx], P0[idx])
        z = Yn / WSCALE
        ok = (res < 1e-8) & np.isfinite(Yn).all(1) & (np.einsum("ij,ij->i", Tn, T[idx]) > 0.9)
        ok &= np.abs(np.hypot(z[:, 4], z[:, 5]) - 1) < 0.5
        acc = idx[ok]; rej = idx[~ok]
        Y[acc] = Yn[ok]; T[acc] = Tn[ok]
        H[acc] = np.where(nit[ok] <= 3, np.minimum(H[acc] * 1.5, HMAX), H[acc])
        H[rej] *= 0.5
        if ok.any():
            am, ld, _ = lead_info(M[ok])
            for q, (j, i) in enumerate(zip(np.where(ok)[0], acc)):
                traces[i].append((z[j], Tn[j][6], am[q], ld[q]))
        zz = Y / WSCALE
        out = alive & ((zz[:, 6] < RATIO_LIM[0]) | (zz[:, 6] > RATIO_LIM[1]))
        end_reason[out] = "parameter-limit"; alive &= ~out
        out = alive & ((zz[:, 0] < X_LIM[0]) | (zz[:, 0] > X_LIM[1]))
        end_reason[out] = "x-limit"; alive &= ~out
        small = alive & (H < HMIN)
        end_reason[small] = "step-collapse"; alive &= ~small
        back = alive & (np.array([len(tr) for tr in traces]) > 30) & \
               (np.linalg.norm(Y - start_pts, axis=1) < 3 * H)
        end_reason[back] = "closed-loop"; alive &= ~back
        if step % 50 == 0:
            print(f"     step {step}: {alive.sum()}/{nb} half-branches alive ({time.time() - t0:.0f} s)")
    out = []
    for i in range(nb):
        out.append(dict(eps=float(P0[i, 1]), dphi=float(P0[i, 2]),
                        Z=np.array([tr[0] for tr in traces[i]]), tl=np.array([tr[1] for tr in traces[i]]),
                        amax=np.array([tr[2] for tr in traces[i]]), lead=np.array([tr[3] for tr in traces[i]]),
                        end=str(end_reason[i])))
    return out

def uncovered(S, P, branches):
    """Grid orbits not lying on any continued branch (same eps and delay)."""
    by = {}
    for b in branches:
        by.setdefault((round(b["eps"], 6), round(b["dphi"], 6)), []).append(b["Z"])
    unc = np.ones(S.shape[0], bool)
    for i in range(S.shape[0]):
        Zs = by.get((round(P[i, 1], 6), round(P[i, 2], 6)))
        if not Zs:
            continue
        Z = np.concatenate(Zs)
        ph = np.arctan2(S[i, 5], S[i, 4])
        dph = np.abs(np.angle(np.exp(1j * (np.arctan2(Z[:, 5], Z[:, 4]) - ph))))
        m = (np.abs(Z[:, 6] - P[i, 0]) < COVER_TOL[0]) & (np.abs(Z[:, 0] - S[i, 0]) < COVER_TOL[1]) & (dph < COVER_TOL[2])
        unc[i] = not m.any()
    return unc

if os.path.exists(ckpt("stageC_branches.npz")):
    d = np.load(ckpt("stageC_branches.npz"), allow_pickle=True)
    BR = d["branches"].tolist()
    print(f"C: loaded {len(BR)} half-branches from checkpoint")
else:
    t0 = time.time()
    SS, SPs = GS[GSTAB], GP[GSTAB]                 # branches through every stable grid orbit
    pick = np.isin(np.round(SPs[:, 0], 6), np.round(GRID_RATIOS[::START_EVERY], 6))
    BR = []
    todo = pick
    for pss in range(N_PASSES):
        if not todo.any():
            break
        print(f"   C pass {pss + 1}: continuing from {int(todo.sum())} stable grid orbits")
        BR += continue_branches(SS[todo], SPs[todo])
        todo = uncovered(SS, SPs, BR)
    n_unc = int(uncovered(SS, SPs, BR).sum()) if SS.shape[0] else 0
    np.savez_compressed(ckpt("stageC_branches.npz"), branches=np.array(BR, dtype=object), n_uncovered=n_unc)
    print(f"C: {len(BR)} half-branches, {sum(len(b['Z']) for b in BR)} points, "
          f"{n_unc} of {SS.shape[0]} stable grid orbits not on a branch ({time.time() - t0:.0f} s)")

# bifurcations along branches
points, bifs, seg_ends = [], [], []
for bi, b in enumerate(BR):
    Z, tl, amax, lead = b["Z"], b["tl"], b["amax"], b["lead"]
    if len(Z) < 2:
        continue
    typ = np.where(amax < 1 - 1e-6, "stable", np.where(np.abs(lead.imag) > 1e-6, "torus",
                   np.where(lead.real < 0, "flip", "fold")))
    st = typ == "stable"
    for j in range(len(Z)):
        points.append((bi, j, b["eps"], b["dphi"], Z[j, 6], Z[j, 0], np.arctan2(Z[j, 5], Z[j, 4]), st[j], amax[j]))
    for j in range(1, len(Z)):
        if np.sign(tl[j]) != np.sign(tl[j - 1]):
            bifs.append(dict(kind="fold", eps=b["eps"], dphi=b["dphi"], ratio=float(0.5 * (Z[j, 6] + Z[j - 1, 6])),
                             x=float(0.5 * (Z[j, 0] + Z[j - 1, 0])), branch=bi))
        if st[j] != st[j - 1]:
            u_ = j if not st[j] else j - 1
            bifs.append(dict(kind="loss-" + str(typ[u_]), eps=b["eps"], dphi=b["dphi"],
                             ratio=float(0.5 * (Z[j, 6] + Z[j - 1, 6])), x=float(0.5 * (Z[j, 0] + Z[j - 1, 0])),
                             branch=bi, lead_re=float(lead[u_].real), lead_im=float(lead[u_].imag)))
    # stable segments and their ends
    j = 0
    while j < len(Z):
        if st[j]:
            k = j
            while k + 1 < len(Z) and st[k + 1]:
                k += 1
            for e_, nb_ in ((j, j - 1), (k, k + 1)):
                if 0 <= nb_ < len(Z):
                    seg_ends.append(dict(kind="bifurcation", type=str(typ[nb_])))
                elif nb_ < 0:
                    seg_ends.append(dict(kind="start", type="branch start (other direction continues)"))
                elif b["end"] in ("parameter-limit", "x-limit", "closed-loop"):
                    seg_ends.append(dict(kind="domain", type=b["end"]))
                else:
                    seg_ends.append(dict(kind="termination", type=b["end"]))
            j = k + 1
        else:
            j += 1
PTS = np.array(points, dtype=float) if points else np.zeros((0, 9))
print(f"C: {PTS.shape[0]} continuation points, {int(PTS[:, 7].sum()) if len(PTS) else 0} stable; "
      f"{sum(b_['kind'] == 'fold' for b_ in bifs)} folds, {sum(b_['kind'].startswith('loss') for b_ in bifs)} stability changes")

# -----------------------------------------------------------------------------
# D. Branch identity of stable continuation points (homotopy, index)
# -----------------------------------------------------------------------------
def homotopy(S, P, etas, fn):
    n = S.shape[0]
    s = S.copy()
    reached = np.ones(n, bool)
    fail_eta = np.full(n, np.nan)
    for eta in etas[1:]:
        idx = np.where(reached)[0]
        if idx.size == 0:
            break
        Pe = P[idx].copy(); Pe[:, 3] = eta
        sn, res, _, _ = batched(fn, s[idx], Pe)
        okh = (res < 1e-7) & valid(sn)
        s[idx[okh]] = sn[okh]
        reached[idx[~okh]] = False
        fail_eta[idx[~okh]] = eta
    return s, reached, fail_eta

def fixed_lock(s_end, P, fn):
    Pf = P.copy(); Pf[:, 3] = 0.0
    sf0 = s_end.copy(); sf0[:, 1] = 0.0
    sf, resf, Mf, _ = batched(fn, sf0, Pf)
    ok = resf < 1e-7
    mu = np.linalg.eigvals(Mf[:, 2:, 2:])
    dI = np.real(np.prod(1.0 - mu, axis=1))
    am = np.abs(mu)
    lead = mu[np.arange(len(mu)), np.argmax(am, 1)]
    ftype = np.where(~ok, "none", np.where(am.max(1) < 1, "node",
                     np.where(np.abs(lead.imag) < 1e-6, "saddle-real", "unstable-complex")))
    return sf, ok, dI, ftype

def relocked_gradient(sf, P, half, fn):
    """a_L: linear fit of the relocked mean force over x_L +/- half (7 points, continued outwards)."""
    n = sf.shape[0]
    off = np.linspace(-half, half, 7)
    Pf = P.copy(); Pf[:, 3] = 0.0
    Fo = np.full((n, off.size), np.nan)
    for side in (+1, -1):
        cur = sf.copy()
        for o in sorted([o for o in off if (o >= 0 if side > 0 else o <= 0)], key=abs):
            cur[:, 0] = sf[:, 0] + o
            cn, rq, _, F = batched(fn, cur, Pf)
            cur = np.where((rq < 1e-7)[:, None], cn, cur)
            Fo[:, int(np.argmin(np.abs(off - o)))] = np.where(rq < 1e-7, F, np.nan)
    aL = np.full(n, np.nan); r2 = np.full(n, np.nan)
    for i in range(n):
        m = np.isfinite(Fo[i])
        if m.sum() >= 5:
            c_ = np.polyfit(off[m], Fo[i, m], 1)
            pr = np.polyval(c_, off[m]); ss = np.sum((Fo[i, m] - Fo[i, m].mean()) ** 2)
            aL[i] = c_[0]; r2[i] = 1 - np.sum((Fo[i, m] - pr) ** 2) / ss if ss > 0 else 0.0
    return aL, r2

def stable_keys(rows):
    return np.stack([rows[:, 2], rows[:, 3], np.round(rows[:, 4] / 0.001),
                     np.round(rows[:, 5] / 0.01), np.round(rows[:, 6] / 0.1)], 1)

if os.path.exists(ckpt("stageD2_index.npz")):
    d = np.load(ckpt("stageD2_index.npz"))
    CL = {k: d[k] for k in d.files}
    print(f"D: loaded {CL['sel'].size} classified points (refined) from checkpoint")
else:
    t0 = time.time()
    stab = np.where(PTS[:, 7] > 0)[0] if len(PTS) else np.array([], int)
    if stab.size:
        _, first = np.unique(stable_keys(PTS[stab]), axis=0, return_index=True)
        stab = stab[np.sort(first)]
    if stab.size > MAX_CLASSIFY:
        stab = np.sort(rng.choice(stab, MAX_CLASSIFY, replace=False))
    n = stab.size
    if n == 0:
        raise SystemExit("No stable continuation points: nothing to classify. Check stages B and C.")
    S = np.array([BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6] for i in stab]).reshape(n, 6)
    P = np.stack([PTS[stab, 4], PTS[stab, 2], PTS[stab, 3], np.ones(n)], 1)
    # homotopy with a fine eta ladder (25 values from 1 to 0.005)
    s_end, reached, fail_eta = homotopy(S, P, ETAS_FINE, J_NHOM)
    sf, okf, dI, ftype = fixed_lock(s_end, P, J_NINT)
    okf &= reached
    ftype = np.where(okf, ftype, "none")
    # a_L at three offset widths; the narrowest that gives a fit is used
    aLs, r2s = {}, {}
    for half in AL_SCALES:
        a_, r_ = np.full(n, np.nan), np.full(n, np.nan)
        idx = np.where(okf)[0]
        if idx.size:
            a_[idx], r_[idx] = relocked_gradient(sf[idx], P[idx], half, J_NINT)
        aLs[half], r2s[half] = a_, r_
    aL = np.full(n, np.nan); aL_half = np.full(n, np.nan)
    for half in AL_SCALES:
        m = np.isnan(aL) & np.isfinite(aLs[half])
        aL[m] = aLs[half][m]; aL_half[m] = half
    index = np.sign(dI) * np.sign(-aL)
    CL = dict(sel=stab, ftype=ftype, dI=dI, aL=aL, aL_half=aL_half, index=index, reached=reached,
              fail_eta=fail_eta, x_L=sf[:, 0],
              **{f"aL_{h}": aLs[h] for h in AL_SCALES}, **{f"r2_{h}": r2s[h] for h in AL_SCALES})
    # suspects: index < 0, or a_L sign differs between offset widths -> recheck at twice the steps per period
    sign_mix = np.zeros(n, bool)
    for h1 in AL_SCALES:
        for h2 in AL_SCALES:
            sign_mix |= np.isfinite(aLs[h1]) & np.isfinite(aLs[h2]) & (np.sign(aLs[h1]) != np.sign(aLs[h2]))
    sus = np.where((np.isfinite(index) & (index < 0)) | sign_mix)[0]
    chk = dict(idx=sus, amax=np.full(sus.size, np.nan), dI=np.full(sus.size, np.nan),
               aL=np.full(sus.size, np.nan), ftype=np.array(["n/a"] * sus.size, dtype=object))
    if sus.size:
        S2, res2, M2, _ = batched(J_NFULL2, S[sus], P[sus])
        chk["amax"] = np.abs(np.linalg.eigvals(M2)).max(1)
        e2, r2_, _ = homotopy(S2, P[sus], ETAS_FINE, J_NHOM2)
        sf2, ok2, dI2, ft2 = fixed_lock(e2, P[sus], J_NINT2)
        ok2 &= r2_
        a2, _ = relocked_gradient(sf2, P[sus], AL_SCALES[0], J_NINT2)
        chk["dI"] = np.where(ok2, dI2, np.nan); chk["aL"] = np.where(ok2, a2, np.nan)
        chk["ftype"] = np.where(ok2, ft2, "none")
    CL.update({f"chk_{k}": np.asarray(v) for k, v in chk.items()})
    CL["chk_ftype"] = CL["chk_ftype"].astype(str); CL["ftype"] = CL["ftype"].astype(str)
    np.savez_compressed(ckpt("stageD2_index.npz"), **CL)
    print(f"D: {n} stable points classified with fine homotopy and narrow a_L offsets; "
          f"{sus.size} suspects rechecked at {SPP_CHECK} steps per period ({time.time() - t0:.0f} s)")

# comparison with the first (coarse) stage D, if it was run
D1 = None
if os.path.exists(ckpt("stageD_index.npz")):
    d = np.load(ckpt("stageD_index.npz"))
    D1 = {k: d[k] for k in d.files}

ft = CL["ftype"]; idxv = CL["index"]; aLv = CL["aL"]
ev = np.isfinite(idxv)
viol = ev & (idxv < 0)
viol_nm = viol & (np.abs(aLv) >= AL_MARGIN)
# the same points after the 2x-steps recheck (stability, branch and a_L recomputed)
chk_i = CL["chk_idx"].astype(int)
viol_chk = np.zeros_like(viol)
chk_rows = []
for q, i in enumerate(chk_i):
    st2 = CL["chk_amax"][q] < 1 - 1e-6
    ix2 = np.sign(CL["chk_dI"][q]) * np.sign(-CL["chk_aL"][q])
    still = bool(st2 and np.isfinite(ix2) and ix2 < 0)
    viol_chk[i] = still and abs(CL["chk_aL"][q]) >= AL_MARGIN
    p_ = int(CL["sel"][i]); b_ = BR[int(PTS[p_, 0])]; j_ = int(PTS[p_, 1])
    amax_b = b_["amax"]; stb = amax_b < 1 - 1e-6
    chg = np.where(stb[1:] != stb[:-1])[0]
    d_bif = int(np.min(np.abs(chg - j_))) if chg.size else None
    chk_rows.append(dict(
        eps=float(PTS[p_, 2]), dphi=float(PTS[p_, 3]), ratio=float(PTS[p_, 4]), x=float(PTS[p_, 5]),
        amax=float(PTS[p_, 8]), steps_to_stability_change=d_bif,
        fixed_type=str(ft[i]), det_I_minus_Mint=float(CL["dI"][i]),
        aL_used=None if np.isnan(aLv[i]) else float(aLv[i]),
        **{f"aL_half_{h}": (None if np.isnan(CL[f"aL_{h}"][i]) else float(CL[f"aL_{h}"][i])) for h in AL_SCALES},
        index=None if np.isnan(idxv[i]) else float(idxv[i]),
        check_2x_steps=dict(amax=float(CL["chk_amax"][q]), fixed_type=str(CL["chk_ftype"][q]),
                            det_I_minus_Mint=None if np.isnan(CL["chk_dI"][q]) else float(CL["chk_dI"][q]),
                            aL=None if np.isnan(CL["chk_aL"][q]) else float(CL["chk_aL"][q]),
                            still_violation=still)))

# -----------------------------------------------------------------------------
# E. Two-parameter picture and areas
# -----------------------------------------------------------------------------
BINS = np.arange(RATIO_LIM[0], RATIO_LIM[1] + 1e-9, AREA_BIN)
d_eps = float(np.diff(EPS_ROWS).min()) if EPS_ROWS.size > 1 else 0.5
key_cls = {tuple(k_): str(t_) for k_, t_ in zip(stable_keys(PTS[CL["sel"]]), ft)}
cls_of_point = {}
stab_all = np.where(PTS[:, 7] > 0)[0]
for p_, k_ in zip(stab_all, stable_keys(PTS[stab_all])):
    c_ = key_cls.get(tuple(k_))
    if c_ is not None:
        cls_of_point[int(p_)] = c_
area = {}
occ = {}
for dp in DPHI_SET:
    for cl in ("any", "node", "saddle-real", "none"):
        occ[(dp, cl)] = np.zeros((EPS_ROWS.size, BINS.size - 1), bool)
pt_index = {}
for k_, row in enumerate(PTS):
    pt_index[(int(row[0]), int(row[1]))] = k_
for bi, b in enumerate(BR):
    Z = b["Z"]
    if len(Z) < 2:
        continue
    ei = int(np.argmin(np.abs(EPS_ROWS - b["eps"])))
    dp = min(DPHI_SET, key=lambda v: abs(v - b["dphi"]))
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
        if c_ in ("node", "saddle-real", "none"):
            occ[(dp, c_)][ei, cells] = True
for dp in DPHI_SET:
    A_ = lambda m_: float(m_.sum() * AREA_BIN * d_eps)
    o_ = lambda c_: occ[(dp, c_)]
    area[f"{dp:.3f}"] = {cl: A_(o_(cl)) for cl in ("any", "node", "saddle-real", "none")}
    area[f"{dp:.3f}"].update(
        node_only=A_(o_("node") & ~o_("saddle-real") & ~o_("none")),
        saddle_only=A_(o_("saddle-real") & ~o_("node") & ~o_("none")),
        none_only=A_(o_("none") & ~o_("node") & ~o_("saddle-real")),
        node_and_none=A_(o_("node") & o_("none")), node_and_saddle=A_(o_("node") & o_("saddle-real")),
        saddle_and_none=A_(o_("saddle-real") & o_("none")),
        unclassified_fraction=(float(1 - (o_("node") | o_("saddle-real") | o_("none")).sum() / o_("any").sum())
                               if o_("any").sum() else None))

# G1: Script 05b stable orbits inside continued stable segments
g1 = None
if os.path.exists(F05B):
    d5 = np.load(F05B, allow_pickle=True)
    st5 = np.array([str(t) == "stable" for t in d5["type"]])
    # only orbits on rows and delays this run covers
    st5 &= np.isin(np.round(d5["eps"], 6), np.round(EPS_ROWS, 6))
    st5 &= np.array([any(abs(v - dp) < 1e-6 for dp in DPHI_SET) for v in d5["dphi"]])
    E5, R5, D5, X5 = d5["eps"][st5], d5["ratio"][st5], d5["dphi"][st5], d5["x"][st5]
    inside = []
    stab_pts = PTS[PTS[:, 7] > 0] if len(PTS) else PTS
    for e_, r_, d_, x_ in zip(E5, R5, D5, X5):
        m = (np.abs(stab_pts[:, 2] - e_) < 1e-6) & (np.abs(stab_pts[:, 3] - d_) < 1e-6) & \
            (np.abs(stab_pts[:, 4] - r_) < 0.003) & (np.abs(stab_pts[:, 5] - x_) < 0.05)
        inside.append(bool(m.any()))
    g1 = dict(n=int(len(inside)), inside=int(sum(inside)),
              fraction=float(np.mean(inside)) if inside else None)
seg_ends = [e_ for e_ in seg_ends if e_["kind"] != "start"]
n_ends = len(seg_ends)
n_bif_ends = sum(e_["kind"] in ("bifurcation", "domain") for e_ in seg_ends)
end_types = {}
for e_ in seg_ends:
    k_ = e_["type"] if e_["kind"] == "bifurcation" else e_["kind"] + ": " + e_["type"]
    end_types[k_] = end_types.get(k_, 0) + 1

# dedupe bifurcation points for reporting
def dedupe_bifs(lst):
    out = []
    for b_ in lst:
        if not any(o["kind"] == b_["kind"] and o["eps"] == b_["eps"] and o["dphi"] == b_["dphi"]
                   and abs(o["ratio"] - b_["ratio"]) < 0.002 and abs(o["x"] - b_["x"]) < 0.02 for o in out):
            out.append(b_)
    return out
BIF = dedupe_bifs(bifs)
bif_counts = {}
for b_ in BIF:
    bif_counts[b_["kind"]] = bif_counts.get(b_["kind"], 0) + 1

# no-lock stable points: do they sit on branches that also carry saddle-type stable points?
nolock_pts = [p_ for p_, t_ in zip(CL["sel"], ft) if t_ == "none"]
br_types = {}
for p_, t_ in zip(CL["sel"], ft):
    br_types.setdefault(int(PTS[p_, 0]), set()).add(str(t_))
nolock_connected = sum(1 for p_ in nolock_pts if "saddle-real" in br_types.get(int(PTS[p_, 0]), set())
                       or "node" in br_types.get(int(PTS[p_, 0]), set()))

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
tag = "" if MODE == "FULL" else "_quick"
COL = {"saddle-real": "#b2182b", "node": "#2166ac", "none": "#999999"}
nd = len(DPHI_SET)
fig, axs = plt.subplots(1, nd, figsize=(3.2 * nd, 3.2), squeeze=False)
for c_, dp in enumerate(DPHI_SET):
    ax = axs[0, c_]
    # categorical map: 0 empty, 1 unclassified, 2 no lock, 3 saddle, 4 node (node drawn on top)
    img = np.zeros(occ[(dp, "any")].shape)
    img[occ[(dp, "any")]] = 1
    img[occ[(dp, "none")]] = 2
    img[occ[(dp, "saddle-real")]] = 3
    img[occ[(dp, "node")]] = 4
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", "#e0e0e0", COL["none"], COL["saddle-real"], COL["node"]])
    e_edges = np.concatenate([EPS_ROWS - d_eps / 2, [EPS_ROWS[-1] + d_eps / 2]])
    ax.pcolormesh(BINS, e_edges, img, cmap=cmap, vmin=-0.5, vmax=4.5, shading="flat", rasterized=True)
    for cl in ("none", "saddle-real", "node"):
        ax.plot([], [], "s", color=COL[cl], ms=4,
                label={"saddle-real": "saddle branch", "node": "node branch", "none": "no fixed lock"}[cl])
    sel_r = np.where(occ[(dp, "any")].any(0))[0]
    if sel_r.size:
        ax.set_xlim(BINS[max(sel_r.min() - 4, 0)], BINS[min(sel_r.max() + 5, BINS.size - 1)])
    for b_ in BIF:
        if abs(b_["dphi"] - dp) > 1e-6:
            continue
        if b_["kind"] == "fold":
            ax.plot(b_["ratio"], b_["eps"], "k.", ms=2)
        elif b_["kind"] == "loss-torus":
            ax.plot(b_["ratio"], b_["eps"], "^", ms=3, mfc="none", mec="#1a9850", mew=0.6)
        elif b_["kind"] == "loss-flip":
            ax.plot(b_["ratio"], b_["eps"], "x", ms=3, color="#762a83")
    if os.path.exists(F05B):
        m5 = np.abs(D5 - dp) < 1e-6
        ax.scatter(R5[m5], E5[m5], s=4, facecolors="none", edgecolors="k", lw=0.3)
    ax.set_xlabel(r"$\omega_c/\omega_w$"); ax.set_title(rf"$\Delta\phi$ = {dp:.2f}", fontsize=8)
    if c_ == 0:
        ax.set_ylabel(r"$\varepsilon$"); ax.legend(frameon=False, fontsize=6, markerscale=3)
fig.suptitle("Stable station holding from continuation (dots: folds, triangles: torus, circles: Script 05b)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"fig01_two_parameter{tag}.png"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, f"fig01_two_parameter{tag}.svg"))
plt.close(fig)

# one-parameter diagram at the eps row with the most stable points (delay 0)
if len(PTS):
    dp0 = DPHI_SET[0]
    rows = [e_ for e_ in EPS_ROWS if ((np.abs(PTS[:, 2] - e_) < 1e-6) & (np.abs(PTS[:, 3] - dp0) < 1e-6) & (PTS[:, 7] > 0)).any()]
    if rows:
        e_best = max(rows, key=lambda e_: ((np.abs(PTS[:, 2] - e_) < 1e-6) & (PTS[:, 7] > 0)).sum())
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for bi, b in enumerate(BR):
            if abs(b["eps"] - e_best) > 1e-6 or abs(b["dphi"] - dp0) > 1e-6 or len(b["Z"]) < 2:
                continue
            Z = b["Z"]; st = b["amax"] < 1 - 1e-6
            for j in range(len(Z) - 1):
                ax.plot(Z[j:j + 2, 6], Z[j:j + 2, 0], "-" if (st[j] and st[j + 1]) else ":",
                        color="#1a9850" if (st[j] and st[j + 1]) else "#bbbbbb", lw=1.0)
        for b_ in BIF:
            if abs(b_["eps"] - e_best) < 1e-6 and abs(b_["dphi"] - dp0) < 1e-6:
                mk = {"fold": "ko", "loss-torus": "g^", "loss-flip": "mx"}.get(b_["kind"])
                if mk:
                    ax.plot(b_["ratio"], b_["x"], mk, ms=3, mfc="none")
        ax.set_xlabel(r"$\omega_c/\omega_w$"); ax.set_ylabel("holding position x")
        ax.set_title(rf"Branches at $\varepsilon$ = {e_best:.3f}, $\Delta\phi$ = {dp0:.2f} (solid: stable)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"fig02_one_parameter{tag}.png"), dpi=300)
        fig.savefig(os.path.join(FIG_DIR, f"fig02_one_parameter{tag}.svg"))
        plt.close(fig)

# -----------------------------------------------------------------------------
# Save and gates
# -----------------------------------------------------------------------------
summary = dict(
    mode=MODE, device=[d.platform for d in jax.devices()], runtime_this_session_s=round(time.time() - t_start, 1),
    eps_rows=[float(EPS_ROWS[0]), float(EPS_ROWS[-1]), d_eps], dphi=DPHI_SET,
    grid=dict(orbits=int(GS.shape[0]), stable=int(GSTAB.sum())),
    continuation=dict(half_branches=len(BR), points=int(PTS.shape[0]),
                      stable_grid_orbits_not_on_branch=int(np.load(ckpt("stageC_branches.npz"), allow_pickle=True)["n_uncovered"]), stable_points=int(PTS[:, 7].sum()) if len(PTS) else 0,
                      end_reasons={r_: sum(b["end"] == r_ for b in BR) for r_ in set(b["end"] for b in BR)}),
    bifurcations=bif_counts,
    stable_segment_ends=dict(n=n_ends, bifurcation=n_bif_ends, by_type=end_types),
    stable_area_by_delay=area,
    classified=dict(n=int(CL["sel"].size), node=int((ft == "node").sum()), saddle_real=int((ft == "saddle-real").sum()),
                    unstable_complex=int((ft == "unstable-complex").sum()), none=int((ft == "none").sum()),
                    evaluated=int(ev.sum()), violations=int(viol.sum()), violations_nonmarginal=int(viol_nm.sum()),
                    violations_after_2x_steps_check=int(viol_chk.sum()),
                    aL_offset_used={str(h): int((CL["aL_half"] == h).sum()) for h in AL_SCALES},
                    fixed_lock_but_no_aL=int(((ft != "none") & ~np.isfinite(aLv)).sum()),
                    homotopy_failed=int((~CL["reached"].astype(bool)).sum()),
                    node_aL_neg=int(((ft == "node") & (aLv < 0)).sum()), saddle_aL_pos=int(((ft == "saddle-real") & (aLv > 0)).sum())),
    no_lock=dict(n=len(nolock_pts), on_branch_with_fixed_lock=int(nolock_connected)),
    gate_G1_05b=g1,
    coarse_stage_D=None if D1 is None else dict(
        evaluated=int(np.isfinite(D1["index"]).sum()),
        violations=int((np.isfinite(D1["index"]) & (D1["index"] < 0)).sum())),
    suspects_rechecked=chk_rows[:40],
)
with open(os.path.join(OUT_DIR, f"summary_script10{tag}.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT_DIR, f"bifurcations_script10{tag}.json"), "w") as f:
    json.dump(BIF, f)
np.savez_compressed(os.path.join(OUT_DIR, f"points_script10{tag}.npz"), points=PTS,
                    cols=np.array(["branch", "j", "eps", "dphi", "ratio", "x", "phase", "stable", "max_abs_mu"]))

print("\n================ SCRIPT 10 SUMMARY ================")
print(json.dumps({k: v for k, v in summary.items() if k not in ("stable_area_by_delay", "suspects_rechecked")}, indent=2))
print("\nStable area in (omega_c/omega_w, eps), by delay and fixed-position branch:")
for dp, a_ in area.items():
    if not a_["any"]:
        print(f"  dphi {dp}: no stable holding"); continue
    print(f"  dphi {dp}: total {a_['any']:.4f} | saddle {a_['saddle-real']:.4f} (only {a_['saddle_only']:.4f}) | "
          f"node {a_['node']:.4f} (only {a_['node_only']:.4f}) | no lock {a_['none']:.4f} (only {a_['none_only']:.4f}) | "
          f"overlaps node+none {a_['node_and_none']:.4f}, node+saddle {a_['node_and_saddle']:.4f}, "
          f"saddle+none {a_['saddle_and_none']:.4f} | unclassified {a_['unclassified_fraction']:.2f}")
if chk_rows:
    print("\nSuspect points (index < 0 or a_L sign depends on offset width), rechecked at 2x steps per period:")
    for r_ in chk_rows[:20]:
        c_ = r_["check_2x_steps"]
        print(f"  eps {r_['eps']:.3f} dphi {r_['dphi']:.2f} ratio {r_['ratio']:.4f} x {r_['x']:.3f} | max|mu| {r_['amax']:.4f}, "
              f"{r_['steps_to_stability_change']} steps from a stability change | {r_['fixed_type']}, det {r_['det_I_minus_Mint']:.2e}, "
              f"a_L by half-width " + ", ".join(f"{h}: {r_[f'aL_half_{h}']}" for h in AL_SCALES)
              + f" | 2x steps: max|mu| {c_['amax']:.4f}, {c_['fixed_type']}, a_L {c_['aL']}, still violation {c_['still_violation']}")
print("\n================ GATES ================")
G1 = g1 is None or g1["n"] == 0 or g1["fraction"] >= 0.95
G2 = n_ends == 0 or n_bif_ends / n_ends >= 0.90
G3 = int(viol_chk.sum()) == 0
g1_txt = ("n/a (05b results not found)" if g1 is None else
          "n/a (no 05b orbits on these rows)" if g1["n"] == 0 else f"{g1['inside']}/{g1['n']}")
print(f"G1 Script 05b stable orbits inside continued stable segments: {g1_txt} -> {'PASS' if G1 else 'FAIL'}")
print(f"G2 stable-segment ends at bifurcations or domain limits (not numerical failure): "
      f"{n_bif_ends}/{n_ends} -> {'PASS' if G2 else 'FAIL'}")
print(f"G3 index violations on continued stable points: {int(viol.sum())} ({int(viol_nm.sum())} non-marginal) of "
      f"{int(ev.sum())}; after the 2x-steps recheck: {int(viol_chk.sum())} -> {'PASS' if G3 else 'FAIL'}")
print(f"\nOutputs: {OUT_DIR}")
