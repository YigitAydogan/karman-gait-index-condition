# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 17: Analyses for the biological questions of the Discussion.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX; runs on the GPU when one is attached (L4: about 10 to 20 min FULL),
# falls back to the CPU (slower). Every part is checkpointed to Drive, so a
# disconnected run resumes where it stopped. No settings need to be changed.
# Needs the outputs of Scripts 02, 02b, 09, 10 and 14 on Drive.
#
# Questions and parts:
#   A. Passive bodies (dead fish): what decides whether a passive body moves
#      upstream to the cylinder, drifts downstream out of the wake, or stays?
#      A1  New simulation, baseline wake: fate of a passive body released at
#          rest at every position x0 in [0.2, 3.0] wavelengths, for body
#          stiffness Omega/omega_w from 0.3 to 4 and a rigid body, at damping
#          ratios 0.05, 0.1 and 0.3; plus the period-averaged force curve at
#          fixed position for every body, whose zero crossings are the
#          equilibria that separate the fates.
#      A2  Reanalysis of Scripts 02 and 02b (four wake families, Latin
#          hypercube sample): direction of the period-averaged force against
#          position relative to the envelope peak, and the stable fraction of
#          passive equilibria by body type and position.
#   B. Where in the street is holding stable? (reanalysis, no simulation)
#      Positions of stable station holding relative to the peak of the vortex
#      envelope and to the formation length, for the baseline (Script 10),
#      every model variant and joint parameter sample (Script 09) and the
#      phase-lag sweep (Script 14), compared with the positions of stable
#      passive equilibria (Scripts 02 and 02b). Positions are also converted
#      to body lengths and cylinder diameters for comparison with experiments.
#   C. Control experiment without the vortex street (uniform flow).
#      In uniform flow the baseline force law produces no thrust (thrust comes
#      from lift tilted by the transverse wake velocity), so the comparison
#      uses the model variant with added reactive thrust (Script 09), which
#      can swim in uniform flow.
#      C1  Stable holding orbits of the reactive-thrust variant in the wake
#          are recovered by direct simulation.
#      C2  The same fish in uniform flow: ground speed against muscle gain g,
#          the gain g* at which thrust balances drag exactly, and the
#          sensitivity of the drift speed to a gain mismatch.
#      C3  Response to a positional kick: in the wake the kick decays, in
#          uniform flow it persists (position is neutral).
#      C4  Noise (CPG phase noise and free-stream speed fluctuations, same in
#          both conditions): time to leave bands of +/- 0.1 and +/- 0.25
#          wavelengths, and growth of the positional variance.
#
# Gates (reported):
#   G1  the JAX model of this script reproduces Script 10 stable orbits
#       (one-period residual below 1e-6 from the stored states)
#   G2  at least half of the reactive-thrust stable orbits are recovered
#   G3  in uniform flow a kick of 0.1 wavelengths persists to within 2%
#
# Outputs: script17_biology_questions/summary_script17.json, tables (CSV),
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

MODE = "FULL"
SEED = 17
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script17_biology_questions")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
for d_ in (FIG_DIR, TAB_DIR, CKPT_DIR):
    os.makedirs(d_, exist_ok=True)
CK = "checkpoints_full"

def P(*a):
    return os.path.join(BASE_DIR, *a)

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

if MODE == "FULL":
    SPP = 100
    X0_GRID = np.round(np.linspace(0.2, 3.0, 57), 6)
    OMR_GRID = np.round(np.geomspace(0.3, 4.0, 22), 6)
    ZE_GRID = [0.05, 0.1, 0.3]
    T_FATE = 300
    XF_GRID = np.round(np.linspace(0.16, 3.6, 87), 6)
    T_FORCE = 40
    N_SEED_PH = 8
    T_RECOVER = 400
    G_SCAN = np.linspace(0.0, 120.0, 61)
    CT_SCAN = [0.0, 0.1, 0.3]
    T_SCAN = 400
    N_ORB_C, N_REP, T_NOISE = 16, 32, 1000
    NOISE_LEVELS = [0.003, 0.01]
    CHUNK = 8192 if ON_GPU else 512
else:
    SPP = 60
    X0_GRID = np.round(np.linspace(0.2, 3.0, 8), 6)
    OMR_GRID = np.round(np.geomspace(0.5, 3.0, 4), 6)
    ZE_GRID = [0.1]
    T_FATE = 60
    XF_GRID = np.round(np.linspace(0.16, 3.6, 12), 6)
    T_FORCE = 10
    N_SEED_PH = 4
    T_RECOVER = 120
    G_SCAN = np.linspace(0.0, 120.0, 9)
    CT_SCAN = [0.0, 0.1]
    T_SCAN = 120
    N_ORB_C, N_REP, T_NOISE = 2, 4, 80
    NOISE_LEVELS = [0.01]
    CHUNK = 512
DT = 1.0 / SPP
X_COLL, X_ESC = 0.15, 4.5
TAU_A = 1.0
OMEGA_W = 2.0 * np.pi

# Unit conversion for comparison with experiments.
# Beal et al. (2006), as used in Script 11: wake wavelength 0.208 m, trout body length 0.178 m.
# Cylinder diameters: lambda / D = (convection speed / U_inf) / St, with St the Strouhal number
# of the cylinder; ST = 0.2 is a typical value and is an assumption.
TROUT = dict(wavelength_m=0.208, body_length_m=0.178)
BL_PER_LAMBDA = TROUT["wavelength_m"] / TROUT["body_length_m"]
ST = 0.20
CONV = 0.87                      # vortex convection speed / U_inf in the model (U_inf = 1/0.87)
D_PER_LAMBDA = CONV / ST

# -----------------------------------------------------------------------------
# 1. Model in JAX (baseline of Scripts 10 and 11, plus reactive thrust)
# -----------------------------------------------------------------------------
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA, LT = 3.0, 0.15, 8.0, 0.5
MU = 2 * np.pi
# parameter vector:
#  0 ratio  1 eps  2 dphi  3 g  4 Om/omega_w  5 zeta  6 rigid  7 wake factor (W)
#  8 deficit factor  9 c_T  10 bounded (collision/escape stops the run)  11 freeze position
NPAR = 12

def W_env(x):
    return W0 * (1.0 - np.exp(-(x / X_R) ** 2)) * np.exp(-x / L_W)

def rhs(t, s, p, psi, aW, aU):
    x, u, th, om, zr, zi = s
    ratio, eps, dphi, g, omr, ze, rigid, wf, df, cT, bnd, frz = p
    V = U_INF * (1.0 + aU) * (1.0 - df * DELTA0 * jnp.exp(-x / L_DELTA)) - u
    w = wf * (1.0 + aW) * W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W) * \
        jnp.sin(OMEGA_W * (x - t) + psi)
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    w_te = w - jnp.abs(V) * th - LT * om
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta) - cT * w_te * w_te * gate
    Om = omr * OMEGA_W
    drive = zr * jnp.cos(dphi) + zi * jnp.sin(dphi)
    dom = (-2 * ze * Om * om - Om ** 2 * th + SIGMA * L + g * drive) * (1.0 - rigid)
    dth = om * (1.0 - rigid)
    r2 = zr * zr + zi * zi
    a = MU * (1.0 - r2)
    b = ratio * OMEGA_W
    inp = eps * th
    dx = u * (1.0 - frz)
    du = Fx * (1.0 - frz)
    return jnp.stack([dx, du, dth, dom, a * zr - b * zi + inp, a * zi + b * zr]), Fx

def make_sim(n_per, noisy):
    """Simulate n_per shedding periods. Returns final state, alive flag, loss
    time, mean force over the second half (alive periods), strobed x per period."""
    def sim(s0, p, nz, key):
        Dpsi, sW, sU, Dz = nz[0], nz[1], nz[2], nz[3]
        def one_period(carry, k):
            s, psi, aW, aU, alive, tloss, Fsum, nacc = carry
            kp = jax.random.fold_in(key, k)
            def step(i, c):
                s_, psi_, aW_, aU_, al_, tl_, Fp_ = c
                t = k + i * DT
                k1, F1 = rhs(t, s_, p, psi_, aW_, aU_)
                k2, _ = rhs(t + DT / 2, s_ + DT / 2 * k1, p, psi_, aW_, aU_)
                k3, _ = rhs(t + DT / 2, s_ + DT / 2 * k2, p, psi_, aW_, aU_)
                k4, _ = rhs(t + DT, s_ + DT * k3, p, psi_, aW_, aU_)
                sn = s_ + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if noisy:
                    nrm = jax.random.normal(jax.random.fold_in(kp, i), (4,))
                    ea = jnp.exp(-DT / TAU_A)
                    psi_n = psi_ + jnp.sqrt(2 * Dpsi * DT) * nrm[0]
                    aW_n = aW_ * ea + sW * jnp.sqrt(1 - ea * ea) * nrm[1]
                    aU_n = aU_ * ea + sU * jnp.sqrt(1 - ea * ea) * nrm[2]
                    ang = jnp.sqrt(2 * Dz * DT) * nrm[3]
                    zr2 = sn[4] * jnp.cos(ang) - sn[5] * jnp.sin(ang)
                    zi2 = sn[4] * jnp.sin(ang) + sn[5] * jnp.cos(ang)
                    sn = sn.at[4].set(zr2).at[5].set(zi2)
                else:
                    psi_n, aW_n, aU_n = psi_, aW_, aU_
                lost = (((sn[0] < X_COLL) | (sn[0] > X_ESC)) & (p[10] > 0.5)) | ~jnp.all(jnp.isfinite(sn))
                tl_n = jnp.where(al_ & lost, t + DT, tl_)
                al_n = al_ & ~lost
                sn = jnp.where(al_n, sn, s_)
                return sn, psi_n, aW_n, aU_n, al_n, tl_n, Fp_ + F1 * DT
            s, psi, aW, aU, alive2, tloss, Fp = jax.lax.fori_loop(
                0, SPP, step, (s, psi, aW, aU, alive, tloss, 0.0))
            use = alive2 & (k >= n_per // 2)
            Fsum = Fsum + jnp.where(use, Fp, 0.0)
            nacc = nacc + jnp.where(use, 1.0, 0.0)
            return (s, psi, aW, aU, alive2, tloss, Fsum, nacc), s[0]
        carry0 = (s0, 0.0, 0.0, 0.0, True, jnp.nan, 0.0, 0.0)
        (s, psi, aW, aU, alive, tloss, Fsum, nacc), xs = jax.lax.scan(one_period, carry0, jnp.arange(n_per))
        return s, alive, tloss, Fsum / jnp.maximum(nacc, 1.0), xs
    return jax.jit(jax.vmap(sim))

SIMS = {}
def get_sim(n_per, noisy=False):
    k = (n_per, noisy)
    if k not in SIMS:
        SIMS[k] = make_sim(n_per, noisy)
    return SIMS[k]

def run(n_per, S0, PP, NZ=None, noisy=False, seed=0, label=None):
    """Batched simulation in chunks. S0 (n,6), PP (n,NPAR), NZ (n,4)."""
    n = S0.shape[0]
    if NZ is None:
        NZ = np.zeros((n, 4))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    fn = get_sim(n_per, noisy)
    outs = []
    tb = time.time()
    for c_, i0 in enumerate(range(0, n, CHUNK)):
        sl = slice(i0, min(n, i0 + CHUNK))
        o = fn(jnp.asarray(S0[sl]), jnp.asarray(PP[sl]), jnp.asarray(NZ[sl]), keys[sl])
        outs.append([np.asarray(v) for v in o])
        if label and c_ % 5 == 0:
            print(f"      {label}: {min(n, i0 + CHUNK)}/{n} runs ({time.time() - tb:.0f} s)", flush=True)
    return [np.concatenate([o[j] for o in outs]) for j in range(5)]

def par(n, **kw):
    base = dict(ratio=1.0, eps=0.0, dphi=0.0, g=16.0, omr=1.2, ze=0.2, rigid=0.0, wf=1.0, df=1.0,
                cT=0.0, bnd=1.0, frz=0.0)
    base.update(kw)
    order = ["ratio", "eps", "dphi", "g", "omr", "ze", "rigid", "wf", "df", "cT", "bnd", "frz"]
    return np.stack([np.broadcast_to(np.asarray(base[k], float), (n,)) for k in order], 1)

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)

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

SUMMARY = dict(mode=MODE, spp=SPP, units=dict(body_lengths_per_wavelength=BL_PER_LAMBDA,
               cylinder_diameters_per_wavelength=D_PER_LAMBDA, strouhal_assumed=ST))
t_start = time.time()
X_PEAK_BASE = float(np.linspace(0, 4, 40001)[np.argmax(W_env(np.linspace(0, 4, 40001)))])
SUMMARY["baseline_envelope_peak"] = X_PEAK_BASE
print(f"Baseline envelope peak at x = {X_PEAK_BASE:.3f} wavelengths")

# -----------------------------------------------------------------------------
# G1. Model check against Script 10 stable orbits
# -----------------------------------------------------------------------------
print("\n=== G1. Model check ===")
G = {}
f10p = P("script10_continuation", "points_script10.npz")
f10b = P("script10_continuation", CK, "stageC_branches.npz")
if os.path.exists(f10p) and os.path.exists(f10b):
    PTS = np.load(f10p)["points"]
    BR = np.load(f10b, allow_pickle=True)["branches"].tolist()
    st_idx = np.where(PTS[:, 7] > 0.5)[0]
    pick = rng.choice(st_idx, min(64, st_idx.size), replace=False)
    S0 = np.array([BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6] for i in pick])
    PP = par(pick.size, ratio=PTS[pick, 4], eps=PTS[pick, 2], dphi=PTS[pick, 3])
    sN = run(1, S0, PP)[0]
    res = np.max(np.abs(sN - S0), 1)
    G["G1_max_residual"] = float(res.max())
    G["G1_pass"] = bool(res.max() < 1e-6)
    print(f"  one-period residual from {pick.size} stored orbits: max {res.max():.2e} "
          f"({'pass' if G['G1_pass'] else 'FAIL'})")
else:
    PTS = None
    print("  Script 10 outputs not found; G1 skipped")

# =============================================================================
# A. Passive bodies
# =============================================================================
print("\n=== A1. Passive fate map (baseline wake) ===")
A = {}
fA = ckpt("A1_fate.npz")
BODIES_A = [(float(r), float(z), 0.0) for z in ZE_GRID for r in OMR_GRID] + [(1.0, 0.1, 1.0)]
if os.path.exists(fA):
    dA = dict(np.load(fA))
    print("  loaded checkpoint")
else:
    nb, nx = len(BODIES_A), X0_GRID.size
    bb = np.repeat(np.arange(nb), nx)
    xx = np.tile(X0_GRID, nb)
    S0 = np.zeros((bb.size, 6)); S0[:, 0] = xx; S0[:, 4] = 1.0
    BA = np.array(BODIES_A)
    PP = par(bb.size, eps=0.0, g=0.0, omr=BA[bb, 0], ze=BA[bb, 1], rigid=BA[bb, 2])
    sN, alive, tloss, _, xs = run(T_FATE, S0, PP, label="fate")
    # mean force at fixed position
    nf = XF_GRID.size
    bf = np.repeat(np.arange(nb), nf)
    xf = np.tile(XF_GRID, nb)
    S0f = np.zeros((bf.size, 6)); S0f[:, 0] = xf; S0f[:, 4] = 1.0
    PPf = par(bf.size, eps=0.0, g=0.0, omr=BA[bf, 0], ze=BA[bf, 1], rigid=BA[bf, 2], frz=1.0, bnd=0.0)
    Fm = run(T_FORCE, S0f, PPf, label="force")[3]
    dA = dict(bodies=BA, bb=bb, x0=xx, xend=sN[:, 0], alive=alive, tloss=tloss,
              xs_last=xs[:, -50:], F=Fm.reshape(nb, nf))
    np.savez(fA, **dA)

BA = dA["bodies"]; nb = BA.shape[0]
fate = np.where(dA["alive"], 0, np.where(dA["xend"] < 1.0, 1, 2))      # 0 stays, 1 upstream (collision), 2 downstream (escape)
span = np.ptp(dA["xs_last"], 1)
stays_still = dA["alive"] & (span < 0.05)
A_rows, A_bodies = [], []
for b in range(nb):
    m = dA["bb"] == b
    F = dA["F"][b]
    sg = np.sign(F)
    zc = np.where(sg[:-1] != sg[1:])[0]
    eqs = [float(XF_GRID[j] - F[j] * (XF_GRID[j + 1] - XF_GRID[j]) / (F[j + 1] - F[j])) for j in zc]
    x0b, fb = dA["x0"][m], fate[m]
    up = x0b[fb == 1]; dn = x0b[fb == 2]; st = x0b[fb == 0]
    rec = dict(omr=None if BA[b, 2] > 0.5 else float(BA[b, 0]), zeta=float(BA[b, 1]), rigid=bool(BA[b, 2] > 0.5),
               equilibria=eqs, n_upstream=int(up.size), n_downstream=int(dn.size), n_stay=int(st.size),
               n_stay_still=int((stays_still[m]).sum()),
               upstream_x0_max=float(up.max()) if up.size else None,
               downstream_x0_min=float(dn.min()) if dn.size else None,
               median_time_to_fate=float(np.nanmedian(dA["tloss"][m])) if np.isfinite(dA["tloss"][m]).any() else None)
    A_bodies.append(rec)
    A_rows.append([rec["omr"] if rec["omr"] is not None else "rigid", rec["zeta"], rec["n_upstream"],
                   rec["n_downstream"], rec["n_stay"], rec["n_stay_still"],
                   ";".join(f"{e:.3f}" for e in eqs), rec["upstream_x0_max"], rec["downstream_x0_min"],
                   rec["median_time_to_fate"]])
save_csv("A1_passive_fates.csv", ["Om_over_omega_w", "zeta", "n_upstream", "n_downstream", "n_stay",
         "n_stay_still", "equilibria_x", "upstream_x0_max", "downstream_x0_min", "median_time_to_fate"], A_rows)
tot = np.bincount(fate, minlength=3)
A["A1"] = dict(n_runs=int(fate.size), upstream=int(tot[1]), downstream=int(tot[2]), stay=int(tot[0]),
               stay_still=int(stays_still.sum()), bodies=A_bodies,
               note="fate 'upstream' = reaches the cylinder (x < 0.15), 'downstream' = leaves the wake (x > 4.5); "
                    "the model has no lateral position, so lateral expulsion from the street is not represented")
# separatrix summary: for every body, the release position that separates upstream and downstream fates
sep = []
for rec in A_bodies:
    if rec["upstream_x0_max"] is not None and rec["downstream_x0_min"] is not None:
        sep.append((rec["omr"], rec["zeta"], rec["upstream_x0_max"], rec["downstream_x0_min"],
                    rec["equilibria"][0] if rec["equilibria"] else None))
A["A1"]["separatrix"] = sep
print(f"  {fate.size} releases: {tot[1]} upstream to the cylinder, {tot[2]} downstream out of the wake, "
      f"{tot[0]} stay ({stays_still.sum()} at rest within 0.05 wavelengths)")
for rec in A_bodies[:: max(1, len(A_bodies) // 8)] + [A_bodies[-1]]:
    lab = "rigid" if rec["rigid"] else f"Om/w {rec['omr']:.2f}, zeta {rec['zeta']:.2f}"
    print(f"    {lab:>26s}: equilibria {[round(e, 3) for e in rec['equilibria']]}, upstream if released "
          f"at x0 <= {rec['upstream_x0_max']}, downstream if x0 >= {rec['downstream_x0_min']}, stay {rec['n_stay']}")

print("\n=== A2. Passive force direction and stability across wake families (Scripts 02, 02b) ===")
f2 = P("script02_passive_robustness", "results_script02.npz")
f2b = P("script02b_passive_cases", "results_script02b.npz")
if os.path.exists(f2):
    d2 = np.load(f2, allow_pickle=True)
    XG2 = np.linspace(0.16, 3.6, d2["lhs_F"].shape[1])
    Fset = d2["lhs_F"]                                   # (sets, positions) period-averaged force at rest
    shape = d2["Plhs_shape"].astype(int)
    xf_ = np.linspace(0.0, 4.0, 801)
    def W_fam(x, s, W0_, xr, lw):
        rise = 1.0 - np.exp(-(x / xr) ** 2)
        if s == 2:
            return W0_ * rise / (1.0 + x / lw)
        if s == 3:
            return W0_ * np.exp(-((x - xr) / (0.5 * lw)) ** 2)
        return W0_ * rise * np.exp(-x / lw)
    xpk_set = np.array([xf_[np.argmax(W_fam(xf_, shape[i], d2["Plhs_W0"][i], d2["Plhs_x_r"][i], d2["Plhs_l_w"][i]))]
                        for i in range(Fset.shape[0])])
    rgrid = XG2[None, :] / xpk_set[:, None]
    BIN = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 100.0])
    rows, A2 = [], []
    for k in range(BIN.size - 1):
        m = (rgrid >= BIN[k]) & (rgrid < BIN[k + 1]) & np.isfinite(Fset)
        if m.sum() == 0:
            continue
        up = float(np.mean(Fset[m] < 0))
        A2.append(dict(r_lo=BIN[k], r_hi=BIN[k + 1], n=int(m.sum()), frac_force_upstream=up))
        rows.append([BIN[k], BIN[k + 1], int(m.sum()), up])
    save_csv("A2_force_direction_by_position.csv", ["x_over_xpeak_lo", "x_over_xpeak_hi", "n", "frac_force_upstream"], rows)
    A["A2_force_direction"] = A2
    # stable fraction of equilibria by body and position, with Script 02b Floquet overrides
    r_eq = d2["lhs_eq_x"] / d2["lhs_xpeak"]
    stable = d2["lhs_cls"] == 0
    floquet_confirmed = np.zeros_like(stable)
    if os.path.exists(f2b):
        b2 = np.load(f2b)
        chk, conv, cf = b2["chk"].astype(int), b2["conv"].astype(bool), b2["cls_f"]
        stable = stable.copy(); stable[chk[conv]] = cf[conv] == 0
        floquet_confirmed[chk[conv]] = cf[conv] == 0
    BODY_NAMES = ["rigid", "soft 0.5", "resonant 0.9", "resonant-damped 0.9", "stiff 1.5", "very stiff 3.0"]
    body = d2["lhs_eq_set"].astype(int) % len(BODY_NAMES)
    rowsb, A2b = [], []
    for bi, bn in enumerate(BODY_NAMES):
        for lab, m0 in (("upstream of peak", r_eq <= 1), ("downstream of peak", r_eq > 1)):
            m = (body == bi) & m0
            k_, n_ = int(stable[m].sum()), int(m.sum())
            p_, lo_, hi_ = wilson(k_, n_)
            A2b.append(dict(body=bn, region=lab, stable=k_, floquet_confirmed=int(floquet_confirmed[m].sum()),
                            n=n_, frac=p_, ci95=[lo_, hi_]))
            rowsb.append([bn, lab, k_, int(floquet_confirmed[m].sum()), n_, p_, lo_, hi_])
    save_csv("A2_passive_stability_by_body.csv", ["body", "region", "stable", "floquet_confirmed", "n",
             "frac", "ci_lo", "ci_hi"], rowsb)
    A["A2_stability_by_body"] = A2b
    A["A2_stable_passive_r"] = dict(median=float(np.median(r_eq[stable])), q10=float(np.quantile(r_eq[stable], 0.1)),
                                    q90=float(np.quantile(r_eq[stable], 0.9)),
                                    floquet_confirmed_median=float(np.median(r_eq[floquet_confirmed]))
                                    if floquet_confirmed.any() else None)
    for e in A2:
        print(f"    x/x_peak in [{e['r_lo']:.2f}, {e['r_hi']:.2f}): force upstream in {100 * e['frac_force_upstream']:.0f}% "
              f"of {e['n']} body-position pairs")
    for e in A2b:
        print(f"    {e['body']:>20s}, {e['region']:>18s}: {e['stable']}/{e['n']} stable "
              f"({e['floquet_confirmed']} Floquet-confirmed)")
else:
    print("  Script 02 outputs not found; A2 skipped")
SUMMARY["A_passive"] = A

# =============================================================================
# B. Where in the street is holding stable?
# =============================================================================
print("\n=== B. Position of stable holding ===")
B = {}
def dist(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return dict(n=int(v.size), min=float(v.min()), q10=float(np.quantile(v, 0.1)), median=float(np.median(v)),
                q90=float(np.quantile(v, 0.9)), max=float(v.max()))
rowsB = []
if PTS is not None:
    for dp, lab in ((0.0, "baseline, phase lag 0"), (np.pi / 2, "baseline, phase lag pi/2")):
        m = (PTS[:, 7] > 0.5) & (np.abs(PTS[:, 3] - dp) < 1e-6)
        x = PTS[m, 5]
        B[lab] = dict(x=dist(x), x_over_peak=dist(x / X_PEAK_BASE), x_over_xr=dist(x / X_R),
                      body_lengths=dist(x * BL_PER_LAMBDA), diameters=dist(x * D_PER_LAMBDA))
        rowsB.append([lab, B[lab]["x"]["n"], B[lab]["x"]["min"], B[lab]["x"]["median"], B[lab]["x"]["max"],
                      B[lab]["x_over_peak"]["min"], B[lab]["x_over_peak"]["median"], B[lab]["x_over_peak"]["max"]])
f9 = P("script09_model_structure", "records_script09_ALL.json")
if os.path.exists(f9):
    R9 = json.load(open(f9))
    STRUCT_OVR = {"plateau deficit": dict(shape=1), "algebraic decay": dict(shape=2), "bump envelope": dict(shape=3)}
    lhs_ovr = {}
    for fn in sorted(os.listdir(P("script09_model_structure", CK))) if os.path.isdir(P("script09_model_structure", CK)) else []:
        if "LHS" in fn:
            c = json.load(open(P("script09_model_structure", CK, fn)))
            lhs_ovr[c["res"]["name"]] = c["res"]["override"]
    xf_ = np.linspace(0.0, 4.0, 4001)
    def peak_of(ov):
        s_, W0_, xr, lw = ov.get("shape", 0), ov.get("W0", W0), ov.get("x_r", X_R), ov.get("l_w", L_W)
        rise = 1.0 - np.exp(-(xf_ / xr) ** 2)
        if s_ == 2:
            Wv = W0_ * rise / (1.0 + xf_ / lw)
        elif s_ == 3:
            Wv = W0_ * np.exp(-((xf_ - xr) / (0.5 * lw)) ** 2)
        else:
            Wv = W0_ * rise * np.exp(-xf_ / lw)
        return float(xf_[np.argmax(Wv)]), float(xr)
    per_var = {}
    allr = []
    for r_ in R9:
        if not r_.get("stable"):
            continue
        v = r_["variant"]
        ov = STRUCT_OVR.get(v, lhs_ovr.get(v, {}))
        xp, xr = peak_of(ov)
        per_var.setdefault(v, []).append((r_["x"], r_["x"] / xp, r_["x"] / xr))
        allr.append(r_["x"] / xp)
    B["variants"] = {}
    for v, arr in per_var.items():
        a = np.array(arr)
        B["variants"][v] = dict(x=dist(a[:, 0]), x_over_peak=dist(a[:, 1]), x_over_xr=dist(a[:, 2]))
        rowsB.append([v, a.shape[0], a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(),
                      a[:, 1].min(), np.median(a[:, 1]), a[:, 1].max()])
    B["all_variants_x_over_peak"] = dist(allr)
    B["frac_all_variants_downstream_of_peak"] = float(np.mean(np.array(allr) > 1.0))
f14 = P("script14_referee_analyses", "summary_script14.json")
if os.path.exists(f14):
    s14 = json.load(open(f14))
    sw = s14.get("I_sweep", s14.get("I_phase_sweep", {}))
    if isinstance(sw, dict):
        B["phase_lag_sweep_x_range"] = {k: v.get("x_range") for k, v in sw.items() if isinstance(v, dict) and v.get("x_range")}
if "A2_stable_passive_r" in A:
    B["passive_stable_x_over_peak"] = A["A2_stable_passive_r"]
save_csv("B_holding_positions.csv", ["set", "n", "x_min", "x_median", "x_max", "xpeak_min", "xpeak_median", "xpeak_max"], rowsB)
for r in rowsB:
    print(f"    {r[0]:>28s}: n {r[1]:4d}, x {r[2]:.2f} to {r[4]:.2f} (median {r[3]:.2f}) wavelengths, "
          f"x/x_peak {r[5]:.2f} to {r[7]:.2f} (median {r[6]:.2f})")
if "all_variants_x_over_peak" in B and B["all_variants_x_over_peak"]:
    print(f"  all variants and samples: {100 * B['frac_all_variants_downstream_of_peak']:.1f}% of stable orbits lie "
          f"downstream of their own envelope peak")
if "A2_stable_passive_r" in A:
    print(f"  stable passive equilibria: median x/x_peak {A['A2_stable_passive_r']['median']:.2f}")
SUMMARY["B_positions"] = B

# =============================================================================
# C. Control experiment: uniform flow
# =============================================================================
print("\n=== C1. Reactive-thrust variant: recover stable orbits in the wake ===")
C = {}
CT_VAR = 0.1
fC1 = ckpt("C1_orbits.npz")
if os.path.exists(fC1):
    dC1 = dict(np.load(fC1))
    print("  loaded checkpoint")
elif os.path.exists(f9):
    recs = [r_ for r_ in R9 if r_["variant"] == "added reactive thrust" and r_.get("stable")]
    n_r = len(recs)
    ph = np.linspace(0, 2 * np.pi, N_SEED_PH, endpoint=False)
    ri = np.repeat(np.arange(n_r), N_SEED_PH)
    pp = np.tile(ph, n_r)
    S0 = np.zeros((ri.size, 6))
    S0[:, 0] = [recs[i]["x"] for i in ri]
    S0[:, 4], S0[:, 5] = np.cos(pp), np.sin(pp)
    PP = par(ri.size, ratio=[recs[i]["ratio"] for i in ri], eps=[recs[i]["eps"] for i in ri],
             dphi=[recs[i]["dphi"] for i in ri], cT=CT_VAR)
    sN, alive, _, _, xs = run(T_RECOVER, S0, PP, label="recover")
    s1 = run(1, sN, PP)[0]
    resid = np.max(np.abs(s1 - sN), 1)
    ok = alive & (np.abs(sN[:, 0] - S0[:, 0]) < 0.02) & (resid < 1e-6)
    keep_s, keep_r = [], []
    for i in range(n_r):
        j = np.where(ok & (ri == i))[0]
        if j.size:
            keep_s.append(sN[j[0]]); keep_r.append(i)
    dC1 = dict(S=np.array(keep_s).reshape(-1, 6), rec=np.array(keep_r, int), n_rec=n_r,
               ratio=np.array([recs[i]["ratio"] for i in keep_r]), eps=np.array([recs[i]["eps"] for i in keep_r]),
               dphi=np.array([recs[i]["dphi"] for i in keep_r]))
    np.savez(fC1, **dC1)
else:
    dC1 = None
if dC1 is not None:
    n_ok, n_rec = dC1["S"].shape[0], int(dC1["n_rec"])
    G["G2_recovered"] = [n_ok, n_rec]
    G["G2_pass"] = bool(n_rec > 0 and n_ok >= 0.5 * n_rec)
    print(f"  recovered {n_ok} of {n_rec} stable orbits ({'pass' if G['G2_pass'] else 'FAIL'}), "
          f"x {dC1['S'][:, 0].min():.2f} to {dC1['S'][:, 0].max():.2f}" if n_ok else "  none recovered")
    C["C1"] = dict(recovered=n_ok, of=n_rec, x=dist(dC1["S"][:, 0]))

if dC1 is not None and dC1["S"].shape[0]:
    sel = np.arange(dC1["S"].shape[0])
    if sel.size > N_ORB_C:
        sel = np.sort(rng.choice(sel, N_ORB_C, replace=False))
    SW = dC1["S"][sel]; RAT, EPS_, DPH = dC1["ratio"][sel], dC1["eps"][sel], dC1["dphi"][sel]
    no = sel.size

    print("\n=== C2. Uniform flow: drift speed against muscle gain ===")
    fC2 = ckpt("C2_scan.npz")
    if os.path.exists(fC2):
        dC2 = dict(np.load(fC2))
        print("  loaded checkpoint")
    else:
        nct, ng = len(CT_SCAN), G_SCAN.size
        oi = np.repeat(np.arange(no), nct * ng)
        ci = np.tile(np.repeat(np.arange(nct), ng), no)
        gi = np.tile(np.arange(ng), no * nct)
        S0 = np.zeros((oi.size, 6)); S0[:, 0] = 1.0; S0[:, 4] = 1.0
        PP = par(oi.size, ratio=RAT[oi], eps=EPS_[oi], dphi=DPH[oi], g=G_SCAN[gi], cT=np.array(CT_SCAN)[ci],
                 wf=0.0, df=0.0, bnd=0.0)
        sN, _, _, _, xs = run(T_SCAN, S0, PP, label="scan")
        nlast = max(10, T_SCAN // 4)
        drift = (xs[:, -1] - xs[:, -1 - nlast]) / nlast
        dC2 = dict(drift=drift.reshape(no, nct, ng))
        # refine the zero crossing of the drift for c_T = CT_VAR by secant iterations
        c_v = CT_SCAN.index(CT_VAR) if CT_VAR in CT_SCAN else None
        gstar = np.full(no, np.nan)
        if c_v is not None:
            for o in range(no):
                d_ = dC2["drift"][o, c_v]
                z_ = np.where(np.sign(d_[:-1]) != np.sign(d_[1:]))[0]
                if z_.size:
                    j = z_[0]
                    gstar[o] = G_SCAN[j] - d_[j] * (G_SCAN[j + 1] - G_SCAN[j]) / (d_[j + 1] - d_[j])
            slope_full = np.full(no, np.nan)
            for it in range(4):
                okg = np.where(np.isfinite(gstar))[0]
                if not okg.size:
                    break
                h = 0.25
                gg = np.concatenate([gstar[okg], gstar[okg] + h])
                oo = np.concatenate([okg, okg])
                S0 = np.zeros((oo.size, 6)); S0[:, 0] = 1.0; S0[:, 4] = 1.0
                PP = par(oo.size, ratio=RAT[oo], eps=EPS_[oo], dphi=DPH[oo], g=gg, cT=CT_VAR, wf=0.0, df=0.0, bnd=0.0)
                xs = run(T_SCAN, S0, PP)[4]
                dr = (xs[:, -1] - xs[:, -1 - nlast]) / nlast
                d0, d1 = dr[:okg.size], dr[okg.size:]
                slope = (d1 - d0) / h
                slope_full[okg] = slope
                step_ = d0 / np.where(np.abs(slope) > 1e-12, slope, np.nan)
                gstar[okg] = gstar[okg] - step_
                gstar[(gstar < 0) | (gstar > 4 * G_SCAN[-1])] = np.nan
            dC2["gstar_slope"] = slope_full
        dC2["gstar"] = gstar
        np.savez(fC2, **dC2)
    gstar = dC2["gstar"]
    C["C2"] = dict(c_T_scan=CT_SCAN, g_scan=[float(G_SCAN[0]), float(G_SCAN[-1])],
                   baseline_force_law_drift=dist(dC2["drift"][:, CT_SCAN.index(0.0), :].ravel()) if 0.0 in CT_SCAN else None,
                   gstar=dist(gstar), gstar_over_wake_gain=dist(gstar / 16.0),
                   n_with_balance=int(np.isfinite(gstar).sum()))
    if 0.0 in CT_SCAN:
        dz = dC2["drift"][:, CT_SCAN.index(0.0), :]
        print(f"  baseline force law (no reactive thrust): drift {dz.min():+.3f} to {dz.max():+.3f} wavelengths "
              f"per period for every gain, the body is always swept downstream")
    for c_i, cT in enumerate(CT_SCAN):
        d_ = dC2["drift"][:, c_i, :]
        print(f"  c_T = {cT}: drift at g = 16 (the wake gain): "
              f"{np.interp(16.0, G_SCAN, np.median(d_, 0)):+.4f} wavelengths per period (median over orbits); "
              f"range over gains {d_.min():+.3f} to {d_.max():+.3f}")
    if np.isfinite(gstar).any():
        sl_ = dC2.get("gstar_slope", np.full(no, np.nan))
        drift_1pct = np.abs(sl_ * 0.01 * gstar)               # wavelengths per period for a 1% gain error
        t_quarter = 0.25 / drift_1pct
        C["C2"]["drift_per_1pct_gain_error"] = dist(drift_1pct)
        C["C2"]["periods_to_drift_quarter_wavelength_1pct"] = dist(t_quarter)
        print(f"  thrust balances drag at g* = {np.nanmedian(gstar):.1f} (median; wake fish uses 16) in "
              f"{np.isfinite(gstar).sum()} of {no} fish; a 1% gain error drifts "
              f"{np.nanmedian(drift_1pct):.2e} wavelengths per period, 0.25 wavelengths in "
              f"{np.nanmedian(t_quarter):.0f} periods")

    print("\n=== C3. Positional kick: wake versus uniform flow ===")
    fC3 = ckpt("C3_kick.npz")
    okg = np.where(np.isfinite(gstar))[0]
    if os.path.exists(fC3):
        dC3 = dict(np.load(fC3))
        print("  loaded checkpoint")
    else:
        KICK = 0.05
        T_K = 200
        # wake: kicked and unkicked from the orbit
        S0w = np.concatenate([SW, SW + np.array([KICK, 0, 0, 0, 0, 0])])
        PPw = par(2 * no, ratio=np.tile(RAT, 2), eps=np.tile(EPS_, 2), dphi=np.tile(DPH, 2), cT=CT_VAR)
        xw = run(T_K, S0w, PPw)[4]
        dC3 = dict(wake_diff=xw[no:] - xw[:no])
        if okg.size:
            S0 = np.zeros((okg.size, 6)); S0[:, 0] = 1.0; S0[:, 4] = 1.0
            PPu = par(okg.size, ratio=RAT[okg], eps=EPS_[okg], dphi=DPH[okg], g=gstar[okg], cT=CT_VAR,
                      wf=0.0, df=0.0, bnd=0.0)
            Su = run(T_SCAN, S0, PPu)[0]
            S0u = np.concatenate([Su, Su + np.array([KICK, 0, 0, 0, 0, 0])])
            xu = run(T_K, S0u, np.concatenate([PPu, PPu]))[4]
            dC3["uniform_diff"] = xu[okg.size:] - xu[:okg.size]
            dC3["uniform_settled"] = Su
        np.savez(fC3, **dC3)
    wd = dC3["wake_diff"]
    C["C3"] = dict(kick=0.05, wake_remaining_after_50=dist(np.abs(wd[:, 49]) / 0.05),
                   wake_remaining_after_200=dist(np.abs(wd[:, -1]) / 0.05))
    print(f"  wake: fraction of the kick remaining after 50 periods, median {np.median(np.abs(wd[:, 49]) / 0.05):.3f}")
    if "uniform_diff" in dC3:
        ud = dC3["uniform_diff"]
        rem = ud[:, -1] / 0.05
        C["C3"]["uniform_remaining_after_200"] = dist(rem)
        G["G3_uniform_kick_remaining"] = dist(rem)
        G["G3_pass"] = bool(np.all(np.abs(rem - 1.0) < 0.02))
        print(f"  uniform flow: fraction remaining after 200 periods, {rem.min():.3f} to {rem.max():.3f} "
              f"({'pass' if G['G3_pass'] else 'FAIL'}: position is neutral)")

    print("\n=== C4. Noise: wake versus uniform flow ===")
    fC4 = ckpt("C4_noise.npz")
    if os.path.exists(fC4):
        dC4 = dict(np.load(fC4))
        print("  loaded checkpoint")
    elif okg.size:
        Su = dC3["uniform_settled"]
        rowsets = []
        S_all, P_all, NZ_all, tags = [], [], [], []
        for lv, D in enumerate(NOISE_LEVELS):
            nz = np.array([0.0, 0.0, np.sqrt(D), D])          # free-stream speed fluctuation and CPG phase noise
            for cond in (0, 1):                               # 0 wake, 1 uniform flow
                if cond == 0:
                    o_ = okg
                    S_ = np.repeat(SW[o_], N_REP, 0)
                    P_ = par(S_.shape[0], ratio=np.repeat(RAT[o_], N_REP), eps=np.repeat(EPS_[o_], N_REP),
                             dphi=np.repeat(DPH[o_], N_REP), cT=CT_VAR)
                else:
                    o_ = okg
                    S_ = np.repeat(Su, N_REP, 0)
                    P_ = par(S_.shape[0], ratio=np.repeat(RAT[o_], N_REP), eps=np.repeat(EPS_[o_], N_REP),
                             dphi=np.repeat(DPH[o_], N_REP), g=np.repeat(gstar[o_], N_REP), cT=CT_VAR,
                             wf=0.0, df=0.0, bnd=0.0)
                S_all.append(S_); P_all.append(P_); NZ_all.append(np.tile(nz, (S_.shape[0], 1)))
                tags.append(np.column_stack([np.full(S_.shape[0], lv), np.full(S_.shape[0], cond),
                                             np.repeat(np.arange(o_.size), N_REP)]))
        S_all, P_all, NZ_all, tags = map(np.concatenate, (S_all, P_all, NZ_all, tags))
        _, alive, tloss, _, xs = run(T_NOISE, S_all, P_all, NZ_all, noisy=True, seed=SEED, label="noise")
        dC4 = dict(tags=tags, alive=alive, tloss=tloss, x0=S_all[:, 0], xs=xs.astype(np.float32))
        np.savez(fC4, **dC4)
    else:
        dC4 = None
    if dC4 is not None:
        C["C4"] = {}
        rows = []
        for lv, D in enumerate(NOISE_LEVELS):
            for cond, cname in ((0, "wake"), (1, "uniform flow")):
                m = (dC4["tags"][:, 0] == lv) & (dC4["tags"][:, 1] == cond)
                dev = (dC4["xs"][m] - dC4["x0"][m][:, None]).astype(float)
                tl_m = dC4["tloss"][m]
                tgrid = np.arange(1, T_NOISE + 1)[None, :]
                dev_alive = np.where(np.isfinite(tl_m)[:, None] & (tgrid >= tl_m[:, None]), np.nan, dev)
                rec = {}
                for band in (0.1, 0.25):
                    out = np.abs(dev) > band
                    first = np.where(out.any(1), out.argmax(1) + 1, np.inf)
                    if cond == 0:
                        lost = np.isfinite(dC4["tloss"][m])
                        first = np.where(lost, np.minimum(first, dC4["tloss"][m]), first)
                    cens = ~np.isfinite(first)
                    # Kaplan-Meier median with censoring at T_NOISE
                    tt = np.where(cens, T_NOISE, first)
                    order = np.argsort(tt)
                    surv, med = 1.0, np.nan
                    n_at = tt.size
                    for t_, c_ in zip(tt[order], cens[order]):
                        if not c_:
                            surv *= (n_at - 1) / n_at
                            if surv <= 0.5 and not np.isfinite(med):
                                med = t_
                        n_at -= 1
                    rec[f"band_{band}"] = dict(km_median=float(med) if np.isfinite(med) else None,
                                               censored_frac=float(cens.mean()))
                var_t = np.nanvar(dev_alive, 0)
                tt_ = np.array([50, 100, 200, 400, min(800, T_NOISE - 1)])
                tt_ = tt_[tt_ < T_NOISE]
                rec["position_variance"] = {int(t_): float(var_t[t_]) for t_ in tt_}
                if cond == 0:
                    lost = np.isfinite(dC4["tloss"][m])
                    rec["residence_lost_frac"] = float(lost.mean())
                C["C4"][f"D={D} {cname}"] = rec
                rows.append([D, cname, rec["band_0.1"]["km_median"], rec["band_0.25"]["km_median"],
                             rec["band_0.25"]["censored_frac"]] + [rec["position_variance"].get(int(t_), "") for t_ in tt_])
                print(f"  D = {D}, {cname:>12s}: median time to leave +/-0.1 {rec['band_0.1']['km_median']}, "
                      f"+/-0.25 {rec['band_0.25']['km_median']} periods; position variance at "
                      + ", ".join(f"{t_}: {rec['position_variance'][int(t_)]:.4f}" for t_ in tt_))
        save_csv("C4_noise_wake_vs_uniform.csv", ["D", "condition", "median_leave_0.1", "median_leave_0.25",
                 "censored_0.25"] + [f"var_x_t{int(t_)}" for t_ in tt_], rows)
SUMMARY["C_uniform_flow"] = C
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
    ze_show = 0.1 if 0.1 in ZE_GRID else ZE_GRID[0]
    bsel = [b for b in range(nb) if abs(BA[b, 1] - ze_show) < 1e-9 and BA[b, 2] < 0.5]
    M = np.array([fate[dA["bb"] == b] for b in bsel])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cm = matplotlib.colors.ListedColormap(["#4caf50", "#1f77b4", "#d62728"])
    ax[0].pcolormesh(X0_GRID, BA[bsel, 0], M, cmap=cm, vmin=-0.5, vmax=2.5, shading="nearest")
    for j, b in enumerate(bsel):
        for e in A_bodies[b]["equilibria"]:
            ax[0].plot(e, BA[b, 0], "k.", ms=4)
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"release position $x_0/\lambda$"); ax[0].set_ylabel(r"$\Omega/\omega_w$")
    ax[0].axvline(X_PEAK_BASE, color="k", ls="--", lw=1)
    ax[0].legend(handles=[matplotlib.patches.Patch(color=c_, label=l_) for c_, l_ in
                          zip(cm.colors, ["stays", "upstream to cylinder", "downstream out of wake"])],
                 loc="upper right", fontsize=10, framealpha=1, edgecolor="k")
    for b in bsel[:: max(1, len(bsel) // 5)]:
        ax[1].plot(XF_GRID, dA["F"][b], lw=1.2, label=f"{BA[b, 0]:.2f}")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(X_PEAK_BASE, color="k", ls="--", lw=1)
    ax[1].set_xlabel(r"$x/\lambda$"); ax[1].set_ylabel(r"$\langle F_x\rangle$")
    ax[1].legend(title=r"$\Omega/\omega_w$", fontsize=9, framealpha=1, edgecolor="k")
    save(fig, "S17_A_passive_fates")
except Exception as e:
    print(f"  figure A skipped: {e}")
try:
    if "C2" in C:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for c_i, cT in enumerate(CT_SCAN):
            ax[0].plot(G_SCAN, np.median(dC2["drift"][:, c_i, :], 0), lw=2, label=f"$c_T={cT}$")
        ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(16, color="k", ls="--", lw=1)
        ax[0].set_xlabel("muscle gain $g$"); ax[0].set_ylabel("drift, wavelengths per period (+ downstream)")
        ax[0].legend(framealpha=1, edgecolor="k")
        if dC4 is not None:
            for lv, D in enumerate(NOISE_LEVELS):
                for cond, ls in ((0, "-"), (1, "--")):
                    m = (dC4["tags"][:, 0] == lv) & (dC4["tags"][:, 1] == cond)
                    dev = dC4["xs"][m] - dC4["x0"][m][:, None]
                    ax[1].plot(np.arange(1, T_NOISE + 1), np.nanvar(dev, 0), ls,
                               label=f"D={D}, {'wake' if cond == 0 else 'uniform'}")
            ax[1].set_xlabel("time, shedding periods"); ax[1].set_ylabel("variance of position")
            ax[1].legend(fontsize=9, framealpha=1, edgecolor="k")
        save(fig, "S17_C_uniform_flow")
except Exception as e:
    print(f"  figure C skipped: {e}")

SUMMARY["runtime_s"] = time.time() - t_start
with open(os.path.join(OUT_DIR, "summary_script17.json"), "w") as f:
    json.dump(to_py(SUMMARY), f, indent=2)

print("\n================ SCRIPT 17 SUMMARY ================")
print(f"Gates: " + ", ".join(f"{k}: {v}" for k, v in G.items() if k.endswith("pass")))
print(f"Outputs: {OUT_DIR}")
print(f"Runtime this session: {time.time() - t_start:.0f} s")
