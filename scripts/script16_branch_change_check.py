# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 16: Identify the fixed-position branch change in the convergence check.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX; runtime about 10 min on a CPU, less on a GPU.
# Needs the Script 10 outputs (points, branches, refined index) on Drive.
#
# Draws exactly the same 500-point stratified sample as Script 12, part A
# (same seed and sampling order), classifies the fixed-position branch of
# every point by homotopy at 100 and at 400 steps per period, and prints every
# point whose branch differs between the two resolutions, with its parameters,
# multipliers, a_L and index. The model and the homotopy are those of
# Script 12, reproduced unchanged below.
#
# Output: script16_branch_change/branch_change_check.npz
# Result in the manuscript (Appendix C): one change, a stable orbit at
# epsilon = 0.25, phase lag pi/2, omega_c/omega_w = 1.006, which reaches an
# unlocked rhythm at 100 steps and the node branch with a_L < 0 at 400 steps.
# =============================================================================

import os, sys, json, time, hashlib, warnings
import numpy as np

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
OUT_DIR = os.path.join(BASE_DIR, "script12_verification")
CKPT_DIR = os.path.join(OUT_DIR, f"checkpoints_{MODE.lower()}")
os.makedirs(CKPT_DIR, exist_ok=True)
Q = "" if MODE == "FULL" else "_quick"
S10_DIR = os.path.join(BASE_DIR, "script10_continuation")
S10_CK = os.path.join(S10_DIR, f"checkpoints_{MODE.lower()}")
REC08 = os.path.join(BASE_DIR, "script08_homotopy", "records_script08.json")
REC09 = os.path.join(BASE_DIR, "script09_model_structure", f"records_script09_ALL{Q}.json")

ON_GPU = any(d.platform == "gpu" for d in jax.devices())
SPP = 100
if MODE == "FULL":
    N_PER_GROUP = 100
    N_EDGE = 100
    SPP_LEVELS = [100, 200, 400]
    MAX_REC = None
    CHUNK = 8192 if ON_GPU else 512
else:
    N_PER_GROUP = 3
    N_EDGE = 3
    SPP_LEVELS = [100, 200]
    MAX_REC = 6
    CHUNK = 256
FD_STEPS = [1e-5, 1e-6, 1e-7]
ETAS_FINE = list(np.geomspace(1.0, 0.005, 25))
AL_SCALES = [0.005, 0.0015, 0.02]
AL_MARGIN = 0.02
N_PHASE_SEEDS = 24
OMEGA_W = 2.0 * np.pi

# -----------------------------------------------------------------------------
# 1. Model variants (identical definitions to Scripts 08 and 09)
# -----------------------------------------------------------------------------
BASE_CFG = dict(
    U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35, l_w=1.50, shape=0,
    c_L=3.0, c_D=0.15, sigma=8.0, lift="qs", tilt=1.0, c_T=0.0, LT=0.5, tau=0.0, m_a=0.0,
    body_ratio=1.2, zeta=0.2, heave=False, Om_y_ratio=1.0, ze_y=0.3,
    mode2=False, Om2_ratio=2.2, w2=0.5,
    cpg="sl", mu=2 * np.pi, c=0.0, mu_v=1.0, g=16.0, wp=1.0, eLL=0.0)

VAR06 = {"baseline": {}, "c = 1": dict(c=1.0), "mu x 0.5": dict(mu=np.pi), "mu x 2": dict(mu=4 * np.pi),
         "g = 12": dict(g=12.0), "g = 24": dict(g=24.0), "body ratio 0.8": dict(body_ratio=0.8),
         "body ratio 1.6": dict(body_ratio=1.6), "zeta 0.1": dict(zeta=0.1), "zeta 0.4": dict(zeta=0.4),
         "W0 0.25": dict(W0=0.25), "W0 0.45": dict(W0=0.45), "c_D 0.10": dict(c_D=0.10), "c_D 0.25": dict(c_D=0.25)}
for r_ in (0.9, 1.0, 1.1, 1.3, 1.4, 1.5):
    for z_ in (0.1, 0.2):
        VAR06[f"body {r_:.1f}, zeta {z_:.1f}"] = dict(body_ratio=r_, zeta=z_)

STRUCT = [("baseline", {}), ("linear lift + lever arm", dict(lift="linear", LT=0.5)),
          ("added reactive thrust", dict(c_T=0.1)), ("reactive thrust only", dict(tilt=0.0, c_T=0.3)),
          ("lift lag", dict(tau=0.08)), ("lift lag + added mass", dict(tau=0.08, m_a=0.5)),
          ("pitch + free heave", dict(heave=True, Om_y_ratio=0.25)),
          ("pitch + resonant heave", dict(heave=True, Om_y_ratio=1.0)),
          ("pitch + stiff heave", dict(heave=True, Om_y_ratio=2.0)), ("two-mode body", dict(mode2=True)),
          ("van der Pol CPG", dict(cpg="vdp")), ("lateral line + proprio", dict(eLL=1.0)),
          ("lateral line only", dict(eLL=1.0, wp=0.0))]
WAKES = [("plateau deficit", dict(shape=1)), ("algebraic decay", dict(shape=2)), ("bump envelope", dict(shape=3))]
LHS_RANGES = dict(body_ratio=(1.0, 1.4), zeta=(0.1, 0.3), g=(12.0, 24.0), mu=(np.pi, 4 * np.pi),
                  W0=(0.25, 0.45), c_D=(0.10, 0.25), c_L=(2.5, 3.5), sigma=(6.0, 10.0),
                  x_r=(0.30, 0.45), l_w=(1.2, 1.8), delta0=(1.2, 1.4))

def lhs_variants(n=20, seed=13):
    """Exactly the Script 09 design (same generator, same first draws)."""
    r_ = np.random.default_rng(seed)
    keys = list(LHS_RANGES)
    u = np.zeros((n, len(keys)))
    for j in range(len(keys)):
        u[:, j] = (r_.permutation(n) + r_.uniform(0, 1, n)) / n
    return [(f"LHS {i:02d}", {k: float(LHS_RANGES[k][0] + u[i, j] * (LHS_RANGES[k][1] - LHS_RANGES[k][0]))
                              for j, k in enumerate(keys)}) for i in range(n)]

VAR09 = dict(STRUCT + WAKES + lhs_variants())

def vdp_cycle(mu_v):
    """Free van der Pol cycle, parametrised by the phase of the z_r fundamental (Script 09)."""
    f = lambda z: np.array([-z[1], z[0] + mu_v * (1.0 - z[0] ** 2) * z[1]])
    z = np.array([2.0, 0.0]); dt = 2 * np.pi / 4000
    def rk(z):
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for _ in range(30 * 4000):
        z = rk(z)
    pts, cr, n = [z.copy()], [], 0
    while len(cr) < 2:
        zn = rk(z); n += 1
        if z[1] < 0 <= zn[1]:
            cr.append(n)
        z = zn; pts.append(z.copy())
    pts = np.array(pts)[cr[0]:cr[1]]
    th = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)
    c1 = np.mean(pts[:, 0] * np.exp(-1j * th))
    return 2 * np.abs(c1), (np.angle(np.exp(1j * (th + np.angle(c1)))), pts[:, 0], pts[:, 1])

# numeric parameter vector (per orbit)
QK = ["ratio", "eps", "dphi", "eta", "U_inf", "delta0", "l_delta", "W0", "x_r", "l_w", "shape",
      "c_L", "c_D", "sigma", "lin", "tilt", "c_T", "LT", "tau", "m_a", "body_ratio", "zeta",
      "Om_y_ratio", "ze_y", "Om2_ratio", "w2", "vdp", "mu", "c", "mu_v", "g", "wp", "eLL", "amp"]
QI = {k: i for i, k in enumerate(QK)}

def layout(cfg):
    names = ["x", "u", "th", "om"]
    if cfg["heave"]:
        names += ["y", "vy"]
    if cfg["mode2"]:
        names += ["th2", "om2"]
    if cfg["tau"] > 0:
        names += ["Lf"]
    return names + ["zr", "zi"]

def qvec(cfg, ratio, eps, dphi, amp):
    v = dict(cfg)
    v.update(ratio=ratio, eps=eps, dphi=dphi, eta=1.0, lin=1.0 if cfg["lift"] == "linear" else 0.0,
             vdp=1.0 if cfg["cpg"] == "vdp" else 0.0, amp=amp)
    return np.array([float(v[k]) for k in QK])

MODELS = {}

def model(lay_key):
    """Compiled functions for one state layout (heave, mode2, lag)."""
    if lay_key in MODELS:
        return MODELS[lay_key]
    heave, mode2, lag = lay_key
    names = ["x", "u", "th", "om"] + (["y", "vy"] if heave else []) + (["th2", "om2"] if mode2 else []) \
            + (["Lf"] if lag else []) + ["zr", "zi"]
    L_ = {n: i for i, n in enumerate(names)}
    D = len(names); IZ = L_["zr"]

    def rhs(t, s, q):
        g_ = lambda k: q[QI[k]]
        x, u, th, om = s[0], s[1], s[2], s[3]
        zr, zi = s[IZ], s[IZ + 1]
        sh = g_("shape")
        d0, ld = g_("delta0"), g_("l_delta")
        dfc = jnp.where(sh == 1, d0 * jnp.exp(-(x / ld) ** 2),
                        jnp.where(sh == 2, d0 / jnp.sqrt(1.0 + (x / ld) ** 2), d0 * jnp.exp(-x / ld)))
        rise = 1.0 - jnp.exp(-(x / g_("x_r")) ** 2)
        Wv = jnp.where(sh == 2, g_("W0") * rise / (1.0 + x / g_("l_w")),
                       jnp.where(sh == 3, g_("W0") * jnp.exp(-((x - g_("x_r")) / (0.5 * g_("l_w"))) ** 2),
                                 g_("W0") * rise * jnp.exp(-x / g_("l_w"))))
        V = g_("U_inf") * (1.0 - dfc) - u
        w = Wv * jnp.sin(OMEGA_W * (x - t))
        vy = s[L_["vy"]] if heave else 0.0
        if mode2:
            th_e = th + g_("w2") * s[L_["th2"]]; om_e = om + g_("w2") * s[L_["om2"]]
        else:
            th_e, om_e = th, om
        gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
        beta = jnp.arctan2(w - vy, jnp.abs(V))
        w_te = w - vy - jnp.abs(V) * th_e - g_("LT") * om_e
        alpha = beta - th_e
        Lqs = g_("c_L") * (V * V + (w - vy) ** 2) * jnp.sin(alpha) * jnp.cos(alpha) * gate
        Llin = g_("c_L") * jnp.abs(V) * w_te * gate
        Lq = jnp.where(g_("lin") > 0.5, Llin, Lqs)
        d = [None] * D
        if lag:
            Lf = s[L_["Lf"]]
            d[L_["Lf"]] = (Lq - Lf) / g_("tau")
            L = Lf
        else:
            L = Lq
        Fx = g_("c_D") * V * jnp.abs(V) - g_("tilt") * L * jnp.sin(beta) - g_("c_T") * w_te * w_te * gate
        drive = (zr * jnp.cos(g_("dphi")) + zi * jnp.sin(g_("dphi"))) / g_("amp")
        OmB, zeB = g_("body_ratio") * OMEGA_W, g_("zeta")
        d[2] = om
        d[3] = -2 * zeB * OmB * om - OmB ** 2 * th + g_("sigma") * L + g_("g") * drive
        if mode2:
            Om2 = g_("Om2_ratio") * OmB
            d[L_["th2"]] = s[L_["om2"]]
            d[L_["om2"]] = -2 * zeB * Om2 * s[L_["om2"]] - Om2 ** 2 * s[L_["th2"]] + g_("sigma") * L + g_("g") * drive
        if heave:
            Omy = g_("Om_y_ratio") * OMEGA_W
            d[L_["y"]] = vy
            d[L_["vy"]] = -2 * g_("ze_y") * Omy * vy - Omy ** 2 * s[L_["y"]] + L * jnp.cos(beta)
        inp = g_("eps") * (g_("wp") * th_e + g_("eLL") * w)
        wc = g_("ratio") * OMEGA_W
        r2 = zr * zr + zi * zi
        a = g_("mu") * (1.0 - r2)
        b = g_("c") * g_("mu") * (1.0 - r2) + wc
        dzr = jnp.where(g_("vdp") > 0.5, -wc * zi + inp, a * zr - b * zi + inp)
        dzi = jnp.where(g_("vdp") > 0.5, wc * (zr + g_("mu_v") * (1.0 - zr * zr) * zi), a * zi + b * zr)
        d[IZ] = dzr; d[IZ + 1] = dzi
        eta = g_("eta")
        d[0] = eta * u
        d[1] = eta * Fx / (1.0 + g_("m_a"))
        return jnp.stack(d), Fx

    def pmap(s, q, spp):
        dt = 1.0 / spp
        def step(i, c):
            s_, F_ = c
            t = i * dt
            k1, f1 = rhs(t, s_, q)
            k2, f2 = rhs(t + dt / 2, s_ + dt / 2 * k1, q)
            k3, f3 = rhs(t + dt / 2, s_ + dt / 2 * k2, q)
            k4, f4 = rhs(t + dt, s_ + dt * k3, q)
            return s_ + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F_ + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        return jax.lax.fori_loop(0, spp, step, (s, 0.0))

    def newton(s, q, mask, n_it, clip, spp):
        I_ = jnp.eye(D)
        def body(i, c):
            s_, _ = c
            s1, _ = pmap(s_, q, spp)
            M = jax.jacfwd(lambda y: pmap(y, q, spp)[0])(s_)
            r = (s1 - s_) * mask
            A = (M - I_) * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask)
            dd = jnp.clip(jnp.nan_to_num(jnp.linalg.solve(A, -r)), -clip, clip) * mask
            return s_ + dd, 0.0
        s, _ = jax.lax.fori_loop(0, n_it, body, (s, 0.0))
        s1, F = pmap(s, q, spp)
        M = jax.jacfwd(lambda y: pmap(y, q, spp)[0])(s)
        return s, jnp.max(jnp.abs((s1 - s) * mask)), M, F

    def fd_monodromy(s, q, h, spp):
        s1, _ = pmap(s, q, spp)
        cols = [(pmap(s + h * jnp.eye(D)[j], q, spp)[0] - s1) / h for j in range(D)]
        return jnp.stack(cols, 1)

    def relax(s, q, n_per):
        def per(k, s_):
            return pmap(s_, q, SPP)[0]
        return jax.lax.fori_loop(0, n_per, per, s)

    mall = jnp.ones(D)
    mint = jnp.ones(D).at[0].set(0.0).at[1].set(0.0)
    fns = {}
    def get(name, spp=SPP, **kw):
        key = (name, spp, tuple(sorted(kw.items())))
        if key not in fns:
            if name == "full":
                f = lambda s, q: newton(s, q, mall, kw.get("n_it", 12), 0.05, spp)
            elif name == "hom":
                f = lambda s, q: newton(s, q, mall, 8, 0.05, spp)
            elif name == "int":
                f = lambda s, q: newton(s, q, mint, 12, 0.2, spp)
            elif name == "fd":
                f = lambda s, q: fd_monodromy(s, q, kw["h"], spp)
            elif name == "relax":
                f = lambda s, q: relax(s, q, kw["n_per"])
            fns[key] = jax.jit(jax.vmap(f))
        return fns[key]
    MODELS[lay_key] = dict(D=D, IZ=IZ, names=names, get=get)
    return MODELS[lay_key]

def lay_key(cfg):
    return (bool(cfg["heave"]), bool(cfg["mode2"]), cfg["tau"] > 0)

def batched(fn, *arrays, label=None):
    n = arrays[0].shape[0]
    if n == 0:
        return None
    outs = []
    for i0 in range(0, n, CHUNK):
        sl = [a[i0:i0 + CHUNK] for a in arrays]
        m = sl[0].shape[0]
        if m < CHUNK:
            sl = [np.concatenate([a, np.repeat(a[-1:], CHUNK - m, axis=0)]) for a in sl]
        o = fn(*[jnp.asarray(a) for a in sl])
        o = o if isinstance(o, tuple) else (o,)
        outs.append([np.asarray(v)[:m] for v in o])
    res = tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))
    return res if len(res) > 1 else res[0]

def valid(S, IZ, amp):
    r = np.hypot(S[:, IZ], S[:, IZ + 1])
    return (S[:, 0] > 0.2) & (S[:, 0] < 3.5) & (r > 0.3 * amp) & (r < 2.5 * amp) & np.isfinite(S).all(1)

def ckpt(name):
    return os.path.join(CKPT_DIR, name)

def eigvals(M):
    """Batch eigenvalues; rows with non-finite matrices give nan."""
    out = np.full(M.shape[:2], np.nan + 0j)
    ok = np.isfinite(M).all((1, 2))
    if ok.any():
        out[ok] = np.linalg.eigvals(M[ok])
    return out

# -----------------------------------------------------------------------------
# 2. Refined index evaluation (Script 10 stage D2, any layout)
# -----------------------------------------------------------------------------
def refined_index(M_, S, Qv, spp=SPP):
    """Fine homotopy -> fixed-position lock -> det(I - M_int), a_L (+/- 0.005 first), index."""
    n = S.shape[0]
    D, IZ = M_["D"], M_["IZ"]
    amp = Qv[:, QI["amp"]]
    s = S.copy()
    reached = np.ones(n, bool)
    for eta in ETAS_FINE[1:]:
        idx = np.where(reached)[0]
        if idx.size == 0:
            break
        Qe = Qv[idx].copy(); Qe[:, QI["eta"]] = eta
        sn, res, _, _ = batched(M_["get"]("hom", spp), s[idx], Qe)
        ok = (res < 1e-7) & valid(sn, IZ, amp[idx])
        s[idx[ok]] = sn[ok]
        reached[idx[~ok]] = False
    Qf = Qv.copy(); Qf[:, QI["eta"]] = 0.0
    sf0 = s.copy(); sf0[:, 1] = 0.0
    sf, resf, Mf, _ = batched(M_["get"]("int", spp), sf0, Qf)
    okf = reached & (resf < 1e-7)
    mu = eigvals(Mf[:, 2:, 2:])
    dI = np.real(np.prod(1.0 - mu, axis=1))
    am = np.abs(mu)
    lead = mu[np.arange(n), np.argmax(am, 1)]
    ftype = np.where(~okf, "none", np.where(am.max(1) < 1, "node",
                     np.where(np.abs(lead.imag) < 1e-6, "saddle-real", "unstable-complex")))
    aLs = {}
    for half in AL_SCALES:
        off = np.linspace(-half, half, 7)
        Fo = np.full((n, 7), np.nan)
        for side in (+1, -1):
            cur = sf.copy()
            for o in sorted([o for o in off if (o >= 0 if side > 0 else o <= 0)], key=abs):
                cur[:, 0] = sf[:, 0] + o
                cn, rq, _, F = batched(M_["get"]("int", spp), cur, Qf)
                cur = np.where((rq < 1e-7)[:, None], cn, cur)
                Fo[:, int(np.argmin(np.abs(off - o)))] = np.where(rq < 1e-7, F, np.nan)
        a_ = np.full(n, np.nan)
        for i in range(n):
            m = np.isfinite(Fo[i])
            if okf[i] and m.sum() >= 5:
                a_[i] = np.polyfit(off[m], Fo[i, m], 1)[0]
        aLs[half] = a_
    aL = np.full(n, np.nan)
    for half in AL_SCALES:
        m = np.isnan(aL) & np.isfinite(aLs[half])
        aL[m] = aLs[half][m]
    return dict(ftype=ftype, dI=np.where(okf, dI, np.nan), aL=aL, aL_wide=aLs[0.02],
                index=np.sign(np.where(okf, dI, np.nan)) * np.sign(-aL), reached=reached)

t_start = time.time()
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}, mode {MODE}")

# -----------------------------------------------------------------------------

import time as _t
PTS = np.load(os.path.join(S10_DIR, f"points_script10{Q}.npz"))["points"]
BR = np.load(os.path.join(S10_CK, "stageC_branches.npz"), allow_pickle=True)["branches"].tolist()
CL = dict(np.load(os.path.join(S10_CK, "stageD2_index.npz")))
SEL, FT = CL["sel"].astype(int), CL["ftype"].astype(str)
pick, grp = [], []
for dp in np.unique(PTS[SEL, 3]):
    for ft_ in ("saddle-real", "node", "none"):
        idx = SEL[(np.abs(PTS[SEL, 3] - dp) < 1e-6) & (FT == ft_)]
        if idx.size:
            ch = rng.choice(idx, min(N_PER_GROUP, idx.size), replace=False)
            pick += list(ch); grp += [f"{dp:.3f}/{ft_}"] * len(ch)
edge = np.where((np.abs(PTS[:, 8] - 1) < 0.02))[0]
if edge.size:
    ch = rng.choice(edge, min(N_EDGE, edge.size), replace=False)
    pick += list(ch); grp += ["near stability change"] * len(ch)
pick = np.array(pick, int); grp = np.array(grp)
print("Script 16: sample", pick.size, {g: int((grp == g).sum()) for g in np.unique(grp)}, flush=True)
S = np.array([BR[int(PTS[i, 0])]["Z"][int(PTS[i, 1]), :6] for i in pick])
cfg = dict(BASE_CFG); M_ = model(lay_key(cfg))
Qv = np.stack([qvec(cfg, PTS[i, 4], PTS[i, 2], PTS[i, 3], 1.0) for i in pick])
out = dict(pick=pick, grp=grp)
for spp in (100, 400):
    t0=_t.time()
    sN, rN, MN, _ = batched(M_["get"]("full", spp, n_it=15), S, Qv)
    ref = refined_index(M_, sN, Qv, spp)
    out[f"ftype_{spp}"] = ref["ftype"]; out[f"index_{spp}"] = ref["index"]; out[f"aL_{spp}"] = ref["aL"]
    out[f"amax_{spp}"] = np.abs(eigvals(MN)).max(1); out[f"x_{spp}"] = sN[:, 0]; out[f"reached_{spp}"] = ref["reached"]
    print("spp", spp, "done", round(_t.time()-t0), "s", flush=True)
OUT16 = os.path.join(BASE_DIR, "script16_branch_change")
os.makedirs(OUT16, exist_ok=True)
np.savez(os.path.join(OUT16, "branch_change_check.npz"), **out)
ch = np.where(out["ftype_100"] != out["ftype_400"])[0]
for i in ch:
    p = pick[i]
    print(dict(group=grp[i], eps=PTS[p,2], dphi=PTS[p,3], ratio=PTS[p,4], x=PTS[p,5], stable=bool(PTS[p,7]), amax_point=PTS[p,8],
               type100=out["ftype_100"][i], type400=out["ftype_400"][i], reached100=out["reached_100"][i], reached400=out["reached_400"][i],
               aL100=out["aL_100"][i], aL400=out["aL_400"][i], idx100=out["index_100"][i], idx400=out["index_400"][i],
               amax100=out["amax_100"][i], amax400=out["amax_400"][i], in_SEL=bool(p in set(SEL))))
