# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 18: Example trajectories of passive bodies and the two locked rhythms
#            at fixed position, for Figs. 2 and 3.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# Uses JAX; a few minutes on a CPU, less on a GPU. Results are checkpointed.
# Independent of the other scripts (it reads no earlier outputs).
#
# Parts:
#   A. Passive bodies (g = 0) in the baseline wake, for the bodies of Fig. 2a
#      (Omega/omega_w = 0.50, 0.93 and 3.09 at damping ratio 0.1, and a rigid
#      body): the equilibrium orbit of every equilibrium is computed by Newton's
#      method on the period map with free surge, classified by its Floquet
#      multipliers, and the body is displaced by +/- 0.01 wavelengths and
#      followed until it collides with the cylinder or leaves the wake.
#      Trajectories are stored at 20 samples per shedding period.
#   B. The two 1:1 locked rhythms at fixed position, in the setting of
#      Script 03 (position x = 0.9, muscle gain 8, phase lag 0, coupling 6):
#      the closed curve of locked solutions is traced by continuation in the
#      CPG phase at the strobe, solving for the intrinsic frequency omega_c,
#      and every solution is classified as node (stable) or saddle by its
#      Floquet multipliers. The curve gives the locked CPG phase and the
#      tail-beat phase against omega_c/omega_w across the locking plateau.
#
# Gates (reported):
#   G1  every equilibrium orbit converges (residual below 1e-10)
#   G2  the locked curve closes after one full turn of the phase, and its two
#       ends at the tongue edges are folds (a real multiplier at +1)
#
# Outputs: script18_examples_and_locked_branches/results_script18.npz and
#          summary_script18.json
# =============================================================================

import os, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script18_examples_and_locked_branches")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "results_script18.npz")
print(f"JAX {jax.__version__}, devices: {[d.platform for d in jax.devices()]}")

OMEGA_W = 2.0 * np.pi
SPP = 100
DT = 1.0 / SPP
U_INF, DELTA0, L_DELTA = 1.0 / 0.87, 1.30, 0.90
W0, X_R, L_W = 0.35, 0.35, 1.50
C_L, C_D, SIGMA = 3.0, 0.15, 8.0
X_COLL, X_ESC = 0.15, 4.5
t0 = time.time()
summary = {}

def W_env(x):
    return W0 * (1.0 - jnp.exp(-(x / X_R) ** 2)) * jnp.exp(-x / L_W)

def U_loc(x):
    return U_INF * (1.0 - DELTA0 * jnp.exp(-x / L_DELTA))

def rk4_period(f, s, p, t0_=0.0, n=SPP):
    def body(i, s_):
        t = t0_ + i * DT
        k1 = f(t, s_, p); k2 = f(t + DT / 2, s_ + DT / 2 * k1, p)
        k3 = f(t + DT / 2, s_ + DT / 2 * k2, p); k4 = f(t + DT, s_ + DT * k3, p)
        return s_ + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return jax.lax.fori_loop(0, n, body, s)

# -----------------------------------------------------------------------------
# A. Passive bodies
# -----------------------------------------------------------------------------
def rhs_passive(t, s, p):
    """State (x, u, th, om); p = (Om/omega_w, zeta, rigid)."""
    x, u, th, om = s
    omr, ze, rigid = p
    V = U_loc(x) - u
    w = W_env(x) * jnp.sin(OMEGA_W * (x - t))
    gate = 0.5 * (1.0 + jnp.tanh(V / 0.05))
    beta = jnp.arctan2(w, jnp.abs(V))
    alpha = beta - th
    L = C_L * (V * V + w * w) * jnp.sin(alpha) * jnp.cos(alpha) * gate
    Fx = C_D * V * jnp.abs(V) - L * jnp.sin(beta)
    Om = omr * OMEGA_W
    dom = (-2 * ze * Om * om - Om ** 2 * th + SIGMA * L) * (1.0 - rigid)
    return jnp.stack([u, Fx, om * (1.0 - rigid), dom])

def rhs_frozen(t, s, p):
    """Fixed position: state (th, om, Fx_integral); p = (x, Om/omega_w, zeta, rigid)."""
    th, om, _ = s
    x, omr, ze, rigid = p
    d = rhs_passive(t, jnp.stack([x, 0.0, th, om]), jnp.stack([omr, ze, rigid]))
    return jnp.stack([d[2], d[3], d[1]])

pmap_passive = jax.jit(lambda s, p: rk4_period(rhs_passive, s, p))
jac_passive = jax.jit(jax.jacfwd(lambda s, p: rk4_period(rhs_passive, s, p)))

@jax.jit
def mean_force(x, p):
    pf = jnp.concatenate([jnp.array([x]), p])
    s = jnp.zeros(3)
    def per(k, s_):
        s_ = s_.at[2].set(0.0)
        return rk4_period(rhs_frozen, s_, pf, t0_=0.0)
    s = jax.lax.fori_loop(0, 30, per, s)
    s = s.at[2].set(0.0)
    s = rk4_period(rhs_frozen, s, pf)
    return s[2], s[:2]

def simulate_passive(s0, p, n_per, sub=20):
    @jax.jit
    def run(s0):
        def per(carry, k):
            s, alive = carry
            def stepc(i, c):
                s_, al = c
                t = k + i * DT
                k1 = rhs_passive(t, s_, p); k2 = rhs_passive(t + DT / 2, s_ + DT / 2 * k1, p)
                k3 = rhs_passive(t + DT / 2, s_ + DT / 2 * k2, p); k4 = rhs_passive(t + DT, s_ + DT * k3, p)
                sn = s_ + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                al2 = al & (sn[0] > X_COLL) & (sn[0] < X_ESC)
                return jnp.where(al2, sn, s_), al2
            xs = []
            st = (s, alive)
            out = []
            for j in range(sub):
                st = jax.lax.fori_loop(0, SPP // sub, stepc, st) if False else st
            return st, st[0][0]
        return None
    # simple, explicit loop (sufficient for a handful of runs)
    s = np.array(s0, float); traj = [s[0]]; t = 0.0
    step_fn = jax.jit(lambda s_, t_: rk4_period(rhs_passive, s_, p, t0_=t_, n=SPP // sub))
    alive = True
    for k in range(n_per * sub):
        if alive:
            s = np.asarray(step_fn(jnp.asarray(s), t))
            if not (X_COLL < s[0] < X_ESC) or not np.all(np.isfinite(s)):
                alive = False
        traj.append(s[0] if alive else np.nan)
        t += 1.0 / sub
    return np.array(traj)

BODIES = [(0.50, 0.1, 0.0), (0.93, 0.1, 0.0), (3.09, 0.1, 0.0), (1.0, 0.1, 1.0)]
XG = np.linspace(0.2, 3.0, 141)
resA = {}
if os.path.exists(OUT):
    print("A: loaded from previous run")
else:
    print("A: passive equilibria and trajectories")
    for bi, b in enumerate(BODIES):
        p = jnp.array(b)
        F = np.array([float(mean_force(x, p)[0]) for x in XG])
        zc = np.where(np.sign(F[:-1]) != np.sign(F[1:]))[0]
        eqs = []
        for j in zc:
            xe = XG[j] - F[j] * (XG[j + 1] - XG[j]) / (F[j + 1] - F[j])
            th0 = mean_force(xe, p)[1]
            s = jnp.array([xe, 0.0, float(th0[0]), float(th0[1])])
            ok = False
            for it in range(40):
                r = pmap_passive(s, p) - s
                if float(jnp.max(jnp.abs(r))) < 1e-11:
                    ok = True; break
                J = jac_passive(s, p) - jnp.eye(4)
                if b[2] > 0.5:                                  # rigid: pitch fixed at zero
                    J = J.at[2, 2].set(1.0).at[3, 3].set(1.0)
                s = s - jnp.linalg.solve(J, r)
            M = np.asarray(jac_passive(s, p))
            mu = np.linalg.eigvals(M[:2, :2] if b[2] > 0.5 else M)
            res_ = float(jnp.max(jnp.abs(pmap_passive(s, p) - s)))
            rec = dict(x=float(s[0]), converged=bool(ok or res_ < 1e-10), residual=res_,
                       mu_max=float(np.abs(mu).max()), stable=bool(np.abs(mu).max() < 1.0))
            if not rec["stable"]:
                trajs = {}
                for sign in (+1, -1):
                    s0 = np.array(s); s0[0] += sign * 0.01
                    trajs[sign] = simulate_passive(s0, p, 40)
                rec["traj_plus"], rec["traj_minus"] = trajs[1], trajs[-1]
            eqs.append(rec)
            print(f"   body {b}: equilibrium x = {rec['x']:.3f}, |mu|max = {rec['mu_max']:.3f}, "
                  f"residual {res_:.1e}, {'stable' if rec['stable'] else 'unstable'}")
        resA[bi] = dict(F=F, eqs=eqs)

# -----------------------------------------------------------------------------
# B. Locked rhythms at fixed position (Script 03 setting)
# -----------------------------------------------------------------------------
X0, G3, EPS3, OMR3, ZE3 = 0.9, 8.0, 6.0, 1.2, 0.2
V0 = float(U_loc(X0)); WX = float(W_env(X0))

def rhs_fixed(t, s, wc):
    th, om, zr, zi = s
    w = WX * jnp.sin(OMEGA_W * (X0 - t))
    beta = jnp.arctan2(w, abs(V0))
    alpha = beta - th
    L = C_L * (V0 * V0 + w * w) * jnp.sin(alpha) * jnp.cos(alpha)
    Om = OMR3 * OMEGA_W
    r2 = zr * zr + zi * zi
    a = 2 * np.pi * (1.0 - r2)
    dzr = a * zr - wc * zi + EPS3 * th
    dzi = a * zi + wc * zr
    dom = -2 * ZE3 * Om * om - Om ** 2 * th + SIGMA * L + G3 * zr
    return jnp.stack([om, dom, dzr, dzi])

Pfix = jax.jit(lambda s, wc: rk4_period(rhs_fixed, s, wc))
Jfix = jax.jit(jax.jacfwd(lambda s, wc: rk4_period(rhs_fixed, s, wc), argnums=(0, 1)))

def solve_at_phase(s, wc, phi):
    """Newton for (s, wc) with P(s) = s and arg z = phi."""
    for it in range(30):
        r = jnp.concatenate([Pfix(s, wc) - s, jnp.array([s[3] * np.cos(phi) - s[2] * np.sin(phi)])])
        if float(jnp.max(jnp.abs(r))) < 1e-11:
            return s, wc, True
        Js, Jw = Jfix(s, wc)
        A = jnp.zeros((5, 5))
        A = A.at[:4, :4].set(Js - jnp.eye(4)).at[:4, 4].set(Jw)
        A = A.at[4, 2].set(-np.sin(phi)).at[4, 3].set(np.cos(phi))
        d = jnp.linalg.solve(A, -r)
        s = s + d[:4]; wc = wc + d[4]
    return s, wc, False

if os.path.exists(OUT):
    print("B: loaded from previous run")
    D = dict(np.load(OUT, allow_pickle=True))
else:
    print("B: locked rhythms at fixed position")
    # start from a locked solution found by simulation in the middle of the plateau
    wc0 = 1.05 * OMEGA_W
    s = jnp.array([0.0, 0.0, 1.0, 0.0])
    for k in range(300):
        s = Pfix(s, wc0)
    phi0 = float(np.arctan2(float(s[3]), float(s[2])))
    s, wc, ok = solve_at_phase(s, wc0, phi0)
    print(f"   start: omega_c/omega_w = {float(wc) / OMEGA_W:.4f}, converged {ok}")
    N_PH = 721
    phis = phi0 + np.linspace(0.0, 2 * np.pi, N_PH)
    S, WC, MU, OK = [], [], [], []
    for phi in phis:
        s, wc, ok = solve_at_phase(s, wc, phi)
        Js, _ = Jfix(s, wc)
        mu = np.linalg.eigvals(np.asarray(Js))
        S.append(np.asarray(s)); WC.append(float(wc)); MU.append(mu); OK.append(ok)
    S, WC, MU, OK = np.array(S), np.array(WC), np.array(MU), np.array(OK)
    amax = np.abs(MU).max(1)
    n_real_out = np.array([np.sum((np.abs(m.imag) < 1e-9) & (m.real > 1.0)) for m in MU])
    kind = np.where(amax < 1.0, "node", np.where(n_real_out == 1, "saddle", "other"))
    # tail-beat phase relative to the transverse velocity at the strobe: from (th, om)
    th_ph = np.arctan2(-S[:, 1] / OMEGA_W, S[:, 0])
    closed = bool(np.max(np.abs(S[-1] - S[0])) < 1e-6 and abs(WC[-1] - WC[0]) < 1e-6)
    # folds: extrema of omega_c along the curve
    dW = np.diff(WC)
    fold_idx = np.where(np.sign(dW[:-1]) != np.sign(dW[1:]))[0] + 1
    fold_mu = [float(np.min(np.abs(MU[i] - 1.0))) for i in fold_idx]
    summary["B"] = dict(n=int(N_PH), converged=int(OK.sum()), closed=closed,
                        ratio_range=[float(WC.min() / OMEGA_W), float(WC.max() / OMEGA_W)],
                        n_node=int((kind == "node").sum()), n_saddle=int((kind == "saddle").sum()),
                        n_other=int((kind == "other").sum()),
                        folds_ratio=[float(WC[i] / OMEGA_W) for i in fold_idx],
                        folds_min_distance_of_multiplier_to_1=fold_mu)
    print(f"   locked curve: {OK.sum()}/{N_PH} converged, closed {closed}, omega_c/omega_w from "
          f"{WC.min() / OMEGA_W:.4f} to {WC.max() / OMEGA_W:.4f}; node {np.sum(kind == 'node')}, "
          f"saddle {np.sum(kind == 'saddle')}, other {np.sum(kind == 'other')}; folds at "
          f"{[round(WC[i] / OMEGA_W, 4) for i in fold_idx]}")
    D = dict(B_phi=phis, B_S=S, B_ratio=WC / OMEGA_W, B_mu=MU, B_kind=kind, B_ok=OK, B_tailphase=th_ph,
             B_settings=np.array([X0, G3, EPS3, OMR3, ZE3]))
    for bi, rec in resA.items():
        D[f"A{bi}_F"] = rec["F"]
        D[f"A{bi}_body"] = np.array(BODIES[bi])
        for ei, e in enumerate(rec["eqs"]):
            D[f"A{bi}_eq{ei}_x"] = e["x"]; D[f"A{bi}_eq{ei}_mu"] = e["mu_max"]; D[f"A{bi}_eq{ei}_stable"] = e["stable"]
            if "traj_plus" in e:
                D[f"A{bi}_eq{ei}_tp"] = e["traj_plus"]; D[f"A{bi}_eq{ei}_tm"] = e["traj_minus"]
        summary.setdefault("A", {})[str(BODIES[bi])] = [{k: v for k, v in e.items() if not k.startswith("traj")}
                                                       for e in rec["eqs"]]
    D["A_xgrid"] = XG
    np.savez(OUT, **D)
    with open(os.path.join(OUT_DIR, "summary_script18.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

print("\n================ SCRIPT 18 SUMMARY ================")
if "B" in summary:
    g1 = all(e["converged"] for v in summary.get("A", {}).values() for e in v)
    g2 = summary["B"]["closed"] and len(summary["B"]["folds_ratio"]) == 2 and \
        all(d < 1e-2 for d in summary["B"]["folds_min_distance_of_multiplier_to_1"])
    print(f"G1 equilibrium orbits converged: {g1}; G2 closed locked curve with two folds: {g2}")
print(f"Outputs: {OUT_DIR}; runtime {time.time() - t0:.0f} s")
