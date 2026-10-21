# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 03: Intrinsic rhythm at fixed position. A Stuart-Landau CPG drives the
#            body mode, and receives proprioceptive feedback from it; the wake
#            forces the body. Where does the CPG lock to the vortex street?
#
# Colab ready. Needs only numpy and matplotlib.
# Runtime: about 15 to 25 min (FULL), about 2.5 min (QUICK).
#
# Same units as Scripts 01 and 02 (lambda = 1, T = 1, omega_w = 2 pi).
# The fish is held at x0 = 0.9 (the passive equilibrium of the baseline wake).
#
# State s = [th, om, zr, zi]:
#   th'' + 2 zeta Omega th' + Omega^2 th = sigma L + g Re(z)        (body)
#   z'  = [mu (1 + i c)(1 - |z|^2) + i omega_c] z + eps th          (CPG)
#   L as in Scripts 01 and 02, with the fish fixed (u = 0) at x0.
#   The CPG free frequency is exactly omega_c at amplitude 1; c is the
#   amplitude-phase coupling (shear); c = 0 is isochronous.
#
# Detuning axis: f_free / f_shed, where f_free is the rhythm of the same fish
# (CPG + body + proprioceptive loop) in uniform flow at the same speed, i.e. the
# free-stream tail-beat frequency. The proprioceptive loop shifts f_free away
# from omega_c, so maps are drawn against f_free, not omega_c.
#
# Parts:
#   A. Arnold tongue maps over (omega_c / omega_w, eps), for c = 0 and c = 1:
#      rotation number rho, 1:1 locking, largest Lyapunov exponent, |z|.
#   B. Devil's staircases rho(omega_c) at three coupling strengths.
#   C. Phase slips at the edge of the 1:1 tongue; inter-slip time scaling
#      (saddle-node on invariant circle predicts exponent -1/2).
#   D. Phase-locking stiffness k (slowest Floquet decay rate) inside the 1:1
#      tongue, and mean streamwise force: direct inputs to Script 04.
#
# Decision gate:
#   A clean 1:1 tongue at plausible coupling, a type-I exponent near -1/2,
#   and a k map. Chaos fraction is reported, not required.
# =============================================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------
MODE = "FULL"            # "FULL" or "QUICK"
SEED = 4
rng = np.random.default_rng(SEED)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
OUT_DIR = os.path.join(BASE_DIR, "script03_cpg_tongues")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OMEGA_W = 2.0 * np.pi
WAKE = dict(U_inf=1.0 / 0.87, delta0=1.30, l_delta=0.90, W0=0.35, x_r=0.35,
            l_w=1.50, c_L=3.0, c_D=0.15, sigma=8.0)
X0 = 0.9
BODY = dict(ratio=1.2, zeta=0.2)     # generic, non-resonant body (Script 01 maps)
CPG = dict(mu=2 * np.pi, g=8.0)      # relaxation ~ one period; active torque gain
C_VALUES = [0.0, 1.0]

if MODE == "FULL":
    SPP = 100
    N_RAT, N_EPS = 100, 40
    T_TRANS, T_MEAS = 150, 150
    N_STAIR, T_STAIR = 400, 300
    N_SLIP, T_SLIP = 15, 1500
else:
    SPP = 80
    N_RAT, N_EPS = 30, 15
    T_TRANS, T_MEAS = 60, 60
    N_STAIR, T_STAIR = 80, 120
    N_SLIP, T_SLIP = 6, 400

RATIO = np.linspace(0.4, 2.0, N_RAT)        # omega_c / omega_w
EPS = np.linspace(0.0, 20.0, N_EPS)         # proprioceptive coupling
LYAP_D0 = 1e-8
LOCK_SPAN = np.pi                           # max excursion of phase difference

# -----------------------------------------------------------------------------
# 1. Model
# -----------------------------------------------------------------------------
def U_loc(x):
    return WAKE["U_inf"] * (1.0 - WAKE["delta0"] * np.exp(-x / WAKE["l_delta"]))

def W_env(x):
    return WAKE["W0"] * (1.0 - np.exp(-(x / WAKE["x_r"]) ** 2)) * np.exp(-x / WAKE["l_w"])

V0 = U_loc(X0)
WX = W_env(X0)
OM_B = BODY["ratio"] * OMEGA_W
ZE_B = BODY["zeta"]

def hydro(t, th, ws):
    w = ws * WX * np.sin(OMEGA_W * (X0 - t))
    beta = np.arctan2(w, abs(V0))
    alpha = beta - th
    L = WAKE["c_L"] * (V0 * V0 + w * w) * np.sin(alpha) * np.cos(alpha)
    Fx = WAKE["c_D"] * V0 * abs(V0) - L * np.sin(beta)
    return L, Fx

def rhs(t, s, wc, eps, c, g, ws):
    th, om, zr, zi = s
    L, Fx = hydro(t, th, ws)
    r2 = zr * zr + zi * zi
    a = CPG["mu"] * (1.0 - r2)
    b = c * CPG["mu"] * (1.0 - r2) + wc
    dzr = a * zr - b * zi + eps * th
    dzi = a * zi + b * zr
    dom = -2 * ZE_B * OM_B * om - OM_B ** 2 * th + WAKE["sigma"] * L + g * zr
    return np.array([om, dom, dzr, dzi]), Fx

def step(t, s, dt, *args):
    k1, F1 = rhs(t, s, *args)
    k2, F2 = rhs(t + dt / 2, s + dt / 2 * k1, *args)
    k3, F3 = rhs(t + dt / 2, s + dt / 2 * k2, *args)
    k4, F4 = rhs(t + dt, s + dt * k3, *args)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (F1 + 2 * F2 + 2 * F3 + F4) / 6

def init_state(B):
    ph = rng.uniform(0, 2 * np.pi, B)
    return np.array([np.zeros(B), np.zeros(B), np.cos(ph), np.sin(ph)])

def run(wc, eps, c, g, T_trans, T_meas, lyap=True, record_strobe=False, spp=SPP, ws=1.0):
    """Batch integration. Returns rho, locked11, lyap, mean |z|, mean Fx, final state.
    ws = 0 removes the vortex street (uniform flow at V0): free swimming rhythm."""
    B = wc.size
    ws = np.broadcast_to(np.asarray(ws, float), (B,))
    s = init_state(B)
    dt = 1.0 / spp
    t = 0.0
    for _ in range(T_trans * spp):
        s, _ = step(t, s, dt, wc, eps, c, g, ws)
        t += dt
    if lyap:
        d = rng.normal(size=s.shape)
        d *= LYAP_D0 / np.linalg.norm(d, axis=0)
        sp = s + d
        lsum = np.zeros(B)
    phase = np.zeros(B)
    zabs = np.zeros(B)
    Fsum = np.zeros(B)
    strobe = np.zeros((T_meas, B))
    for n in range(T_meas):
        for _ in range(spp):
            z_old = s[2] + 1j * s[3]
            s, F = step(t, s, dt, wc, eps, c, g, ws)
            if lyap:
                sp, _ = step(t, sp, dt, wc, eps, c, g, ws)
            z_new = s[2] + 1j * s[3]
            phase += np.angle(z_new * np.conj(z_old))
            zabs += np.abs(z_new)
            Fsum += F
            t += dt
        strobe[n] = phase - OMEGA_W * (n + 1)          # phase relative to wake
        if lyap:
            dv = sp - s
            dn = np.linalg.norm(dv, axis=0)
            lsum += np.log(dn / LYAP_D0)
            sp = s + dv * (LYAP_D0 / dn)
    rho = phase / (OMEGA_W * T_meas)
    span = strobe.max(0) - strobe.min(0)
    out = dict(rho=rho, locked11=span < LOCK_SPAN,
               lyap=lsum / T_meas if lyap else None,
               zabs=zabs / (T_meas * spp), Fx=Fsum / (T_meas * spp), s=s, t=t)
    if record_strobe:
        out["strobe"] = strobe
    return out

def period_map(s, wc, eps, c, g, t0, spp=SPP):
    dt = 1.0 / spp
    t = t0
    ws = np.ones(s.shape[1])
    for _ in range(spp):
        s, _ = step(t, s, dt, wc, eps, c, g, ws)
        t += dt
    return s

def floquet_locked(s0, wc, eps, c, g, t0, h=1e-6, spp=SPP):
    """Floquet multipliers of the period-1 stroboscopic map at a converged orbit."""
    B = s0.shape[1]
    s1 = period_map(s0, wc, eps, c, g, t0, spp)
    res = np.abs(s1 - s0).max(axis=0)
    J = np.zeros((B, 4, 4))
    for j in range(4):
        ds = np.zeros_like(s0); ds[j] = h
        J[:, :, j] = ((period_map(s0 + ds, wc, eps, c, g, t0, spp) - s1) / h).T
    mu = np.linalg.eigvals(J)
    a = np.sort(np.abs(mu), axis=1)[:, ::-1]
    return a, res

# -----------------------------------------------------------------------------
# 2. Part A: tongue maps
# -----------------------------------------------------------------------------
t_start = time.time()
print(f"Mode {MODE}: x0 = {X0}, V0 = {V0:.3f}, W(x0) = {WX:.3f}")

# passive reference at x0 (no CPG torque): mean thrust and body amplitude
ref = run(np.array([OMEGA_W]), np.array([0.0]), np.array([0.0]), 0.0, T_TRANS, 20, lyap=False)
F_passive = float(ref["Fx"][0])
print(f"Passive mean streamwise force at x0: {F_passive:.4f}")

RR, EE = np.meshgrid(RATIO, EPS, indexing="ij")
maps = {}
for c in C_VALUES:
    t0 = time.time()
    wc = RR.ravel() * OMEGA_W
    ep = EE.ravel()
    cc = np.full(wc.size, c)
    R = run(wc, ep, cc, CPG["g"], T_TRANS, T_MEAS)
    maps[c] = {k: (v.reshape(N_RAT, N_EPS) if isinstance(v, np.ndarray) and v.ndim == 1 else v)
               for k, v in R.items()}
    lk = maps[c]["locked11"]
    ly = maps[c]["lyap"]
    # confirm apparent chaos with a 3x longer measurement window
    cand = np.where(ly.ravel() > 0.01)[0]
    confirmed = np.zeros(N_RAT * N_EPS, bool)
    if cand.size:
        Rc = run(wc[cand], ep[cand], cc[cand], CPG["g"], T_TRANS, 3 * T_MEAS)
        confirmed[cand] = Rc["lyap"] > 0.01
    maps[c]["chaos_confirmed"] = confirmed.reshape(N_RAT, N_EPS)
    # free-swimming rhythm: same fish in uniform flow (no vortex street)
    Rf = run(wc, ep, cc, CPG["g"], T_TRANS, T_MEAS, lyap=False, ws=0.0)
    maps[c]["rho_free"] = Rf["rho"].reshape(N_RAT, N_EPS)
    print(f"A [c={c}]: 1:1 locked {lk.mean()*100:.1f}%, chaotic (lambda>0.01) "
          f"{np.mean(ly > 0.01)*100:.2f}%, confirmed with 3x window {confirmed.mean()*100:.2f}%, "
          f"quenched |z|<0.2 {np.mean(maps[c]['zabs'] < 0.2)*100:.2f}%, {time.time() - t0:.0f} s")

def tongue_width(lk, rf):
    """Width of the 1:1 locked set measured in free-swimming frequency
    f_free / f_shed (sum of local grid spacings over locked cells), and its centre."""
    w = np.zeros(N_EPS)
    ctr = np.full(N_EPS, np.nan)
    for j in range(N_EPS):
        dr = np.abs(np.gradient(rf[:, j]))
        w[j] = np.sum(dr[lk[:, j]])
        if lk[:, j].any():
            ctr[j] = np.mean(rf[lk[:, j], j])
    return w, ctr

widths, centres = {}, {}
for c in C_VALUES:
    widths[c], centres[c] = tongue_width(maps[c]["locked11"], maps[c]["rho_free"])

# -----------------------------------------------------------------------------
# 3. Part B: devil's staircases (c = 0)
# -----------------------------------------------------------------------------
print("B: devil's staircases ...")
EPS_STAIR = [2.0, 6.0, 12.0]
RAT_STAIR = np.linspace(0.4, 2.0, N_STAIR)
stairs = {}
for e in EPS_STAIR:
    R = run(RAT_STAIR * OMEGA_W, np.full(N_STAIR, e), np.zeros(N_STAIR), CPG["g"],
            T_TRANS, T_STAIR, lyap=True)
    stairs[e] = dict(rho=R["rho"], locked11=R["locked11"], lyap=R["lyap"])

# -----------------------------------------------------------------------------
# 4. Part C: phase slips at the upper edge of the 1:1 tongue (c = 0, eps = 6)
# -----------------------------------------------------------------------------
print("C: phase slips at the tongue edge ...")
EPS_SLIP = 6.0
lk = stairs[EPS_SLIP]["locked11"]
edge = None
if lk.any():
    i_up = np.where(lk)[0].max()
    lo, hi = RAT_STAIR[i_up], RAT_STAIR[min(i_up + 1, N_STAIR - 1)]
    for _ in range(2):                          # two vectorised refinement scans
        grid = np.linspace(lo, hi, 41)
        R = run(grid * OMEGA_W, np.full(grid.size, EPS_SLIP), np.zeros(grid.size),
                CPG["g"], T_TRANS, 300, lyap=False)
        lk_g = R["locked11"]
        if lk_g.any() and not lk_g.all():
            k_ = np.where(lk_g)[0].max()
            lo, hi = grid[k_], grid[min(k_ + 1, grid.size - 1)]
        else:
            break
    edge = hi
slip = None
if edge is not None:
    delta = np.geomspace(2e-4, 5e-2, N_SLIP)
    wc = (edge + delta) * OMEGA_W
    R = run(wc, np.full(N_SLIP, EPS_SLIP), np.zeros(N_SLIP), CPG["g"], T_TRANS, T_SLIP,
            lyap=False, record_strobe=True)
    strobe = R["strobe"]
    n_slip = np.abs(strobe[-1] - strobe[0]) / (2 * np.pi)
    T_s = T_SLIP / np.maximum(n_slip, 1e-9)
    ok = n_slip >= 3
    if ok.sum() >= 3:
        coef = np.polyfit(np.log(delta[ok]), np.log(T_s[ok]), 1)
        slip = dict(edge_ratio=float(edge), delta=delta.tolist(), n_slips=n_slip.tolist(),
                    T_slip=T_s.tolist(), exponent=float(coef[0]), used_points=int(ok.sum()),
                    strobe_example=strobe[:, [int(np.where(ok)[0][0])]].ravel().tolist())
        print(f"   edge at omega_c/omega_w = {edge:.5f}, inter-slip exponent {coef[0]:.3f} "
              f"(SNIC prediction -0.5)")
    else:
        print("   too few slipping runs for a fit")

# -----------------------------------------------------------------------------
# 5. Part D: phase-locking stiffness k inside the 1:1 tongue (c = 0)
# -----------------------------------------------------------------------------
print("D: locking stiffness from Floquet multipliers ...")
M = maps[0.0]
lk = M["locked11"] & (M["lyap"] < -1e-3)
ii, jj = np.where(lk)
k_map = np.full((N_RAT, N_EPS), np.nan)
lead_map = np.full((N_RAT, N_EPS), np.nan)
if ii.size:
    idx = ii * N_EPS + jj
    s0 = M["s"][:, idx]
    wc = RATIO[ii] * OMEGA_W
    ep = EPS[jj]
    cc = np.zeros(ii.size)
    t0 = M["t"]
    # converge a little further onto the orbit
    for _ in range(20):
        s0 = period_map(s0, wc, ep, cc, CPG["g"], t0)
    a, res = floquet_locked(s0, wc, ep, cc, CPG["g"], t0)
    good = res < 1e-5
    k_vals = -np.log(a[:, 0])                  # slowest decay per period
    k_map[ii[good], jj[good]] = k_vals[good]
    lead_map[ii[good], jj[good]] = a[good, 0]
    print(f"   {good.sum()}/{ii.size} locked orbits converged; "
          f"k range {np.nanmin(k_vals[good]):.4f} to {np.nanmax(k_vals[good]):.3f} per period")

# time-step convergence on a small subset
print("Convergence check (spp 100 vs 200) ...")
sub = rng.choice(N_RAT * N_EPS, size=min(60, N_RAT * N_EPS), replace=False)
Ra = run(RR.ravel()[sub] * OMEGA_W, EE.ravel()[sub], np.zeros(sub.size), CPG["g"],
         T_TRANS, T_MEAS, lyap=False, spp=SPP)
Rb = run(RR.ravel()[sub] * OMEGA_W, EE.ravel()[sub], np.zeros(sub.size), CPG["g"],
         T_TRANS, T_MEAS, lyap=False, spp=2 * SPP)
conv = dict(max_abs_drho=float(np.max(np.abs(Ra["rho"] - Rb["rho"]))),
            lock_flag_agreement=float(np.mean(Ra["locked11"] == Rb["locked11"])))
print(f"   max |d rho| = {conv['max_abs_drho']:.2e}, lock agreement {conv['lock_flag_agreement']:.3f}")

# -----------------------------------------------------------------------------
# 6. Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".svg"))
    plt.close(fig)
EE_T = EE  # eps grid, shape (N_RAT, N_EPS)

def pm(axh, Xf, C, **kw):
    """Map drawn in free-swimming frequency coordinates (f_free / f_shed, eps)."""
    return axh.pcolormesh(Xf, EE_T, C, shading="nearest", rasterized=True, **kw)

fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.8))
for col, c in enumerate(C_VALUES):
    M = maps[c]
    Xf = M["rho_free"]
    im = pm(ax[0, col], Xf, M["rho"], cmap="twilight_shifted", vmin=0.4, vmax=2.0)
    ax[0, col].contour(Xf, EE_T, M["locked11"].astype(float), levels=[0.5], colors="k",
                       linewidths=0.8)
    ax[0, col].set_title(rf"Rotation number $\rho$, c = {c}", fontsize=9)
    plt.colorbar(im, ax=ax[0, col], fraction=0.046)
    ly = np.clip(M["lyap"], -0.5, 0.2)
    im = pm(ax[1, col], Xf, ly, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.2))
    ax[1, col].contour(Xf, EE_T, M["chaos_confirmed"].astype(float), levels=[0.5], colors="k",
                       linewidths=0.6)
    ax[1, col].set_title(rf"Largest Lyapunov exponent, c = {c}", fontsize=9)
    plt.colorbar(im, ax=ax[1, col], fraction=0.046)
for a_ in ax.ravel():
    a_.set_xlabel(r"$f_{free}/f_{shed}$")
    a_.set_ylabel(r"$\varepsilon$")
    a_.axvline(1.0, color="w", lw=0.6, ls="--")
    a_.set_xlim(0.4, 2.0)
fig.tight_layout()
save(fig, "fig01_tongues_lyapunov")

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
cols = ["#1b9e77", "#d95f02", "#7570b3"]
for col, e in zip(cols, EPS_STAIR):
    ax[0].plot(RAT_STAIR, stairs[e]["rho"], color=col, lw=0.9, label=rf"$\varepsilon$={e}")
ax[0].plot(RAT_STAIR, RAT_STAIR, color="0.7", lw=0.6, ls=":")
ax[0].set_xlabel(r"$\omega_c/\omega_w$"); ax[0].set_ylabel(r"$\rho$")
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("Devil's staircases (c = 0)", fontsize=9)
if slip is not None:
    ax[1].plot(np.array(slip["strobe_example"]) / (2 * np.pi), color="#d95f02", lw=0.8)
    ax[1].set_xlabel("period"); ax[1].set_ylabel(r"$(\phi - \omega_w t)/2\pi$")
    ax[1].set_title("Phase slips past the edge", fontsize=9)
    d_ = np.array(slip["delta"]); T_ = np.array(slip["T_slip"]); n_ = np.array(slip["n_slips"])
    ok = n_ >= 3
    ax[2].loglog(d_[ok], T_[ok], "o", ms=3, color="#7570b3")
    ax[2].loglog(d_[ok], np.exp(np.polyval(np.polyfit(np.log(d_[ok]), np.log(T_[ok]), 1),
                                            np.log(d_[ok]))), "k--", lw=0.8)
    ax[2].set_xlabel(r"distance past edge $\delta$"); ax[2].set_ylabel("inter-slip time")
    ax[2].set_title(f"slope {slip['exponent']:.3f} (theory -0.5)", fontsize=9)
fig.tight_layout()
save(fig, "fig02_staircase_slips")

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
Xf0 = maps[0.0]["rho_free"]
im = pm(ax[0], Xf0, k_map, cmap="viridis")
ax[0].set_title(r"Locking stiffness $k$ (per period)", fontsize=9)
plt.colorbar(im, ax=ax[0], fraction=0.046)
Fm = maps[0.0]["Fx"]
lim = np.nanmax(np.abs(Fm - F_passive))
im = pm(ax[1], Xf0, Fm - F_passive, cmap="PuOr_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
ax[1].contour(Xf0, EE_T, maps[0.0]["locked11"].astype(float), levels=[0.5], colors="k",
              linewidths=0.6)
ax[1].set_title(r"$\bar F_x$ minus passive (<0: more thrust)", fontsize=9)
plt.colorbar(im, ax=ax[1], fraction=0.046)
for c, ls in zip(C_VALUES, ("-", "--")):
    ax[2].plot(EPS, widths[c], ls, color="k", label=f"c = {c}")
ax[2].set_xlabel(r"$\varepsilon$"); ax[2].set_ylabel(r"1:1 width in $f_{free}/f_{shed}$")
ax[2].legend(frameon=False, fontsize=7)
ax[2].set_title("Tongue width", fontsize=9)
for a_ in ax[:2]:
    a_.set_xlabel(r"$f_{free}/f_{shed}$"); a_.set_ylabel(r"$\varepsilon$")
    a_.set_xlim(0.4, 2.0)
fig.tight_layout()
save(fig, "fig03_stiffness_force_width")

# -----------------------------------------------------------------------------
# 7. Save and decision gate
# -----------------------------------------------------------------------------
def frac(a):
    return float(np.mean(a))

rationals = {}
for e in EPS_STAIR:
    r = stairs[e]["rho"]; per = stairs[e]["lyap"] < -1e-3
    found = {}
    for q in range(1, 5):
        for pnum in range(1, 9):
            v = pnum / q
            if 0.4 <= v <= 2.0 and np.gcd(pnum, q) == 1:
                n = int(np.sum(per & (np.abs(r - v) < 2e-3)))
                if n >= 2:
                    found[f"{pnum}:{q}"] = n
    rationals[str(e)] = found

summary = dict(
    mode=MODE, runtime_s=round(time.time() - t_start, 1), x0=X0, body=BODY, cpg=CPG,
    wake=WAKE, spp=SPP, grid=dict(n_ratio=N_RAT, n_eps=N_EPS, T_trans=T_TRANS, T_meas=T_MEAS),
    passive_mean_force_at_x0=F_passive,
    A={str(c): dict(frac_locked11=frac(maps[c]["locked11"]),
                    frac_chaotic=frac(maps[c]["lyap"] > 0.01),
                    frac_chaotic_confirmed=frac(maps[c]["chaos_confirmed"]),
                    frac_quasiperiodic=frac((np.abs(maps[c]["lyap"]) <= 0.01) & ~maps[c]["locked11"]),
                    frac_quenched=frac(maps[c]["zabs"] < 0.2),
                    max_lyap=float(np.max(maps[c]["lyap"])),
                    tongue_width_free_freq_at_eps={f"{EPS[j]:.2f}": float(widths[c][j])
                                                   for j in range(0, N_EPS, max(1, N_EPS // 10))},
                    tongue_centre_free_freq_at_eps={f"{EPS[j]:.2f}": float(centres[c][j])
                                                    for j in range(0, N_EPS, max(1, N_EPS // 10))})
       for c in C_VALUES},
    B_locked_plateaus_by_eps=rationals,
    C_phase_slips=None if slip is None else {k: v for k, v in slip.items() if k != "strobe_example"},
    D_locking_stiffness=dict(n=int(np.isfinite(k_map).sum()),
                             k_min=float(np.nanmin(k_map)) if np.isfinite(k_map).any() else np.nan,
                             k_median=float(np.nanmedian(k_map)) if np.isfinite(k_map).any() else np.nan,
                             k_max=float(np.nanmax(k_map)) if np.isfinite(k_map).any() else np.nan),
    convergence=conv,
)
np.savez_compressed(os.path.join(OUT_DIR, "results_script03.npz"),
                    RATIO=RATIO, EPS=EPS, k_map=k_map, lead_map=lead_map,
                    **{f"c{int(c)}_{k}": v for c in C_VALUES for k, v in maps[c].items()
                       if isinstance(v, np.ndarray) and k != "s"},
                    **{f"stair_{int(e)}_{k}": v for e in EPS_STAIR for k, v in stairs[e].items()},
                    RAT_STAIR=RAT_STAIR)
with open(os.path.join(OUT_DIR, "summary_script03.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n================ SCRIPT 03 SUMMARY ================")
print(json.dumps(summary, indent=2))
print("\n================ DECISION GATE ================")
w0 = widths[0.0]
j_mid = np.argmin(np.abs(EPS - 6.0))
ok_tongue = bool(np.any(w0 > 0.05))
reach = {}
lk0, rf0 = maps[0.0]["locked11"], maps[0.0]["rho_free"]
lock_range = {}
for j in range(N_EPS):
    if lk0[:, j].any():
        lock_range[f"{EPS[j]:.2f}"] = [float(rf0[lk0[:, j], j].min()), float(rf0[lk0[:, j], j].max())]
for target in (0.8, 0.9, 1.1, 1.2):
    hit = [EPS[j] for j in range(N_EPS) if lk0[:, j].any()
           and (rf0[lk0[:, j], j].max() >= target if target > 1 else rf0[lk0[:, j], j].min() <= target)]
    reach[target] = float(min(hit)) if hit else None
ok_slip = slip is not None and abs(slip["exponent"] + 0.5) < 0.1
summary["gate"] = dict(width_at_eps6=float(w0[j_mid]), max_width=float(w0.max()),
                       locked_range_free_freq_by_eps=lock_range,
                       min_eps_to_lock_f_free_over_f_shed={str(k): v for k, v in reach.items()})
with open(os.path.join(OUT_DIR, "summary_script03.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"1:1 tongue width (in f_free/f_shed): {w0[j_mid]:.3f} at eps = {EPS[j_mid]:.1f}, "
      f"max {w0.max():.3f} ({'PASS' if ok_tongue else 'FAIL'}: need > 0.05 somewhere)")
for k, v in reach.items():
    print(f"   smallest eps whose tongue reaches f_free/f_shed = {k}: {v}")
print(f"Type-I slip exponent: {slip['exponent'] if slip else float('nan'):.3f} "
      f"({'PASS' if ok_slip else 'CHECK'}: theory -0.5)")
print(f"Confirmed chaos (c=0 / c=1): {100*frac(maps[0.0]['chaos_confirmed']):.2f}% / "
      f"{100*frac(maps[1.0]['chaos_confirmed']):.2f}% of the map")
if ok_tongue:
    print("Proceed to Script 04: free position + CPG, using the k map to test the")
    print("entrainment versus station-keeping prediction (constant term -a k).")
else:
    print("Tongue too narrow at plausible coupling: revisit g, mu or the body mode first.")
print(f"\nOutputs written to: {OUT_DIR}")
