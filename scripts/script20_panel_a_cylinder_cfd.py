"""Run a 2D, cylinder-only CFD illustration aligned to Fig1_model.png.

Requirements: Python 3, numpy, scipy, matplotlib, Pillow, and GCC or Clang.
GCC with OpenMP is used when available. A serial build is also supported.
Usage: python panel_a_cylinder_cfd.py Fig1_model.png
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


CROP = (0, 0, 2048, 780)
CENTER_PX = (358.5, 426.5)
DIAMETER_PX = 110.0
REYNOLDS = 100.0
INLET = 0.085
DPI = 600
MAX_DISPLAY_SPEED = 1.5
DIRECTIONS = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
              (1, 1), (-1, 1), (-1, -1), (1, -1))
WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "panel_speed", ["#428fb8", "#fbfcfc", "#e8a33f"], N=1025
)
SPEED_NORM = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=MAX_DISPLAY_SPEED)

# Compiled inner loop for the Python solver. Direction order follows DIRECTIONS.
# The population buffers are direction-major, with a ghost column on each side.
KERNEL = r"""
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void relax_pair(float *plus_out, float *minus_out,
                              float plus, float minus, float eq_plus,
                              float eq_minus, float omega_e, float omega_o) {
    float even = 0.5f * omega_e * (plus + minus - eq_plus - eq_minus);
    float odd = 0.5f * omega_o * (plus - minus - eq_plus + eq_minus);
    *plus_out = plus - even - odd;
    *minus_out = minus - even + odd;
}

void advance(float *a, float *b, const uint16_t *flags, const float *sponge,
             int nx, int ny, int steps, float inlet, float omega_e,
             float omega_o, int threads) {
    int pitch = nx + 2, plane = pitch * ny;
    const int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    const float weight[9] = {4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9,
                             1.f/36, 1.f/36, 1.f/36, 1.f/36};
    const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    float far[9];
    for (int q = 0; q < 9; q++) {
        float cu = cx[q] * inlet;
        far[q] = weight[q] * (1 + 3*cu + 4.5f*cu*cu - 1.5f*inlet*inlet);
    }

    // Each advance call uses an even step count so the result lands in a.
    for (int step = 0; step < steps; step++) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int y = 0; y < ny; y++) {
            int row = y*pitch;
            int prev = (y == 0 ? ny-1 : y-1)*pitch;
            int next = (y == ny-1 ? 0 : y+1)*pitch;
            for (int x = 1; x <= nx; x++) {
                int i = row + x;
                uint16_t boundary = flags[i];
                if (boundary & 1) continue;
                float f[9] = {a[i], a[plane+i-1], a[2*plane+prev+x],
                    a[3*plane+i+1], a[4*plane+next+x], a[5*plane+prev+x-1],
                    a[6*plane+prev+x+1], a[7*plane+next+x+1],
                    a[8*plane+next+x-1]};
                // Halfway bounce-back: incoming solid links reflect at mid-link.
                if (boundary) {
                    for (int q = 1; q < 9; q++) {
                        if (boundary & (1u << q)) f[q] = a[opposite[q]*plane+i];
                    }
                }
                if (x == 1) {
                    // Zou-He prescribed left-boundary velocity, v = 0.
                    float rho = (f[0]+f[2]+f[4]+2*(f[3]+f[6]+f[7]))/(1-inlet);
                    f[1] = f[3] + (2.f/3)*rho*inlet;
                    f[5] = f[7] + 0.5f*(f[4]-f[2]) + rho*inlet/6;
                    f[8] = f[6] + 0.5f*(f[2]-f[4]) + rho*inlet/6;
                }
                if (x == nx) {
                    // Zero-gradient ghost populations at the outlet.
                    f[3] = a[3*plane+i];
                    f[6] = a[6*plane+prev+x];
                    f[7] = a[7*plane+next+x];
                }
                float rho = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8];
                float u = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])/rho;
                float v = (f[2]-f[4]+f[5]+f[6]-f[7]-f[8])/rho;
                float base = 1 - 1.5f*(u*u+v*v);
                float eq[9];
                for (int q = 0; q < 9; q++) {
                    float cu;
                    switch (q) {
                        case 1: cu = u; break;
                        case 2: cu = v; break;
                        case 3: cu = -u; break;
                        case 4: cu = -v; break;
                        case 5: cu = u+v; break;
                        case 6: cu = -u+v; break;
                        case 7: cu = -u-v; break;
                        case 8: cu = u-v; break;
                        default: cu = 0;
                    }
                    eq[q] = rho*weight[q]*(base+3*cu+4.5f*cu*cu);
                }
                float post[9];
                post[0] = f[0] - omega_e*(f[0] - eq[0]);
                relax_pair(&post[1], &post[3], f[1], f[3], eq[1], eq[3], omega_e, omega_o);
                relax_pair(&post[2], &post[4], f[2], f[4], eq[2], eq[4], omega_e, omega_o);
                relax_pair(&post[5], &post[7], f[5], f[7], eq[5], eq[7], omega_e, omega_o);
                relax_pair(&post[6], &post[8], f[6], f[8], eq[6], eq[8], omega_e, omega_o);
                float damping = sponge[x];
                for (int q = 0; q < 9; q++) {
                    b[q*plane+i] = post[q] + damping*(far[q]-post[q]);
                }
            }
        }
        float *temp = a;
        a = b;
        b = temp;
    }
}
"""


def build_solver(tempdir):
    source = Path(tempdir) / "advance.c"
    shared = Path(tempdir) / ("advance.dll" if os.name == "nt" else "advance.so")
    source.write_text(KERNEL, encoding="utf-8")
    compiler = os.environ.get("CC", "gcc")
    command = [compiler, "-O3", "-ffast-math", "-march=native",
               "-shared", "-fPIC", str(source), "-o", str(shared)]
    result = subprocess.run(command + ["-fopenmp"], capture_output=True, text=True)
    if result.returncode:
        subprocess.run(command, check=True)
        print("OpenMP unavailable, using a serial solver.", flush=True)
    library = ctypes.CDLL(str(shared))
    advance = library.advance
    float_array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    uint16_array = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
    advance.argtypes = [float_array, float_array, uint16_array, float_array,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]
    advance.restype = None
    return advance


def macros(populations):
    rho = np.sum(populations, axis=0)
    _, e, n, w, s, ne, nw, sw, se = populations
    u = (e - w + ne - nw - sw + se) / rho
    v = (n - s + ne + nw - sw - se) / rho
    return rho, u, v


def run_flow(diameter, duration, threads):
    # Coordinates in cylinder diameters, with y positive upwards.
    x = np.arange(-7 * diameter, 24 * diameter + 1) / diameter
    y = np.arange(-9 * diameter, 9 * diameter + 1) / diameter
    xx, yy = np.meshgrid(x, y)
    solid = xx**2 + yy**2 <= 0.25
    mask = solid.astype(np.uint16)
    for q, (cx, cy) in enumerate(DIRECTIONS[1:], 1):
        mask |= np.roll(solid, (cy, cx), axis=(0, 1)).astype(np.uint16) << q
    flags = np.pad(mask, ((0, 0), (1, 1)))

    # A small, initial wake disturbance breaks exact reflection symmetry.
    u = np.where(solid, 0, INLET)
    v = 0.12 * INLET * np.exp(-((xx - 1.2)**2 / 0.75 + yy**2 / 0.4))
    v[solid] = 0
    a = np.empty((9, y.size, x.size + 2), dtype=np.float32)
    for q, ((cx, cy), weight) in enumerate(zip(DIRECTIONS, WEIGHTS)):
        cu = cx * u + cy * v
        a[q, :, 1:-1] = weight * (1 + 3*cu + 4.5*cu**2 - 1.5*(u**2 + v**2))
    a[:, :, 0] = a[:, :, 1]
    a[:, :, -1] = a[:, :, -2]
    b = a.copy()

    # Sponge is downstream of the entire visible panel and starts at x/D=19.
    sponge = np.pad(0.035 * np.clip((x-19)/5, 0, 1)**3, 1).astype(np.float32)
    nu_lattice = INLET * diameter / REYNOLDS
    tau_even = 0.5 + 3 * nu_lattice
    tau_odd = 0.5 + (3/16) / (tau_even - 0.5)
    batch = max(2, int(0.22 * diameter / INLET) // 2 * 2)
    total = int(duration * diameter / INLET) // batch * batch
    y_probe = int(np.argmin(np.abs(y)))
    x_probe = int(np.argmin(np.abs(x-4))) + 1
    samples, snapshots = [], []
    start = time.monotonic()
    next_report = 0.0
    with tempfile.TemporaryDirectory(prefix="cylinder_lbm_") as tempdir:
        advance = build_solver(tempdir)
        for step in range(batch, total+1, batch):
            advance(a, b, flags, sponge, x.size, y.size, batch,
                    INLET, 1/tau_even, 1/tau_odd, threads)
            t_star = step * INLET / diameter
            _, _, probe_v = macros(a[:, y_probe, x_probe])
            if not np.isfinite(probe_v):
                raise RuntimeError("The wake velocity became non-finite.")
            samples.append((t_star, float(probe_v) / INLET))
            if t_star >= duration - 14:
                rho, u, v = macros(a[:, :, 1:-1])
                u[solid], v[solid] = 0, 0
                snapshots.append((t_star, u / INLET, v / INLET))
            if t_star >= next_report or step == total:
                print(f"tU∞/D={t_star:6.1f}/{duration:g}, "
                      f"wake v/U∞={probe_v / INLET:+.3f}, "
                      f"elapsed={time.monotonic()-start:.0f}s", flush=True)
                next_report += 12
    if not np.isfinite(rho[~solid]).all() or rho[~solid].min() <= 0:
        raise RuntimeError("The final fluid density is invalid.")
    return x, y, solid, snapshots, np.array(samples), rho


def choose_snapshot(x, y, snapshots):
    # The first upper rotation arrow in the artwork indicates clockwise
    # rotation. Choose a late snapshot with clockwise vorticity in that area.
    xx, yy = np.meshgrid(x, y)
    upper_wake = ((xx > 3.6) & (xx < 5.0) & (yy > 0.05) & (yy < 1.65))
    spacing = x[1] - x[0]
    def phase_score(snapshot):
        _, u, v = snapshot
        omega = np.gradient(v, spacing, axis=1) - np.gradient(u, spacing, axis=0)
        return float(np.mean(omega[upper_wake]))
    return min(snapshots, key=phase_score)


def sample_field(field, x, y, x_pixels, y_pixels):
    x_d = (x_pixels - CENTER_PX[0]) / DIAMETER_PX
    y_d = (CENTER_PX[1] - y_pixels) / DIAMETER_PX
    sx = (x_d - x[0]) / (x[1] - x[0])
    sy = (y_d - y[0]) / (y[1] - y[0])
    return map_coordinates(field, [sy, sx], order=1, mode="nearest")


def draw_background(x, y, u, v, outdir):
    width, height = CROP[2], CROP[3]
    py, px = np.mgrid[0:height, 0:width].astype(np.float32)
    px += 0.5
    py += 0.5
    speed = np.hypot(sample_field(u, x, y, px, py), sample_field(v, x, y, px, py))
    circle_distance = np.hypot(px-CENTER_PX[0], py-CENTER_PX[1])
    image = SPEED_CMAP(SPEED_NORM(np.clip(speed, 0, MAX_DISPLAY_SPEED)), bytes=True)
    image[..., 3] = np.round(255*np.clip(circle_distance-DIAMETER_PX/2+0.5, 0, 1))

    fig = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    ax.imshow(image, extent=(0, width, height, 0), interpolation="none")
    # Broader spacing in the outer flow, finer sampling through the wake.
    outer_y = np.array([76, 167, 254, 603, 697])
    wake_y = np.array([294, 334, 373, 409, 445, 481, 519, 560])
    ox, oy = np.meshgrid(np.arange(52, width-20, 77), outer_y)
    wx, wy = np.meshgrid(np.arange(52, width-20, 56), wake_y)
    ax_px = np.concatenate((ox.ravel(), wx.ravel()))
    ay_px = np.concatenate((oy.ravel(), wy.ravel()))
    qu = sample_field(u, x, y, ax_px, ay_px)
    qv = sample_field(v, x, y, ax_px, ay_px)
    magnitude = np.hypot(qu, qv)
    keep = (np.hypot(ax_px-CENTER_PX[0], ay_px-CENTER_PX[1])
            > DIAMETER_PX/2+15) & (magnitude >= 0.04)
    # Speed is encoded by the color field. Arrow length only aids legibility.
    length = 18 + 16*np.clip(magnitude, 0, 1.35)
    ax.quiver(ax_px[keep], ay_px[keep], (qu*length/np.maximum(magnitude, 1e-6))[keep],
              (-qv*length/np.maximum(magnitude, 1e-6))[keep],
              color=(0.13, 0.32, 0.40, 0.44), angles="xy", scale_units="xy",
              scale=1, units="xy", width=1.85, headwidth=4.6, headlength=4.9,
              headaxislength=4.2, pivot="middle")
    path = outdir / "Panel_A_CFD_Background.png"
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    with Image.open(path) as background:
        assert background.size == (width, height)
    return path, float(speed[circle_distance > DIAMETER_PX/2+3].max())


def transparent_artwork(crop):
    # The source artwork is rasterized against pure white. Keep solid artwork
    # opaque, and remove the white matte from light antialiasing pixels.
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255
    # Preserve colored fill and gray vortex cores. Fade pixels that are within
    # 18/255 of white in every channel to retain smooth edges.
    contrast = 1 - rgb.min(axis=2)
    alpha = np.clip(contrast / (18/255), 0, 1)
    color = 1 - (1 - rgb) / np.maximum(alpha[..., None], 1e-8)
    rgba = np.dstack((np.clip(color, 0, 1), alpha))
    return Image.fromarray(np.round(255*rgba).astype(np.uint8), "RGBA")


def draw_legend(outdir):
    fig = plt.figure(figsize=(2.56, 0.72), dpi=DPI)
    ax = fig.add_axes((0.085, 0.47, 0.83, 0.20))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=SPEED_NORM, cmap=SPEED_CMAP),
                       cax=ax, orientation="horizontal")
    # Use a linear speed axis with the same piecewise color mapping as the field.
    ax.set_xscale("linear")
    bar.set_ticks([0, 0.5, 1.0, MAX_DISPLAY_SPEED], labels=["0", "0.5", "1.0", f"{MAX_DISPLAY_SPEED:g}"])
    bar.ax.tick_params(length=2.2, width=0.65, labelsize=8, pad=2)
    bar.outline.set_linewidth(0.65)
    fig.text(0.5, 0.93, r"Speed $|\mathbf{u}|/U_\infty$", ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.07, "Arrows show instantaneous flow direction",
             ha="center", va="bottom", fontsize=7)
    fig.savefig(outdir / "Panel_A_CFD_Speed_Key.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", nargs="?", type=Path, default=Path("Fig1_model.png"))
    parser.add_argument("--outdir", type=Path, default=Path("panel_a_cfd"))
    parser.add_argument("--cells-per-diameter", type=int, default=36)
    parser.add_argument("--duration", type=float, default=132)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--field-in", type=Path, help="Replot a previously computed field.")
    args = parser.parse_args()
    if args.cells_per_diameter < 16 or args.duration < 20 or args.threads < 1:
        parser.error("Use at least 16 cells per diameter, 20 time units, and one thread.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.figure) as original:
        if original.size != (2048, 1635):
            parser.error("This registration is for the supplied 2048 × 1635 source figure.")
        crop = original.convert("RGBA").crop(CROP)
    crop.save(args.outdir / "Panel_A_Crop.png", dpi=(DPI, DPI))
    artwork = transparent_artwork(crop)
    artwork.save(args.outdir / "Panel_A_Transparent_Artwork.png", dpi=(DPI, DPI))

    field_path = args.outdir / "Panel_A_CFD_Field.npz"
    summary_path = args.outdir / "Panel_A_CFD_Run.json"
    if args.field_in is None:
        x, y, solid, snapshots, probe, rho = run_flow(
            args.cells_per_diameter, args.duration, args.threads
        )
        t_star, u, v = choose_snapshot(x, y, snapshots)
        late = probe[probe[:, 0] > args.duration - 36]
        peaks, _ = find_peaks(late[:, 1], prominence=0.12)
        periods = np.diff(late[peaks, 0])
        strouhal = float(1 / periods.mean()) if periods.size else None
        summary = {
            "reynolds_number": REYNOLDS,
            "cells_per_diameter": args.cells_per_diameter,
            "lattice_shape_y_x": list(solid.shape),
            "domain_x_D": [float(x[0]), float(x[-1])],
            "domain_y_D": [float(y[0]), float(y[-1])],
            "lattice_inlet_speed": INLET,
            "final_t_Uinf_over_D": float(probe[-1, 0]),
            "snapshot_t_Uinf_over_D": t_star,
            "probe_xy_D": [4, 0],
            "late_probe_v_over_Uinf_range": [float(late[:, 1].min()),
                                              float(late[:, 1].max())],
            "strouhal_number_from_late_probe": strouhal,
            "late_peak_intervals_t_Uinf_over_D": periods.tolist(),
            "final_fluid_density_range": [float(rho[~solid].min()),
                                           float(rho[~solid].max())],
            "panel_crop_box_px": list(CROP),
            "cylinder_center_from_top_left_px": list(CENTER_PX),
            "cylinder_diameter_px": DIAMETER_PX,
            "color_scale_speed_over_Uinf": [0, MAX_DISPLAY_SPEED]
        }
        np.savez_compressed(field_path, x_over_D=x, y_over_D=y,
                            u_over_Uinf=u, v_over_Uinf=v, cylinder=solid,
                            probe_t_Uinf_over_D=probe[:, 0],
                            probe_v_over_Uinf=probe[:, 1],
                            summary_json=json.dumps(summary))
    else:
        with np.load(args.field_in) as data:
            x, y = data["x_over_D"], data["y_over_D"]
            u, v = data["u_over_Uinf"], data["v_over_Uinf"]
            summary = json.loads(data["summary_json"].item())

    background_path, speed_max = draw_background(x, y, u, v, args.outdir)
    with Image.open(background_path) as background:
        white = Image.new("RGBA", crop.size, "white")
        composite = Image.alpha_composite(Image.alpha_composite(white, background), artwork)
        composite.convert("RGB").save(args.outdir / "Panel_A_CFD_Composite.png", dpi=(DPI, DPI))
    draw_legend(args.outdir)
    summary["color_scale_speed_over_Uinf"] = [0, MAX_DISPLAY_SPEED]
    summary["rendered_panel_max_speed_over_Uinf"] = speed_max
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
