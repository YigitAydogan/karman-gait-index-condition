# =============================================================================
# KARMAN GAIT AS ACTIVE-PASSIVE LOCKING
# Script 15: Collect every result and cache needed for figure generation into
#            one zip file on Drive, so the figures can be produced elsewhere.
#
# Colab ready. Paste into one cell and run (Drive is mounted by the script).
# No computation and no GPU needed; runtime is a few minutes, set by copying.
#
# What goes in the zip:
#   1. Required files: everything Script 13 reads (Scripts 01 to 12 outputs)
#      and all Script 14 outputs (summary, tables, checkpoints, figures).
#   2. Optional files: every other .json, .npz, .csv, .tex, .md, .txt file in
#      the project folder, plus the Script 13 figure folder (png, pdf, svg).
#   Files larger than MAX_FILE_MB are skipped and listed. If the total would
#   exceed MAX_TOTAL_MB, the largest optional files are dropped first; required
#   files are never dropped.
#   A manifest (path, size, SHA-256 checksum, required or optional) is written
#   into the zip as MANIFEST.json, and a list of skipped files as SKIPPED.json.
#
# Output: MyDrive/karman_gait_bundle_<date>.zip
# =============================================================================

import os, sys, json, time, hashlib, zipfile, fnmatch

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/karman_gait_project"
    OUT_PARENT = "/content/drive/MyDrive"
else:
    BASE_DIR = os.path.abspath("./karman_gait_project")
    OUT_PARENT = os.path.abspath(".")

MAX_FILE_MB = 400          # single files above this are skipped
MAX_TOTAL_MB = 1800        # target size of the bundle before compression
CK = "checkpoints_full"

REQUIRED = [
    "script01_passive_baseline/results_script01.npz",
    "script01_passive_baseline/summary_script01.json",
    "script02_passive_robustness/results_script02.npz",
    "script02_passive_robustness/summary_script02.json",
    "script02b_passive_cases/results_script02b.npz",
    "script02b_passive_cases/summary_script02b.json",
    "script03_cpg_tongues/results_script03.npz",
    "script03_cpg_tongues/summary_script03.json",
    "script05b_holding_dense/results_script05.npz",
    "script05b_holding_dense/results_script05b.npz",
    "script06_robustness/summary_script06_*.json",
    "script06_robustness/orbits_script06_*.json",
    "script06c_body_aL/*.json",
    "script06c_body_aL/*.npz",
    "script07_index_condition/*.json",
    "script08_homotopy/records_script08.json",
    "script08_homotopy/summary_script08.json",
    "script09_model_structure/records_script09_ALL.json",
    "script09_model_structure/summary_script09_ALL.json",
    f"script09_model_structure/{CK}/*.json",
    "script10_continuation/points_script10.npz",
    "script10_continuation/summary_script10.json",
    "script10_continuation/bifurcations_script10.json",
    f"script10_continuation/{CK}/stageC_branches.npz",
    f"script10_continuation/{CK}/stageD2_index.npz",
    f"script10_continuation/{CK}/stageD_index.npz",
    f"script10_continuation/{CK}/stageB_grid.npz",
    "script11_noise_predictions/summary_script11.json",
    f"script11_noise_predictions/{CK}/*.npz",
    "script12_verification/summary_script12.json",
    "script12_verification/manifest.json",
    f"script12_verification/{CK}/*.json",
    "script14_referee_analyses/summary_script14.json",
    "script14_referee_analyses/tables/*",
    "script14_referee_analyses/figures/*",
    f"script14_referee_analyses/{CK}/*.npz",
]
OPTIONAL_EXT = (".json", ".npz", ".csv", ".tex", ".md", ".txt")
OPTIONAL_DIRS = ["script13_figures"]          # whole folder (figures, tables, captions)
SKIP_DIRS = ("__pycache__", ".ipynb_checkpoints")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def mb(n):
    return n / 1024 / 1024

t0 = time.time()
if not os.path.isdir(BASE_DIR):
    raise SystemExit(f"Project folder not found: {BASE_DIR}")

all_files = []
for root, dirs, files in os.walk(BASE_DIR):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fn in files:
        full = os.path.join(root, fn)
        rel = os.path.relpath(full, BASE_DIR).replace(os.sep, "/")
        if fn.endswith(".tmp") or fn.startswith("karman_gait_bundle"):
            continue
        all_files.append(rel)

def is_required(rel):
    return any(fnmatch.fnmatch(rel, pat) for pat in REQUIRED)

def is_optional(rel):
    if any(rel.startswith(d + "/") for d in OPTIONAL_DIRS):
        return True
    return rel.lower().endswith(OPTIONAL_EXT)

req, opt, skipped = [], [], []
for rel in sorted(all_files):
    size = os.path.getsize(os.path.join(BASE_DIR, rel))
    if size > MAX_FILE_MB * 1024 * 1024:
        skipped.append(dict(path=rel, size_mb=round(mb(size), 1), reason=f"larger than {MAX_FILE_MB} MB"))
        continue
    if is_required(rel):
        req.append((rel, size))
    elif is_optional(rel):
        opt.append((rel, size))

missing = [pat for pat in REQUIRED if not any(fnmatch.fnmatch(r, pat) for r, _ in req)]
total_req = sum(s for _, s in req)
budget = MAX_TOTAL_MB * 1024 * 1024 - total_req
opt.sort(key=lambda t: t[1])                       # smallest first
kept_opt = []
for rel, size in opt:
    if size <= budget:
        kept_opt.append((rel, size)); budget -= size
    else:
        skipped.append(dict(path=rel, size_mb=round(mb(size), 1), reason="bundle size limit"))

stamp = time.strftime("%Y%m%d_%H%M")
zip_path = os.path.join(OUT_PARENT, f"karman_gait_bundle_{stamp}.zip")
manifest = []
print(f"Required files: {len(req)} ({mb(total_req):.1f} MB); optional files: {len(kept_opt)} "
      f"({mb(sum(s for _, s in kept_opt)):.1f} MB); skipped: {len(skipped)}")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
    for i, (rel, size) in enumerate(req + kept_opt):
        full = os.path.join(BASE_DIR, rel)
        # npz, png and pdf files are already compressed: store them as they are
        ctype = zipfile.ZIP_STORED if rel.lower().endswith((".npz", ".png", ".pdf", ".zip")) else zipfile.ZIP_DEFLATED
        z.write(full, arcname="karman_gait_project/" + rel, compress_type=ctype)
        manifest.append(dict(path=rel, size_bytes=size, sha256=sha256(full), required=i < len(req)))
        if i % 50 == 0:
            print(f"   {i + 1}/{len(req) + len(kept_opt)} files ({time.time() - t0:.0f} s)")
    z.writestr("karman_gait_project/MANIFEST.json", json.dumps(dict(
        created=stamp, base_dir=BASE_DIR, n_files=len(manifest), files=manifest,
        required_patterns_not_found=missing), indent=2))
    z.writestr("karman_gait_project/SKIPPED.json", json.dumps(skipped, indent=2))

print("\n================ SCRIPT 15 SUMMARY ================")
print(f"Bundle: {zip_path}")
print(f"Size on disk: {mb(os.path.getsize(zip_path)):.1f} MB, files: {len(manifest)}")
if missing:
    print("Required patterns with no file (fine if that script never wrote it):")
    for pat in missing:
        print(f"   {pat}")
if skipped:
    print(f"Skipped {len(skipped)} files (listed in SKIPPED.json inside the zip); largest:")
    for s_ in sorted(skipped, key=lambda d: -d["size_mb"])[:10]:
        print(f"   {s_['path']} ({s_['size_mb']} MB, {s_['reason']})")
print(f"Runtime: {time.time() - t0:.0f} s")
