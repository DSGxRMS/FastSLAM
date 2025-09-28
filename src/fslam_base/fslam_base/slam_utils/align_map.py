# Offline SE(2) alignment: known_map.csv -> cone_snapshot.csv frame
# - Assumes start-box cones are labeled "orange_large" in both files.
# - Tries permutations to pick the best 4-corner correspondence (no ICP).
# - Outputs transform, saves it to data/alignment_transform.json,
#   and plots snapshot vs transformed map for a quick sanity check.

import csv
import json
import math
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
KNOWN_MAP_PATH = Path("data/known_map.csv")         # your static map file
SNAPSHOT_PATH = Path("data/cone_snapshot.csv")      # the file you just saved
OUT_TRANSFORM_JSON = Path("data/alignment_transform.json")
PLOT_PATH = Path("data/alignment_overlay.png")

# Column names we try to detect
X_KEYS = ["x", "x_m", "x_meter", "xcoord"]
Y_KEYS = ["y", "y_m", "y_meter", "ycoord"]
COLOR_KEYS = ["color", "colour"]

# Color labels we expect
LABELS = {
    "blue": "blue",
    "yellow": "yellow",
    "orange": "orange",
    "orange_large": "orange_large",
    "unknown": "unknown",
}

START_LABELS = {"orange_large"}   # must be 4 points in known_map
FALLBACK_START_LABELS = {"orange"}  # used if snapshot has <4 orange_large

# ---------- utils ----------
def read_cones_csv(path: Path):
    """Reads a CSV with columns (x, y, color/colour). Returns list of dict rows."""
    rows = []
    with open(path, "r", newline="") as f:
        # strip commented header/meta lines
        clean_lines = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    if not clean_lines:
        return rows
    rdr = csv.DictReader(clean_lines)
    # figure columns
    keys = [k.lower() for k in rdr.fieldnames]
    def resolve(colset, default=None):
        for cand in colset:
            if cand in keys:
                return cand
        return default
    xk = resolve(X_KEYS); yk = resolve(Y_KEYS); ck = resolve(COLOR_KEYS)
    if xk is None or yk is None or ck is None:
        raise ValueError(f"CSV {path} missing columns. Found {keys}")

    for r in rdr:
        try:
            x = float(r[xk]); y = float(r[yk]); c = r[ck].strip().lower()
        except Exception:
            continue
        rows.append({"x": x, "y": y, "color": c})
    return rows

def group_by_color(rows):
    g = defaultdict(list)
    for r in rows:
        lab = r["color"]
        if lab not in LABELS:
            lab = "unknown"
        g[lab].append([r["x"], r["y"]])
    # convert to np arrays
    for k in g:
        g[k] = np.asarray(g[k], dtype=float)
    return g

def centroid(X):
    return np.mean(X, axis=0)

def kabsch_se2(A, B):
    """
    Return R(2x2), t(2,), rms that best aligns A (Nx2) onto B (Nx2):
    B ≈ R @ A + t. No reflection, no scale.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    assert A.shape == B.shape and A.shape[1] == 2 and A.shape[0] >= 2

    ca = centroid(A)
    cb = centroid(B)
    A0 = A - ca
    B0 = B - cb

    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # enforce proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca

    err = B - (A @ R.T + t)  # (Nx2)
    rms = math.sqrt(np.mean(np.sum(err**2, axis=1)))
    return R, t, rms

def transform_pts(A, R, t):
    A = np.asarray(A, float)
    return A @ R.T + t

def pick_four_snapshot_starts(snap_groups):
    """Prefer 4 orange_large; fallback to 4 orange closest to their centroid."""
    if "orange_large" in snap_groups and len(snap_groups["orange_large"]) >= 4:
        # choose the 4 closest to their own centroid (in case >4)
        X = snap_groups["orange_large"]
        c = centroid(X)
        d = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(d)[:4]
        return X[idx]
    # fallback to small orange
    if "orange" in snap_groups and len(snap_groups["orange"]) >= 4:
        X = snap_groups["orange"]
        c = centroid(X)
        d = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(d)[:4]
        return X[idx]
    raise ValueError("Snapshot does not contain >=4 start-box oranges (orange_large or orange).")

def pick_four_map_starts(map_groups):
    """Must have exactly or at least 4 orange_large in the known map."""
    if "orange_large" in map_groups and len(map_groups["orange_large"]) >= 4:
        X = map_groups["orange_large"]
        if len(X) == 4:
            return X
        # if >4, pick 4 closest to centroid
        c = centroid(X)
        d = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(d)[:4]
        return X[idx]
    raise ValueError("Known map must contain >=4 'orange_large' start-box cones.")

def best_alignment_from_four(A4_map, B4_snap):
    """
    Try all permutations mapping A4_map (fixed order) onto B4_snap.
    Returns (R, t, rms, perm).
    """
    best = None
    for perm in itertools.permutations(range(4)):
        B = B4_snap[list(perm)]
        R, t, rms = kabsch_se2(A4_map, B)
        if (best is None) or (rms < best[2]):
            best = (R, t, rms, perm)
    return best

def as_homogeneous(R, t):
    H = np.eye(3)
    H[:2,:2] = R
    H[:2, 2] = t
    return H

# ---------- main ----------
def main():
    # Read inputs
    known_rows = read_cones_csv(KNOWN_MAP_PATH)
    snap_rows = read_cones_csv(SNAPSHOT_PATH)
    if not known_rows or not snap_rows:
        raise SystemExit("Empty input. Check file paths and CSV content.")

    known_g = group_by_color(known_rows)
    snap_g = group_by_color(snap_rows)

    # Pick start-box corners
    A4 = pick_four_map_starts(known_g)     # (4,2) in map frame
    B4 = pick_four_snapshot_starts(snap_g) # (4,2) in snapshot/world frame

    # Solve best transform over permutations
    R, t, rms, perm = best_alignment_from_four(A4, B4)

    # Report
    theta = math.atan2(R[1,0], R[0,0])  # yaw
    H = as_homogeneous(R, t)
    print("\n=== SE(2) alignment (map → snapshot/world) ===")
    print(f"RMS on 4 start cones: {rms:.3f} m")
    print(f"Rotation (yaw): {math.degrees(theta):.3f} deg")
    print(f"Translation t: [{t[0]:.3f}, {t[1]:.3f}] m")
    print("Homogeneous T =")
    with np.printoptions(precision=6, suppress=True):
        print(H)

    # Save transform
    OUT_TRANSFORM_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRANSFORM_JSON, "w") as f:
        json.dump({
            "rotation": [[float(R[0,0]), float(R[0,1])],
                         [float(R[1,0]), float(R[1,1])]],
            "translation": [float(t[0]), float(t[1])],
            "yaw_deg": float(math.degrees(theta)),
            "rms_on_starts_m": float(rms),
            "perm_used": list(perm),
            "source": {
                "known_map": str(KNOWN_MAP_PATH),
                "snapshot": str(SNAPSHOT_PATH),
                "labels_assumed": list(LABELS.keys()),
                "start_labels": list(START_LABELS),
            }
        }, f, indent=2)
    print(f"\nSaved transform → {OUT_TRANSFORM_JSON}")

    # Transform the whole known map and plot overlay
    all_map = np.array([[r["x"], r["y"]] for r in known_rows], float)
    all_map_colors = [r["color"] if r["color"] in LABELS else "unknown" for r in known_rows]
    all_map_T = transform_pts(all_map, R, t)

    snap_all = np.array([[r["x"], r["y"]] for r in snap_rows], float)
    snap_cols = [r["color"] if r["color"] in LABELS else "unknown" for r in snap_rows]

    # style
    filled = {
        "blue": dict(marker="o", linestyle="None", markersize=6, label="blue", color="blue"),
        "yellow": dict(marker="o", linestyle="None", markersize=6, label="yellow", color="gold"),
        "orange": dict(marker="^", linestyle="None", markersize=7, label="orange", color="orange"),
        "orange_large": dict(marker="s", linestyle="None", markersize=8, label="orange_large", color="orange"),
        "unknown": dict(marker="x", linestyle="None", markersize=6, label="unknown", color="black"),
    }
    hollow = {
        k: dict(marker=filled[k]["marker"], linestyle="None", markersize=filled[k]["markersize"],
                markerfacecolor="none", markeredgecolor=filled[k]["color"],
                label=f"{k} (map→world)") for k in filled
    }

    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # plot snapshot (filled)
    bycol = defaultdict(list)
    for p, c in zip(snap_all, snap_cols):
        bycol[c].append(p)
    for c, arr in bycol.items():
        arr = np.asarray(arr)
        ax.plot(arr[:,0], arr[:,1], **filled.get(c, filled["unknown"]))

    # plot transformed map (hollow)
    bycolT = defaultdict(list)
    for p, c in zip(all_map_T, all_map_colors):
        bycolT[c].append(p)
    for c, arr in bycolT.items():
        arr = np.asarray(arr)
        ax.plot(arr[:,0], arr[:,1], **hollow.get(c, hollow["unknown"]))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Known map (transformed) over snapshot (world)")
    # clean legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); H2 = []; L2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            H2.append(h); L2.append(l); seen.add(l)
    ax.legend(H2, L2, loc="best")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    print(f"Saved overlay plot → {PLOT_PATH}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
