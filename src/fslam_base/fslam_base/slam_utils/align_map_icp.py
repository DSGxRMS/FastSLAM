#!/usr/bin/env python3
# align_map_snapshot.py
# Skidpad map→world alignment (simple + deterministic):
# 1) Anchor on 4 small-orange start cones (ROI in known map) → initial R,t (translation pinned to start centroid)
# 2) Colored-only ICP (same-color NN, trimmed LS), translation RE-PINNED each iter
# 3) Unknown-only ICP (rotation-only, trimmed), translation RE-PINNED each iter
# 4) One-shot yaw disambiguation using start heading + colored counts (keeps translation pinned)
# Saves: data/alignment_transform.json + data/alignment_overlay.png

import csv
import json
import math
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Paths ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_MAP_PATH     = DATA_DIR / "known_map.csv"
SNAPSHOT_PATH      = DATA_DIR / "cone_snapshot.csv"
OUT_TRANSFORM_JSON = DATA_DIR / "alignment_transform.json"
PLOT_PATH          = DATA_DIR / "alignment_overlay.png"

# --------------------------- Config --------------------------
# ROI for start box in KNOWN MAP frame
ROI_Y_MIN, ROI_Y_MAX = -15.0, 0.0
ROI_X_ABS_MAX = 2.0

# ICP settings
MAX_ITERS_COLORED = 10
MAX_ITERS_UNKNOWN = 10
NN_RADIUS_COLORED = 2.0      # meters
NN_RADIUS_UNKNOWN = 2.0      # meters
TRIM_FRACTION     = 0.20     # drop worst 20% pairs each iter
CONV_EPS_YAW_RAD  = 1e-3     # ~0.057 deg
CONV_EPS_T        = 1e-3     # meters

# Column aliases + colors
X_KEYS = ["x", "x_m", "x_meter", "xcoord"]
Y_KEYS = ["y", "y_m", "y_meter", "ycoord"]
C_KEYS = ["color", "colour"]

LABELS = {"blue", "yellow", "orange", "orange_large", "unknown"}
COLOR_ALIASES = {
    'bluecone':'blue', 'b':'blue', 'blue':'blue',
    'yellowcone':'yellow', 'y':'yellow', 'yellow':'yellow',
    'orange_small':'orange', 'small_orange':'orange', 'orange':'orange',
    'orange_big':'orange_large', 'big_orange':'orange_large', 'orange_large':'orange_large'
}

# ----------------------- IO helpers --------------------------
def read_cones_csv(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        clean = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    if not clean:
        return rows
    rdr = csv.DictReader(clean)
    keys = [k.lower() for k in rdr.fieldnames]
    def pick(cands):
        for c in cands:
            if c in keys: return c
        return None
    xk, yk, ck = pick(X_KEYS), pick(Y_KEYS), pick(C_KEYS)
    if not (xk and yk and ck):
        raise SystemExit(f"{path} missing required columns. Found: {keys}")
    for r in rdr:
        try:
            x = float(r[xk]); y = float(r[yk])
            c_raw = r[ck].strip().lower()
            c = COLOR_ALIASES.get(c_raw, c_raw)
            if c not in LABELS: c = "unknown"
            rows.append({"x": x, "y": y, "color": c})
        except Exception:
            continue
    return rows

# --------------------- Geometry helpers ----------------------
def yaw_from_R(R):
    return float(math.atan2(R[1,0], R[0,0]))  # radians

def R_from_yaw(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s],[s, c]], float)

def kabsch_se2(A, B):
    """Return R(2x2), t(2,), rms that aligns A→B (no scale, det>0)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca = np.mean(A, axis=0); cb = np.mean(B, axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    err = B - (A @ R.T + t)
    rms = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return R, t, rms

def kabsch_rotation_only(A, B):
    """Return R(2x2) that best rotates centered A to centered B (translation = 0)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    return R

def as_homogeneous(R, t):
    H = np.eye(3); H[:2,:2] = R; H[:2,2] = t
    return H

def four_point_rect_features(P):
    """Given 4 points (4x2), return (a,b,d). a<=b are side lengths, d=mean diagonal."""
    P = np.asarray(P, float)
    d2 = []
    for i in range(4):
        for j in range(i+1,4):
            d2.append(np.sum((P[i]-P[j])**2))
    d2 = np.sort(np.array(d2))
    edges = np.sqrt(d2[:4]); diags = np.sqrt(d2[4:])
    edges = np.sort(edges)
    a = float((edges[0]+edges[1])/2.0)
    b = float((edges[2]+edges[3])/2.0)
    if a > b: a, b = b, a
    d = float(np.mean(diags))
    return a, b, d

def start_heading_from_rect_map(A4_map):
    """Lane-forward vector from the 4 start oranges in MAP: front-mid − rear-mid (by y in map)."""
    A = np.asarray(A4_map, float)
    idx = np.argsort(A[:,1])
    rear = A[idx[:2], :]; front = A[idx[2:], :]
    mid_rear  = np.mean(rear, axis=0)
    mid_front = np.mean(front, axis=0)
    h_map = mid_front - mid_rear
    n = np.linalg.norm(h_map)
    return h_map / (n + 1e-9)

# -------------------- Selection / matching -------------------
def select_map_start(known_rows):
    cands = [r for r in known_rows
             if r["color"]=="orange"
             and (ROI_Y_MIN <= r["y"] <= ROI_Y_MAX)
             and (abs(r["x"]) <= ROI_X_ABS_MAX)]
    if len(cands) != 4:
        raise SystemExit(f"Map ROI expected 4 small-orange start cones, got {len(cands)}. Adjust ROI or map.")
    A4 = np.array([[r["x"], r["y"]] for r in cands], float)
    return A4

def select_snapshot_start(snap_rows, A4_map):
    oranges = [r for r in snap_rows if r["color"]=="orange"]
    if len(oranges) < 4:
        raise SystemExit("Snapshot must contain >=4 small-orange cones.")
    a_map, b_map, d_map = four_point_rect_features(A4_map)
    best_B4 = None; best_cost = 1e18
    for combo in itertools.combinations(oranges, 4):
        B4 = np.array([[r["x"], r["y"]] for r in combo], float)
        a_s, b_s, d_s = four_point_rect_features(B4)
        cost = abs((a_s/max(b_s,1e-6)) - (a_map/max(b_map,1e-6))) + abs(d_s/max(d_map,1e-6) - 1.0)
        if cost < best_cost:
            best_cost = cost; best_B4 = B4
    return best_B4

def pin_translation(R, c_map, c_snap):
    """Keep start centroid fixed: R*c_map + t = c_snap."""
    return (c_snap - R @ c_map)

def nn_correspondences(X, Y, radius):
    """Brute-force nearest-neighbor pairs from X to Y within radius. Returns indices and distances."""
    if X.shape[0]==0 or Y.shape[0]==0:
        return [], []
    idxs = []; dists = []
    for i, xi in enumerate(X):
        diffs = Y - xi
        ds = np.linalg.norm(diffs, axis=1)
        j = int(np.argmin(ds))
        if ds[j] <= radius:
            idxs.append((i,j)); dists.append(float(ds[j]))
    return idxs, dists

def trim_pairs(pairs, dists, trim_frac):
    """Drop worst fraction by distance."""
    if not pairs: return [], []
    n = len(pairs)
    k = int(max(1, round(n*(1.0-trim_frac))))
    order = np.argsort(dists)[:k]
    return [pairs[i] for i in order], [dists[i] for i in order]

# ------------------------ ICP stages -------------------------
def icp_colored(R_init, t_init, known_rows, snap_rows, A4_map, B4_snap):
    """SE(2) ICP using colored cones only (same color). Translation is re-pinned each iter."""
    R = np.array(R_init, float); t = np.array(t_init, float)
    c_map  = np.mean(A4_map, axis=0)
    c_snap = np.mean(B4_snap, axis=0)

    # colored sets
    mapC = [(i, r) for i, r in enumerate(known_rows) if r["color"] in ("blue","yellow")]
    snapC = [(j, r) for j, r in enumerate(snap_rows) if r["color"] in ("blue","yellow")]
    Xm = np.array([[r["x"], r["y"]] for _, r in mapC], float)
    Ym = [r["color"] for _, r in mapC]
    Xs = np.array([[r["x"], r["y"]] for _, r in snapC], float)
    Ys = [r["color"] for _, r in snapC]

    for _ in range(MAX_ITERS_COLORED):
        # transform map colored
        Xm_world = (Xm @ R.T) + t

        # split by color for NN
        pairs_all = []; dists_all = []
        for col in ("blue","yellow"):
            idx_m = [i for i,c in enumerate(Ym) if c==col]
            idx_s = [j for j,c in enumerate(Ys) if c==col]
            if not idx_m or not idx_s:
                continue
            Xm_c = Xm_world[idx_m]
            Xs_c = Xs[idx_s]
            pairs, dists = nn_correspondences(Xm_c, Xs_c, NN_RADIUS_COLORED)
            # map back to global indices
            pairs = [ (idx_m[i], idx_s[j]) for (i,j) in pairs ]
            pairs_all += pairs; dists_all += dists

        # trim
        pairs_all, dists_all = trim_pairs(pairs_all, dists_all, TRIM_FRACTION)
        if len(pairs_all) < 2:
            break  # not enough pairs to update

        A = np.array([ Xm[p] for p,_ in pairs_all ], float)  # map (pre-transform, body frame)
        B = np.array([ Xs[q] for _,q in pairs_all ], float)  # snapshot/world

        # closed-form SE(2)
        R_new, t_new, _ = kabsch_se2(A, B)
        # keep ONLY rotation; re-pin translation
        t_new = pin_translation(R_new, c_map, c_snap)

        dtheta = abs(yaw_from_R(R_new) - yaw_from_R(R))
        dt = np.linalg.norm(t_new - t)

        R, t = R_new, t_new
        if dtheta < CONV_EPS_YAW_RAD and dt < CONV_EPS_T:
            break
    return R, t

def icp_unknown_rotation_only(R_init, t_init, known_rows, snap_rows, A4_map, B4_snap):
    """Rotation-only ICP using unknown-color cones. Translation re-pinned each iter."""
    R = np.array(R_init, float); t = np.array(t_init, float)
    c_map  = np.mean(A4_map, axis=0)
    c_snap = np.mean(B4_snap, axis=0)

    mapU = np.array([[r["x"], r["y"]] for r in known_rows if r["color"]=="unknown"], float)
    snapU = np.array([[r["x"], r["y"]] for r in snap_rows  if r["color"]=="unknown"], float)

    if mapU.shape[0] < 2 or snapU.shape[0] < 2:
        return R, t  # nothing to do

    for _ in range(MAX_ITERS_UNKNOWN):
        # transform map unknowns
        M = (mapU @ R.T) + t

        pairs, dists = nn_correspondences(M, snapU, NN_RADIUS_UNKNOWN)
        pairs, dists = trim_pairs(pairs, dists, TRIM_FRACTION)
        if len(pairs) < 2:
            break

        A = np.array([ M[i] for i,_ in pairs ], float)  # already in world
        B = np.array([ snapU[j] for _,j in pairs ], float)

        # rotation-only: center both and solve R
        Ac = A - np.mean(A, axis=0)
        Bc = B - np.mean(B, axis=0)
        R_delta = kabsch_rotation_only(Ac, Bc)
        R_new = R_delta @ R
        # re-pin translation
        t_new = pin_translation(R_new, c_map, c_snap)

        dtheta = abs(yaw_from_R(R_new) - yaw_from_R(R))
        dt = np.linalg.norm(t_new - t)

        R, t = R_new, t_new
        if dtheta < CONV_EPS_YAW_RAD and dt < CONV_EPS_T:
            break
    return R, t

# ------------------ Yaw disambiguation (π flip) ------------------
def score_colored_handedness(R, t, h_map, c_map, snap_rows):
    """Count colored cones ahead of start that are on correct side:
       +1 if yellow RIGHT (negative), +1 if blue LEFT (positive)."""
    h = (R @ h_map).astype(float)
    h /= max(np.linalg.norm(h), 1e-9)
    c_world = (R @ np.array(c_map, float)) + t
    def side(v): return float(h[0]*v[1] - h[1]*v[0])  # right<0, left>0
    score = 0
    for r in snap_rows:
        if r["color"] not in ("yellow","blue"):
            continue
        v = np.array([r["x"], r["y"]]) - c_world
        proj = float(np.dot(h, v)) / (np.linalg.norm(v)+1e-9)
        if proj <= 0.1:  # ahead-ish
            continue
        s = side(v)
        if r["color"]=="yellow" and s < 0: score += 1
        if r["color"]=="blue"   and s > 0: score += 1
    return score

# --------------------------- Plot ----------------------------
def plot_overlay(snap_rows, map_rows_T, plot_path: Path):
    filled = {
        "blue": dict(marker="o",  ls="None", ms=6,  label="blue",          color="blue"),
        "yellow": dict(marker="o",ls="None", ms=6,  label="yellow",        color="gold"),
        "orange": dict(marker="^",ls="None", ms=7,  label="orange",        color="orange"),
        "orange_large": dict(marker="s",ls="None", ms=8, label="orange_large", color="orange"),
        "unknown": dict(marker="x",ls="None", ms=6,  label="unknown",       color="black"),
    }
    hollow = {
        k: dict(marker=filled[k]["marker"], ls="None", ms=filled[k]["ms"],
                markerfacecolor="none", markeredgecolor=filled[k]["color"],
                label=f"{k} (map→world)") for k in filled
    }
    snap_xy = np.array([[r["x"], r["y"]] for r in snap_rows], float)
    snap_c  = [r["color"] for r in snap_rows]
    map_xyT = np.array([[r["x"], r["y"]] for r in map_rows_T], float)
    map_c   = [r["color"] for r in map_rows_T]
    fig = plt.figure(figsize=(7,7)); ax = plt.gca()
    bycol = defaultdict(list)
    for p, c in zip(snap_xy, snap_c): bycol[c].append(p)
    for c, arr in bycol.items():
        arr = np.asarray(arr); ax.plot(arr[:,0], arr[:,1], **filled.get(c, filled["unknown"]))
    bycolT = defaultdict(list)
    for p, c in zip(map_xyT, map_c): bycolT[c].append(p)
    for c, arr in bycolT.items():
        arr = np.asarray(arr); ax.plot(arr[:,0], arr[:,1], **hollow.get(c, hollow["unknown"]))
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Known map (transformed) over snapshot (world)")
    # dedup legend
    h, l = ax.get_legend_handles_labels(); seen=set(); h2=[]; l2=[]
    for hi, li in zip(h, l):
        if li not in seen: h2.append(hi); l2.append(li); seen.add(li)
    ax.legend(h2, l2, loc="best")
    plt.tight_layout(); plt.savefig(plot_path, dpi=160)
    print(f"Saved overlay plot → {plot_path}")
    try: plt.show()
    except Exception: pass

# --------------------------- Main ----------------------------
def main():
    # Load
    known_rows = read_cones_csv(KNOWN_MAP_PATH)
    snap_rows  = read_cones_csv(SNAPSHOT_PATH)
    if not known_rows: raise SystemExit(f"No data in {KNOWN_MAP_PATH}")
    if not snap_rows:  raise SystemExit(f"No data in {SNAPSHOT_PATH}")

    # Step 1: Anchor on start box
    A4_map  = select_map_start(known_rows)
    B4_snap = select_snapshot_start(snap_rows, A4_map)
    c_map   = np.mean(A4_map, axis=0)
    c_snap  = np.mean(B4_snap, axis=0)

    # initial R via 4-point alignment (permute B to best RMS)
    best = None
    for perm in itertools.permutations(range(4)):
        B = B4_snap[list(perm)]
        R0, t0, rms0 = kabsch_se2(A4_map, B)
        # pin translation to start centroid
        t0 = pin_translation(R0, c_map, c_snap)
        if (best is None) or (rms0 < best[0]):
            best = (rms0, R0, t0)
    _, R, t = best

    # Step 2: Colored-only ICP (R,t updated; t re-pinned each iter)
    R, t = icp_colored(R, t, known_rows, snap_rows, A4_map, B4_snap)

    # Step 3: Unknown-only ICP (rotation-only; t re-pinned each iter)
    R, t = icp_unknown_rotation_only(R, t, known_rows, snap_rows, A4_map, B4_snap)

    # Step 4: π-flip disambiguation with start heading + colored counts (t stays pinned)
    h_map = start_heading_from_rect_map(A4_map)
    score_A = score_colored_handedness(R, t, h_map, c_map, snap_rows)

    # flip yaw by π, keep start centroid fixed
    R_flip = -R  # 2D rotation by π
    t_flip = pin_translation(R_flip, c_map, c_snap)
    score_B = score_colored_handedness(R_flip, t_flip, h_map, c_map, snap_rows)

    if score_B > score_A:
        R, t = R_flip, t_flip
        print("[info] π-flip applied based on start-zone color counts.")

    # Report + save
    yaw_deg = math.degrees(yaw_from_R(R))
    H = as_homogeneous(R, t)
    print("\n=== SE(2) alignment (map → snapshot/world) ===")
    print(f"Yaw: {yaw_deg:.3f} deg")
    print(f"Translation t: [{t[0]:.3f}, {t[1]:.3f}] m")
    with np.printoptions(precision=6, suppress=True):
        print("Homogeneous T =\n", H)

    with open(OUT_TRANSFORM_JSON, "w") as f:
        json.dump({
            "rotation": [[float(R[0,0]), float(R[0,1])],
                         [float(R[1,0]), float(R[1,1])]],
            "translation": [float(t[0]), float(t[1])],
            "yaw_deg": float(yaw_deg),
            "roi": {"y_min": ROI_Y_MIN, "y_max": ROI_Y_MAX, "x_abs_max": ROI_X_ABS_MAX},
            "icp": {
                "iters_colored": MAX_ITERS_COLORED,
                "iters_unknown": MAX_ITERS_UNKNOWN,
                "nn_radius_colored": NN_RADIUS_COLORED,
                "nn_radius_unknown": NN_RADIUS_UNKNOWN,
                "trim_fraction": TRIM_FRACTION
            }
        }, f, indent=2)
    print(f"Saved transform → {OUT_TRANSFORM_JSON}")

    # Overlay plot
    all_map_xy = np.array([[r["x"], r["y"]] for r in known_rows], float)
    all_map_cols = [r["color"] for r in known_rows]
    all_map_xy_T = all_map_xy @ R.T + t
    map_rows_T = [{"x": float(p[0]), "y": float(p[1]), "color": c} for p, c in zip(all_map_xy_T, all_map_cols)]
    plot_overlay(snap_rows, map_rows_T, PLOT_PATH)

if __name__ == "__main__":
    main()
