#!/usr/bin/env python3
# yaw_icp_obs_all.py
# Step-1 anchor kept. SIMPLE yaw-only ICP where:
#   • Translation is ALWAYS pinned to the start-box centroid.
#   • Correspondences: OBSERVED → nearest TRANSFORMED KNOWN (every observed must match).
#   • Rotation via Kabsch on chosen pairs (map-frame A -> world-frame B).
#   • Hard gate: ALL observed NN error ≤ RADIUS_REQ to declare success.
#   • If the loop STALLS (dθ < YAW_EPS_RAD but max_err > RADIUS_REQ), do a ONE-TIME π-FLIP
#     (yaw += π), re-pin t, and continue iterations.

import csv, json, math, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_MAP_PATH     = DATA_DIR / "known_map.csv"
SNAPSHOT_PATH      = DATA_DIR / "cone_snapshot.csv"
OUT_TRANSFORM_JSON = DATA_DIR / "alignment_transform.json"
PLOT_PATH          = DATA_DIR / "alignment_overlay.png"
ERR_CSV_PATH       = DATA_DIR / "obs_nn_errors.csv"

# ---------------- Config ----------------
# Start-box ROI (MAP frame)
ROI_Y_MIN, ROI_Y_MAX = -15.0, 0.0
ROI_X_ABS_MAX = 2.0

# ICP (yaw-only)
MAX_ITERS     = 20
RADIUS_REQ    = 0.80     # hard requirement for ALL observed at convergence
YAW_EPS_RAD   = 1e-3     # ~0.057°
RANGE_GATE_M  = 60.0     # generous gate around start (WORLD)

# ---------------- Columns & colors ----------------
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

# ---------------- IO ----------------
def read_cones_csv(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        clean = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    if not clean: return rows
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

# ---------------- Geometry ----------------
def yaw_from_R(R):
    return float(math.atan2(R[1,0], R[0,0]))  # rad

def R_from_yaw(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s],[s, c]], float)

def kabsch_se2(A, B):
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

def as_homogeneous(R, t):
    H = np.eye(3); H[:2,:2] = R; H[:2,2] = t
    return H

# ---------------- Start selection ----------------
def four_point_rect_features(P):
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

def select_map_start(known_rows):
    cands = [r for r in known_rows
             if r["color"]=="orange"
             and (ROI_Y_MIN <= r["y"] <= ROI_Y_MAX)
             and (abs(r["x"]) <= ROI_X_ABS_MAX)]
    if len(cands) != 4:
        raise SystemExit(f"Map ROI expected 4 small-orange, got {len(cands)}. Adjust ROI/map.")
    return np.array([[r["x"], r["y"]] for r in cands], float)

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
    return (c_snap - R @ c_map)

# ---------------- ICP (yaw-only; ALL observed must match) ----------------
def icp_yaw_obs_all(R_init, t_init, known_rows, snap_rows, A4_map, B4_snap):
    """
    Iterative:
      1) Transform ALL known-map points by current (R,t).
      2) For EACH observed y_j, pick nearest transformed map x_i'.
      3) Build A (map-frame points i) and B (observed y_j); Kabsch → R_new.
      4) Re-pin t to start centroid.
      5) Hard gate: ALL observed NN distances ≤ RADIUS_REQ to declare success.
      6) STALL rule: if dθ < YAW_EPS_RAD while max_err > RADIUS_REQ → ONE-TIME π-FLIP (yaw += π),
         re-pin t, continue.
    Returns: R, t, ok_flag, final_errors (per-observed NN distances over FULL snapshot)
    """
    R = np.array(R_init, float); t = np.array(t_init, float)
    c_map  = np.mean(A4_map, axis=0)
    c_snap = np.mean(B4_snap, axis=0)

    M_map = np.array([[r["x"], r["y"]] for r in known_rows], float)      # (Nm,2), MAP
    Y_obs_full = np.array([[r["x"], r["y"]] for r in snap_rows], float)  # (No,2), WORLD

    # gate observed for ICP (keep full for reporting)
    keep = np.linalg.norm(Y_obs_full - c_snap, axis=1) <= RANGE_GATE_M
    Y_obs = Y_obs_full[keep]
    if Y_obs.shape[0] == 0:
        raise SystemExit("No observed cones within range gate around start.")

    prev_yaw = yaw_from_R(R)
    flipped_once = False
    final_errors = None

    for it in range(MAX_ITERS):
        # map→world
        M_world = (M_map @ R.T) + t

        # OBS→NN(map')
        pairs_A = []
        pairs_B = []
        for y in Y_obs:
            ds = np.linalg.norm(M_world - y, axis=1)
            j = int(np.argmin(ds))
            pairs_A.append(M_map[j])  # MAP
            pairs_B.append(y)         # WORLD
        A = np.asarray(pairs_A, float)
        B = np.asarray(pairs_B, float)

        # Kabsch (rotation only), re-pin t
        R_new, _, _ = kabsch_se2(A, B)
        t_new = pin_translation(R_new, c_map, c_snap)

        # update
        R, t = R_new, t_new
        cur_yaw = yaw_from_R(R)
        dtheta = abs(cur_yaw - prev_yaw)

        # measure hard-gate metrics on gated set
        M_world = (M_map @ R.T) + t
        ds_all = np.linalg.norm(M_world[:,None,:] - Y_obs[None,:,:], axis=2).min(axis=0)
        max_dist = float(np.max(ds_all))
        med_dist = float(np.median(ds_all))
        p95_dist = float(np.percentile(ds_all, 95))

        print(f"[iter {it:02d}] yaw={math.degrees(cur_yaw):.3f}°, max={max_dist:.3f}m, med={med_dist:.3f}m, p95={p95_dist:.3f}m")

        final_errors = ds_all.tolist()
        # success
        if (max_dist <= RADIUS_REQ) and (dtheta < YAW_EPS_RAD):
            return R, t, True, final_errors

        # STALL → ONE-TIME π-FLIP and continue
        if (dtheta < YAW_EPS_RAD) and (max_dist > RADIUS_REQ) and (not flipped_once):
            print("[info] Stalled with high error. Applying ONE-TIME π flip and continuing...")
            R_pi = -np.eye(2)                  # 180° rotation
            R = R @ R_pi
            t = pin_translation(R, c_map, c_snap)
            prev_yaw = yaw_from_R(R)
            flipped_once = True
            continue

        prev_yaw = cur_yaw

    # final check after loop (report over FULL snapshot)
    M_world = (M_map @ R.T) + t
    ds_all_full = np.linalg.norm(M_world[:,None,:] - Y_obs_full[None,:,:], axis=2).min(axis=0)
    final_errors = ds_all_full.tolist()
    ok = (float(np.max(ds_all_full)) <= RADIUS_REQ) and (ds_all_full.shape[0] == Y_obs_full.shape[0])
    return R, t, ok, final_errors

# ---------------- Plot ----------------
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

# ---------------- Main ----------------
def main():
    known_rows = read_cones_csv(KNOWN_MAP_PATH)
    snap_rows  = read_cones_csv(SNAPSHOT_PATH)
    if not known_rows: raise SystemExit(f"No data in {KNOWN_MAP_PATH}")
    if not snap_rows:  raise SystemExit(f"No data in {SNAPSHOT_PATH}")

    # Step-1: anchor on start box
    A4_map  = select_map_start(known_rows)
    B4_snap = select_snapshot_start(snap_rows, A4_map)
    c_map   = np.mean(A4_map, axis=0)
    c_snap  = np.mean(B4_snap, axis=0)

    # initial R via best permutation
    best = None
    for perm in itertools.permutations(range(4)):
        R0, _, rms0 = kabsch_se2(A4_map, B4_snap[list(perm)])
        if (best is None) or (rms0 < best[0]):
            best = (rms0, R0)
    R = best[1]
    t = pin_translation(R, c_map, c_snap)

    print(f"[init] yaw={math.degrees(yaw_from_R(R)):.3f}°, t=({t[0]:.3f},{t[1]:.3f})")

    # ICP with stall-based π-flip
    R_fin, t_fin, ok, final_errors = icp_yaw_obs_all(R, t, known_rows, snap_rows, A4_map, B4_snap)
    yaw_deg = math.degrees(yaw_from_R(R_fin))

    # Final error stats (OVER FULL SNAPSHOT)
    M_map = np.array([[r["x"], r["y"]] for r in known_rows], float)
    M_world = (M_map @ R_fin.T) + t_fin
    Y_obs_full = np.array([[r["x"], r["y"]] for r in snap_rows], float)
    ds_all_full = np.linalg.norm(M_world[:,None,:] - Y_obs_full[None,:,:], axis=2).min(axis=0)

    N_obs = ds_all_full.shape[0]
    min_e = float(np.min(ds_all_full))
    med_e = float(np.median(ds_all_full))
    mean_e = float(np.mean(ds_all_full))
    p95_e = float(np.percentile(ds_all_full, 95))
    max_e = float(np.max(ds_all_full))
    within_req = int(np.sum(ds_all_full <= RADIUS_REQ))

    print("\n--- FINAL OBSERVED→MAP NN ERROR STATS ---")
    print(f"observed cones: {N_obs}")
    print(f"min: {min_e:.3f} m | median: {med_e:.3f} m | mean: {mean_e:.3f} m | p95: {p95_e:.3f} m | max: {max_e:.3f} m")
    print(f"≤ RADIUS_REQ ({RADIUS_REQ:.2f} m): {within_req}/{N_obs} cones")
    if not ok:
        print("[warn] Hard gate FAILED during ICP (even after possible flip). Inspect stats/overlay.")

    # Save per-observed errors
    with open(ERR_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["obs_index", "nn_error_m"])
        for i, e in enumerate(ds_all_full.tolist()): w.writerow([i, e])
    print(f"Saved per-observed errors → {ERR_CSV_PATH}")

    # Save transform
    H = as_homogeneous(R_fin, t_fin)
    with open(OUT_TRANSFORM_JSON, "w") as f:
        json.dump({
            "rotation": [[float(R_fin[0,0]), float(R_fin[0,1])],
                         [float(R_fin[1,0]), float(R_fin[1,1])]],
            "translation": [float(t_fin[0]), float(t_fin[1])],
            "yaw_deg": float(yaw_deg),
            "icp": {
                "max_iters": MAX_ITERS,
                "radius_req": RADIUS_REQ,
                "yaw_eps_rad": YAW_EPS_RAD,
                "range_gate_m": RANGE_GATE_M,
                "flip_on_stall": True
            },
            "hard_gate_passed": bool(ok),
            "final_error_stats": {
                "N_obs": N_obs, "min": min_e, "median": med_e,
                "mean": mean_e, "p95": p95_e, "max": max_e,
                "within_radius_req": within_req
            }
        }, f, indent=2)
    print(f"Saved transform → {OUT_TRANSFORM_JSON}")

    # Overlay
    all_map_xy = np.array([[r["x"], r["y"]] for r in known_rows], float)
    all_map_cols = [r["color"] for r in known_rows]
    all_map_xy_T = all_map_xy @ R_fin.T + t_fin
    map_rows_T = [{"x": float(p[0]), "y": float(p[1]), "color": c} for p, c in zip(all_map_xy_T, all_map_cols)]
    plot_overlay(snap_rows, map_rows_T, PLOT_PATH)

if __name__ == "__main__":
    main()
