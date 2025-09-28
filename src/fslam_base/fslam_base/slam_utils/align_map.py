#!/usr/bin/env python3
# align_map_with_flips_and_colorcheck.py
# Step-1 anchor (translation pin), yaw-only ICP (obs→NN map), iterative π-flip trials,
# then an independent color-overlap check with optional y-axis flip (map pre-flip).

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
RADIUS_REQ    = 0.80     # ALL observed must be ≤ this to declare success
YAW_EPS_RAD   = 1e-3     # stall threshold (~0.057°)
RANGE_GATE_M  = 60.0     # gate around start (WORLD)

# π-flip trials (toggle basin repeatedly; keep best)
N_FLIP_TRIALS = 3        # try initial + 2 flips = 3 basins; increase if you want

# Color overlap check (independent)
COLOR_MATCH_RADIUS = 1.0     # meters for color NN check
COLOR_MIN_FRAC     = 0.60    # require at least 60% colored obs to match right color
# Side rule is not enforced here; we only check color identity via NN.

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
    Returns: R, t, ok_flag, ds_all_full (NN errors for FULL snapshot)
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
        ds_gate = np.linalg.norm(M_world[:,None,:] - Y_obs[None,:,:], axis=2).min(axis=0)
        max_dist = float(np.max(ds_gate))
        med_dist = float(np.median(ds_gate))
        p95_dist = float(np.percentile(ds_gate, 95))
        print(f"[iter {it:02d}] yaw={math.degrees(cur_yaw):.3f}°, max={max_dist:.3f}m, med={med_dist:.3f}m, p95={p95_dist:.3f}m")

        # success?
        if (max_dist <= RADIUS_REQ) and (dtheta < YAW_EPS_RAD):
            break

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

    # final errors over FULL snapshot for reporting / ranking
    M_world = (M_map @ R.T) + t
    ds_all_full = np.linalg.norm(M_world[:,None,:] - Y_obs_full[None,:,:], axis=2).min(axis=0)
    ok = (float(np.max(ds_all_full)) <= RADIUS_REQ) and (ds_all_full.shape[0] == Y_obs_full.shape[0])
    return R, t, ok, ds_all_full

# ---------------- Trials manager (iterative π flips; pick best) ----------------
def run_flip_trials(R0, t0, known_rows, snap_rows, A4_map, B4_snap):
    c_map  = np.mean(A4_map, axis=0)
    c_snap = np.mean(B4_snap, axis=0)

    trials = []
    R_seed = np.array(R0); t_seed = np.array(t0)

    for k in range(N_FLIP_TRIALS):
        print(f"\n=== Flip trial {k+1}/{N_FLIP_TRIALS} ===")
        Rk, tk, ok, ds = icp_yaw_obs_all(R_seed, t_seed, known_rows, snap_rows, A4_map, B4_snap)
        max_e = float(np.max(ds)); med_e = float(np.median(ds)); mean_e = float(np.mean(ds))
        p95_e = float(np.percentile(ds, 95))
        trials.append({
            "R": Rk, "t": tk, "ok": bool(ok),
            "max": max_e, "med": med_e, "mean": mean_e, "p95": p95_e, "ds": ds
        })
        # toggle basin for next trial
        R_seed = R_seed @ (-np.eye(2))
        t_seed = pin_translation(R_seed, c_map, c_snap)

    # pick best by minimal max error (then p95 as tie-break)
    trials.sort(key=lambda d: (d["max"], d["p95"], d["mean"]))
    best = trials[0]
    print(f"\n=== Best of {N_FLIP_TRIALS} trials ===")
    print(f"max={best['max']:.3f} m, p95={best['p95']:.3f} m, med={best['med']:.3f} m, mean={best['mean']:.3f} m, ok={best['ok']}")
    return best

# ---------------- Color overlap check (+ optional y-axis map flip) ----------------
def color_overlap_fraction(R, t, known_rows, snap_rows, radius=COLOR_MATCH_RADIUS):
    # transform known map
    M = np.array([[r["x"], r["y"]] for r in known_rows], float)
    Ck = [r["color"] for r in known_rows]
    M_w = (M @ R.T) + t

    obs_colored = [r for r in snap_rows if r["color"] in ("blue","yellow","orange","orange_large")]
    if not obs_colored:
        return 1.0, 0  # nothing to test

    Y  = np.array([[r["x"], r["y"]] for r in obs_colored], float)
    Co = [r["color"] for r in obs_colored]

    ok = 0
    for j, y in enumerate(Y):
        ds = np.linalg.norm(M_w - y, axis=1)
        i = int(np.argmin(ds))
        if ds[i] <= radius and Ck[i] == Co[j]:
            ok += 1
    return ok / len(obs_colored), len(obs_colored)

def try_y_axis_flip_if_color_bad(best, A4_map, B4_snap, known_rows, snap_rows):
    """
    If color overlap is below threshold, pre-flip map about y-axis:
      This is equivalent to R' = R @ Fy, t' = pin_translation(R', c_map, c_snap),
      where Fy = diag([-1, 1]) applied to MAP points prior to rotation.
    Accept the flip only if color overlap improves AND geometric max error doesn't explode.
    """
    frac0, Ncol = color_overlap_fraction(best["R"], best["t"], known_rows, snap_rows)
    print(f"\n[Color check] colored-match frac = {frac0:.2f} ({Ncol} colored obs), threshold={COLOR_MIN_FRAC:.2f}")

    if Ncol == 0 or frac0 >= COLOR_MIN_FRAC:
        return best, {"color_frac": frac0, "used_y_axis_flip": False}

    # construct y-axis flip for map
    Fy = np.diag([-1.0, 1.0])
    c_map  = np.mean(np.array([[r["x"], r["y"]] for r in known_rows if r["color"]=="orange"
                                and (ROI_Y_MIN <= r["y"] <= ROI_Y_MAX)
                                and (abs(r["x"]) <= ROI_X_ABS_MAX)], float), axis=0)
    c_snap = np.mean(B4_snap, axis=0)

    R_alt = best["R"] @ Fy
    t_alt = pin_translation(R_alt, c_map, c_snap)

    # geometric errors for alt
    M_map = np.array([[r["x"], r["y"]] for r in known_rows], float)
    Y_obs = np.array([[r["x"], r["y"]] for r in snap_rows], float)
    M_world_alt = (M_map @ R_alt.T) + t_alt
    ds_alt = np.linalg.norm(M_world_alt[:,None,:] - Y_obs[None,:,:], axis=2).min(axis=0)
    max_alt = float(np.max(ds_alt)); p95_alt = float(np.percentile(ds_alt,95)); mean_alt=float(np.mean(ds_alt)); med_alt=float(np.median(ds_alt))

    frac1, _ = color_overlap_fraction(R_alt, t_alt, known_rows, snap_rows)
    print(f"[Color check] alt (y-axis flip) colored-match frac = {frac1:.2f}; geom max={max_alt:.3f}m p95={p95_alt:.3f}m")

    # Decide: require color improves and geometry not worse by > 10%
    if (frac1 > frac0) and (max_alt <= 1.10 * best["max"]):
        print("[Color check] Accepting y-axis flip (better color, geometry OK).")
        return {
            "R": R_alt, "t": t_alt, "ok": best["ok"],
            "max": max_alt, "p95": p95_alt, "mean": mean_alt, "med": med_alt, "ds": ds_alt
        }, {"color_frac": frac1, "used_y_axis_flip": True}

    print("[Color check] Keeping original (flip not beneficial).")
    return best, {"color_frac": frac0, "used_y_axis_flip": False}

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
    best_perm = None
    best_rms  = 1e18
    for perm in itertools.permutations(range(4)):
        R0, _, rms0 = kabsch_se2(A4_map, B4_snap[list(perm)])
        if rms0 < best_rms:
            best_rms, best_perm, R_init = rms0, perm, R0
    R = R_init
    t = pin_translation(R, c_map, c_snap)
    print(f"[init] yaw={math.degrees(yaw_from_R(R)):.3f}°, t=({t[0]:.3f},{t[1]:.3f}), rms(4-start)={best_rms:.3f}m")

    # π-flip trials → pick best
    best = run_flip_trials(R, t, known_rows, snap_rows, A4_map, B4_snap)

    # Color overlap check with optional y-axis flip (map pre-flip)
    best2, color_meta = try_y_axis_flip_if_color_bad(best, A4_map, B4_snap, known_rows, snap_rows)

    # Final reporting (FULL snapshot)
    R_fin, t_fin = best2["R"], best2["t"]
    yaw_deg = math.degrees(yaw_from_R(R_fin))

    # Errors over FULL snapshot
    M_map = np.array([[r["x"], r["y"]] for r in known_rows], float)
    Y_obs = np.array([[r["x"], r["y"]] for r in snap_rows], float)
    M_world = (M_map @ R_fin.T) + t_fin
    ds_all_full = np.linalg.norm(M_world[:,None,:] - Y_obs[None,:,:], axis=2).min(axis=0)

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
    print(f"Final yaw: {yaw_deg:.3f}°  |  used_y_axis_flip: {color_meta['used_y_axis_flip']}  |  color_match_frac: {color_meta['color_frac']:.2f}")

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
                "flip_trials": N_FLIP_TRIALS
            },
            "final_error_stats": {
                "N_obs": N_obs, "min": min_e, "median": med_e,
                "mean": mean_e, "p95": p95_e, "max": max_e,
                "within_radius_req": within_req
            },
            "color_check": {
                "used_y_axis_flip": color_meta["used_y_axis_flip"],
                "match_radius": COLOR_MATCH_RADIUS,
                "min_frac_required": COLOR_MIN_FRAC,
                "final_match_frac": color_meta["color_frac"]
            }
        }, f, indent=2)
    print(f"Saved transform → {OUT_TRANSFORM_JSON}")

    # Overlay
    all_map_cols = [r["color"] for r in known_rows]
    all_map_xy_T = M_world
    map_rows_T = [{"x": float(p[0]), "y": float(p[1]), "color": c} for p, c in zip(all_map_xy_T, all_map_cols)]
    plot_overlay(snap_rows, map_rows_T, PLOT_PATH)

if __name__ == "__main__":
    main()
