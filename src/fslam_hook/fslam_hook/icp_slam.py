#!/usr/bin/env python3
# icp_da_node_fast.py
#
# ICP-based DA over a known (fixed) cone map — optimized
# Hooks added:
#  - SensorData QoS (BEST_EFFORT) to prevent backlog
#  - Throttled logs (log_period_s)
#  - Uniform-grid spatial hash:
#      * fast union-of-candidates (map crop) per frame
#      * fast per-iteration local NN (instead of O(N*M))
#  - Mahalanobis via Cholesky "solve" (no explicit inverse)
#
# Accuracy unchanged: same inflation (alpha/beta), same per-pair chi2 gate,
# same greedy uniqueness for the final associations.

import math, json, time, bisect
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

# ---------- helpers ----------
def rot2d(th: float) -> np.ndarray:
    c,s = math.cos(th), math.sin(th)
    return np.array([[c,-s],[s,c]], float)

def wrap(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def load_known_map(csv_path: Path):
    import csv as _csv
    rows=[]
    with open(csv_path, "r", newline="") as f:
        clean = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    rdr = _csv.DictReader(clean)
    lk = [k.lower() for k in rdr.fieldnames]
    def pick(*cands):
        for c in cands:
            if c in lk: return c
        return None
    xk=pick("x","x_m","xmeter"); yk=pick("y","y_m","ymeter"); ck=pick("color","colour","tag")
    for r in rdr:
        try:
            x=float(r[xk]); y=float(r[yk]); c=str(r[ck]).strip().lower()
        except Exception:
            continue
        rows.append((x,y,c))
    return rows

# ---------- ICP node ----------
class ICPDANode(Node):
    def __init__(self):
        super().__init__("fslam_icp")

        # Core params
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)
        # ICP params
        self.declare_parameter("icp_max_iter", 10)
        self.declare_parameter("icp_tol_xy", 1e-3)
        self.declare_parameter("icp_tol_yaw_deg", 0.05)
        self.declare_parameter("icp_trim_frac", 0.25)
        self.declare_parameter("icp_nn_clip_m", 3.0)
        # Hooks
        self.declare_parameter("log_period_s", 0.5)
        self.declare_parameter("grid_cell_m", 6.0)         # spatial hash cell size
        self.declare_parameter("nn_search_r_m", 4.0)       # per-iter local NN search radius

        p = self.get_parameter
        self.detections_frame = str(p("detections_frame").value).lower()
        self.map_crop_m  = float(p("map_crop_m").value)
        self.chi2_gate   = float(p("chi2_gate_2d").value)
        self.sigma0      = float(p("meas_sigma_floor_xy").value)
        self.alpha       = float(p("alpha_trans").value)
        self.beta        = float(p("beta_rot").value)
        self.odom_buffer_sec = float(p("odom_buffer_sec").value)
        self.extrap_cap  = float(p("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(p("target_cone_hz").value)
        self.max_pairs_print = int(p("max_pairs_print").value)

        self.icp_max_iter = int(p("icp_max_iter").value)
        self.icp_tol_xy   = float(p("icp_tol_xy").value)
        self.icp_tol_yaw  = math.radians(float(p("icp_tol_yaw_deg").value))
        self.icp_trim     = float(p("icp_trim_frac").value)
        self.icp_nn_clip  = float(p("icp_nn_clip_m").value)

        self.log_period   = float(p("log_period_s").value)
        self.grid_cell_m  = float(p("grid_cell_m").value)
        self.nn_search_r  = float(p("nn_search_r_m").value)

        # Map load + alignment
        data_dir = (Path(__file__).resolve().parents[0] / str(p("map_data_dir").value)).resolve()
        align_json = data_dir / "alignment_transform.json"
        known_map_csv = data_dir / "known_map.csv"
        if not align_json.exists() or not known_map_csv.exists():
            raise RuntimeError(f"Missing map assets under {data_dir}")
        with open(align_json, "r") as f:
            T = json.load(f)
        R_map = np.array(T["rotation"], float)
        t_map = np.array(T["translation"], float)
        rows = load_known_map(known_map_csv)
        M = np.array([[x,y] for (x,y,_) in rows], float)
        self.MAP = (M @ R_map.T) + t_map    # [Nmap,2]

        # Build spatial hash for the global map once
        self._build_grid_index(self.MAP)

        # State
        self.odom_buf: deque = deque()  # (t,x,y,yaw,v,yawrate)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_proc_wall = 0.0
        self._last_t = None
        self._last_log_wall = 0.0

        # QoS — SensorData style (BEST_EFFORT)
        q_cones = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)
        q_odom  = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)

        # GT topics (adjust if needed)
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q_odom)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q_cones)

        self.get_logger().info(f"ICP up. Map cones: {self.MAP.shape[0]} (SensorData QoS, grid accel)")

    # ---------- spatial hash over fixed MAP ----------
    def _build_grid_index(self, pts: np.ndarray):
        self._xmin = float(np.min(pts[:,0]))
        self._ymin = float(np.min(pts[:,1]))
        self._cell = max(1e-3, self.grid_cell_m)
        buckets = defaultdict(list)
        ix = np.floor((pts[:,0] - self._xmin)/self._cell).astype(int)
        iy = np.floor((pts[:,1] - self._ymin)/self._cell).astype(int)
        for i, (gx,gy) in enumerate(zip(ix,iy)):
            buckets[(int(gx), int(gy))].append(i)
        self._grid = {k: np.array(v, dtype=int) for k,v in buckets.items()}

    def _grid_query_ball_ids(self, x: float, y: float, r: float) -> np.ndarray:
        cs = self._cell
        gx = int(math.floor((x - self._xmin)/cs))
        gy = int(math.floor((y - self._ymin)/cs))
        rad_cells = int(math.ceil(r / cs)) + 1
        cand_lists = []
        for dx in range(-rad_cells, rad_cells+1):
            for dy in range(-rad_cells, rad_cells+1):
                key = (gx+dx, gy+dy)
                arr = self._grid.get(key, None)
                if arr is not None and arr.size:
                    cand_lists.append(arr)
        if not cand_lists:
            return np.empty((0,), dtype=int)
        return np.unique(np.concatenate(cand_lists))

    # ---------- odom ----------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        v, yr = 0.0, 0.0
        if self.odom_buf:
            t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[-1]
            dt = max(1e-3, t - t0)
            v = math.hypot(x-x0, y-y0) / dt
            yr = wrap(yaw - yaw0) / dt
        self.pose_latest = np.array([x,y,yaw], float)
        self.odom_buf.append((t,x,y,yaw,v,yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

    def _pose_at(self, t_query: float):
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf'), 0.0, 0.0
        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)
        if idx == 0:
            t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[0]
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        if idx >= len(self.odom_buf):
            t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[-1]
            dt = t_query - t1
            dt_c = max(0.0, min(dt, self.extrap_cap))
            if abs(yr1) < 1e-3:
                x = x1 + v1*dt_c*math.cos(yaw1)
                y = y1 + v1*dt_c*math.sin(yaw1)
                yaw = wrap(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt_c) - math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt_c) - math.cos(yaw1))
                yaw = wrap(yaw1 + yr1*dt_c)
            return np.array([x,y,yaw], float), abs(dt), v1, yr1
        t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[idx-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[idx]
        if t1==t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a*dyaw)
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    # ---------- rigid solver ----------
    @staticmethod
    def _rigid_solve_2d(A: np.ndarray, B: np.ndarray):
        if A.shape[0] == 0:
            return np.eye(2), np.zeros(2)
        ca = A.mean(axis=0); cb = B.mean(axis=0)
        A0 = A - ca; B0 = B - cb
        H = A0.T @ B0
        U,S,Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1,:] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        return R, t

    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    # ---------- ICP + DA ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        # throttle to ~target_cone_hz
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build obs (WORLD) and base-range
        obs_pw=[]; obs_pb=[]; obs_S0=[]
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma0**2, self.sigma0**2])
                if self.detections_frame == "base":
                    pb = np.array([px,py], float)
                    pw = np.array([x,y]) + Rwb @ pb
                    S0 = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y]))
                    S0 = Pw
                obs_pw.append(pw); obs_pb.append(pb); obs_S0.append(S0)
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)

        n_obs = len(obs_pw)
        if n_obs == 0:
            return

        obs_pw = np.vstack(obs_pw)        # Nx2
        obs_pb = np.vstack(obs_pb)        # Nx2
        obs_S0 = np.stack(obs_S0)         # Nx2x2

        # --- Fast union-of-candidates from spatial hash ---
        crop_r = self.map_crop_m
        union_ids = set()
        for pw in obs_pw:
            idxs = self._grid_query_ball_ids(pw[0], pw[1], crop_r)
            if idxs.size:
                # refine: keep within radius
                diffs = self.MAP[idxs] - pw
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                keep = idxs[d2 <= (crop_r*crop_r)]
                union_ids.update(int(i) for i in keep.tolist())

        if not union_ids:
            self._print_line(t_z, dt_pose, v, n_obs, 0, 0, 0.0, int((time.perf_counter()-t0)*1000), [])
            return

        cand_ids = np.array(sorted(list(union_ids)), dtype=int)   # global map indices
        C = self.MAP[cand_ids]                                    # Mx2
        # index: global map id -> row in C
        idx_in_C = {int(mid): i for i, mid in enumerate(cand_ids)}

        # Build a grid for C (subset) by reusing the global grid areas implicitly:
        # we'll query global grid and then filter to subset via idx_in_C.

        # --- ICP iterations (obs -> map), with LOCAL NN via grid ---
        R_icp = np.eye(2); t_icp = np.zeros(2)
        Pw_cur = obs_pw.copy()
        iters = 0
        for it in range(self.icp_max_iter):
            iters = it + 1

            # For each transformed obs point, fetch nearby candidate map points from grid
            # within nn_search_r (≥ icp_nn_clip). Then choose nearest among those (not all M).
            nn_idx = np.empty(Pw_cur.shape[0], dtype=int)
            nn_d   = np.empty(Pw_cur.shape[0], dtype=float)

            r_nn = max(self.nn_search_r, self.icp_nn_clip)
            for i, p in enumerate(Pw_cur):
                idxs = self._grid_query_ball_ids(p[0], p[1], r_nn)
                if idxs.size == 0:
                    nn_idx[i] = 0
                    nn_d[i] = 1e9
                    continue
                # keep only indices that are members of C
                # map global->row in C
                rows = [idx_in_C.get(int(mid), -1) for mid in idxs]
                rows = np.array([r for r in rows if r >= 0], dtype=int)
                if rows.size == 0:
                    nn_idx[i] = 0
                    nn_d[i] = 1e9
                    continue
                cand = C[rows]                # kx2
                d2 = np.sum((cand - p)**2, axis=1)
                jloc = int(np.argmin(d2))
                nn_idx[i] = rows[jloc]        # row in C
                nn_d[i] = math.sqrt(float(d2[jloc]))

            keep = np.ones(n_obs, dtype=bool)
            if self.icp_trim > 0.0 and n_obs >= 4:
                k = int((1.0 - self.icp_trim) * n_obs)
                k = max(3, min(n_obs, k))
                order = np.argsort(nn_d)
                keep[:] = False
                keep[order[:k]] = True

            if self.icp_nn_clip > 0.0:
                keep &= (nn_d <= self.icp_nn_clip)

            if keep.sum() < 3:
                break

            A = Pw_cur[keep]
            B = C[nn_idx[keep]]
            R_d, t_d = self._rigid_solve_2d(A, B)

            R_icp = R_d @ R_icp
            t_icp = R_d @ t_icp + t_d
            Pw_cur = (obs_pw @ R_icp.T) + t_icp

            dx, dy = t_d
            dyaw = math.atan2(R_d[1,0], R_d[0,0])
            if (math.hypot(dx,dy) < self.icp_tol_xy) and (abs(dyaw) < self.icp_tol_yaw):
                break

        # Final local NN (one more time) to compute RMSE and candidates
        r_nn = max(self.nn_search_r, self.icp_nn_clip)
        d_list=[]; nn_idx_final = np.empty(n_obs, dtype=int)
        for i, p in enumerate(Pw_cur):
            idxs = self._grid_query_ball_ids(p[0], p[1], r_nn)
            if idxs.size == 0:
                nn_idx_final[i] = 0
                d_list.append(1e9)
                continue
            rows = [idx_in_C.get(int(mid), -1) for mid in idxs]
            rows = np.array([r for r in rows if r >= 0], dtype=int)
            if rows.size == 0:
                nn_idx_final[i] = 0
                d_list.append(1e9)
                continue
            cand = C[rows]
            d2 = np.sum((cand - p)**2, axis=1)
            jloc = int(np.argmin(d2))
            nn_idx_final[i] = rows[jloc]
            d_list.append(math.sqrt(float(d2[jloc])))

        nn_d_final = np.array(d_list, float)
        if nn_d_final.size:
            rmse = float(np.sqrt(np.mean(np.minimum(nn_d_final**2, (self.icp_nn_clip**2)))))
        else:
            rmse = 0.0

        # --- Per-obs inflate + Mahalanobis gate (Cholesky solve) + one-to-one greedy ---
        matches = []
        for i in range(n_obs):
            j = int(nn_idx_final[i])
            if j < 0 or j >= C.shape[0]:
                continue
            # inflate
            r_obs = float(np.linalg.norm(obs_pb[i]))
            S_eff = self._inflate_cov(obs_S0[i], v, yawrate, dt_pose, r_obs, yaw)
            # m² via Cholesky
            innov = Pw_cur[i] - C[j]
            try:
                L = np.linalg.cholesky(S_eff)
                y = np.linalg.solve(L, innov)   # L y = innov
                m2 = float(y @ y)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S_eff)
                m2 = float(innov @ Sinv @ innov)
            if m2 <= self.chi2_gate:
                matches.append((i, int(cand_ids[j]), m2))

        # uniqueness: greedily by smallest m²
        matches.sort(key=lambda t: t[2])
        used_cols = set()
        final=[]
        for (oi, mid, m2) in matches:
            if mid in used_cols: continue
            used_cols.add(mid)
            final.append((oi, mid, m2))

        show=[]
        for (oi, mid, m2) in final[:min(self.max_pairs_print, len(final))]:
            show.append(f"(obs{oi}->map{mid}, √m²={math.sqrt(max(0.0,m2)):.2f})")

        proc_ms = int((time.perf_counter()-t0)*1000)
        self._print_line(t_z, dt_pose, v, n_obs, len(final), iters, rmse, proc_ms, show)

    # ---------- logging ----------
    def _print_line(self, t_z, dt_pose, v, n_obs, n_match, iters, rmse, proc_ms, show_pairs):
        now = time.time()
        if (now - self._last_log_wall) < self.log_period:
            return
        self._last_log_wall = now
        pass_dt = 0.0 if self._last_t is None else (t_z - self._last_t)
        self._last_t = t_z
        base = (f"[icp] pass_dt={int(pass_dt*1000)}ms  Δt_pose={int(dt_pose*1000)}ms  v={v:.2f}m/s  "
                f"obs={n_obs}  matched={n_match}  iters={iters}  rmse={rmse:.3f}m  proc={proc_ms}ms")
        if show_pairs:
            base += " | " + ", ".join(show_pairs)
        print(base)

# ---------- main ----------
def main():
    rclpy.init()
    node = ICPDANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
