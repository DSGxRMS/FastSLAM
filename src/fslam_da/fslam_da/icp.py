#!/usr/bin/env python3
# icp_da_node.py
#
# ICP-based DA over a known (fixed) cone map.
# - Build local candidate map set (crop + per-obs dynamic gate).
# - Estimate SE(2) transform (obs -> map) via point-to-point ICP (SVD/Kabsch).
# - Final one-to-one associations by nearest neighbor + Mahalanobis gate.
# - Same dynamic covariance inflation as your filter (alpha/beta, speed, yawrate).
# - No CSV/RViz; CLI prints only.
#
# Run:
#   ros2 run fslam_da icp
#
import math, json, time, bisect
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

def rot2d(th):
    c,s = math.cos(th), math.sin(th)
    return np.array([[c,-s],[s,c]], float)

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x, y, z, w):
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
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
        except Exception: continue
        rows.append((x,y,c))
    return rows

class ICPDANode(Node):
    def __init__(self):
        super().__init__("fslam_icp")

        # --- Parameters (align with your filter) ---
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)     # per-pair gate
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)

        # ICP-specific
        self.declare_parameter("icp_max_iter", 10)
        self.declare_parameter("icp_tol_xy", 1e-3)        # meters
        self.declare_parameter("icp_tol_yaw_deg", 0.05)   # degrees
        self.declare_parameter("icp_trim_frac", 0.25)     # robust: drop largest residuals each iter (0..0.5)
        self.declare_parameter("icp_nn_clip_m", 3.0)      # ignore pairs farther than this Euclid (for stability)

        p = self.get_parameter
        self.detections_frame = str(p("detections_frame").value).lower()
        self.map_crop_m = float(p("map_crop_m").value)
        self.chi2_gate = float(p("chi2_gate_2d").value)
        self.sigma0 = float(p("meas_sigma_floor_xy").value)
        self.alpha = float(p("alpha_trans").value)
        self.beta  = float(p("beta_rot").value)
        self.odom_buffer_sec = float(p("odom_buffer_sec").value)
        self.extrap_cap = float(p("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(p("target_cone_hz").value)
        self.max_pairs_print = int(p("max_pairs_print").value)

        self.icp_max_iter = int(p("icp_max_iter").value)
        self.icp_tol_xy = float(p("icp_tol_xy").value)
        self.icp_tol_yaw = math.radians(float(p("icp_tol_yaw_deg").value))
        self.icp_trim = float(p("icp_trim_frac").value)
        self.icp_nn_clip = float(p("icp_nn_clip_m").value)

        # Known map (aligned)
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
        self.MAP = (M @ R_map.T) + t_map

        # State
        self.odom_buf: deque = deque()
        self.pose_latest = np.array([0.0,0.0,0.0], float)

        q = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q)

        self._last_proc_wall = 0.0
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_t = None
        self.frame_idx = 0

        self.get_logger().info(f"ICP up. Map cones: {self.MAP.shape[0]}")

    # ------------ ROS ------------
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

    # ------------ ICP core ------------
    @staticmethod
    def _rigid_solve_2d(A: np.ndarray, B: np.ndarray):
        """Solve R,t minimizing ||R A + t - B|| (point-to-point).
        A,B: Nx2"""
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
        # same inflation as your gate filter
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    def cb_cones(self, msg: ConeArrayWithCovariance):
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build obs (WORLD) and base-range for inflation
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

        # Build candidate map set: crop per-observation, then union
        MAP = self.MAP
        crop2 = self.map_crop_m**2
        union_ids = set()
        for pw in obs_pw:
            diffs = MAP - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            idx = np.where(d2 <= crop2)[0]
            union_ids.update(int(i) for i in idx.tolist())
        if not union_ids:
            # no local map — nothing to do
            self._print_line(t_z, dt_pose, v, n_obs, 0, 0, 0, 0.0, [])
            self.frame_idx += 1
            return
        C = MAP[sorted(list(union_ids)), :]            # Mx2
        cand_ids = np.array(sorted(list(union_ids)))   # map IDs aligned to C rows

        # --- ICP iterations (obs -> map) ---
        R_icp = np.eye(2); t_icp = np.zeros(2)
        iters = 0
        rmse = 0.0
        Pw_cur = obs_pw.copy()

        for it in range(self.icp_max_iter):
            iters = it+1
            # NN search (brute-force is fine at these sizes)
            # distances and indices
            d2 = np.sum((Pw_cur[:,None,:] - C[None,:,:])**2, axis=2)   # [N,M]
            nn_idx = np.argmin(d2, axis=1)
            nn_d = np.sqrt(d2[np.arange(n_obs), nn_idx])

            # optional robust trimming
            keep = np.ones(n_obs, dtype=bool)
            if self.icp_trim > 0.0 and n_obs >= 4:
                k = int((1.0 - self.icp_trim) * n_obs)
                k = max(3, min(n_obs, k))
                order = np.argsort(nn_d)
                keep[:] = False
                keep[order[:k]] = True

            # clip very far pairs (stability)
            if self.icp_nn_clip > 0.0:
                keep &= (nn_d <= self.icp_nn_clip)

            A = Pw_cur[keep]
            B = C[nn_idx[keep]]
            if A.shape[0] < 3:
                break

            R_delta, t_delta = self._rigid_solve_2d(A, B)

            # update transform
            R_icp = R_delta @ R_icp
            t_icp = R_delta @ t_icp + t_delta

            # apply
            Pw_cur = (obs_pw @ R_icp.T) + t_icp

            # check convergence
            dx, dy = t_delta
            dyaw = math.atan2(R_delta[1,0], R_delta[0,0])
            if (math.hypot(dx,dy) < self.icp_tol_xy) and (abs(dyaw) < self.icp_tol_yaw):
                break

        # final residuals
        d2_final = np.sum((Pw_cur[:,None,:] - C[None,:,:])**2, axis=2)
        nn_idx_final = np.argmin(d2_final, axis=1)
        nn_d_final = np.sqrt(d2_final[np.arange(n_obs), nn_idx_final])
        if nn_d_final.size:
            rmse = float(np.sqrt(np.mean(np.minimum(nn_d_final**2, (self.icp_nn_clip**2)))))

        # --- Build Mahalanobis-gated, one-to-one matches (greedy, smallest m² first) ---
        # compute per-obs inflated S_eff with runtime kinematics
        S_invs = []
        for i in range(n_obs):
            r_obs = float(np.linalg.norm(obs_pb[i]))
            S_eff = self._inflate_cov(obs_S0[i], v, yawrate, dt_pose, r_obs, yaw)
            try: S_inv = np.linalg.inv(S_eff)
            except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S_eff)
            S_invs.append(S_inv)
        S_invs = np.stack(S_invs)

        # candidate pairs (obs i -> map col j) with m² and apply gate
        pairs=[]
        for i in range(n_obs):
            j = int(nn_idx_final[i])
            innov = Pw_cur[i] - C[j]
            m2 = float(innov.T @ S_invs[i] @ innov)
            if m2 <= self.chi2_gate:
                pairs.append((i, j, m2))

        # enforce uniqueness of map columns greedily
        pairs.sort(key=lambda t: t[2])
        used_cols = set()
        matches=[]
        for (i,j,m2) in pairs:
            if j in used_cols: continue
            used_cols.add(j)
            matches.append((i, int(cand_ids[j]), m2))  # map id in global indexing

        # pretty print
        show=[]
        for (oi, mid, m2) in matches[:min(self.max_pairs_print, len(matches))]:
            show.append(f"(obs{oi}->map{mid}, √m²={math.sqrt(max(0.0,m2)):.2f})")

        proc_ms = int((time.perf_counter()-t0)*1000)
        self._print_line(t_z, dt_pose, v, n_obs, len(matches), iters, rmse, proc_ms, show)
        self.frame_idx += 1

    def _print_line(self, t_z, dt_pose, v, n_obs, n_match, iters, rmse, proc_ms, show_pairs):
        pass_dt = 0.0 if self._last_t is None else (t_z - self._last_t)
        self._last_t = t_z
        base = (f"[icp] pass_dt={int(pass_dt*1000)}ms  Δt_pose={int(dt_pose*1000)}ms  v={v:.2f}m/s  "
                f"obs={n_obs}  matched={n_match}  iters={iters}  rmse={rmse:.3f}m  proc={proc_ms}ms")
        if show_pairs:
            base += " | " + ", ".join(show_pairs)
        print(base)

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
