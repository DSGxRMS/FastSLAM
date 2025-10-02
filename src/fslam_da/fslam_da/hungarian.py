#!/usr/bin/env python3
# hungarian_da_node.py
#
# Hungarian DA with dynamic gating:
# - Cost = Mahalanobis^2
# - Only map points that passed the gate for ≥1 obs are in the matrix
# - 1+ dummy columns let obs remain unmatched
# - No CSVs/RViz; prints only.
#
# Run:
#   ros2 run fslam_da hungarian
#
import math, json, time, bisect
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

BASE_DIR = Path(__file__).resolve().parents[0]

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

# --- Minimal Hungarian for square matrices ---
def hungarian_min_cost(cost: np.ndarray):
    cost = cost.copy()
    N = cost.shape[0]
    cost -= cost.min(axis=1, keepdims=True)
    cost -= cost.min(axis=0, keepdims=True)

    mask = np.zeros_like(cost, dtype=np.int8)  # 0 free, 1 star, 2 prime
    row_cov = np.zeros(N, dtype=bool)
    col_cov = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if cost[i,j]==0 and (not row_cov[i]) and (not col_cov[j]):
                mask[i,j]=1; row_cov[i]=True; col_cov[j]=True; break
    row_cov[:]=False; col_cov[:]=False

    def cover_cols_with_stars():
        col_cov[:] = False
        for j in range(N):
            if np.any(mask[:,j]==1): col_cov[j]=True
    cover_cols_with_stars()

    def find_uncovered_zero():
        for i in range(N):
            if row_cov[i]: continue
            for j in range(N):
                if (not col_cov[j]) and cost[i,j]==0:
                    return i,j
        return None,None

    def find_star_in_row(r):
        js = np.where(mask[r]==1)[0]
        return js[0] if js.size else None
    def find_star_in_col(c):
        is_ = np.where(mask[:,c]==1)[0]
        return is_[0] if is_.size else None
    def find_prime_in_row(r):
        js = np.where(mask[r]==2)[0]
        return js[0] if js.size else None

    while True:
        if col_cov.sum() == N: break
        while True:
            r,c = find_uncovered_zero()
            if r is None:
                # adjust
                m = cost[~row_cov][:, ~col_cov].min()
                cost[~row_cov,:] -= m
                cost[:, col_cov] += m
            else:
                mask[r,c]=2
                c_star = find_star_in_row(r)
                if c_star is None:
                    # augment
                    path=[(r,c)]
                    while True:
                        r_star = find_star_in_col(path[-1][1])
                        if r_star is None: break
                        path.append((r_star, path[-1][1]))
                        c_prime = find_prime_in_row(r_star)
                        path.append((r_star, c_prime))
                    for (ri,ci) in path:
                        mask[ri,ci] = 1 if mask[ri,ci]==2 else 0
                    row_cov[:]=False; col_cov[:]=False
                    mask[mask==2]=0
                    cover_cols_with_stars()
                    break
                else:
                    row_cov[r]=True; col_cov[c_star]=False

    row_assign = -np.ones(N, dtype=int)
    col_assign = -np.ones(N, dtype=int)
    for i in range(N):
        js = np.where(mask[i]==1)[0]
        if js.size:
            row_assign[i]=js[0]; col_assign[js[0]]=i
    return row_assign, col_assign

class HungarianDANode(Node):
    def __init__(self):
        super().__init__("fslam_hungarian")

        # params
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("min_candidates", 0)
        self.declare_parameter("unmatched_cost", 25.0)  # m²
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)

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
        self.min_candidates = int(p("min_candidates").value)
        self.unmatched_cost = float(p("unmatched_cost").value)
        self.max_pairs_print = int(p("max_pairs_print").value)

        # map
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

        # state
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

        self.get_logger().info(f"Hungarian up. Map cones: {self.MAP.shape[0]}")

    # --- ROS ---
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

    def _pose_at(self, t_query: float) -> Tuple[np.ndarray, float, float, float]:
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

        # Build obs in WORLD
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
        if n_obs==0:
            return

        # Candidates
        crop2 = self.map_crop_m**2
        MAP = self.MAP
        cands: List[List[Tuple[int,float]]] = []
        total_cands=0
        for pw, S0, pb in zip(obs_pw, obs_S0, obs_pb):
            r = float(np.linalg.norm(pb))
            S_eff = self._inflate_cov(S0, v, yawrate, dt_pose, r, yaw)
            try: S_inv = np.linalg.inv(S_eff)
            except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S_eff)
            diffs = MAP - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            near = np.where(d2 <= crop2)[0]
            cc=[]
            for mid in near:
                innov = pw - MAP[mid]
                m2 = float(innov.T @ S_inv @ innov)
                if m2 <= self.chi2_gate:
                    cc.append((int(mid), m2))
            if not cc and self.min_candidates>0 and near.size>0:
                k = min(self.min_candidates, near.size)
                by = sorted([(int(mid), d2[mid]) for mid in near], key=lambda t:t[1])[:k]
                # give them high (but finite) cost
                cc = [(mid, self.chi2_gate + 1.0) for (mid,_) in by]
            cc.sort(key=lambda t:t[1])
            cands.append(cc)
            total_cands += len(cc)

        if total_cands==0:
            self._print_line(t_z, dt_pose, v, n_obs, 0, total_cands, 0, [])
            self.frame_idx += 1
            return

        # Compact columns: union of all candidate map ids
        cand_ids = sorted(set(mid for lst in cands for mid,_ in lst))
        id2col = {mid:i for i,mid in enumerate(cand_ids)}
        base_cols = len(cand_ids)

        # Build cost matrix with one dummy column (unmatched)
        BIG = 1e6
        cost = np.full((n_obs, base_cols+1), BIG, float)
        for i,lst in enumerate(cands):
            for (mid, m2) in lst:
                cost[i, id2col[mid]] = m2
        # dummy unmatched column
        cost[:, base_cols] = float(self.unmatched_cost)

        # Pad to square if needed
        rows, cols = cost.shape
        if cols < rows:
            pad = rows - cols
            cost = np.hstack([cost, np.full((rows,pad), float(self.unmatched_cost), float)])
        elif cols > rows:
            pad = cols - rows
            cost = np.vstack([cost, np.zeros((pad, cols), float)])

        # Solve
        rs, _ = hungarian_min_cost(cost)

        # Decode (only real rows)
        matches=[]
        for i in range(n_obs):
            j = int(rs[i])
            if j < base_cols:
                mid = cand_ids[j]
                m2  = cost[i,j]
                if m2 <= self.chi2_gate + 1e-9:
                    matches.append((i, mid, m2))

        # Pretty pairs
        show=[]
        for (oi, mid, m2) in matches[:min(self.max_pairs_print, len(matches))]:
            show.append(f"(obs{oi}->map{mid}, √m²={math.sqrt(max(0.0,m2)):.2f})")

        proc_ms = int((time.perf_counter()-t0)*1000)
        self._print_line(t_z, dt_pose, v, n_obs, len(matches), total_cands, proc_ms, show)
        self.frame_idx += 1

    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    def _print_line(self, t_z, dt_pose, v, n_obs, n_match, n_cands_total, proc_ms, show_pairs):
        pass_dt = 0.0 if not hasattr(self, "_last_t") or self._last_t is None else (t_z - self._last_t)
        self._last_t = t_z
        mean_cands = n_cands_total / max(1, n_obs)
        base = (f"[hung] pass_dt={int(pass_dt*1000)}ms  Δt_pose={int(dt_pose*1000)}ms  "
                f"v={v:.2f}m/s  obs={n_obs}  mean_cands/obs={mean_cands:.2f}  "
                f"matched={n_match}  proc={proc_ms}ms")
        if show_pairs:
            base += " | " + ", ".join(show_pairs)
        print(base)

def main():
    rclpy.init()
    node = HungarianDANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
