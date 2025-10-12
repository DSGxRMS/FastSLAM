#!/usr/bin/env python3
# pf_localizer_hungarian.py — Mahalanobis-gated Hungarian DA + pose correction on predicted odom
#
# IO:
#   in  pose (predicted): /odometry_integration/car_state   (Odometry)
#   in  cones:            /ground_truth/cones               (ConeArrayWithCovariance)  # name-only "ground_truth"
#   out corrected odom:   /pf_localizer/odom_corrected      (Odometry)
#   out corrected pose2d: /pf_localizer/pose2d              (Pose2D)
#
# Behavior:
#   - Build per-obs candidate set via dynamic inflation + χ² gating against known map.
#   - Solve global assignment with Hungarian (adds unmatched column).
#   - If enough matched pairs: estimate rigid 2D transform (SVD) and
#     update EMA-smoothed Δ=[dx,dy,dθ]. Otherwise keep previous Δ.
#   - Always publish (at cones rate): odom corrected by current Δ; if Δ≈0 → as-is.
#
# Logs:
#   [hung] obs=<N> matched=<K> pass_dt=<ms> proc=<ms>   (throttled + heartbeat)
#
import math, json, time, bisect, sys
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
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3

BASE_DIR = Path(__file__).resolve().parents[0]

# ---------- helpers ----------
def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

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

    # Greedy star of zeros
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
                m = cost[~row_cov][:, ~col_cov].min()
                cost[~row_cov,:] -= m
                cost[:, col_cov] += m
            else:
                mask[r,c]=2
                c_star = find_star_in_row(r)
                if c_star is None:
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

# --- rigid fit (SVD) ---
def svd_rigid_2d(A: np.ndarray, B: np.ndarray):
    """
    Given paired points A (Nx2) and B (Nx2), find R(2x2), t(2,)
    minimizing || R*A_i + t - B_i ||^2.
    """
    if A.shape[0] == 0:
        return np.eye(2), np.zeros(2)
    ca = np.mean(A, axis=0)
    cb = np.mean(B, axis=0)
    A0 = A - ca
    B0 = B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t

class PFLocalizerHungarian(Node):
    def __init__(self):
        super().__init__("pf_localizer_hungarian", automatically_declare_parameters_from_overrides=True)

        # params
        self.declare_parameter("detections_frame", "base")     # "base" or "world"
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("min_candidates", 0)
        self.declare_parameter("unmatched_cost", 25.0)        # m² (>= chi2_gate_2d)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)

        # topics + logging
        self.declare_parameter("pose_topic", "/odometry_integration/car_state")
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("log_period_s", 0.5)

        # correction / smoothing
        self.declare_parameter("corr.min_k_for_update", 6)    # min matched pairs
        self.declare_parameter("corr.ema_alpha", 0.25)        # EMA for Δ

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
        self.pose_topic = str(p("pose_topic").value)
        self.cones_topic = str(p("cones_topic").value)
        self.log_period = float(p("log_period_s").value)

        self.min_k_update = int(p("corr.min_k_for_update").value)
        self.ema_alpha    = float(p("corr.ema_alpha").value)

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

        # state / runtime stats
        self.odom_buf: deque = deque()
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self._last_proc_wall = 0.0
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_t = None
        self._last_log_wall = -1e9   # ensure first print
        self._last_status = {"obs":0, "matched":0, "proc_ms":0, "t_z":None}
        self._warned_no_pubs = False

        # correction delta (kept across frames)
        self.delta = np.zeros(3, float)  # dx, dy, dθ

        # pubs
        self.pub_out = self.create_publisher(Odometry, "/pf_localizer/odom_corrected", 10)
        self.pub_pose2d = self.create_publisher(Pose2D, "/pf_localizer/pose2d", 10)

        # QoS: SensorData style
        q_cones = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE,
                             history=QoSHistoryPolicy.KEEP_LAST)
        q_odom  = QoSProfile(depth=20,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE,
                             history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry, self.pose_topic, self.cb_odom, q_odom)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_cones)

        # Heartbeat timer – prints even if callbacks never fire
        self.create_timer(self.log_period, self._heartbeat)

        self.get_logger().info(
            f"[hung] up. Map cones: {self.MAP.shape[0]} | pose={self.pose_topic} | cones={self.cones_topic}"
        )

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

    # ---- inflation ----
    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    # ---- main DA + correction/publish ----
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

        # Build obs in WORLD (+ BASE vec for range)
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
            # no detections -> publish as-is with current Δ and update heartbeat
            self._publish_corrected(t_z, pose_w_b)
            self._update_status(0, 0, int((time.perf_counter()-t0)*1000), t_z)
            return

        crop2 = self.map_crop_m**2
        MAP = self.MAP
        cands: List[List[Tuple[int,float]]] = []
        total_cands=0

        # Build candidate lists with dynamic inflation + gate
        for pw, S0, pb in zip(obs_pw, obs_S0, obs_pb):
            r = float(np.linalg.norm(pb))
            S_eff = self._inflate_cov(S0, v, yawrate, dt_pose, r, yaw)

            # per-obs Cholesky
            use_chol = True
            try:
                L = np.linalg.cholesky(S_eff)
                Linv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                use_chol = False
                Sinv = np.linalg.pinv(S_eff)

            diffs = MAP - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            near = np.where(d2 <= crop2)[0]

            cc=[]
            if near.size:
                innovs = pw - MAP[near]  # [K,2]
                if use_chol:
                    y = innovs @ Linv.T
                    m2 = np.einsum('ij,ij->i', y, y)
                else:
                    m2 = np.einsum('ij,ij->i', innovs @ Sinv, innovs)
                keep = np.where(m2 <= self.chi2_gate)[0]
                if keep.size:
                    order = keep[np.argsort(m2[keep])]
                    mids = near[order].astype(int)
                    cc = list(zip(mids.tolist(), m2[order].astype(float).tolist()))

            if not cc and self.min_candidates>0 and near.size>0:
                k = min(self.min_candidates, near.size)
                by = np.argsort(d2[near])[:k]
                for idx in by:
                    cc.append((int(near[idx]), self.chi2_gate + 1.0))  # “expensive” fallback

            cands.append(cc)
            total_cands += len(cc)

        if total_cands==0:
            # publish as-is (no new correction)
            self._publish_corrected(t_z, pose_w_b)
            self._update_status(n_obs, 0, int((time.perf_counter()-t0)*1000), t_z)
            return

        # Compact columns: union of map IDs
        cand_ids = sorted(set(mid for lst in cands for mid,_ in lst))
        id2col = {mid:i for i,mid in enumerate(cand_ids)}
        base_cols = len(cand_ids)

        # Build cost matrix with one dummy column for unmatched
        BIG = 1e6
        cost = np.full((n_obs, base_cols+1), BIG, float)
        for i,lst in enumerate(cands):
            for (mid, m2) in lst:
                cost[i, id2col[mid]] = m2
        cost[:, base_cols] = float(self.unmatched_cost)

        # Pad to square
        rows, cols = cost.shape
        if cols < rows:
            pad = rows - cols
            cost = np.hstack([cost, np.full((rows,pad), float(self.unmatched_cost), float)])
        elif cols > rows:
            pad = cols - rows
            cost = np.vstack([cost, np.zeros((pad, cols), float)])

        rs, _ = hungarian_min_cost(cost)

        # Decode matches for real rows
        matches=[]
        for i in range(n_obs):
            j = int(rs[i])
            if j < base_cols:
                mid = cand_ids[j]
                m2  = cost[i,j]
                if m2 <= self.chi2_gate + 1e-9:
                    matches.append((i, mid, m2))

        # If enough matches, compute rigid correction (SVD)
        k = len(matches)
        if k >= self.min_k_update:
            A = np.stack([obs_pw[i] for (i,_,_) in matches], axis=0)   # observed (WORLD)
            B = self.MAP[np.array([mid for (_,mid,_) in matches], int), :]  # map
            R_fit, t_fit = svd_rigid_2d(A, B)
            dth = math.atan2(R_fit[1,0], R_fit[0,0])
            p_o = np.array([x, y], float)
            p_c = R_fit @ p_o + t_fit
            dx, dy = float(p_c[0] - x), float(p_c[1] - y)
            # EMA blend
            a = self.ema_alpha
            self.delta[0] = (1.0 - a)*self.delta[0] + a*dx
            self.delta[1] = (1.0 - a)*self.delta[1] + a*dy
            self.delta[2] = wrap((1.0 - a)*self.delta[2] + a*dth)
        # else keep previous delta

        # Publish corrected odom (or as-is if delta small / unchanged)
        self._publish_corrected(t_z, pose_w_b)

        proc_ms = int((time.perf_counter()-t0)*1000)
        self._update_status(n_obs, k, proc_ms, t_z)

    # ---- publishing ----
    def _publish_corrected(self, t: float, odom_pose):
        dx, dy, dth = self.delta
        x_o, y_o, yaw_o = odom_pose
        x_c, y_c, yaw_c = x_o + dx, y_o + dy, wrap(yaw_o + dth)
        # use nearest odom twist; rotate linear to corrected yaw
        _, _, v_o, yr_o = self._pose_at(t)

        od = Odometry()
        od.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose = PoseWithCovariance()
        od.twist = TwistWithCovariance()
        od.pose.pose = Pose(position=Point(x=x_c, y=y_c, z=0.0), orientation=quat_from_yaw(yaw_c))
        od.twist.twist = Twist(
            linear=Vector3(x=v_o * math.cos(yaw_c), y=v_o * math.sin(yaw_c), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=yr_o)
        )
        self.pub_out.publish(od)

        p2 = Pose2D(); p2.x = x_c; p2.y = y_c; p2.theta = yaw_c
        self.pub_pose2d.publish(p2)

    # ---- status & logging ----
    def _update_status(self, n_obs, n_match, proc_ms, t_z):
        self._last_status.update({"obs": n_obs, "matched": n_match, "proc_ms": proc_ms, "t_z": t_z})
        now = time.time()
        if (now - self._last_log_wall) >= self.log_period or self._last_log_wall < -1e8:
            self._last_log_wall = now
            pass_dt = 0.0 if self._last_t is None else (t_z - self._last_t)
            self._last_t = t_z
            self.get_logger().info(f"[hung] obs={n_obs} matched={n_match} pass_dt={int(pass_dt*1000)}ms proc={proc_ms}ms")
            sys.stdout.flush()

    def _heartbeat(self):
        # warn once if no publishers on cones topic
        pubs = self.get_publishers_info_by_topic(self.cones_topic)
        if not pubs and not self._warned_no_pubs:
            self._warned_no_pubs = True
            self.get_logger().warn(f"[hung] No publishers on {self.cones_topic}. You will see only heartbeat logs.")
        # periodic line even if callbacks are silent
        st = self._last_status
        if st["t_z"] is None:
            self.get_logger().info("[hung] obs=0 matched=0 pass_dt=0ms proc=0ms")
        else:
            now = time.time()
            if (now - self._last_log_wall) >= self.log_period:
                self._last_log_wall = now
                self.get_logger().info(f"[hung] obs={st['obs']} matched={st['matched']} pass_dt=0ms proc={st['proc_ms']}ms")
        sys.stdout.flush()

# --- main ---
def main():
    rclpy.init()
    node = PFLocalizerHungarian()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
