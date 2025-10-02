#!/usr/bin/env python3
# ijcbb_min.py — Minimal iJCBB DA runner (no RViz, no CSV; concise CLI logs only)
# - SensorData QoS to avoid backlog (BEST_EFFORT, KEEP_LAST, depth=1)
# - Throttled console logs
# - Vectorized gating + per-obs Cholesky reuse
# - Same dynamic covariance inflation as your filter (alpha/beta on v, yawrate, dt_pose, range)

import math, json, time, bisect
from pathlib import Path
from collections import deque
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

# ---------- Map assets ----------
BASE_DIR = Path(__file__).resolve().parents[0]

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
    xk = pick("x","x_m","xmeter"); yk = pick("y","y_m","ymeter"); ck = pick("color","colour","tag")
    for r in rdr:
        try:
            x=float(r[xk]); y=float(r[yk]); c=str(r[ck]).strip().lower()
        except Exception:
            continue
        rows.append((x,y,c))
    return rows

# ---------- Math helpers ----------
def rot2d(theta: float) -> np.ndarray:
    c,s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]], float)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def chi2_thresh_2d(sig=0.99):
    # 2 dof
    if abs(sig-0.99) < 1e-6: return 9.2103
    if abs(sig-0.997) < 1e-6: return 11.829
    if abs(sig-0.95) < 1e-6: return 5.9915
    # Wilson–Hilferty approx fallback
    dof=2
    z = {0.95:1.64485, 0.99:2.32635, 0.997:2.96774}.get(sig, 2.32635)
    return dof * (1 - 2/(9*dof) + z*math.sqrt(2/(9*dof)))**3

# ---------- Minimal iJCBB ----------
def ijcbb_assign(cand_per_obs: List[np.ndarray], mahal_costs: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    cand_per_obs[i] -> array of map_ids for obs i (shape [Ki])
    mahal_costs[i]  -> array of Mahalanobis^2 for those candidates (shape [Ki])
    We run a greedy-ordered joint-compatibility backtracking that prefers low individual m2 and more matches.
    Returns: (assign list of map_id or -1, matched_count)
    NOTE: We skip full JCBB chi-square over joint set for speed; with tight gates Ki is small.
    """
    n = len(cand_per_obs)
    # Order by best individual cost
    best_first = np.array([c[0] if c.size>0 else np.inf for c in mahal_costs])
    order = np.argsort(best_first)
    best_assign = [-1]*n
    best_matched = 0
    best_score = float('inf')
    used = set()
    assign = [-1]*n

    def dfs(k:int, matched:int, score:float):
        nonlocal best_matched, best_score, best_assign
        if k == n:
            if matched > best_matched or (matched==best_matched and score < best_score):
                best_matched = matched
                best_score = score
                best_assign = assign.copy()
            return
        oi = order[k]
        mids = cand_per_obs[oi]
        costs = mahal_costs[oi]
        # try candidates in ascending cost
        for j in range(mids.size):
            mid = int(mids[j])
            if mid in used: continue
            c = float(costs[j])
            used.add(mid); assign[oi]=mid
            dfs(k+1, matched+1, score+c)
            assign[oi]=-1; used.remove(mid)
        # allow unmatched
        dfs(k+1, matched, score)

    dfs(0,0,0.0)
    return best_assign, best_matched

# ---------- Node ----------
class IJCBBCore(Node):
    def __init__(self):
        super().__init__("fslam_ijcbb")

        # Params (kept minimal)
        self.declare_parameter("detections_frame", "base")   # "base" or "world"
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)        # gating on 2 dof
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("log_every_n_frames", 5)      # throttle CLI logs

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
        self.log_every = max(1, int(p("log_every_n_frames").value))

        # Map & alignment
        data_dir = (BASE_DIR / str(p("map_data_dir").value)).resolve()
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
        self.map_xy = (M @ R_map.T) + t_map
        self.n_map = self.map_xy.shape[0]

        # State
        self.odom_buf: deque = deque()  # (t, x, y, yaw, v, yawrate)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_proc_wall = 0.0
        self._last_t = None
        self.frame_idx = 0

        # QoS — SensorData style to prevent backlog
        q_cones = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        q_odom = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q_odom)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q_cones)

        log_root = (BASE_DIR.parents[1] / "logs" / "da_eval")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_root / ts / "algo=ijcbb"
        self.get_logger().info(f"iJCBB up. Map cones: {self.n_map}. Logs: {self.log_dir}")

    # --------- Odom handling ----------
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
            yr = wrap_angle(yaw - yaw0) / dt
        self.pose_latest = np.array([x,y,yaw], float)
        self.odom_buf.append((t,x,y,yaw,v,yr))
        # trim
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
                yaw = wrap_angle(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt_c) - math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt_c) - math.cos(yaw1))
                yaw = wrap_angle(yaw1 + yr1*dt_c)
            return np.array([x,y,yaw], float), abs(dt), v1, yr1
        t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[idx-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[idx]
        if t1==t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap_angle(yaw1 - yaw0); yaw = wrap_angle(yaw0 + a*dyaw)
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    # --------- Cones callback (main hot path) ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        # Input-rate guard: process at ~target_cone_hz
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build observations in WORLD with base ranges
        obs_pw = []      # list of 2-vectors
        obs_pb_r = []    # scalar ranges in BASE
        Sigma0_w = np.diag([self.sigma0**2, self.sigma0**2])

        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma0**2, self.sigma0**2])
                if self.detections_frame == "base":
                    pb = np.array([px,py], float)
                    pw = np.array([x,y]) + Rwb @ pb
                    Sw = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y]))
                    Sw = Pw
                obs_pw.append(pw)
                obs_pb_r.append(float(np.linalg.norm(pb)))
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)

        n_obs = len(obs_pw)
        if n_obs == 0:
            return

        # Vectorized nearby crop
        crop2 = self.map_crop_m**2
        map_xy = self.map_xy

        cand_per_obs: List[np.ndarray] = []
        mahal_costs:  List[np.ndarray] = []

        # Precompute per-obs inflated cov + its Cholesky
        for i in range(n_obs):
            pw = obs_pw[i]
            r  = obs_pb_r[i]
            # dynamic inflation
            trans_var = self.alpha * (v**2) * (dt_pose**2)
            lateral_var = self.beta * (yawrate**2) * (dt_pose**2) * (r**2)
            Jrot = Rwb @ np.array([[0.0,0.0],[0.0,lateral_var]]) @ Rbw
            S = Sigma0_w + np.diag([trans_var, trans_var]) + Jrot
            # crop candidates by Euclid
            diffs = map_xy - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            near = np.where(d2 <= crop2)[0]
            if near.size == 0:
                cand_per_obs.append(np.empty((0,), dtype=int))
                mahal_costs.append(np.empty((0,), dtype=float))
                continue
            # Mahalanobis with Cholesky solve
            try:
                L = np.linalg.cholesky(S)
                Linv = np.linalg.inv(L)
                Sinv = Linv.T @ Linv
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

            innovs = pw - map_xy[near]        # shape [K,2]
            m2 = np.einsum('ij,ij->i', innovs @ Sinv, innovs)
            keep = np.where(m2 <= self.chi2_gate)[0]
            if keep.size == 0:
                cand_per_obs.append(np.empty((0,), dtype=int))
                mahal_costs.append(np.empty((0,), dtype=float))
            else:
                # sort by cost asc
                kk = keep[np.argsort(m2[keep])]
                cand_per_obs.append(near[kk].astype(int))
                mahal_costs.append(m2[kk])

        # Run iJCBB-lite (order + backtrack with uniqueness)
        assign, matched = ijcbb_assign(cand_per_obs, mahal_costs)

        # logging (throttled)
        if self._last_t is None:
            pass_dt = 0.0
        else:
            pass_dt = (t_z - self._last_t)*1000.0
        self._last_t = t_z

        total_cands = sum(len(c) for c in cand_per_obs)
        mean_cands = total_cands / max(1, n_obs)
        proc_ms = int((time.perf_counter() - t0)*1000)

        # Print every N frames or if we matched < obs (interesting)
        if (self.frame_idx % self.log_every == 0) or (matched < n_obs):
            print(f"[ijcbb] pass_dt={pass_dt:.0f}ms  Δt_pose={dt_pose*1000:.0f}ms  "
                  f"v={v:.2f}m/s  obs={n_obs}  mean_cands/obs={mean_cands:.2f}  "
                  f"matched={matched}  proc={proc_ms}ms")

        self.frame_idx += 1

# ---------- main ----------
def main():
    rclpy.init()
    node = IJCBBCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
