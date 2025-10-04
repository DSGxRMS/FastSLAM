#!/usr/bin/env python3
# ijcbb_min_fast.py — iJCBB-lite (GT odom + GT cones), optimized
# Speed-ups:
#  - Uniform-grid spatial hash for map crop (O(1)-ish candidate lookup)
#  - Cholesky "solve" (no explicit Σ^-1)
#  - Cap candidates per obs after gating (top-K)
#  - Strong pruning in DFS
# Accuracy:
#  - Same gating + inflation + assignment policy as your iJCBB-lite

import math, json, time, bisect
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

BASE_DIR = Path(__file__).resolve().parents[0]

# ---------- io/helpers ----------
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

def rot2d(theta: float) -> np.ndarray:
    c,s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]], float)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

# ---------- iJCBB-lite assignment (unchanged policy) ----------
def ijcbb_assign(cand_per_obs: List[np.ndarray], mahal_costs: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    Greedy-ordered backtracking preferring more matches then lower total m2.
    Enforces unique map assignments. No joint chi-square set test.
    """
    n = len(cand_per_obs)
    # order by best individual cost (empties last)
    best_first = np.array([c[0] if c.size>0 else np.inf for c in mahal_costs])
    order = np.argsort(best_first)

    best_assign = [-1]*n
    best_matched = 0
    best_score = float('inf')
    used = set()
    assign = [-1]*n

    def dfs(k:int, matched:int, score:float):
        nonlocal best_matched, best_score, best_assign
        # bound: even if we match everything left, can we beat current best?
        if matched + (n - k) < best_matched:
            return
        if k == n:
            if matched > best_matched or (matched == best_matched and score < best_score):
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
            if mid in used: 
                continue
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

        # Params
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
        self.declare_parameter("log_period_s", 0.5)

        # Speed-tuning knobs (safe defaults)
        self.declare_parameter("top_k_per_obs", 24)          # cap on candidates after gating
        self.declare_parameter("grid_cell_m", 6.0)           # spatial-hash cell size (≈ cone spacing)

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
        self.log_period = float(p("log_period_s").value)

        self.top_k_per_obs = int(p("top_k_per_obs").value)
        self.grid_cell_m   = float(p("grid_cell_m").value)

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

        # Build spatial hash (uniform grid)
        self._build_grid_index()

        # State
        self.odom_buf: deque = deque()  # (t, x, y, yaw, v, yawrate)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_proc_wall = 0.0
        self._last_t = None
        self._last_log_wall = 0.0

        # QoS — SensorData style (GT topics)
        q_cones = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)
        q_odom = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q_odom)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q_cones)

        self.get_logger().info(f"iJCBB up. Map cones: {self.n_map} (GT odom+cones)")

    # ---------- grid index ----------
    def _build_grid_index(self):
        pts = self.map_xy
        self._xmin = float(np.min(pts[:,0]))
        self._ymin = float(np.min(pts[:,1]))
        self._cell = max(1e-3, self.grid_cell_m)
        # map from (ix,iy) -> np.array of global indices
        buckets = defaultdict(list)
        ix = np.floor((pts[:,0] - self._xmin)/self._cell).astype(int)
        iy = np.floor((pts[:,1] - self._ymin)/self._cell).astype(int)
        for i, (gx,gy) in enumerate(zip(ix,iy)):
            buckets[(int(gx), int(gy))].append(i)
        # convert to arrays
        self._grid = {k: np.array(v, dtype=int) for k,v in buckets.items()}

    def _grid_query_ball(self, x: float, y: float, r: float) -> np.ndarray:
        """Return candidate map indices within bbox around (x,y) of radius r using grid buckets."""
        cs = self._cell
        gx = int(math.floor((x - self._xmin)/cs))
        gy = int(math.floor((y - self._ymin)/cs))
        rad_cells = int(math.ceil(r / cs)) + 1  # +1 to be safe
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

    # --------- Odom ----------
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

    # --------- Cones (hot path) ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        # input-rate guard
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build observations in WORLD + base ranges
        obs_pw = []
        obs_pb_r = []
        sigma0 = self.sigma0
        Sigma0_w = np.array([[sigma0*sigma0, 0.0],[0.0, sigma0*sigma0]], float)  # constant addend

        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                if self.detections_frame == "base":
                    pb = np.array([px,py], float)
                    pw = np.array([x,y]) + Rwb @ pb
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y]))
                obs_pw.append(pw)
                obs_pb_r.append(float(np.linalg.norm(pb)))
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)

        n_obs = len(obs_pw)
        if n_obs == 0:
            return

        map_xy = self.map_xy
        crop_r = self.map_crop_m
        crop2 = crop_r*crop_r
        topK = self.top_k_per_obs

        cand_per_obs: List[np.ndarray] = []
        mahal_costs:  List[np.ndarray] = []

        # Per-obs inflation + grid crop + Mahalanobis (via Cholesky solve)
        for i in range(n_obs):
            pw = obs_pw[i]
            r  = obs_pb_r[i]

            trans_var   = self.alpha * (v**2) * (dt_pose**2)
            lateral_var = self.beta  * (yawrate**2) * (dt_pose**2) * (r**2)
            Jrot = Rwb @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ Rbw
            S = Sigma0_w + np.array([[trans_var,0.0],[0.0,trans_var]], float) + Jrot

            # spatial hash query -> nearby cell candidates
            near_idx = self._grid_query_ball(pw[0], pw[1], crop_r)
            if near_idx.size == 0:
                cand_per_obs.append(np.empty((0,), dtype=int))
                mahal_costs.append(np.empty((0,), dtype=float))
                continue

            # Euclidean prefilter inside r (tighten from bbox)
            diffs = map_xy[near_idx] - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            within = near_idx[d2 <= crop2]
            if within.size == 0:
                cand_per_obs.append(np.empty((0,), dtype=int))
                mahal_costs.append(np.empty((0,), dtype=float))
                continue

            # Cholesky; compute m2 via solve (no explicit inverse)
            try:
                L = np.linalg.cholesky(S)
                # Solve L * Y = innov^T   -> Y has shape (2,K)
                innov = (pw - map_xy[within]).T  # (2,K)
                Y = np.linalg.solve(L, innov)    # (2,K)
                m2 = np.einsum('ik,ik->k', Y, Y)
            except np.linalg.LinAlgError:
                # If S not PD, fall back to pseudo-inverse path
                Sinv = np.linalg.pinv(S)
                innov = pw - map_xy[within]      # (K,2)
                m2 = np.einsum('ij,ij->i', innov @ Sinv, innov)

            # gate by chi2
            keep = np.where(m2 <= self.chi2_gate)[0]
            if keep.size == 0:
                cand_per_obs.append(np.empty((0,), dtype=int))
                mahal_costs.append(np.empty((0,), dtype=float))
                continue

            # sort by cost asc; cap to topK (keeps accuracy w/ tight gates)
            if keep.size > topK:
                kk = keep[np.argpartition(m2[keep], topK)[:topK]]
                kk = kk[np.argsort(m2[kk])]
            else:
                kk = keep[np.argsort(m2[keep])]

            cand_per_obs.append(within[kk].astype(int))
            mahal_costs.append(m2[kk])

        # iJCBB-lite assign
        _, matched = ijcbb_assign(cand_per_obs, mahal_costs)

        # Minimal log every 0.5 s
        now = time.time()
        if (now - self._last_log_wall) >= self.log_period:
            self._last_log_wall = now
            pass_dt = 0.0 if self._last_t is None else (t_z - self._last_t)
            self._last_t = t_z
            proc_ms = int((time.perf_counter() - t0)*1000)
            print(f"[ijcbb] obs={n_obs} matched={matched} pass_dt={int(pass_dt*1000)}ms proc={proc_ms}ms")

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
