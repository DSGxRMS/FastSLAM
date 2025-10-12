#!/usr/bin/env python3
# pf_localizer_ijcbb.py
#
# Particle-filter drift corrector (Δ over SE(2)) that:
#   - Subscribes to predicted odom: /odometry_integration/car_state
#   - Subscribes to cones: /ground_truth/cones  (name-only "ground_truth")
#   - Runs iJCBB-lite (fast DA) to build matches
#   - Publishes corrected odom: /pf_localizer/odom_corrected  and  /pf_localizer/pose2d
#
# Behavior:
#   - Always publishes a pose on each cones frame (fast). If no good DA, it keeps the last Δ (or zero
#     before first correction), so output == prediction until corrections are available.
#   - PF predict every cones frame; PF update only when enough matches (K>=min_k_for_update).
#   - Logs: [ijcbb] obs, matched, pass_dt, proc_ms (throttled), plus 1Hz PF state line.
#
# iJCBB-lite details:
#   - Spatial hash crop (grid_cell_m)
#   - Per-observation inflation with (alpha, beta) using v, yawrate, dt, range
#   - Cholesky-based Mahalanobis m²
#   - Gate by chi2_gate_2d, keep top_k_per_obs per observation
#   - Greedy-ordered backtracking with unique-map constraint (no joint χ² set test)

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
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3

# ---------- small helpers ----------
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
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
        except Exception:
            continue
        rows.append((x,y,c))
    return rows

# ---------- iJCBB-lite (assignment) ----------
def ijcbb_assign(cand_per_obs: List[np.ndarray], mahal_costs: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    Greedy-ordered backtracking: maximize matches, then minimize sum of costs.
    Unique map IDs enforced; no joint chi² set gate.
    """
    n = len(cand_per_obs)
    best_first = np.array([c[0] if c.size>0 else np.inf for c in mahal_costs])
    order = np.argsort(best_first)

    best_assign = [-1]*n
    best_matched = 0
    best_score = float('inf')
    used = set()
    assign = [-1]*n

    def dfs(k:int, matched:int, score:float):
        nonlocal best_matched, best_score, best_assign
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
        for j in range(mids.size):
            mid = int(mids[j])
            if mid in used:  # unique constraint
                continue
            c = float(costs[j])
            used.add(mid); assign[oi]=mid
            dfs(k+1, matched+1, score+c)
            assign[oi]=-1; used.remove(mid)
        # unmatched branch
        dfs(k+1, matched, score)

    dfs(0,0,0.0)
    return best_assign, best_matched

# ---------- Node ----------
class PFLocalizerIJ(Node):
    def __init__(self):
        super().__init__("pf_localizer", automatically_declare_parameters_from_overrides=True)

        # ---- Topics ----
        self.declare_parameter("pose_topic", "/odometry_integration/car_state")  # predicted odom
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("out_odom_topic", "/pf_localizer/odom_corrected")
        self.declare_parameter("out_pose2d_topic", "/pf_localizer/pose2d")

        # ---- iJCBB params (fast DA) ----
        self.declare_parameter("detections_frame", "base")     # "base" or "world"
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
        # iJCBB speed knobs
        self.declare_parameter("top_k_per_obs", 24)
        self.declare_parameter("grid_cell_m", 6.0)
        self.declare_parameter("ijcbb.log_period_s", 0.5)

        # ---- PF params (Δ over SE(2)) ----
        self.declare_parameter("pf.num_particles", 250)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        self.declare_parameter("pf.process_std_xy_m", 0.02)
        self.declare_parameter("pf.process_std_yaw_rad", 0.002)
        self.declare_parameter("pf.likelihood_floor", 1e-12)
        self.declare_parameter("pf.min_k_for_update", 3)   # require at least K matches for PF update

        # ---- CLI log ----
        self.declare_parameter("log.cli_hz", 1.0)

        # ---- read params ----
        gp = self.get_parameter
        self.pose_topic = str(gp("pose_topic").value)
        self.cones_topic = str(gp("cones_topic").value)
        self.out_odom_topic = str(gp("out_odom_topic").value)
        self.out_pose2d_topic = str(gp("out_pose2d_topic").value)

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.map_crop_m = float(gp("map_crop_m").value)
        self.chi2_gate = float(gp("chi2_gate_2d").value)
        self.sigma0 = float(gp("meas_sigma_floor_xy").value)
        self.alpha = float(gp("alpha_trans").value)
        self.beta  = float(gp("beta_rot").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(gp("target_cone_hz").value)
        self.max_pairs_print = int(gp("max_pairs_print").value)

        self.top_k_per_obs = int(gp("top_k_per_obs").value)
        self.grid_cell_m   = float(gp("grid_cell_m").value)
        self.ijcbb_log_period = float(gp("ijcbb.log_period_s").value)

        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)
        self.min_k_for_update = int(gp("pf.min_k_for_update").value)

        self.cli_hz = float(gp("log.cli_hz").value)

        # ---- load map (+ alignment like before) ----
        data_dir = (Path(__file__).resolve().parents[0] / str(gp("map_data_dir").value)).resolve()
        align_json = data_dir / "alignment_transform.json"
        known_map_csv = data_dir / "known_map.csv"
        if not align_json.exists() or not known_map_csv.exists():
            raise RuntimeError(f"[pf_localizer] Missing map assets under {data_dir}")
        with open(align_json, "r") as f:
            T = json.load(f)
        R_map = np.array(T["rotation"], float)
        t_map = np.array(T["translation"], float)
        rows = load_known_map(known_map_csv)
        M = np.array([[x, y] for (x, y, _) in rows], float)
        self.MAP = (M @ R_map.T) + t_map
        self.n_map = self.MAP.shape[0]

        # ---- build spatial hash for map ----
        self._grid_build()

        # ---- state ----
        self.odom_buf: deque = deque()  # (t, x, y, yaw, v, yawrate)
        self.pose_latest = np.array([0.0, 0.0, 0.0], float)

        # PF state (Δ particles & weights)
        self.particles = np.zeros((self.Np, 3), float)
        self.weights = np.ones(self.Np, float) / self.Np

        # timing / logs
        self.last_cone_ts = None
        self._last_cli_wall = time.time()
        self._last_ij_log_wall = 0.0
        self._last_t_z = None
        self._last_obs = 0
        self._last_match = 0

        # ROS I/O
        q_rel = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                           durability=QoSDurabilityPolicy.VOLATILE,
                           history=QoSHistoryPolicy.KEEP_LAST)
        q_fast = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.pose_topic, self.cb_odom, q_fast)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_fast)
        self.pub_out = self.create_publisher(Odometry, self.out_odom_topic, 10)
        self.pub_pose2d = self.create_publisher(Pose2D, self.out_pose2d_topic, 10)

        # CLI
        self.create_timer(max(0.05, 1.0 / max(1e-3, self.cli_hz)), self.on_cli_timer)

        self.get_logger().info(f"[pf_localizer/ijcbb] up | pose={self.pose_topic} | cones={self.cones_topic} | map={self.n_map}")

    # ---------- spatial hash ----------
    def _grid_build(self):
        pts = self.MAP
        self._xmin = float(np.min(pts[:,0])); self._ymin = float(np.min(pts[:,1]))
        self._cell = max(1e-3, self.grid_cell_m)
        buckets = defaultdict(list)
        ix = np.floor((pts[:,0] - self._xmin)/self._cell).astype(int)
        iy = np.floor((pts[:,1] - self._ymin)/self._cell).astype(int)
        for i, (gx,gy) in enumerate(zip(ix,iy)):
            buckets[(int(gx), int(gy))].append(i)
        self._grid = {k: np.array(v, dtype=int) for k,v in buckets.items()}

    def _grid_ball(self, x: float, y: float, r: float) -> np.ndarray:
        cs = self._cell
        gx = int(math.floor((x - self._xmin)/cs))
        gy = int(math.floor((y - self._ymin)/cs))
        rad_cells = int(math.ceil(r / cs)) + 1
        packs=[]
        for dx in range(-rad_cells, rad_cells+1):
            for dy in range(-rad_cells, rad_cells+1):
                arr = self._grid.get((gx+dx, gy+dy), None)
                if arr is not None and arr.size:
                    packs.append(arr)
        if not packs:
            return np.empty((0,), dtype=int)
        return np.unique(np.concatenate(packs))

    # ---------- odom buffering ----------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        v, yr = 0.0, 0.0
        if self.odom_buf:
            t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[-1]
            dt = max(1e-3, t - t0)
            v = math.hypot(x - x0, y - y0) / dt
            yr = wrap(yaw - yaw0) / dt
        self.pose_latest = np.array([x, y, yaw], float)
        self.odom_buf.append((t, x, y, yaw, v, yr))
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
            t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[0]
            return np.array([x0, y0, yaw0], float), abs(t_query - t0), v0, yr0
        if idx >= len(self.odom_buf):
            t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[-1]
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
        t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[idx-1]
        t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[idx]
        if t1 == t0:
            return np.array([x0, y0, yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a*dyaw)
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    # ---------- PF core ----------
    def _pf_predict(self):
        self.particles[:, 0] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 1] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 2] += np.random.normal(0.0, self.p_std_yaw, size=self.Np)
        self.particles[:, 2] = np.array([wrap(th) for th in self.particles[:, 2]])

    def _pf_update(self, pairs, odom_pose, obs_pb):
        if len(pairs) < max(2, self.min_k_for_update):
            return
        idxs = [pi for (pi, mid) in pairs]
        mids = [mid for (pi, mid) in pairs]
        PB = np.stack([obs_pb[i] for i in idxs], axis=0)  # (K,2) base coords
        MW = self.MAP[np.array(mids, int), :]              # (K,2) world map points

        x_o, y_o, yaw_o = odom_pose
        new_weights = np.zeros_like(self.weights)
        sigma2 = self.sigma0 ** 2
        for i in range(self.Np):
            dx, dy, dth = self.particles[i]
            yaw_c = wrap(yaw_o + dth)
            Rc = rot2d(yaw_c)
            pc = np.array([x_o + dx, y_o + dy], float)
            PW_pred = pc[None, :] + (PB @ Rc.T)
            diffs = PW_pred - MW
            m2_sum = float(np.sum(np.einsum('ij,ij->i', diffs, diffs) / sigma2))
            like = self.like_floor if not math.isfinite(m2_sum) else max(math.exp(-0.5*m2_sum), self.like_floor)
            new_weights[i] = self.weights[i] * like
        sw = float(np.sum(new_weights))
        if sw <= 0.0 or not math.isfinite(sw):
            self.weights[:] = 1.0 / self.Np
        else:
            self.weights[:] = new_weights / sw

        # resample
        neff = 1.0 / float(np.sum(self.weights * self.weights))
        if neff < self.neff_ratio * self.Np:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.Np
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, 'i')
        csum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < csum[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights[:] = 1.0 / N

    def _estimate_delta(self):
        w = self.weights
        dx = float(np.sum(w * self.particles[:, 0]))
        dy = float(np.sum(w * self.particles[:, 1]))
        cs = float(np.sum(w * np.cos(self.particles[:, 2])))
        sn = float(np.sum(w * np.sin(self.particles[:, 2])))
        dth = math.atan2(sn, cs)
        return dx, dy, dth

    # ---------- Cones (iJCBB + PF + publish) ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        # rate limit to ~target_cone_hz
        now_wall = time.time()
        if self.last_cone_ts is not None:
            if (RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9 - self.last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.last_cone_ts = t_z

        # Pose @ cones time
        odom_pose, dt_pose, v, yawrate = self._pose_at(t_z)
        x, y, yaw = odom_pose
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build observations (WORLD points + base ranges + base coords)
        obs_pw = []; obs_pb = []; obs_pb_r = []
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                if self.detections_frame == "base":
                    pb = np.array([px, py], float)
                    pw = np.array([x, y]) + Rwb @ pb
                else:
                    pw = np.array([px, py], float)
                    pb = Rbw @ (pw - np.array([x, y]))
                obs_pw.append(pw); obs_pb.append(pb); obs_pb_r.append(float(np.linalg.norm(pb)))
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)

        n_obs = len(obs_pw)
        if n_obs == 0:
            # Even with no obs, still publish predicted (Δ unchanged)
            self._publish_corrected(t_z, odom_pose)
            return

        # iJCBB fast candidates per obs
        crop_r = self.map_crop_m
        crop2 = crop_r*crop_r
        topK = self.top_k_per_obs
        cand_per_obs: List[np.ndarray] = []
        cost_per_obs: List[np.ndarray] = []

        sigma0 = self.sigma0
        Sigma0_w = np.array([[sigma0*sigma0, 0.0],[0.0, sigma0*sigma0]], float)

        for i in range(n_obs):
            pw = obs_pw[i]; r = obs_pb_r[i]
            # Inflate cov
            trans_var   = self.alpha * (v**2) * (dt_pose**2)
            lateral_var = self.beta  * (yawrate**2) * (dt_pose**2) * (r**2)
            Jrot = Rwb @ np.array([[0.0,0.0],[0.0, lateral_var]]) @ Rbw
            S = Sigma0_w + np.array([[trans_var,0.0],[0.0,trans_var]], float) + Jrot

            near_idx = self._grid_ball(pw[0], pw[1], crop_r)
            if near_idx.size == 0:
                cand_per_obs.append(np.empty((0,), int)); cost_per_obs.append(np.empty((0,), float)); continue

            diffs_bbox = self.MAP[near_idx] - pw
            d2 = np.einsum('ij,ij->i', diffs_bbox, diffs_bbox)
            within = near_idx[d2 <= crop2]
            if within.size == 0:
                cand_per_obs.append(np.empty((0,), int)); cost_per_obs.append(np.empty((0,), float)); continue

            try:
                L = np.linalg.cholesky(S)
                innov = (pw - self.MAP[within]).T  # (2,K)
                Y = np.linalg.solve(L, innov)
                m2 = np.einsum('ik,ik->k', Y, Y)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
                innov = pw - self.MAP[within]
                m2 = np.einsum('ij,ij->i', innov @ Sinv, innov)

            keep = np.where(m2 <= self.chi2_gate)[0]
            if keep.size == 0:
                cand_per_obs.append(np.empty((0,), int)); cost_per_obs.append(np.empty((0,), float)); continue

            if keep.size > topK:
                kk = keep[np.argpartition(m2[keep], topK)[:topK]]
                kk = kk[np.argsort(m2[kk])]
            else:
                kk = keep[np.argsort(m2[keep])]

            cand_per_obs.append(within[kk].astype(int))
            cost_per_obs.append(m2[kk])

        # iJCBB-lite assignment
        assign, matched = ijcbb_assign(cand_per_obs, cost_per_obs)
        pairs = [(i, int(assign[i])) for i in range(len(assign)) if assign[i] != -1]
        self._last_obs, self._last_match = n_obs, matched

        # PF predict + (conditional) update
        self._pf_predict()
        self._pf_update(pairs, odom_pose, obs_pb)

        # publish (pred if no update happened; corrected otherwise)
        self._publish_corrected(t_z, odom_pose)

        # iJCBB log (throttled)
        now = time.time()
        if (now - self._last_ij_log_wall) >= self.ijcbb_log_period:
            self._last_ij_log_wall = now
            pass_dt = 0.0 if self._last_t_z is None else (t_z - self._last_t_z)
            self._last_t_z = t_z
            proc_ms = int((time.perf_counter() - t0)*1000)
            print(f"[ijcbb] obs={n_obs} matched={matched} pass_dt={int(pass_dt*1000)}ms proc={proc_ms}ms")

    # ---------- publish ----------
    def _publish_corrected(self, t: float, odom_pose):
        dx, dy, dth = self._estimate_delta()
        x_o, y_o, yaw_o = odom_pose
        x_c, y_c, yaw_c = x_o + dx, y_o + dy, wrap(yaw_o + dth)
        # keep twist from nearest odom, just rotate to corrected yaw for linear xy
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

    # ---------- CLI ----------
    def on_cli_timer(self):
        if self.last_cone_ts is None:
            return
        now = time.time()
        if now - self._last_cli_wall < (1.0 / max(1e-3, self.cli_hz)):
            return
        self._last_cli_wall = now

        # report corrected pose & Δ and Neff
        odom_pose, _, v_o, _ = self._pose_at(self.last_cone_ts)
        dx, dy, dth = self._estimate_delta()
        x_o, y_o, yaw_o = odom_pose
        x_c, y_c, yaw_c = x_o + dx, y_o + dy, wrap(yaw_o + dth)
        neff = 1.0 / float(np.sum(self.weights * self.weights))
        self.get_logger().info(
            f"[t={self.last_cone_ts:7.2f}s] PF: x={x_c:6.2f} y={y_c:6.2f} yaw={math.degrees(yaw_c):6.1f}° v={v_o:4.2f} | "
            f"Δ=[{dx:+.3f},{dy:+.3f},{math.degrees(dth):+.2f}°] | "
            f"ijcbb obs/match={self._last_obs}/{self._last_match} | Neff={neff:.1f}/{self.Np}"
        )

# ---------- main ----------
def main():
    rclpy.init()
    node = PFLocalizerIJ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
