#!/usr/bin/env python3
# pf_localizer_icp.py
#
# Drift corrector using 2D point-to-point ICP between observed cones (via predicted odom)
# and the known cone map. Publishes continuously; corrects when ICP converges.
#
# IO (same as before):
#   - in  pose (predicted): /odometry_integration/car_state   (Odometry)
#   - in  cones:            /ground_truth/cones               (ConeArrayWithCovariance)  # name-only "ground_truth"
#   - out corrected odom:   /pf_localizer/odom_corrected      (Odometry)
#   - out corrected pose2d: /pf_localizer/pose2d              (Pose2D)
#
# Behavior:
#   - Every cones frame: compute ICP transform T_icp (R, t) such that MAP ≈ R*PW + t
#     where PW are observed cone positions in WORLD using predicted pose.
#   - Apply pose correction: yaw_c = yaw_o + theta_icp; p_c = R* p_o + t
#     (so output == prediction when ICP has too few/poor matches).
#   - Optional EMA smoothing of the delta.
#
# Speed:
#   - Spatial hash crop around observations
#   - KNN radius search in cropped map
#   - Few ICP iterations (3–7 typical)
#
# Logs:
#   [icp] obs=<N> matched=<K> iters=<i> rmse=<e> pass_dt=<ms> proc=<ms>

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

# ---------- ICP helpers ----------
def svd_rigid_2d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    # enforce proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t

def pair_knn_radius(P: np.ndarray, Q: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Brute-force KNN with radius clip (small sets; after spatial crop).
    For each p in P, find nearest q in Q with ||p-q|| <= r.
    Returns indices (i_in_P, j_in_Q, dists).
    """
    if P.size == 0 or Q.size == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)
    # (N, M) squared distances
    d2 = np.sum((P[:,None,:] - Q[None,:,:])**2, axis=2)
    j = np.argmin(d2, axis=1)
    dmin2 = d2[np.arange(P.shape[0]), j]
    keep = np.where(dmin2 <= r*r)[0]
    return keep, j[keep], np.sqrt(dmin2[keep])

# ---------- Node ----------
class PFLocalizerICP(Node):
    def __init__(self):
        super().__init__("pf_localizer_icp", automatically_declare_parameters_from_overrides=True)

        # ---- Topics ----
        self.declare_parameter("pose_topic", "/odometry_integration/car_state")  # predicted odom
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("out_odom_topic", "/pf_localizer/odom_corrected")
        self.declare_parameter("out_pose2d_topic", "/pf_localizer/pose2d")

        # ---- General params ----
        self.declare_parameter("detections_frame", "base")     # "base" or "world"
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)

        # ---- Spatial crop ----
        self.declare_parameter("grid_cell_m", 6.0)
        self.declare_parameter("map_crop_m", 25.0)  # radial crop around obs

        # ---- ICP params ----
        self.declare_parameter("icp.max_iters", 6)
        self.declare_parameter("icp.nn_radius_m", 3.5)
        self.declare_parameter("icp.outlier_thresh_m", 1.2)   # drop pairs with residual > this
        self.declare_parameter("icp.min_k_for_update", 6)     # minimal good pairs to accept correction
        self.declare_parameter("icp.top_k_pairs", 80)         # optional cap on pairs per iter
        self.declare_parameter("icp.ema_alpha", 0.25)         # smoothing of delta [0..1], 0=off

        # ---- CLI ----
        self.declare_parameter("log.cli_hz", 1.0)
        self.declare_parameter("icp.log_period_s", 0.5)

        # ---- read params ----
        gp = self.get_parameter
        self.pose_topic = str(gp("pose_topic").value)
        self.cones_topic = str(gp("cones_topic").value)
        self.out_odom_topic = str(gp("out_odom_topic").value)
        self.out_pose2d_topic = str(gp("out_pose2d_topic").value)

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(gp("target_cone_hz").value)

        self.grid_cell_m = float(gp("grid_cell_m").value)
        self.map_crop_m  = float(gp("map_crop_m").value)

        self.icp_max_iters = int(gp("icp.max_iters").value)
        self.icp_nn_radius = float(gp("icp.nn_radius_m").value)
        self.icp_outlier = float(gp("icp.outlier_thresh_m").value)
        self.icp_min_k = int(gp("icp.min_k_for_update").value)
        self.icp_top_k = int(gp("icp.top_k_pairs").value)
        self.icp_ema_alpha = float(gp("icp.ema_alpha").value)

        self.cli_hz = float(gp("log.cli_hz").value)
        self.icp_log_period = float(gp("icp.log_period_s").value)

        # ---- load map (+ alignment) ----
        data_dir = (Path(__file__).resolve().parents[0] / str(gp("map_data_dir").value)).resolve()
        align_json = data_dir / "alignment_transform.json"
        known_map_csv = data_dir / "known_map.csv"
        if not align_json.exists() or not known_map_csv.exists():
            raise RuntimeError(f"[pf_localizer_icp] Missing map assets under {data_dir}")
        with open(align_json, "r") as f:
            T = json.load(f)
        R_map = np.array(T["rotation"], float)
        t_map = np.array(T["translation"], float)
        rows = load_known_map(known_map_csv)
        M = np.array([[x, y] for (x, y, _) in rows], float)
        self.MAP = (M @ R_map.T) + t_map
        self.n_map = self.MAP.shape[0]

        # ---- spatial hash grid ----
        self._grid_build()

        # ---- state ----
        self.odom_buf: deque = deque()  # (t, x, y, yaw, v, yawrate)
        self.pose_latest = np.array([0.0, 0.0, 0.0], float)

        # correction Δ (ema-smoothed)
        self.delta = np.zeros(3, float)  # [dx, dy, dth] to add on top of odom pose

        # timing / logs
        self.last_cone_ts = None
        self._last_cli_wall = time.time()
        self._last_icp_log_wall = 0.0
        self._last_t_z = None
        self._last_obs = 0
        self._last_match = 0
        self._last_icp_rmse = float('nan')
        self._last_icp_iters = 0

        # ROS I/O
        q_fast = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.pose_topic, self.cb_odom, q_fast)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_fast)
        self.pub_out = self.create_publisher(Odometry, self.out_odom_topic, 10)
        self.pub_pose2d = self.create_publisher(Pose2D, self.out_pose2d, 10)

        # CLI
        self.create_timer(max(0.05, 1.0 / max(1e-3, self.cli_hz)), self.on_cli_timer)

        self.get_logger().info(f"[pf_localizer/icp] up | pose={self.pose_topic} | cones={self.cones_topic} | map={self.n_map}")

    # ---------- grid ----------
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

    # ---------- Cones → ICP → publish ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        # rate limit (by cones stamp)
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_cone_ts is not None:
            if (t_z - self.last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        self.last_cone_ts = t_z

        tic = time.perf_counter()

        # Pose @ t_z and transforms
        odom_pose, dt_pose, v, yawrate = self._pose_at(t_z)
        x, y, yaw = odom_pose
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Observations in WORLD
        obs_pw = []
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                if self.detections_frame == "base":
                    pb = np.array([px, py], float)
                    pw = np.array([x, y]) + Rwb @ pb
                else:
                    pw = np.array([px, py], float)
                obs_pw.append(pw)
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)
        if not obs_pw:
            self._publish_corrected(t_z, odom_pose)  # no change
            return

        P = np.stack(obs_pw, axis=0)  # (N,2)
        Nobs = P.shape[0]

        # Spatial crop of map around obs bbox (via grid)
        pxm, pym = np.mean(P[:,0]), np.mean(P[:,1])
        near_idx = self._grid_ball(pxm, pym, self.map_crop_m)
        if near_idx.size == 0:
            self._publish_corrected(t_z, odom_pose)
            self._last_obs = Nobs; self._last_match = 0; self._last_icp_iters = 0; self._last_icp_rmse = float('nan')
            self._log_icp(t_z, tic)
            return
        Q0 = self.MAP[near_idx]  # candidate map pts

        # ICP loop
        R_acc = np.eye(2); t_acc = np.zeros(2)
        Pk = P.copy()
        matched = 0
        rmse = float('nan')
        iters_done = 0

        for it in range(self.icp_max_iters):
            iters_done = it + 1
            # pair within radius
            ii, jj, d = pair_knn_radius(Pk, Q0, self.icp_nn_radius)
            if ii.size < self.icp_min_k:
                break

            # optional cap on pairs (take best)
            if ii.size > self.icp_top_k:
                sel = np.argsort(d)[:self.icp_top_k]
                ii, jj, d = ii[sel], jj[sel], d[sel]

            A = Pk[ii]
            B = Q0[jj]

            # outlier rejection by residual
            resid = np.linalg.norm(A - B, axis=1)
            keep = np.where(resid <= self.icp_outlier)[0]
            if keep.size < self.icp_min_k:
                break
            A = A[keep]; B = B[keep]
            matched = int(A.shape[0])

            # rigid fit
            R_step, t_step = svd_rigid_2d(A, B)

            # accumulate transform
            R_acc = R_step @ R_acc
            t_acc = R_step @ t_acc + t_step

            # apply to Pk
            Pk = (P @ R_acc.T) + t_acc

            # rmse
            rmse = float(np.sqrt(np.mean(np.sum((A - B)**2, axis=1))))

            # small early-stop if very low error
            if rmse < 0.05 and matched >= self.icp_min_k:
                break

        # Accept correction only if enough pairs were used in the last iter
        self._last_obs = Nobs
        self._last_match = matched
        self._last_icp_iters = iters_done
        self._last_icp_rmse = rmse

        if matched >= self.icp_min_k and np.isfinite(rmse):
            # Pose correction:
            # Given MAP ≈ R_acc * PW + t_acc, corrected pose is:
            #   yaw_c = yaw_o + theta(R_acc)
            #   p_c   = R_acc * p_o + t_acc
            dth = math.atan2(R_acc[1,0], R_acc[0,0])
            p_o = np.array([x, y], float)
            p_c = R_acc @ p_o + t_acc
            dx, dy = float(p_c[0] - x), float(p_c[1] - y)

            # EMA blend to avoid jitter (set ema_alpha=0 to disable)
            if self.icp_ema_alpha > 0.0:
                self.delta[0] = (1.0 - self.icp_ema_alpha)*self.delta[0] + self.icp_ema_alpha*dx
                self.delta[1] = (1.0 - self.icp_ema_alpha)*self.delta[1] + self.icp_ema_alpha*dy
                self.delta[2] = wrap((1.0 - self.icp_ema_alpha)*self.delta[2] + self.icp_ema_alpha*dth)
            else:
                self.delta[:] = [dx, dy, dth]
        # else: keep previous delta (or zeros initially)

        # publish
        self._publish_corrected(t_z, odom_pose)
        self._log_icp(t_z, tic)

    # ---------- publish ----------
    def _publish_corrected(self, t: float, odom_pose):
        dx, dy, dth = self.delta
        x_o, y_o, yaw_o = odom_pose
        x_c, y_c, yaw_c = x_o + dx, y_o + dy, wrap(yaw_o + dth)
        # keep twist from nearest odom, rotate linear to corrected yaw
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

    # ---------- logs ----------
    def _log_icp(self, t_z: float, tic: float):
        now = time.time()
        if (now - self._last_icp_log_wall) >= self.icp_log_period:
            self._last_icp_log_wall = now
            pass_dt = 0.0 if self._last_t_z is None else (t_z - self._last_t_z)
            self._last_t_z = t_z
            proc_ms = int((time.perf_counter() - tic)*1000)
            print(f"[icp] obs={self._last_obs} matched={self._last_match} iters={self._last_icp_iters} "
                  f"rmse={self._last_icp_rmse:.3f} pass_dt={int(pass_dt*1000)}ms proc={proc_ms}ms")

    def on_cli_timer(self):
        if self.last_cone_ts is None:
            return
        now = time.time()
        if now - self._last_cli_wall < (1.0 / max(1e-3, self.cli_hz)):
            return
        self._last_cli_wall = now

        odom_pose, _, v_o, _ = self._pose_at(self.last_cone_ts)
        dx, dy, dth = self.delta
        x_o, y_o, yaw_o = odom_pose
        x_c, y_c, yaw_c = x_o + dx, y_o + dy, wrap(yaw_o + dth)
        self.get_logger().info(
            f"[t={self.last_cone_ts:7.2f}s] ICP: x={x_c:6.2f} y={y_c:6.2f} yaw={math.degrees(yaw_c):6.1f}° v={v_o:4.2f} | "
            f"Δ=[{dx:+.3f},{dy:+.3f},{math.degrees(dth):+.2f}°] | obs/match={self._last_obs}/{self._last_match} "
            f"| iters={self._last_icp_iters} rmse={self._last_icp_rmse:.3f}"
        )

# ---------- main ----------
def main():
    rclpy.init()
    node = PFLocalizerICP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
