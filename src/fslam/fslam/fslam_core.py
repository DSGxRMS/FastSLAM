#!/usr/bin/env python3
# fslam_core.py
#
# FastSLAM (PF + per-particle landmark EKFs) with:
#  - JCBB (color-strict) + greedy fallback
#  - Adaptive gating (context + residual-driven)
#  - Confidence-based retention + early map publish
#  - Drift nudging (small Gauss-Newton step on Δ per particle; guarded)
#
# Inputs:
#   /odometry_integration/car_state (nav_msgs/Odometry)  -> proposal  (swap to GT via param)
#   /ground_truth/cones             (eufs_msgs/ConeArrayWithCovariance) -> detections (BASE by default)
#
# Outputs (for your visualizer):
#   /fastslam/odom       (nav_msgs/Odometry)
#   /fastslam/map_cones  (eufs_msgs/ConeArrayWithCovariance)
#   /fastslam/pose2d     (geometry_msgs/Pose2D)
#
import math
import time
import bisect
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3, Pose2D
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance


# --------------------- helpers ---------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw / 2.0); q.w = math.cos(yaw / 2.0)
    return q

def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return  float('inf')
    a1=-3.969683028665376e+01; a2= 2.209460984245205e+02; a3=-2.759285104469687e+02
    a4= 1.383577518672690e+02; a5=-3.066479806614716e+01; a6= 2.506628277459239e+00
    b1=-5.447609879822406e+01; b2= 1.615858368580409e+02; b3=-1.556989798598866e+02
    b4= 6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00
    d4= 3.754408661907416e+00
    plow  = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p - 0.5; r = q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)


# --------------------- landmark + particle data ---------------------
@dataclass
class Landmark:
    mean: np.ndarray              # (2,) world coords in this particle's corrected frame
    cov: np.ndarray               # (2x2)
    color: str                    # 'blue' | 'yellow' | 'orange' | 'big'
    promoted: bool = False
    last_seen_t: float = -1.0
    last_hit_t: float = -1.0
    distinct_hits: int = 0
    conf: float = 0.0             # EWMA [0..1]
    retained: bool = False        # never deleted if True

@dataclass
class Particle:
    delta: np.ndarray             # Δ = [dx, dy, dθ]
    weight: float
    lms: List[Landmark] = field(default_factory=list)


# --------------------- FastSLAM core node ---------------------
class FastSLAMCore(Node):
    def __init__(self):
        super().__init__("fastslam_core", automatically_declare_parameters_from_overrides=True)

        # ---- topics ----
        self.declare_parameter("topics.pose_in", "/odometry_integration/car_state")   # override to /ground_truth/odom for GT tests
        self.declare_parameter("topics.cones_in", "/ground_truth/cones")
        self.declare_parameter("topics.odom_out", "/fastslam/odom")
        self.declare_parameter("topics.map_out", "/fastslam/map_cones")
        self.declare_parameter("topics.pose2d_out", "/fastslam/pose2d")

        # ---- QoS ----
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 200)

        # ---- DA / measurement ----
        self.declare_parameter("detections_frame", "base")   # "base" (car sees) or "world"
        self.declare_parameter("chi2_gate_2d", 11.83)        # ~99% for 2 dof (base gate)
        self.declare_parameter("joint_sig", 0.95)
        self.declare_parameter("joint_relax", 1.4)
        self.declare_parameter("min_k_for_joint", 3)
        self.declare_parameter("meas_sigma_floor_xy", 0.30)
        self.declare_parameter("alpha_trans", 0.9)
        self.declare_parameter("beta_rot", 0.9)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 60.0)
        self.declare_parameter("max_pairs_print", 6)
        self.declare_parameter("drop_stale_ms", 200.0)       # NEW: skip old cone frames to avoid snaps

        # ---- PF ----
        self.declare_parameter("pf.num_particles", 400)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        self.declare_parameter("pf.process_std_xy_m", 0.02)
        self.declare_parameter("pf.process_std_yaw_rad", 0.002)
        self.declare_parameter("pf.likelihood_floor", 1e-12)
        self.declare_parameter("pf.init_std_xy_m", 0.0)
        self.declare_parameter("pf.init_std_yaw_rad", 0.0)
        self.declare_parameter("seed", 7)

        # ---- Birth / Promotion / Deletion ----
        self.declare_parameter("birth.single_gate_sig", 0.95)
        self.declare_parameter("promotion.k_in_n", [2, 3])    # use only k (distinct hits >= k)
        self.declare_parameter("promotion_min_dt_s", 0.25)
        self.declare_parameter("delete.cov_trace_max", 1.0)

        # ---- Retention / Confidence ----
        self.declare_parameter("retain.conf_beta", 0.15)
        self.declare_parameter("retain.decay_hz", 0.05)
        self.declare_parameter("retain.retain_conf_thr", 0.80)
        self.declare_parameter("retain.retain_min_hits", 3)
        self.declare_parameter("retain.retain_cov_tr_max", 0.20)
        self.declare_parameter("retain.prune_conf_thr", 0.10)
        self.declare_parameter("retain.max_unseen_s", 30.0)
        self.declare_parameter("retain.merge_m2_thr", 6.0)

        # ---- JCBB bounds & logging ----
        self.declare_parameter("jcbb.max_obs_total", 9)
        self.declare_parameter("jcbb.max_obs_per_color", 3)
        self.declare_parameter("jcbb.cand_per_obs", 5)
        self.declare_parameter("jcbb.time_budget_ms", 700.0)  # total budget (shared) per cones frame
        self.declare_parameter("jcbb.crop_radius_m", 25.0)
        self.declare_parameter("log.jcbb", True)

        # ---- Map publish thresholds ----
        self.declare_parameter("map.publish_min_conf", 0.15)
        self.declare_parameter("map.publish_min_hits", 1)

        # ---- Adaptive gating params ----
        self.declare_parameter("gate.adapt_enable", True)
        self.declare_parameter("gate.scale_min", 0.6)
        self.declare_parameter("gate.scale_max", 2.5)
        self.declare_parameter("gate.target_cands", 2.0)     # aim ~2 candidates per obs
        self.declare_parameter("gate.k_pos", 0.10)            # per-meter scaling of chi²
        self.declare_parameter("gate.k_yaw", 0.50)            # per-rad scaling of chi²

        # ---- Drift nudge ----
        self.declare_parameter("nudge.enable", True)
        self.declare_parameter("nudge.gain_xy", 0.20)
        self.declare_parameter("nudge.gain_yaw", 0.12)
        self.declare_parameter("nudge.max_xy_step", 0.20)
        self.declare_parameter("nudge.max_yaw_step", 5.0)     # deg clamp per update

        # ---- Output smoothing (visual only) ----
        self.declare_parameter("odom.smooth_alpha", 0.0)      # 0 = off; try 0.15–0.25 to smooth tiny twitches

        # ---- read params ----
        gp = self.get_parameter
        self.pose_topic_in    = str(gp("topics.pose_in").value)
        self.cones_topic_in   = str(gp("topics.cones_in").value)
        self.odom_topic_out   = str(gp("topics.odom_out").value)
        self.map_topic_out    = str(gp("topics.map_out").value)
        self.pose2d_topic_out = str(gp("topics.pose2d_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE),
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.chi2_gate_base = float(gp("chi2_gate_2d").value)
        self.joint_sig = float(gp("joint_sig").value)
        self.joint_relax = float(gp("joint_relax").value)
        self.min_k_for_joint = int(gp("min_k_for_joint").value)
        self.sigma0 = float(gp("meas_sigma_floor_xy").value)
        self.alpha = float(gp("alpha_trans").value)
        self.beta = float(gp("beta_rot").value)
        self.target_cone_hz = float(gp("target_cone_hz").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0
        self.max_pairs_print = int(gp("max_pairs_print").value)
        self.drop_stale_s = float(gp("drop_stale_ms").value) / 1000.0

        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)
        self.init_std_xy = float(gp("pf.init_std_xy_m").value)
        self.init_std_yaw = float(gp("pf.init_std_yaw_rad").value)
        self.seed = int(gp("seed").value)

        self.birth_sig = float(gp("birth.single_gate_sig").value)
        self.prom_k, _ = [int(x) for x in gp("promotion.k_in_n").value]
        self.prom_dt_min = float(gp("promotion_min_dt_s").value)
        self.del_covtr = float(gp("delete.cov_trace_max").value)

        self.conf_beta     = float(gp("retain.conf_beta").value)
        self.decay_hz      = float(gp("retain.decay_hz").value)
        self.retain_thr    = float(gp("retain.retain_conf_thr").value)
        self.retain_hits   = int(gp("retain.retain_min_hits").value)
        self.retain_cov_tr = float(gp("retain.retain_cov_tr_max").value)
        self.prune_thr     = float(gp("retain.prune_conf_thr").value)
        self.max_unseen_s  = float(gp("retain.max_unseen_s").value)
        self.merge_m2_thr  = float(gp("retain.merge_m2_thr").value)

        self.max_obs_total   = int(gp("jcbb.max_obs_total").value)
        self.max_obs_per_col = int(gp("jcbb.max_obs_per_color").value)
        self.cand_per_obs    = int(gp("jcbb.cand_per_obs").value)
        self.jcbb_budget_s   = float(gp("jcbb.time_budget_ms").value) / 1000.0
        self.crop_radius_m   = float(gp("jcbb.crop_radius_m").value)
        self.log_jcbb        = bool(gp("log.jcbb").value)

        self.map_pub_conf = float(gp("map.publish_min_conf").value)
        self.map_pub_hits = int(gp("map.publish_min_hits").value)

        self.adapt_gate    = bool(gp("gate.adapt_enable").value)
        self.gate_min      = float(gp("gate.scale_min").value)
        self.gate_max      = float(gp("gate.scale_max").value)
        self.gate_target   = float(gp("gate.target_cands").value)
        self.gate_k_pos    = float(gp("gate.k_pos").value)
        self.gate_k_yaw    = float(gp("gate.k_yaw").value)

        self.nudge_enable  = bool(gp("nudge.enable").value)
        self.nudge_xy      = float(gp("nudge.gain_xy").value)
        self.nudge_yaw     = float(gp("nudge.gain_yaw").value)
        self.nudge_max_xy  = float(gp("nudge.max_xy_step").value)
        self.nudge_max_yaw = math.radians(float(gp("nudge.max_yaw_step").value))  # deg->rad

        self.odom_alpha    = float(gp("odom.smooth_alpha").value)

        # ---- state ----
        np.random.seed(self.seed)
        self.particles: List[Particle] = [
            Particle(delta=np.zeros(3, float), weight=1.0 / self.Np) for _ in range(self.Np)
        ]
        if self.init_std_xy > 0.0 or self.init_std_yaw > 0.0:
            for p in self.particles:
                p.delta[0] = np.random.normal(0.0, self.init_std_xy)
                p.delta[1] = np.random.normal(0.0, self.init_std_xy)
                p.delta[2] = np.random.normal(0.0, self.init_std_yaw)

        self.delta_hat = np.zeros(3, float)
        self.delta_pub = np.zeros(3, float)   # smoothed-for-publish copy (visual only)
        self.have_pose_feed = False

        # odom buffer: (t, x, y, yaw, v, yawrate)
        self.odom_buf: deque = deque()
        self.pose_latest = np.array([0.0, 0.0, 0.0], float)

        self.last_cone_ts: Optional[float] = None

        # ---- I/O ----
        self.create_subscription(Odometry, self.pose_topic_in, self.cb_odom, self.qos)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic_in, self.cb_cones, self.qos)
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic_out, 10)
        self.pub_map  = self.create_publisher(ConeArrayWithCovariance, self.map_topic_out, 10)
        self.pub_p2d  = self.create_publisher(Pose2D, self.pose2d_topic_out, 10)

        self.get_logger().info(f"[fastslam] up. in:pose={self.pose_topic_in} cones={self.cones_topic_in} | out:odom={self.odom_topic_out} map={self.map_topic_out}")
        self.get_logger().info(f"[fastslam] Np={self.Np} QoS={'BEST_EFFORT' if best_effort else 'RELIABLE'} detections_frame={self.detections_frame}")

    # -------------------- odom buffer helpers --------------------
    def _pose_at(self, t_query: float):
        """Return pose at t_query (x,y,yaw), plus a timing mismatch dt_pose (used for inflation),
           and the v, yawrate at the *anchor* used. Δt semantics:
           - If interpolating between two stamps -> dt_pose = 0.0
           - If extrapolating beyond the newest -> clamp to self.extrap_cap and return that clamped dt
           - If before the first -> dt_pose = 0.0 (we don't inflate for unknown history)
        """
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf'), 0.0, 0.0

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        # Before first: no inflation
        if idx == 0:
            t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[0]
            return np.array([x0, y0, yaw0], float), 0.0, v0, yr0

        # After last: clamp extrapolation
        if idx >= len(self.odom_buf):
            t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[-1]
            dt = t_query - t1
            dt_c = max(0.0, min(dt, self.extrap_cap))
            if abs(yr1) < 1e-3:
                x = x1 + v1 * dt_c * math.cos(yaw1)
                y = y1 + v1 * dt_c * math.sin(yaw1)
                yaw = wrap(yaw1)
            else:
                x = x1 + (v1 / yr1) * (math.sin(yaw1 + yr1 * dt_c) - math.sin(yaw1))
                y = y1 - (v1 / yr1) * (math.cos(yaw1 + yr1 * dt_c) - math.cos(yaw1))
                yaw = wrap(yaw1 + yr1 * dt_c)
            return np.array([x, y, yaw], float), abs(dt_c), v1, yr1

        # Interpolation at exact query -> zero mismatch
        t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[idx]
        if t1 == t0:
            return np.array([x0, y0, yaw0], float), 0.0, v0, yr0
        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a * dyaw)
        v = (1 - a) * v0 + a * v1
        yr = (1 - a) * yr0 + a * yr1
        return np.array([x, y, yaw], float), 0.0, v, yr

    # -------------------- cap observations to bound JCBB time --------------------
    def _cap_observations(self, obs: List[Tuple[np.ndarray, np.ndarray, str]]) \
            -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        obs: list of (pb, Sb, color) in BODY frame.
        Keep nearest-by-range per color up to jcbb.max_obs_per_color, then
        globally cap to jcbb.max_obs_total by nearest range.
        """
        if not obs:
            return obs

        def _r(o): return float(np.linalg.norm(o[0]))

        by_color = {"blue": [], "yellow": [], "orange": [], "big": []}
        for o in obs:
            col = o[2]
            if col not in by_color:
                by_color[col] = []
            by_color[col].append(o)

        capped = []
        per_col = max(1, self.max_obs_per_col)
        for col, arr in by_color.items():
            if not arr:
                continue
            arr.sort(key=_r)
            capped.extend(arr[:per_col])

        total_cap = max(1, self.max_obs_total)
        if len(capped) <= total_cap:
            return capped
        capped.sort(key=_r)
        return capped[:total_cap]

    # -------------------- Odometry IN → publish corrected odom at same rate --------------------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v  = math.hypot(vx, vy)
        yr = float(msg.twist.twist.angular.z)

        self.pose_latest = np.array([x, y, yaw], float)
        self.odom_buf.append((t, x, y, yaw, v, yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

        if not self.have_pose_feed:
            self.have_pose_feed = True
            self.get_logger().info("[fastslam] proposal feed ready.")

        dx, dy, dth = self.delta_pub  # publish smoothed (visual) delta if enabled
        yaw_c = wrap(yaw + dth)
        x_c = x + dx
        y_c = y + dy

        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose = PoseWithCovariance()
        od.twist = TwistWithCovariance()
        od.pose.pose = Pose(position=Point(x=x_c, y=y_c, z=0.0), orientation=quat_from_yaw(yaw_c))
        od.twist.twist = Twist(
            linear=Vector3(x=v * math.cos(yaw_c), y=v * math.sin(yaw_c), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=yr)
        )
        self.pub_odom.publish(od)

        p2 = Pose2D(); p2.x = x_c; p2.y = y_c; p2.theta = yaw_c
        self.pub_p2d.publish(p2)

        self._publish_map(t)

    # -------------------- Cones IN → per-particle JCBB + EKF map update --------------------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.have_pose_feed:
            return

        t_frame_start = time.perf_counter()

        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_cone_ts is not None:
            if (t_z - self.last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        self.last_cone_ts = t_z

        # Drop stale frames to avoid backlog-induced snaps
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if (now_s - t_z) > self.drop_stale_s:
            return

        # Build obs in BODY frame (pb, Sb, color). Convert to world per particle.
        obs: List[Tuple[np.ndarray, np.ndarray, str]] = []

        odom_pose, dt_pose, v, yawrate = self._pose_at(t_z)
        x_o, y_o, yaw_o = odom_pose
        Rwb = rot2d(yaw_o); Rbw = Rwb.T

        def add(cones, color: str):
            for c in cones:
                px = float(c.point.x); py = float(c.point.y)
                Sraw = np.array([[c.covariance[0], c.covariance[1]],
                                 [c.covariance[2], c.covariance[3]]], float)
                # BODY noise floor
                Sraw = Sraw + np.diag([self.sigma0**2, self.sigma0**2])
                if self.detections_frame == "base":
                    pb = np.array([px, py], float); Sb = Sraw
                else:
                    pw = np.array([px, py], float)
                    pb = Rbw @ (pw - np.array([x_o, y_o]))
                    Sb = Rbw @ Sraw @ Rbw.T
                obs.append((pb, Sb, color))

        add(msg.blue_cones, "blue")
        add(msg.yellow_cones, "yellow")
        add(msg.orange_cones, "orange")
        add(msg.big_orange_cones, "big")
        if not obs:
            return

        # NEW: enforce configured caps to bound JCBB time
        obs = self._cap_observations(obs)

        # PF predict on Δ
        self._pf_predict()

        # --------- Adaptive chi² gate for the frame ---------
        chi2_gate = self.chi2_gate_base
        if self.adapt_gate:
            drift_pos = math.hypot(self.delta_hat[0], self.delta_hat[1])
            drift_yaw = abs(self.delta_hat[2])
            scale = 1.0 + self.gate_k_pos * drift_pos + self.gate_k_yaw * drift_yaw
            scale = min(max(scale, self.gate_min), self.gate_max)
            chi2_gate *= scale

        # --------- Per-frame JCBB budget sharing (keeps proc time bounded) ---------
        # Distribute total budget across particles; clamp per-particle slice to [1ms, 15ms]
        pp_budget = self.jcbb_budget_s / float(min(self.Np, 64))
        pp_budget = max(0.001, min(pp_budget, 0.015))

        new_w = np.zeros(self.Np, float)

        for i, p in enumerate(self.particles):
            dx, dy, dth = p.delta
            yaw_c = wrap(yaw_o + dth)
            Rc = rot2d(yaw_c)
            pc = np.array([x_o + dx, y_o + dy], float)

            # ---------- build candidates (color-strict), proper gating with S = P + R ----------
            cands: List[List[Tuple[int, float]]] = []
            total_cands = 0
            gate_eff = chi2_gate
            J = np.array([[0.0, -1.0],
                          [1.0,  0.0]])  # 90° rot

            for (pb, Sb, color) in obs:
                pw_pred = pc + Rc @ pb

                # measurement covariance in world
                S0_w = Rc @ Sb @ Rc.T

                # kinematic inflations (correct yaw shape)
                trans_var = self.alpha * (v ** 2) * (dt_pose ** 2)
                sigma_theta2 = self.beta * ((yawrate * dt_pose) ** 2)
                u = Rc @ (J @ pb)  # direction perpendicular to measurement ray (world)
                S_yaw = sigma_theta2 * np.outer(u, u)
                S_trans = np.eye(2) * trans_var
                R = S0_w + S_trans + S_yaw
                R[0,0] += 1e-9; R[1,1] += 1e-9

                # coarse crop
                near_idx = []
                if p.lms:
                    for j, lm in enumerate(p.lms):
                        if lm.color != color:
                            continue
                        if (abs(lm.mean[0] - pw_pred[0]) > self.crop_radius_m) or (abs(lm.mean[1] - pw_pred[1]) > self.crop_radius_m):
                            continue
                        if (lm.mean - pw_pred) @ (lm.mean - pw_pred) > (self.crop_radius_m ** 2):
                            continue
                        near_idx.append(j)

                lst = []
                # IMPORTANT: gate using S = P + R (NOT R alone)
                for j in near_idx:
                    lm = p.lms[j]
                    innov = pw_pred - lm.mean
                    S_gate = lm.cov + R
                    try:
                        Sinv = np.linalg.inv(S_gate)
                    except np.linalg.LinAlgError:
                        Sinv = np.linalg.pinv(S_gate)
                    m2 = float(innov.T @ Sinv @ innov)
                    if m2 <= gate_eff:
                        lst.append((j, m2))

                if lst:
                    lst.sort(key=lambda t: t[1])
                    if self.cand_per_obs > 0:
                        lst = lst[:self.cand_per_obs]
                total_cands += len(lst)
                cands.append(lst)

            # Fine adapt & optional pruning to make it bite without rebuilds
            avg_c = (total_cands / max(1, len(obs)))
            if self.adapt_gate:
                if avg_c > (self.gate_target * 1.5):
                    gate_eff = max(self.gate_min * self.chi2_gate_base, gate_eff * 0.8)
                    # prune tails
                    keep = max(1, int(math.ceil(self.gate_target)))
                    for k in range(len(cands)):
                        if cands[k]:
                            cands[k] = cands[k][:keep]
                elif avg_c < 0.5:
                    gate_eff = min(self.gate_max * self.chi2_gate_base, gate_eff * 1.25)

            # ---------- JCBB with per-particle timeout (shared per-frame) ----------
            order = sorted(range(len(obs)), key=lambda k: len(cands[k]) if cands[k] else 9999)
            cands_ord = [cands[k] for k in order]
            best_pairs: List[Tuple[int, int, float]] = []
            best_K = 0
            used = set()

            t_deadline = time.perf_counter() + pp_budget

            def dfs(idx, cur_pairs, cur_sum):
                nonlocal best_pairs, best_K
                if time.perf_counter() > t_deadline:
                    return
                cur_K = len(cur_pairs)
                if cur_K + (len(cands_ord) - idx) < best_K:
                    return
                if cur_K >= self.min_k_for_joint:
                    df = 2 * cur_K
                    thresh = self.joint_relax * chi2_quantile(df, self.joint_sig)
                    if cur_sum > thresh:
                        return
                if idx == len(cands_ord):
                    if cur_K > best_K:
                        best_K = cur_K
                        best_pairs = cur_pairs.copy()
                    return
                # skip
                dfs(idx + 1, cur_pairs, cur_sum)
                # take
                for (lmj, m2) in cands_ord[idx]:
                    if lmj in used:
                        continue
                    used.add(lmj)
                    cur_pairs.append((order[idx], lmj, m2))
                    dfs(idx + 1, cur_pairs, cur_sum + m2)
                    cur_pairs.pop()
                    used.remove(lmj)

            dfs(0, [], 0.0)

            # ---------- Fallback: greedy NN if nothing ----------
            if len(best_pairs) == 0:
                used_lm = set()
                for oi in order:
                    if not cands[oi]:
                        continue
                    lmj, m2 = min(cands[oi], key=lambda t: t[1])
                    if lmj in used_lm:
                        continue
                    if m2 <= gate_eff:
                        best_pairs.append((oi, lmj, m2))
                        used_lm.add(lmj)

            # ---------- Likelihood ----------
            m2_sum = float(sum(m2 for (_, _, m2) in best_pairs))
            K = len(best_pairs)
            df = 2 * max(1, K)
            gate_joint = self.joint_relax * chi2_quantile(df, self.joint_sig)
            have_valid = (K >= max(2, self.min_k_for_joint) and m2_sum <= gate_joint)
            like = self.like_floor if not have_valid else max(math.exp(-0.5 * m2_sum), self.like_floor)
            if K == 1 and m2_sum <= gate_eff:  # allow small weight from singletons
                like = max(math.exp(-0.5 * m2_sum), self.like_floor)
            new_w[i] = p.weight * like

            # ---------- Conf decay for unmatched ----------
            decay = math.exp(-self.decay_hz * max(0.0, float(dt_pose)))
            matched_lm_ids = {lmj for (_, lmj, _) in best_pairs}
            for lj, lm in enumerate(p.lms):
                if lj not in matched_lm_ids:
                    lm.conf *= decay

            # ---------- EKF updates + promotion/retention ----------
            innovations_world = []   # for drift nudge
            yaw_terms = []           # (Rc@pb, innov)
            for (oi, lmj, _) in best_pairs:
                pb, Sb, color = obs[oi]
                pw_pred = pc + Rc @ pb
                lm = p.lms[lmj]

                # measurement covariance (consistent with gating)
                S0_w = Rc @ Sb @ Rc.T
                trans_var = self.alpha * (v ** 2) * (dt_pose ** 2)
                sigma_theta2 = self.beta * ((yawrate * dt_pose) ** 2)
                u = Rc @ (J @ pb)
                S_yaw = sigma_theta2 * np.outer(u, u)
                S_trans = np.eye(2) * trans_var
                R = S0_w + S_trans + S_yaw
                R[0,0] += 1e-9; R[1,1] += 1e-9

                P = lm.cov
                S = P + R
                try: Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError: Sinv = np.linalg.pinv(S)
                Kmat = P @ Sinv
                innov = pw_pred - lm.mean
                lm.mean = lm.mean + Kmat @ innov
                lm.cov  = (np.eye(2) - Kmat) @ P

                # collect for nudging
                innovations_world.append(innov)
                yaw_terms.append((Rc @ pb, innov))

                lm.conf = (1.0 - self.conf_beta) * lm.conf + self.conf_beta * 1.0
                if (lm.last_hit_t < 0.0) or ((t_z - lm.last_hit_t) >= self.prom_dt_min):
                    lm.distinct_hits += 1
                    lm.last_hit_t = t_z
                lm.last_seen_t = t_z

                if (not lm.promoted) and (lm.distinct_hits >= self.prom_k):
                    lm.promoted = True

                if (not lm.retained and lm.promoted and
                    lm.conf >= self.retain_thr and
                    lm.distinct_hits >= self.retain_hits and
                    float(np.trace(lm.cov)) <= self.retain_cov_tr):
                    lm.retained = True

            # ---------- births / merges for unmatched obs ----------
            matched_obs_idx = set([oi for (oi, _, _) in best_pairs])
            for oi, (pb, Sb, color) in enumerate(obs):
                if oi in matched_obs_idx:
                    continue
                pw_pred = pc + Rc @ pb

                # measurement covariance (same as above)
                S0_w = Rc @ Sb @ Rc.T
                trans_var = self.alpha * (v ** 2) * (dt_pose ** 2)
                sigma_theta2 = self.beta * ((yawrate * dt_pose) ** 2)
                u = Rc @ (J @ pb)
                S_yaw = sigma_theta2 * np.outer(u, u)
                S_trans = np.eye(2) * trans_var
                R = S0_w + S_trans + S_yaw
                R[0,0] += 1e-9; R[1,1] += 1e-9

                # try merge
                best_j, best_m2 = -1, float("inf")
                for j, lm2 in enumerate(p.lms):
                    if lm2.color != color:
                        continue
                    Ssum = lm2.cov + R
                    try: Sinv = np.linalg.inv(Ssum)
                    except np.linalg.LinAlgError: Sinv = np.linalg.pinv(Ssum)
                    m2 = float((pw_pred - lm2.mean).T @ Sinv @ (pw_pred - lm2.mean))
                    if m2 < best_m2:
                        best_m2, best_j = m2, j

                if best_m2 <= self.merge_m2_thr and best_j >= 0:
                    lm2 = p.lms[best_j]
                    Kb = lm2.cov @ np.linalg.pinv(lm2.cov + R)
                    lm2.mean = lm2.mean + Kb @ (pw_pred - lm2.mean)
                    lm2.cov  = (np.eye(2) - Kb) @ lm2.cov
                    lm2.last_seen_t = t_z
                    if (lm2.last_hit_t < 0.0) or ((t_z - lm2.last_hit_t) >= self.prom_dt_min):
                        lm2.distinct_hits += 1
                        lm2.last_hit_t = t_z
                    lm2.conf = (1.0 - self.conf_beta) * lm2.conf + self.conf_beta * 1.0
                    if (not lm2.promoted) and lm2.distinct_hits >= self.prom_k:
                        lm2.promoted = True
                    if (not lm2.retained and lm2.promoted and
                        lm2.conf >= self.retain_thr and
                        lm2.distinct_hits >= self.retain_hits and
                        float(np.trace(lm2.cov)) <= self.retain_cov_tr):
                        lm2.retained = True
                else:
                    # birth: use R (full measurement noise) as initial covariance; not just S0_w
                    p.lms.append(Landmark(mean=pw_pred.copy(),
                                          cov=R.copy(),
                                          color=color,
                                          promoted=False,
                                          last_seen_t=t_z,
                                          last_hit_t=t_z,
                                          distinct_hits=1,
                                          conf=self.conf_beta,
                                          retained=False))

            # ---------- drift nudge on Δ (small, bounded) – only when association is strong ----------
            if self.nudge_enable:
                if have_valid and len(innovations_world) >= 2:
                    innov_mean = np.mean(np.vstack(innovations_world), axis=0)  # world xy
                    step_xy = -self.nudge_xy * innov_mean
                    # yaw step via cross((Rc@pb), innov) / ||pb||^2 averaged
                    yaw_accum = 0.0; denom = 0.0
                    for pwb, inn in yaw_terms:
                        r2 = float(np.dot(pwb, pwb)) + 1e-6
                        # For small angles, innovation ~ dθ * [ -y_b, x_b ], use z-cross:
                        yaw_accum += (pwb[0]*inn[1] - pwb[1]*inn[0]) / r2
                        denom += 1.0
                    step_yaw = -self.nudge_yaw * (yaw_accum / max(1.0, denom))

                    # clamp
                    nrm = float(np.linalg.norm(step_xy))
                    if nrm > self.nudge_max_xy:
                        step_xy *= (self.nudge_max_xy / nrm)
                    if abs(step_yaw) > self.nudge_max_yaw:
                        step_yaw = math.copysign(self.nudge_max_yaw, step_yaw)

                    p.delta[0] += float(step_xy[0])
                    p.delta[1] += float(step_xy[1])
                    p.delta[2] = wrap(p.delta[2] + float(step_yaw))

            # ---------- deletion ----------
            keep = []
            for lm in p.lms:
                if lm.retained:
                    keep.append(lm)
                    continue
                if (t_z - lm.last_seen_t) > self.max_unseen_s:
                    continue
                if lm.conf < self.prune_thr:
                    continue
                if float(np.trace(lm.cov)) > self.del_covtr:
                    continue
                keep.append(lm)
            p.lms = keep

        # normalize + resample + Δ̂
        sw = float(np.sum(new_w))
        if sw <= 0.0 or not math.isfinite(sw):
            for p in self.particles:
                p.weight = 1.0 / self.Np
        else:
            for i, p in enumerate(self.particles):
                p.weight = new_w[i] / sw

        neff = 1.0 / float(np.sum([p.weight * p.weight for p in self.particles]))
        if neff < self.neff_ratio * self.Np:
            self._systematic_resample()

        self.delta_hat = self._estimate_delta()

        # Visual-only smoothing for published odom
        if 0.0 < self.odom_alpha < 1.0:
            a = self.odom_alpha
            self.delta_pub[0] = (1-a)*self.delta_pub[0] + a*self.delta_hat[0]
            self.delta_pub[1] = (1-a)*self.delta_pub[1] + a*self.delta_hat[1]
            cs = (1-a)*math.cos(self.delta_pub[2]) + a*math.cos(self.delta_hat[2])
            sn = (1-a)*math.sin(self.delta_pub[2]) + a*math.sin(self.delta_hat[2])
            self.delta_pub[2] = math.atan2(sn, cs)
        else:
            self.delta_pub = self.delta_hat.copy()

        proc_ms = int((time.perf_counter() - t_frame_start) * 1000)
        if self.log_jcbb:
            self.get_logger().info(
                f"[fastslam/jcbb] t={t_z:7.2f}s obs={len(obs)} proc={proc_ms}ms "
                f"neff={neff:.1f} Δ̂=[{self.delta_hat[0]:+.3f},{self.delta_hat[1]:+.3f},{math.degrees(self.delta_hat[2]):+.1f}°] "
                f"pp_budget={pp_budget*1000:.1f}ms"
            )

    # -------------------- PF core over Δ --------------------
    def _pf_predict(self):
        if self.p_std_xy > 0.0:
            noise_xy = np.random.normal(0.0, self.p_std_xy, size=(self.Np, 2))
        else:
            noise_xy = np.zeros((self.Np, 2))
        if self.p_std_yaw > 0.0:
            noise_y = np.random.normal(0.0, self.p_std_yaw, size=(self.Np,))
        else:
            noise_y = np.zeros((self.Np,))
        for i, p in enumerate(self.particles):
            p.delta[0] += noise_xy[i, 0]
            p.delta[1] += noise_xy[i, 1]
            p.delta[2] = wrap(p.delta[2] + noise_y[i])

    def _systematic_resample(self):
        N = self.Np
        w = np.array([p.weight for p in self.particles], float)
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, dtype=int)
        cs = np.cumsum(w)
        i, j = 0, 0
        while i < N:
            if positions[i] < cs[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        new_particles: List[Particle] = []
        for idx in indexes:
            src = self.particles[idx]
            new_lms = [Landmark(mean=lm.mean.copy(),
                                cov=lm.cov.copy(),
                                color=lm.color,
                                promoted=lm.promoted,
                                last_seen_t=lm.last_seen_t,
                                last_hit_t=lm.last_hit_t,
                                distinct_hits=lm.distinct_hits,
                                conf=lm.conf,
                                retained=lm.retained) for lm in src.lms]
        # NOTE: deepcopy of landmarks above; copy deltas and reset weights
            new_particles.append(Particle(delta=src.delta.copy(), weight=1.0/N, lms=new_lms))
        self.particles = new_particles

    def _estimate_delta(self) -> np.ndarray:
        w = np.array([p.weight for p in self.particles], float)
        dx = float(np.sum(w * np.array([p.delta[0] for p in self.particles])))
        dy = float(np.sum(w * np.array([p.delta[1] for p in self.particles])))
        ang = np.array([p.delta[2] for p in self.particles], float)
        cs = float(np.sum(w * np.cos(ang)))
        sn = float(np.sum(w * np.sin(ang)))
        dth = math.atan2(sn, cs)
        return np.array([dx, dy, dth], float)

    # -------------------- Publish map from best particle --------------------
    def _publish_map(self, t_now: float):
        if not self.particles:
            return
        idx_best = int(np.argmax([p.weight for p in self.particles]))
        p = self.particles[idx_best]

        out = ConeArrayWithCovariance()
        out.header.stamp = rclpy.time.Time(seconds=t_now).to_msg()
        out.header.frame_id = "map"

        for lm in p.lms:
            # publish retained OR promoted, OR early if minimally supported
            if not (lm.retained or lm.promoted or lm.conf >= self.map_pub_conf or lm.distinct_hits >= self.map_pub_hits):
                continue
            c = ConeWithCovariance()
            c.point.x = float(lm.mean[0]); c.point.y = float(lm.mean[1]); c.point.z = 0.0
            c.covariance = [float(lm.cov[0,0]), float(lm.cov[0,1]),
                            float(lm.cov[1,0]), float(lm.cov[1,1])]
            if lm.color == "blue":
                out.blue_cones.append(c)
            elif lm.color == "yellow":
                out.yellow_cones.append(c)
            elif lm.color == "orange":
                out.orange_cones.append(c)
            elif lm.color == "big":
                out.big_orange_cones.append(c)

        self.pub_map.publish(out)


# --------------------- main ---------------------
def main():
    rclpy.init()
    node = FastSLAMCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
