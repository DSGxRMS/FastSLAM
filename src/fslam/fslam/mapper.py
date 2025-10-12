#!/usr/bin/env python3
# fslam_mapper.py — minimal online cone mapper + light loop-closure (big_orange gate)
#
# Subscribes:
#   - /odometry_integration/car_state (nav_msgs/Odometry)
#   - /ground_truth/cones (eufs_msgs/ConeArrayWithCovariance)
#
# Publishes:
#   - /mapper/known_map (eufs_msgs/ConeArrayWithCovariance)
#
# Loop-closure will ONLY trigger if:
#   - car is physically back near the start gate (pos within loop.close_radius_m),
#   - car is effectively stopped (|v| <= loop.stop_speed_mps),
#   - a big_orange pair near the anchor centroid is seen,
#   - the observed gate width matches the anchor width within loop.sep_tol_m.

import math
from typing import List, Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance

# ---------- helpers ----------
def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return  float('inf')
    a1=-3.969683028665376e+01; a2= 2.209460984245205e+02; a3=-2.759285104469687e+02
    a4= 1.383577518672690e+02; a5=-3.066479806614716e+01; a6= 2.506628277459239e+00
    b1=-5.447609879822406e+01; b2= 1.615858368580409e+02; b3=-1.556989798598866e+02
    b4= 6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00; d4= 3.754408661907416e+00
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p>phigh:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q=p-0.5; r=q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)

def add_cone(arr: ConeArrayWithCovariance, x: float, y: float, P: np.ndarray, color: str):
    a = float(P[0,0]); d = float(P[1,1]); b = float(P[0,1]); c = float(P[1,0])
    pt = ConeWithCovariance()
    pt.point.x = x; pt.point.y = y; pt.point.z = 0.0
    pt.covariance = [a, b, c, d]
    cl = (color or "").lower().strip()
    if ("big" in cl) and ("orange" in cl):
        arr.big_orange_cones.append(pt)
    elif "orange" in cl or cl.startswith("o"):
        arr.orange_cones.append(pt)
    elif "yellow" in cl or cl.startswith("y"):
        arr.yellow_cones.append(pt)
    else:
        arr.blue_cones.append(pt)

# ---------- node ----------
class OnlineMapper(Node):
    def __init__(self):
        super().__init__("fslam_online_mapper", automatically_declare_parameters_from_overrides=True)

        # I/O topics
        self.declare_parameter("topics.pose_in", "/odometry_integration/car_state")
        self.declare_parameter("topics.cones_in", "/ground_truth/cones")
        self.declare_parameter("topics.map_out", "/mapper/known_map")

        # Frames & DA
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("da.gate_p", 0.99)               # χ² gate (2 dof)
        self.declare_parameter("da.crop_radius_m", 25.0)
        self.declare_parameter("da.use_color", True)
        self.declare_parameter("da.min_separation_m", 0.0)

        # Perception throttle (Hz). 0 => no throttle.
        self.declare_parameter("da.target_cone_hz", 0.0)

        # Measurement handling
        self.declare_parameter("meas.sigma_floor_xy", 0.35)
        self.declare_parameter("meas.inflate_xy", 0.05)
        self.declare_parameter("init.landmark_var", 0.30)

        # --- Loop-closure (simple gate-based) ---
        self.declare_parameter("loop.enable", True)
        self.declare_parameter("loop.min_gate_sep_m", 2.0)
        self.declare_parameter("loop.max_gate_sep_m", 8.0)
        self.declare_parameter("loop.close_radius_m", 10.0)     # car & gate centroid must be within this
        self.declare_parameter("loop.sep_tol_m", 0.6)
        self.declare_parameter("loop.min_lm_before_close", 30)
        self.declare_parameter("loop.cooldown_s", 5.0)
        self.declare_parameter("loop.require_stop", True)        # NEW: require near-zero speed
        self.declare_parameter("loop.stop_speed_mps", 0.5)       # NEW: ≤ 0.5 m/s counts as “stopped”

        # Read params
        gp = self.get_parameter
        self.pose_topic = str(gp("topics.pose_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)
        self.map_topic = str(gp("topics.map_out").value)
        self.det_frame = str(gp("detections_frame").value).lower()

        self.gate_p = float(gp("da.gate_p").value)
        self.crop_r2 = float(gp("da.crop_radius_m").value) ** 2
        self.use_color = bool(gp("da.use_color").value)
        self.min_sep = float(gp("da.min_separation_m").value)
        self.target_cone_hz = float(gp("da.target_cone_hz").value)

        self.sigma_floor = float(gp("meas.sigma_floor_xy").value)
        self.inflate_xy = float(gp("meas.inflate_xy").value)
        self.init_lm_var = float(gp("init.landmark_var").value)

        self.loop_enable = bool(gp("loop.enable").value)
        self.loop_min_sep = float(gp("loop.min_gate_sep_m").value)
        self.loop_max_sep = float(gp("loop.max_gate_sep_m").value)
        self.loop_close_r = float(gp("loop.close_radius_m").value)
        self.loop_sep_tol = float(gp("loop.sep_tol_m").value)
        self.loop_min_lm = int(gp("loop.min_lm_before_close").value)
        self.loop_cooldown = float(gp("loop.cooldown_s").value)
        self.loop_require_stop = bool(gp("loop.require_stop").value)
        self.loop_stop_speed = float(gp("loop.stop_speed_mps").value)

        self.chi2_gate = chi2_quantile(2, self.gate_p)

        # State
        self.pose_ready = False
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.speed = 0.0  # NEW: from odom twist
        self._last_cone_ts = None

        # Landmarks
        self.landmarks: List[dict] = []

        # Loop-closure state
        self.anchor_gate: Optional[Tuple[np.ndarray,np.ndarray]] = None   # (p1, p2)
        self.anchor_sep: Optional[float] = None
        self.anchor_centroid: Optional[np.ndarray] = None
        self.loop_frozen = False
        self._last_loop_event_ts = -1e9

        # ROS I/O
        q_rel = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        q_fast = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry, self.pose_topic, self.cb_pose, q_rel)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_fast)
        self.pub_map = self.create_publisher(ConeArrayWithCovariance, self.map_topic, 10)

        self.get_logger().info(
            f"[mapper] pose_in={self.pose_topic} cones={self.cones_topic} map_out={self.map_topic}")
        self.get_logger().info(
            f"[mapper] frame={self.det_frame} | χ²(2,{self.gate_p})={self.chi2_gate:.2f} | crop={math.sqrt(self.crop_r2):.1f}m | "
            f"colorDA={self.use_color} | target_cone_hz={self.target_cone_hz:.1f} | loop_enable={self.loop_enable}"
        )

    # --- pose feed (latest only) ---
    def cb_pose(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        # speed from twist (m/s)
        vx = float(msg.twist.twist.linear.x) if msg.twist and msg.twist.twist else 0.0
        vy = float(msg.twist.twist.linear.y) if msg.twist and msg.twist.twist else 0.0
        self.speed = math.hypot(vx, vy)
        if not self.pose_ready:
            self.pose_ready = True
            self.get_logger().info("[mapper] predictor pose feed ready.")

    # --- cones feed ---
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.pose_ready:
            return

        tz = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        # Throttle: process at ~target_cone_hz (0 = no throttle)
        if self.target_cone_hz > 0.0 and self._last_cone_ts is not None:
            min_dt = 1.0 / max(1e-6, self.target_cone_hz)
            if (tz - self._last_cone_ts) < min_dt:
                return
        self._last_cone_ts = tz

        # If map already frozen by loop-closure: just republish and return (cheap)
        if self.loop_frozen:
            self._publish_map()
            return

        Rwb = rot2d(self.yaw); pos = np.array([self.x, self.y], float)

        # Build detection list in WORLD
        dets = []  # (pw, Sw, color)
        big_orange_world = []  # store separately for loop-closure
        def add(arr, color_hint):
            nonlocal big_orange_world
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma_floor**2, self.sigma_floor**2]) + np.eye(2)*self.inflate_xy
                if self.det_frame == "base":
                    pb = np.array([px, py], float)
                    pw = pos + (Rwb @ pb)
                    Sw = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px, py], float)
                    Sw = Pw
                dets.append((pw, Sw, color_hint))
                if color_hint == "big_orange":
                    big_orange_world.append(pw)

        add(msg.blue_cones, "blue")
        add(msg.yellow_cones, "yellow")
        add(msg.orange_cones, "orange")
        add(msg.big_orange_cones, "big_orange")

        # --- Light loop-closure check (uses only big_orange cones) ---
        if self.loop_enable and len(big_orange_world) >= 2:
            self._maybe_loop_close(tz, big_orange_world)

        # continue normal DA / mapping (unless frozen just now)
        if self.loop_frozen:
            self._publish_map()
            return

        LM = np.array([lm['m'] for lm in self.landmarks], float) if self.landmarks else np.zeros((0,2), float)

        upd = 0; new = 0
        for (pw, Sw, col) in dets:
            match = -1
            best_m2 = 1e18
            nearest_euclid2 = 1e18

            if LM.shape[0] > 0:
                diffs = LM - pw[None,:]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                near = np.where(d2 <= self.crop_r2)[0]
                if self.use_color:
                    near = np.array([j for j in near if self._color_ok(self.landmarks[j]['color'], col)], dtype=int)
                if near.size > 0:
                    nearest_euclid2 = float(np.min(d2[near]))
                    for j in near:
                        P = self.landmarks[j]['P']
                        S = P + Sw
                        try: S_inv = np.linalg.inv(S)
                        except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S)
                        innov = pw - self.landmarks[j]['m']
                        m2 = float(innov.T @ S_inv @ innov)
                        if m2 < best_m2 and m2 <= self.chi2_gate:
                            best_m2 = m2; match = int(j)

            create = (match < 0)
            if not create and self.min_sep > 0.0 and nearest_euclid2 > (self.min_sep**2):
                create = True

            if create:
                P0 = Sw + np.eye(2)*self.init_lm_var
                self.landmarks.append({'m': pw.copy(), 'P': P0, 'color': col, 'hits': 1, 'last_seen': tz})
                new += 1
                LM = np.vstack([LM, pw[None,:]]) if LM.size else pw.reshape(1,2)
            else:
                lm = self.landmarks[match]
                m = lm['m']; P = lm['P']
                S = P + Sw
                try: S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S)
                K = P @ S_inv
                innov = pw - m
                m_new = m + K @ innov
                P_new = (np.eye(2) - K) @ P
                lm['m'] = m_new
                lm['P'] = 0.5*(P_new + P_new.T)
                lm['hits'] += 1
                lm['last_seen'] = tz
                upd += 1
                LM[match,:] = m_new

        self._publish_map()
        self.get_logger().info(
            f"[mapper] t={tz:7.2f}s det={len(dets)} upd={upd} new={new} LM={len(self.landmarks)}"
            + (f" | throttled to ~{self.target_cone_hz:.1f} Hz" if self.target_cone_hz > 0.0 else "")
        )

    # -------- loop-closure utilities --------
    def _maybe_loop_close(self, tz: float, gate_pts: List[np.ndarray]):
        """
        gate_pts: WORLD positions of currently seen big_orange cones
        """
        # Build candidate big_orange pairs with plausible separation
        cand_pairs: List[Tuple[np.ndarray,np.ndarray,float,np.ndarray]] = []
        n = len(gate_pts)
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = gate_pts[i], gate_pts[j]
                d = float(np.linalg.norm(p2 - p1))
                if self.loop_min_sep <= d <= self.loop_max_sep:
                    c = 0.5*(p1 + p2)  # centroid
                    cand_pairs.append((p1, p2, d, c))
        if not cand_pairs:
            return

        # No anchor yet? Set from widest plausible pair.
        if self.anchor_gate is None and len(self.landmarks) >= 2:
            p1, p2, d, c = max(cand_pairs, key=lambda t: t[2])
            self.anchor_gate = (p1.copy(), p2.copy())
            self.anchor_sep = d
            self.anchor_centroid = c.copy()
            self._last_loop_event_ts = tz
            self.get_logger().info(f"[loop] Anchor gate set: sep={d:.2f} m @ {self.anchor_centroid.round(2).tolist()}")
            return

        if self.anchor_gate is None:
            return

        # Cooldown
        if (tz - self._last_loop_event_ts) < self.loop_cooldown:
            return

        # Must have a reasonably sized map
        if len(self.landmarks) < self.loop_min_lm:
            return

        # Car must be back near the anchor AND (optionally) stopped
        pos = np.array([self.x, self.y], float)
        if float(np.linalg.norm(pos - self.anchor_centroid)) > self.loop_close_r:
            return
        if self.loop_require_stop and (self.speed > self.loop_stop_speed):
            return

        # Choose candidate pair whose centroid AND separation match the anchor
        best = None; best_cost = 1e9
        for p1, p2, d, c in cand_pairs:
            if float(np.linalg.norm(c - self.anchor_centroid)) > self.loop_close_r:
                continue
            sep_err = abs(d - self.anchor_sep)
            if sep_err <= self.loop_sep_tol and sep_err < best_cost:
                best = (p1, p2); best_cost = sep_err
        if best is None:
            return

        # Compute SE(2) transform CURRENT gate -> ANCHOR gate
        (a1, a2) = self.anchor_gate
        (b1, b2) = best
        T_R, T_t = self._se2_from_pairs(src1=b1, src2=b2, dst1=a1, dst2=a2)

        # Apply to all landmarks (positions and covariances)
        for lm in self.landmarks:
            m = lm['m']
            lm['m'] = (T_R @ m) + T_t
            P = lm['P']
            lm['P'] = T_R @ P @ T_R.T

        self.loop_frozen = True
        self._last_loop_event_ts = tz
        self.get_logger().info(
            f"[loop] CLOSED at start gate (car near & stopped). Applied rigid correction; map frozen. "
            f"Δθ={math.degrees(math.atan2(T_R[1,0], T_R[0,0])):+.2f}°  t={T_t.round(3).tolist()}"
        )

    @staticmethod
    def _se2_from_pairs(src1: np.ndarray, src2: np.ndarray,
                        dst1: np.ndarray, dst2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given two 2D point correspondences (src->dst) assuming no scale,
        compute rotation R and translation t such that: R*src + t ≈ dst.
        """
        v_src = src2 - src1
        v_dst = dst2 - dst1
        ang_src = math.atan2(v_src[1], v_src[0])
        ang_dst = math.atan2(v_dst[1], v_dst[0])
        dth = wrap(ang_dst - ang_src)
        R = rot2d(dth)
        c_src = 0.5*(src1 + src2)
        c_dst = 0.5*(dst1 + dst2)
        t = c_dst - (R @ c_src)
        return R, t

    # ---- publish ----
    def _publish_map(self):
        if not self.landmarks:
            return
        out = ConeArrayWithCovariance()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        for lm in self.landmarks:
            add_cone(out, float(lm['m'][0]), float(lm['m'][1]), lm['P'], lm['color'])
        self.pub_map.publish(out)

    def _color_ok(self, lm_c: str, obs_c: str) -> bool:
        if not lm_c or not obs_c: return True
        if lm_c == obs_c: return True
        if ("orange" in lm_c) and ("orange" in obs_c): return True
        return False

# ---------- main ----------
def main():
    rclpy.init()
    node = OnlineMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
