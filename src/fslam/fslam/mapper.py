#!/usr/bin/env python3
# fslam_mapper.py — minimal online cone mapper (no CSV, no buffering, loose gates)
#
# Subscribes:
#   - /odometry_integration/car_state (nav_msgs/Odometry)         <-- predictor pose
#   - /ground_truth/cones (eufs_msgs/ConeArrayWithCovariance)     <-- perception (or your detector)
#
# Publishes:
#   - /mapper/known_map (eufs_msgs/ConeArrayWithCovariance)       <-- live landmark map (cones only)
#
# Notes:
#   - No loop closure. No time buffering. Uses latest odom only.
#   - DA: colour-aware, chi-square gate, crop area.
#   - Colours handled explicitly for: blue, yellow, orange, big_orange.
#   - NEW: Perception throttle via da.target_cone_hz (0 = no throttle).

import math
from typing import List
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
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00
    d4= 3.754408661907416e+00
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
    """Route to the correct colour array; ensure big_orange doesn't fall through."""
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
        self.declare_parameter("detections_frame", "base")      # "base" or "world"
        self.declare_parameter("da.gate_p", 0.99)               # χ² gate (2 dof)
        self.declare_parameter("da.crop_radius_m", 25.0)
        self.declare_parameter("da.use_color", True)
        self.declare_parameter("da.min_separation_m", 0.0)

        # NEW: perception throttle (Hz). 0 => no throttle.
        self.declare_parameter("da.target_cone_hz", 0.0)

        # Measurement handling
        self.declare_parameter("meas.sigma_floor_xy", 0.35)
        self.declare_parameter("meas.inflate_xy", 0.05)
        self.declare_parameter("init.landmark_var", 0.30)

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

        self.chi2_gate = chi2_quantile(2, self.gate_p)

        # State: latest pose; last processed cone timestamp (for throttle)
        self.pose_ready = False
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self._last_cone_ts = None  # seconds (float)

        # Landmarks
        self.landmarks: List[dict] = []

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
            f"colorDA={self.use_color} | target_cone_hz={self.target_cone_hz:.1f}"
        )

    # --- pose feed (latest only) ---
    def cb_pose(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
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

        Rwb = rot2d(self.yaw); pos = np.array([self.x, self.y], float)

        # Build detection list in WORLD
        dets = []  # (pw, Sw, color)
        def add(arr, color_hint):
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

        add(msg.blue_cones, "blue")
        add(msg.yellow_cones, "yellow")
        add(msg.orange_cones, "orange")
        add(msg.big_orange_cones, "big_orange")

        if not dets:
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

        # Publish immediately (no timers, no files)
        out = ConeArrayWithCovariance()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        for lm in self.landmarks:
            add_cone(out, float(lm['m'][0]), float(lm['m'][1]), lm['P'], lm['color'])
        self.pub_map.publish(out)

        self.get_logger().info(
            f"[mapper] t={tz:7.2f}s det={len(dets)} upd={upd} new={new} LM={len(self.landmarks)}"
            + (f" | throttled to ~{self.target_cone_hz:.1f} Hz" if self.target_cone_hz > 0.0 else "")
        )

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
