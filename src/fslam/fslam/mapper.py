#!/usr/bin/env python3
# fslam_mapper.py  (time-aligned, colour-aware, proper gate S=P+Sw)

import math, csv, time, bisect, signal, os
from collections import deque
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance

# -------- helpers --------
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

def cone_array_append(arr: ConeArrayWithCovariance, x: float, y: float, cov_2x2: np.ndarray, color: str):
    a = float(cov_2x2[0,0]); d = float(cov_2x2[1,1]); b = float(cov_2x2[0,1]); c = float(cov_2x2[1,0])
    pt = ConeWithCovariance()
    pt.point.x = x; pt.point.y = y; pt.point.z = 0.0
    pt.covariance = [a,b,c,d]
    color = (color or "").lower()
    if color.startswith("y"): arr.yellow_cones.append(pt)
    elif color.startswith("o") and "big" in color: arr.big_orange_cones.append(pt)
    elif color.startswith("o"): arr.orange_cones.append(pt)
    else: arr.blue_cones.append(pt)

# -------- node --------
class OnlineMapper(Node):
    def __init__(self):
        super().__init__("fslam_online_mapper", automatically_declare_parameters_from_overrides=True)

        # topics
        self.declare_parameter("topics.pose_in", "/odometry_integration/car_state")
        self.declare_parameter("topics.cones_in", "/ground_truth/cones")
        self.declare_parameter("topics.map_out", "/mapper/known_map")

        # frames & DA
        self.declare_parameter("detections_frame", "base")    # "base" or "world"
        self.declare_parameter("da.gate_p", 0.997)            # χ² prob for 2 dof
        self.declare_parameter("da.crop_radius_m", 28.0)
        self.declare_parameter("da.use_color", True)          # restrict candidates by colour
        self.declare_parameter("da.min_separation_m", 1.0)    # if nearest landmark farther than this, prefer new

        # noise handling
        self.declare_parameter("meas.sigma_floor_xy", 0.25)   # added diag
        self.declare_parameter("meas.inflate_xy", 0.0)        # extra diag inflation
        self.declare_parameter("init.landmark_var", 0.20)     # initial var added to Sw for new landmarks

        # publishing & saving
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("save.csv_path", "slam_utils/data/online_map.csv")
        self.declare_parameter("save.every_s", 5.0)

        # read params
        P = self.get_parameter
        self.pose_topic = str(P("topics.pose_in").value)
        self.cones_topic = str(P("topics.cones_in").value)
        self.map_topic  = str(P("topics.map_out").value)
        self.det_frame  = str(P("detections_frame").value).lower()

        self.gate_p     = float(P("da.gate_p").value)
        self.crop_r2    = float(P("da.crop_radius_m").value)**2
        self.use_color  = bool(P("da.use_color").value)
        self.min_sep    = float(P("da.min_separation_m").value)

        self.sigma_floor = float(P("meas.sigma_floor_xy").value)
        self.inflate_xy  = float(P("meas.inflate_xy").value)
        self.init_lm_var = float(P("init.landmark_var").value)

        self.pub_hz     = float(P("publish_hz").value)
        self.csv_path   = Path(str(P("save.csv_path").value)).resolve()
        self.save_every = float(P("save.every_s").value)

        self.chi2_gate = chi2_quantile(2, self.gate_p)

        # state: odom buffer for time alignment
        self.pose_ready = False
        self.odom_buf: deque = deque()  # (t, x, y, yaw)
        self.odom_window = 2.0          # seconds to keep for interpolation
        self.last_save = time.time()

        # landmarks = [{'m': [x,y], 'P': 2x2, 'color': str, 'hits': int, 'last_seen': t}]
        self.landmarks: List[dict] = []

        # ROS I/O
        q_rel = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.RELIABLE,
                           durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)
        q_fast = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry, self.pose_topic, self.cb_pose, q_rel)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_fast)
        self.pub_map = self.create_publisher(ConeArrayWithCovariance, self.map_topic, 10)

        if self.pub_hz > 0.0:
            self.create_timer(max(0.02, 1.0/self.pub_hz), self._publish_map)

        try:
            signal.signal(signal.SIGINT, self._on_sig)
            signal.signal(signal.SIGTERM, self._on_sig)
        except Exception:
            pass

        self.get_logger().info(f"[mapper] pose_in={self.pose_topic} cones={self.cones_topic} map_out={self.map_topic}")
        self.get_logger().info(f"[mapper] frame={self.det_frame} | χ²(2,{self.gate_p})={self.chi2_gate:.3f} | crop={math.sqrt(self.crop_r2):.1f}m | colorDA={self.use_color}")

    # ---- odom feed ----
    def cb_pose(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.odom_buf.append((t, x, y, yaw))
        tmin = t - self.odom_window
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()
        if not self.pose_ready:
            self.pose_ready = True
            self.get_logger().info("[mapper] predictor pose feed ready.")

    def _pose_at(self, tq: float) -> Tuple[np.ndarray, float]:
        if not self.odom_buf:
            return np.array([0.0,0.0,0.0], float), float('inf')
        ts = [it[0] for it in self.odom_buf]
        i = bisect.bisect_left(ts, tq)
        if i == 0:
            t0,x0,y0,yaw0 = self.odom_buf[0]
            return np.array([x0,y0,yaw0], float), abs(tq - t0)
        if i >= len(self.odom_buf):
            t1,x1,y1,yaw1 = self.odom_buf[-1]
            return np.array([x1,y1,yaw1], float), abs(tq - t1)
        t0,x0,y0,yaw0 = self.odom_buf[i-1]
        t1,x1,y1,yaw1 = self.odom_buf[i]
        if t1 == t0:
            return np.array([x0,y0,yaw0], float), abs(tq - t0)
        a = (tq - t0)/(t1 - t0)
        x = x0 + a*(x1 - x0)
        y = y0 + a*(y1 - y0)
        dyaw = wrap(yaw1 - yaw0)
        yaw = wrap(yaw0 + a*dyaw)
        dt_near = min(abs(tq - t0), abs(tq - t1))
        return np.array([x,y,yaw], float), dt_near

    # ---- cones feed ----
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.pose_ready:
            return
        tz = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose = self._pose_at(tz)
        xw, yw, yaw = pose_w_b
        Rwb = rot2d(yaw); pos = np.array([xw, yw], float)

        detections = []  # (pw, Sw, color)
        def add_arr(arr, color_hint: str):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                # meas floor + optional inflation
                Pw = Pw + np.diag([self.sigma_floor**2, self.sigma_floor**2])
                if self.inflate_xy > 0.0:
                    Pw = Pw + np.eye(2)*self.inflate_xy
                if self.det_frame == "base":
                    pb = np.array([px,py], float)
                    pw = pos + (Rwb @ pb)
                    Sw = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    Sw = Pw
                detections.append((pw, Sw, color_hint))

        add_arr(msg.blue_cones, "blue")
        add_arr(msg.yellow_cones, "yellow")
        add_arr(msg.orange_cones, "orange")
        add_arr(msg.big_orange_cones, "big_orange")

        if not detections:
            return

        # Prepare landmark matrix for quick distance crop
        LM = np.array([lm['m'] for lm in self.landmarks], float) if self.landmarks else np.zeros((0,2), float)

        upd = 0; new = 0
        for (pw, Sw, col) in detections:
            match_idx = -1
            best_m2 = 1e18
            nearest_euclid2 = 1e18

            if LM.shape[0] > 0:
                diffs = LM - pw[None,:]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                near = np.where(d2 <= self.crop_r2)[0]

                # colour filter (optional)
                if self.use_color:
                    near = np.array([j for j in near if self._colors_compatible(self.landmarks[j]['color'], col)], dtype=int)

                if near.size > 0:
                    nearest_euclid2 = float(np.min(d2[near]))
                    # proper gate with S = P + Sw
                    for j in near:
                        P = self.landmarks[j]['P']
                        S = P + Sw
                        try: S_inv = np.linalg.inv(S)
                        except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S)
                        innov = pw - self.landmarks[j]['m']
                        m2 = float(innov.T @ S_inv @ innov)
                        if m2 < best_m2 and m2 <= self.chi2_gate:
                            best_m2 = m2; match_idx = int(j)

            # create vs update decision
            create = (match_idx < 0)
            if not create and self.min_sep > 0.0:
                # if it's unusually far, prefer a new landmark (prevents fusing distant cones)
                if nearest_euclid2 > (self.min_sep**2):
                    create = True

            if create:
                P0 = Sw + np.eye(2)*(self.init_lm_var)
                self.landmarks.append({
                    'm': pw.copy(),
                    'P': P0,
                    'color': col,
                    'hits': 1,
                    'last_seen': tz
                })
                new += 1
                LM = np.vstack([LM, pw[None,:]]) if LM.size else pw.reshape(1,2)
            else:
                lm = self.landmarks[match_idx]
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
                LM[match_idx,:] = m_new  # keep cache consistent

        if (upd + new) > 0:
            self.get_logger().info(f"[mapper] t={tz:7.2f}s det={len(detections)} upd={upd} new={new} LM={len(self.landmarks)} dt_pose={dt_pose*1e3:.1f}ms")

        now = time.time()
        if (now - self.last_save) >= self.save_every:
            self._save_csv()
            self.last_save = now

    def _colors_compatible(self, lm_color: str, obs_color: str) -> bool:
        if not lm_color or not obs_color: return True
        # strict equality except allow "big_orange" with "orange"
        if lm_color == obs_color: return True
        if ("orange" in lm_color) and ("orange" in obs_color): return True
        return False

    # ---- publish/save ----
    def _publish_map(self):
        if not self.landmarks:
            return
        msg = ConeArrayWithCovariance()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for lm in self.landmarks:
            cone_array_append(msg, float(lm['m'][0]), float(lm['m'][1]), lm['P'], lm['color'])
        self.pub_map.publish(msg)

    def _save_csv(self):
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x","y","color","var_x","var_y","cov_xy","hits","last_seen_s"])
                for lm in self.landmarks:
                    P = lm['P']
                    w.writerow([
                        f"{lm['m'][0]:.6f}",
                        f"{lm['m'][1]:.6f}",
                        lm['color'],
                        f"{P[0,0]:.6f}",
                        f"{P[1,1]:.6f}",
                        f"{P[0,1]:.6f}",
                        lm['hits'],
                        f"{lm['last_seen']:.3f}"
                    ])
            os.replace(tmp, self.csv_path)
            self.get_logger().info(f"[mapper] map saved: {self.csv_path} ({len(self.landmarks)} landmarks)")
        except Exception as e:
            self.get_logger().warn(f"[mapper] save failed: {e}")

    def _on_sig(self, *_):
        self.get_logger().info("[mapper] signal caught, saving map...")
        try: self._save_csv()
        finally: rclpy.shutdown()

# ---- main ----
def main():
    rclpy.init()
    node = OnlineMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._save_csv()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
