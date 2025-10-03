#!/usr/bin/env python3
# imu_quat_predictor_tester.py — full quaternion-based predictor (bias lock + plots)

import math, time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.time import Time as RclTime

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

G = 9.80665

# -------- Quaternion helpers (w, x, y, z convention internally) --------
def q_normalize(q):
    return q / (np.linalg.norm(q) + 1e-12)

def q_mul(q1, q2):
    # (w x y z) * (w x y z)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def q_from_imu_msg(qx,qy,qz,qw):
    # switch from (x y z w) to (w x y z)
    return np.array([qw,qx,qy,qz], float)

def q_to_rot(q):
    # rotation matrix R(q): world_from_body
    w,x,y,z = q
    ww,xx,yy,zz = w*w, x*x, y*y, z*z
    wx,wy,wz = w*x, w*y, w*z
    xy,xz,yz = x*y, x*z, y*z
    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz]
    ], dtype=float)

def q_from_omega_dt(wx, wy, wz, dt):
    # small-angle quaternion from angular rate in body frame
    th = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
    if th < 1e-12:
        # first-order
        half = 0.5*dt
        return q_normalize(np.array([1.0, half*wx, half*wy, half*wz], float))
    u = np.array([wx,wy,wz], float) / (th + 1e-12)
    half_th = 0.5*th
    return np.array([math.cos(half_th), *(math.sin(half_th)*u)], float)

def yaw_from_q(q):
    # extract yaw (Z) from quaternion
    R = q_to_rot(q)
    return math.atan2(R[1,0], R[0,0])

def unwrap(vec):
    if len(vec)==0: return np.array([])
    return np.unwrap(np.array(vec))

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

class EstState:
    def __init__(self):
        self.t = 0.0
        self.q_wb = np.array([1.0,0.0,0.0,0.0], float)  # (w x y z)
        self.v = np.zeros(3)  # world
        self.p = np.zeros(3)  # world

class ImuQuatPredictorTester(Node):
    def __init__(self):
        super().__init__("imu_quat_predictor_tester")

        # Params
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("gt_topic", "/ground_truth/odom")
        self.declare_parameter("duration_s", 35.0)
        self.declare_parameter("log_period_s", 0.5)
        self.declare_parameter("bias_window_s", 5.0)
        self.declare_parameter("max_dt_imu", 0.05)
        self.declare_parameter("vel_leak_hz", 0.0)   # optional bleed

        p = self.get_parameter
        self.imu_topic = str(p("imu_topic").value)
        self.gt_topic  = str(p("gt_topic").value)
        self.duration_s = float(p("duration_s").value)
        self.log_period_s = float(p("log_period_s").value)
        self.bias_window_s = float(p("bias_window_s").value)
        self.max_dt_imu = float(p("max_dt_imu").value)
        self.vel_leak_hz = float(p("vel_leak_hz").value)

        # Subs
        self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.gt_topic, self.cb_gt, QoSProfile(depth=50))

        # State
        self.est = EstState()
        self._last_t_imu = None

        # Bias estimation (stationary)
        self._bias_collect = True
        self._bias_t0 = None
        self.acc_b = np.zeros(3)  # body-frame accel bias
        self.gyro_b = np.zeros(3) # (bx, by, bz) if you want full; we’ll mainly use bz
        self._acc_buf = []
        self._w_buf   = []

        # Logs (IMU pred + GT)
        self.tt = []; self.pos = []; self.speed = []; self.yaws = []
        self.tgt = []; self.xg = []; self.yg = []; self.vg = []; self.yawg = []

        # Timers
        self._t_start = time.time()
        self._t_last_log = self._t_start
        self.create_timer(0.05, self._check_end)

        self._init_q_from_imu = False  # set true once we see a valid IMU quaternion

        self.get_logger().info(f"IMU Quaternion Predictor up. duration={self.duration_s:.1f}s | imu={self.imu_topic} | odom={self.gt_topic}")

    # ---------- Callbacks ----------
    def cb_gt(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        q = msg.pose.pose.orientation
        # yaw from GT orientation
        yawg = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.tgt.append(t); self.xg.append(x); self.yg.append(y)
        self.vg.append(math.hypot(vx,vy)); self.yawg.append(yawg)

    def cb_imu(self, msg: Imu):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self._bias_t0 is None: self._bias_t0 = t

        ax,ay,az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        wx,wy,wz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

        # If IMU provides a valid orientation, use it once to initialize q
        if not self._init_q_from_imu:
            if msg.orientation_covariance[0] >= 0.0:  # not -1
                qx,qy,qz,qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                self.est.q_wb = q_normalize(q_from_imu_msg(qx,qy,qz,qw))
                self._init_q_from_imu = True

        # Stationary bias collection (first bias_window_s seconds)
        if self._bias_collect and (t - self._bias_t0) <= self.bias_window_s:
            a = np.array([ax,ay,az], float)
            w = np.array([wx,wy,wz], float)
            # crude stationary test: low angular rate and accel near g
            if np.linalg.norm(w) < 0.2 and abs(np.linalg.norm(a) - G) < 0.5:
                self._acc_buf.append(a); self._w_buf.append(w)
        elif self._bias_collect:
            if self._acc_buf:
                self.acc_b = np.mean(np.array(self._acc_buf), axis=0)
            if self._w_buf:
                self.gyro_b = np.mean(np.array(self._w_buf), axis=0)
            self._bias_collect = False
            self.get_logger().info(f"Bias locked: acc_b={self.acc_b.round(4).tolist()} m/s², gyro_b={self.gyro_b.round(5).tolist()} rad/s")

        # timing
        if self._last_t_imu is None:
            self._last_t_imu = t
            if not self._init_q_from_imu:
                # initialize if no IMU orientation: assume identity (flat, yaw=0)
                self.est.q_wb = np.array([1.0,0.0,0.0,0.0], float)
            return

        dt = t - self._last_t_imu
        self._last_t_imu = t
        if dt <= 0.0 or dt > self.max_dt_imu:
            return

        # Bias-correct body signals
        a_b = np.array([ax,ay,az], float) - self.acc_b
        w_b = np.array([wx,wy,wz], float) - self.gyro_b

        # Propagate quaternion: q_{k+1} = q_k ⊗ exp(0.5 * w_b * dt)
        dq = q_from_omega_dt(w_b[0], w_b[1], w_b[2], dt)
        self.est.q_wb = q_normalize(q_mul(self.est.q_wb, dq))

        # Rotate accel to world & remove gravity
        Rwb = q_to_rot(self.est.q_wb)
        a_w = Rwb @ a_b - np.array([0.0,0.0,G])

        # Optional small velocity leak
        if self.vel_leak_hz > 0.0:
            leak = math.exp(-self.vel_leak_hz * dt)
            self.est.v *= leak

        # Integrate velocity & position (world frame)
        self.est.v += a_w * dt
        self.est.p += self.est.v * dt
        self.est.t  = t

        # Logs
        self.tt.append(t)
        self.pos.append(self.est.p.copy())
        self.speed.append(float(np.linalg.norm(self.est.v[:2])))
        self.yaws.append(yaw_from_q(self.est.q_wb))

        # Periodic CLI
        now = time.time()
        if (now - self._t_last_log) >= self.log_period_s:
            self._t_last_log = now
            xg, yg, vg, ygaw = self._nearest_gt(t)
            x, y = self.est.p[0], self.est.p[1]
            v = self.speed[-1]
            yaw = self.yaws[-1]
            pos_err = math.hypot(x - xg, y - yg) if xg is not None else float('nan')
            yaw_err = wrap(yaw - ygaw) if ygaw is not None else float('nan')
            dv = (v - vg) if vg is not None else float('nan')
            print(f"[pred/q] t={t:.2f}s dt={dt*1000:.1f}ms | "
                  f"bias_acc={self.acc_b.round(3).tolist()} gyro_b={self.gyro_b.round(5).tolist()} | "
                  f"yaw={yaw:.2f} v={v:.2f} | pos=({x:.2f},{y:.2f}) | "
                  f"err pos={pos_err:.2f}m yaw={math.degrees(yaw_err):.1f}° dv={dv:.2f}m/s")

    def _nearest_gt(self, t):
        if not self.tgt: return None, None, None, None
        idx = np.searchsorted(self.tgt, t)
        if idx <= 0: i = 0
        elif idx >= len(self.tgt): i = -1
        else:
            i = idx if abs(self.tgt[idx]-t) < abs(self.tgt[idx-1]-t) else idx-1
        return self.xg[i], self.yg[i], self.vg[i], self.yawg[i]

    # ---------- End & plots ----------
    def _check_end(self):
        if (time.time() - self._t_start) >= self.duration_s:
            self.get_logger().info("Duration reached — generating plots…")
            try: self._plots()
            finally:
                rclpy.shutdown()

    def _plots(self):
        if not self.tt or not self.tgt:
            self.get_logger().warn("Insufficient data to plot.")
            return
        # IMU pred
        T = np.array(self.tt) - self.tt[0]
        P = np.array(self.pos)
        V = np.array(self.speed)
        Y = unwrap(self.yaws)

        # GT
        Tg = np.array(self.tgt) - self.tgt[0]
        Xg = np.array(self.xg); Yg = np.array(self.yg)
        Vg = np.array(self.vg); Yawg = unwrap(self.yawg)

        plt.figure(figsize=(8,7))
        plt.plot(Xg, Yg, label="GT XY")
        plt.plot(P[:,0], P[:,1], label="IMU quat XY")
        plt.axis('equal'); plt.grid(True, ls='--', alpha=0.3)
        plt.legend(); plt.title("XY Trajectory")

        plt.figure(figsize=(10,4))
        plt.plot(Tg, Vg, label="GT speed")
        plt.plot(T,  V,  label="IMU quat speed")
        plt.grid(True, ls='--', alpha=0.3); plt.legend()
        plt.xlabel("time (s)"); plt.ylabel("m/s"); plt.title("Speed vs Time")

        plt.figure(figsize=(10,4))
        plt.plot(Tg, Yawg, label="GT yaw (rad)")
        plt.plot(T,  Y,    label="IMU quat yaw (rad)")
        plt.grid(True, ls='--', alpha=0.3); plt.legend()
        plt.xlabel("time (s)"); plt.ylabel("rad"); plt.title("Yaw vs Time")
        plt.show()

def main():
    rclpy.init()
    n = ImuQuatPredictorTester()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
