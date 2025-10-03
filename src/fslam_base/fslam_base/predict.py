#!/usr/bin/env python3
# imu_pose_predictor_tester.py (gravity-comp + robust biasing + unwrap plots)

import math, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

G = 9.80665  # m/s^2

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi
def rot3_from_quat(qx,qy,qz,qw):
    # standard quaternion -> rotation matrix (world_from_body)
    x,y,z,w = qx,qy,qz,qw
    xx,yy,zz = x*x,y*y,z*z
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

def yaw_from_quat(qx,qy,qz,qw):
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def unwrap(vec):
    if len(vec)==0: return np.array([])
    out = np.unwrap(np.array(vec))
    return out

class Est:
    def __init__(self):
        self.t=0.0; self.x=0.0; self.y=0.0; self.vx=0.0; self.vy=0.0; self.yaw=0.0

class ImuPosePredictorTester(Node):
    def __init__(self):
        super().__init__("imu_pose_predictor_tester")

        # Params
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("gt_topic", "/ground_truth/odom")
        self.declare_parameter("duration_s", 35.0)
        self.declare_parameter("log_period_s", 0.5)
        self.declare_parameter("bias_window_s", 5.0)
        self.declare_parameter("max_dt_imu", 0.05)
        self.declare_parameter("vel_leak_hz", 0.0)   # small bleed to tame drift (0 disables)

        p = self.get_parameter
        self.imu_topic = str(p("imu_topic").value)
        self.gt_topic = str(p("gt_topic").value)
        self.duration_s = float(p("duration_s").value)
        self.log_period_s = float(p("log_period_s").value)
        self.bias_window_s = float(p("bias_window_s").value)
        self.max_dt_imu = float(p("max_dt_imu").value)
        self.vel_leak_hz = float(p("vel_leak_hz").value)

        # Subs
        self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.gt_topic, self.cb_gt, QoSProfile(depth=50))

        # State
        self.est = Est()
        self._last_t_imu = None

        # Bias estimation (only when stationary)
        self._bias_collect = True
        self._bias_t0 = None
        self.acc_b = np.zeros(3)  # body-frame accel bias
        self.gyro_bz = 0.0

        self._acc_buf = []   # 3D body acc samples
        self._gz_buf  = []   # gyro z samples

        # Logs
        self.t_list=[]; self.x_list=[]; self.y_list=[]; self.v_list=[]; self.yaw_list=[]
        self.tgt=[]; self.xgt=[]; self.ygt=[]; self.vgt=[]; self.yawgt=[]

        # Timers
        self._t_start = time.time()
        self._t_last_log = self._t_start
        self.create_timer(0.05, self._check_end)

        self.get_logger().info(f"IMU Pose Predictor Tester up. duration={self.duration_s:.1f}s | imu={self.imu_topic} | odom={self.gt_topic}")

    # ---------- Callbacks ----------
    def cb_gt(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x,q.y,q.z,q.w)
        self.tgt.append(t); self.xgt.append(x); self.ygt.append(y)
        self.vgt.append(math.hypot(vx,vy)); self.yawgt.append(yaw)

    def cb_imu(self, msg: Imu):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self._bias_t0 is None: self._bias_t0 = t

        ax,ay,az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx,gy,gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        q = msg.orientation
        Rwb = rot3_from_quat(q.x,q.y,q.z,q.w)  # world_from_body
        yaw_q = yaw_from_quat(q.x,q.y,q.z,q.w)

        # Stationary detection for biasing (first bias_window_s):
        if self._bias_collect and (t - self._bias_t0) <= self.bias_window_s:
            a = np.array([ax,ay,az], float)
            w = np.array([gx,gy,gz], float)
            if abs(gz) < 0.05 and np.linalg.norm(w) < 0.2 and abs(np.linalg.norm(a)-G) < 0.5:
                self._acc_buf.append(a); self._gz_buf.append(gz)
        elif self._bias_collect:
            # finalize
            if self._acc_buf:
                self.acc_b = np.mean(np.array(self._acc_buf), axis=0)
            if self._gz_buf:
                self.gyro_bz = float(np.mean(self._gz_buf))
            self._bias_collect = False
            self.get_logger().info(f"Bias locked: acc_b={self.acc_b.round(4).tolist()} m/s², gyro_bz={self.gyro_bz:.5f} rad/s")

        # dt
        if self._last_t_imu is None:
            self._last_t_imu = t
            # initialize yaw from quaternion so we start aligned
            self.est.yaw = yaw_q
            return
        dt = t - self._last_t_imu
        self._last_t_imu = t
        if dt <= 0.0 or dt > self.max_dt_imu: return

        # Bias correction (body frame)
        a_body = np.array([ax,ay,az], float) - self.acc_b
        gz_c = gz - self.gyro_bz

        # Gravity compensation: a_world = Rwb * a_body - [0,0,g]
        a_world3 = Rwb @ a_body - np.array([0.0,0.0,G])
        ax_w, ay_w = float(a_world3[0]), float(a_world3[1])

        # Yaw integration (keep planar yaw, but start from IMU yaw)
        self.est.yaw = wrap(self.est.yaw + gz_c*dt)

        # Velocity leak (optional small bleed against drift)
        if self.vel_leak_hz > 0.0:
            leak = math.exp(-self.vel_leak_hz * dt)
            self.est.vx *= leak; self.est.vy *= leak

        # Integrate v & p in WORLD
        self.est.vx += ax_w * dt
        self.est.vy += ay_w * dt
        self.est.x  += self.est.vx * dt
        self.est.y  += self.est.vy * dt
        self.est.t   = t

        # Trace
        self.t_list.append(t)
        self.x_list.append(self.est.x); self.y_list.append(self.est.y)
        self.v_list.append(math.hypot(self.est.vx,self.est.vy))
        self.yaw_list.append(self.est.yaw)

        # Periodic CLI
        now = time.time()
        if (now - self._t_last_log) >= 0.5:
            self._t_last_log = now
            xg,yg,vg,yawg = self._nearest_gt(t)
            pos_err = math.hypot(self.est.x-xg, self.est.y-yg) if xg is not None else float('nan')
            yaw_err = wrap(self.est.yaw - yawg) if yawg is not None else float('nan')
            dv = (math.hypot(self.est.vx,self.est.vy) - vg) if vg is not None else float('nan')
            print(f"[pred] t={t:.2f}s dt={dt*1000:.1f}ms | "
                  f"bias_acc={self.acc_b.round(3).tolist()} gyro_bz={self.gyro_bz:.5f} | "
                  f"yaw={self.est.yaw:.2f} v={math.hypot(self.est.vx,self.est.vy):.2f} | "
                  f"pos=({self.est.x:.2f},{self.est.y:.2f}) | "
                  f"err pos={pos_err:.2f}m yaw={math.degrees(yaw_err):.1f}° dv={dv:.2f}m/s")

    def _nearest_gt(self, t):
        if not self.tgt: return None,None,None,None
        idx = np.searchsorted(self.tgt, t)
        if idx<=0: i=0
        elif idx>=len(self.tgt): i=-1
        else:
            i = idx if abs(self.tgt[idx]-t) < abs(self.tgt[idx-1]-t) else idx-1
        return self.xgt[i], self.ygt[i], self.vgt[i], self.yawgt[i]

    def _check_end(self):
        if (time.time() - self._t_start) >= self.duration_s:
            self.get_logger().info("Duration reached — generating plots…")
            try: self._plots()
            finally:
                rclpy.shutdown()

    # ---------- Plots ----------
    def _plots(self):
        if not self.t_list or not self.tgt:
            self.get_logger().warn("Insufficient data to plot.")
            return
        t0 = self.t_list[0]
        T  = np.array(self.t_list) - t0
        X  = np.array(self.x_list); Y = np.array(self.y_list)
        V  = np.array(self.v_list); Yaw = unwrap(self.yaw_list)

        t0g = self.tgt[0]
        Tg  = np.array(self.tgt) - t0g
        Xg  = np.array(self.xgt); Yg=np.array(self.ygt)
        Vg  = np.array(self.vgt); Ygaw=unwrap(self.yawgt)

        plt.figure(figsize=(8,7))
        plt.plot(Xg,Yg,label="GT XY")
        plt.plot(X,Y,label="IMU pred XY")
        plt.axis('equal'); plt.grid(True,ls='--',alpha=0.3); plt.legend(); plt.title("XY Trajectory")

        plt.figure(figsize=(10,4))
        plt.plot(Tg,Vg,label="GT speed"); plt.plot(T,V,label="IMU pred speed")
        plt.grid(True,ls='--',alpha=0.3); plt.legend(); plt.xlabel("time (s)"); plt.ylabel("m/s"); plt.title("Speed vs Time")

        plt.figure(figsize=(10,4))
        plt.plot(Tg,Ygaw,label="GT yaw (rad)"); plt.plot(T,Yaw,label="IMU pred yaw (rad)")
        plt.grid(True,ls='--',alpha=0.3); plt.legend(); plt.xlabel("time (s)"); plt.ylabel("rad"); plt.title("Yaw vs Time")
        plt.show()

def main():
    rclpy.init()
    n = ImuPosePredictorTester()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
