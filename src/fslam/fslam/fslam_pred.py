import math, time, bisect
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclTime

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from eufs_msgs.msg import WheelSpeedsStamped
from geometry_msgs.msg import Quaternion as QuaternionMsg
from geometry_msgs.msg import Pose2D

G = 9.80665  # m/s^2

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def rot3_from_quat(qx,qy,qz,qw):
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

def quat_from_yaw(yaw) -> QuaternionMsg:
    q = QuaternionMsg()
    q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

class ImuWheelEKF(Node):
    def __init__(self):
        super().__init__("fastslam_localizer", automatically_declare_parameters_from_overrides=True)

        # ---- Params ----
        self.declare_parameter("topics.imu", "/imu/data")
        self.declare_parameter("topics.wheels", "/ros_can/wheel_speeds")
        self.declare_parameter("topics.gt_odom", "/ground_truth/odom")
        self.declare_parameter("topics.out_odom", "/odometry_integration/car_state")
        self.declare_parameter("topics.out_pose2d", "/odometry_integration/pose2d")

        self.declare_parameter("run.bias_window_s", 5.0)
        self.declare_parameter("log.cli_hz", 1.0)

        self.declare_parameter("input.imu_use_rate_hz", 0.0)
        self.declare_parameter("input.target_wheel_hz", 400.0)

        # Keep defaults at 0.0 so prediction matches your predictor exactly.
        self.declare_parameter("proc.sigma_acc", 0.0)
        self.declare_parameter("proc.sigma_yaw", 0.0)
        self.declare_parameter("init.sigma_v", 0.0)
        self.declare_parameter("vel_leak_hz", 0.0)

        self.declare_parameter("wheel.radius_m", 0.253)
        self.declare_parameter("wheel.rpm_min", 5.0)
        self.declare_parameter("wheel.sigma_base", 0.35)
        self.declare_parameter("wheel.k_turn", 1.0)
        self.declare_parameter("wheel.k_acc", 0.5)

        # Yaw scale calibration params (robust)
        self.declare_parameter("yawscale.enable", True)
        self.declare_parameter("yawscale.alpha", 0.04)        # EMA for k_yaw
        self.declare_parameter("yawscale.v_min", 1.0)          # m/s, need motion
        self.declare_parameter("yawscale.gz_min", 0.05)        # rad/s, need turn
        self.declare_parameter("yawscale.k_min", 0.6)
        self.declare_parameter("yawscale.k_max", 2.4)
        self.declare_parameter("yawscale.aperp_min", 0.4)      # m/s^2, reject noise
        self.declare_parameter("yawscale.par_over_perp_max", 0.5)  # |a_par| <= r * |a_perp|

        P = lambda k: self.get_parameter(k).value
        self.topic_imu   = str(P("topics.imu"))
        self.topic_wheel = str(P("topics.wheels"))
        self.topic_gt    = str(P("topics.gt_odom"))
        self.topic_out   = str(P("topics.out_odom"))
        self.topic_pose2d= str(P("topics.out_pose2d"))

        self.bias_window_s = float(P("run.bias_window_s"))
        self.cli_hz = float(P("log.cli_hz"))

        self.imu_use_rate_hz = float(P("input.imu_use_rate_hz"))
        self.target_wheel_hz = float(P("input.target_wheel_hz"))

        self.sigma_acc = float(P("proc.sigma_acc"))
        self.sigma_yaw = float(P("proc.sigma_yaw"))
        self.init_sigma_v = float(P("init.sigma_v"))
        self.vel_leak_hz = float(P("vel_leak_hz"))

        self.Rw = float(P("wheel.radius_m"))
        self.rpm_min = float(P("wheel.rpm_min"))
        self.w_sigma_base = float(P("wheel.sigma_base"))
        self.w_k_turn = float(P("wheel.k_turn"))
        self.w_k_acc  = float(P("wheel.k_acc"))

        # Yaw scale cfg
        self.yawscale_enable = bool(P("yawscale.enable"))
        self.yawscale_alpha  = float(P("yawscale.alpha"))
        self.yawscale_v_min  = float(P("yawscale.v_min"))
        self.yawscale_gz_min = float(P("yawscale.gz_min"))
        self.yawscale_k_min  = float(P("yawscale.k_min"))
        self.yawscale_k_max  = float(P("yawscale.k_max"))
        self.yawscale_aperp_min = float(P("yawscale.aperp_min"))
        self.yawscale_par_over_perp_max = float(P("yawscale.par_over_perp_max"))

        # ---- EKF State ----
        # x = [x, y, yaw, vx, vy]
        self.x = np.zeros(5, float)
        self.P = np.diag([1e-4, 1e-4, 1e-4, max(1e-4, self.init_sigma_v**2), max(1e-4, self.init_sigma_v**2)])

        # Bias & timing
        self.bias_locked = False
        self.bias_t0 = None
        self.acc_b = np.zeros(3)
        self.gz_b  = 0.0
        self._acc_buf=[]; self._gz_buf=[]
        self.last_imu_t = None
        self._last_imu_proc_t = None

        # GT logs
        self.gt_t=[]; self.gt_x=[]; self.gt_y=[]; self.gt_v=[]; self.gt_yaw=[]

        # Logs
        self.pred_log = []   # (t, x, y, yaw, v_pred)
        self.corr_log = []   # (t, x, y, yaw, v_corr)

        self.last_cli = time.time()
        self.last_yawrate = 0.0
        self.last_axw = 0.0

        self._last_wheel_t = None
        self.last_output = None

        # Yaw scale state
        self.k_yaw = 1.0  # multiplicative scale applied to gyro z

        qos_fast = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=200,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        qos_rel = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Imu, self.topic_imu, self.cb_imu, qos_fast)
        self.create_subscription(WheelSpeedsStamped, self.topic_wheel, self.cb_wheel, qos_fast)
        self.create_subscription(Odometry, self.topic_gt, self.cb_gt, qos_rel)

        self.pub_out = self.create_publisher(Odometry, self.topic_out, 10)
        self.pub_pose2d = self.create_publisher(Pose2D, self.topic_pose2d, 10)
        self.create_timer(max(0.05, 1.0/self.cli_hz), self.on_cli_timer)

        self.get_logger().info(
            f"[WS-EKF] imu={self.topic_imu} | wheel={self.topic_wheel} | out={self.topic_out}\n"
            f"         bias_window={self.bias_window_s}s | run=∞"
        )

    # ---------- Callbacks ----------
    def cb_gt(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x); y = float(msg.pose.pose.position.y)
        vx = float(msg.twist.twist.linear.x); vy = float(msg.twist.twist.linear.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x,q.y,q.z,q.w)
        self.gt_t.append(t); self.gt_x.append(x); self.gt_y.append(y)
        self.gt_v.append(math.hypot(vx,vy)); self.gt_yaw.append(yaw)

    def cb_imu(self, msg: Imu):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.bias_t0 is None:
            self.bias_t0 = t

        ax,ay,az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx,gy,gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        q = msg.orientation
        Rwb = rot3_from_quat(q.x,q.y,qz:=q.z,q.w)  # world_from_body
        yaw_q = yaw_from_quat(q.x,q.y,qz,q.w)

        # Bias lock window
        if not self.bias_locked and (t - self.bias_t0) <= self.bias_window_s:
            a = np.array([ax,ay,az], float); w = np.array([gx,gy,gz], float)
            if abs(gz) < 0.05 and np.linalg.norm(w) < 0.2 and abs(np.linalg.norm(a)-G) < 0.5:
                self._acc_buf.append(a); self._gz_buf.append(gz)
            self.last_imu_t = t
            return
        elif not self.bias_locked:
            if self._acc_buf: self.acc_b = np.mean(np.array(self._acc_buf), axis=0)
            if self._gz_buf:  self.gz_b  = float(np.mean(self._gz_buf))
            self.bias_locked = True
            self.get_logger().info(f"[WS-EKF] Bias locked: acc_b={self.acc_b.round(4).tolist()} m/s², gyro_bz={self.gz_b:.5f} rad/s")
            self.x[2] = yaw_q
            if self.init_sigma_v > 0.0:
                self.x[3] = np.random.normal(0.0, self.init_sigma_v)
                self.x[4] = np.random.normal(0.0, self.init_sigma_v)
            self.last_imu_t = t
            self._last_imu_proc_t = None
            return

        # Optional IMU downsample
        if self.imu_use_rate_hz > 0.0 and self._last_imu_proc_t is not None:
            if (t - self._last_imu_proc_t) < (1.0 / self.imu_use_rate_hz):
                self.last_imu_t = t
                return

        if self.last_imu_t is None:
            self.last_imu_t = t
            self.x[2] = yaw_q
            self._last_imu_proc_t = t
            return
        dt = t - self.last_imu_t
        if dt <= 0.0 or dt > 0.05:
            self.last_imu_t = t
            return
        self.last_imu_t = t
        self._last_imu_proc_t = t

        # Bias-correct IMU
        a_body = np.array([ax,ay,az], float) - self.acc_b
        gz_c = gz - self.gz_b

        # Gravity compensation in WORLD
        a_world3 = Rwb @ a_body - np.array([0.0,0.0,G])
        ax_w, ay_w = float(a_world3[0]), float(a_world3[1])

        # -------- Prediction (matches your equations) --------
        # (1) Yaw integrate with SCALE correction (robust centripetal update)
        vx_prev, vy_prev = self.x[3], self.x[4]
        v_prev = math.hypot(vx_prev, vy_prev)
        gz_used = gz_c

        if (self.yawscale_enable
            and v_prev > self.yawscale_v_min
            and abs(gz_c) > self.yawscale_gz_min):

            # unit vectors along velocity and its left-normal
            th_v = math.atan2(vy_prev, vx_prev)
            vx_h, vy_h = math.cos(th_v), math.sin(th_v)
            nx, ny = -vy_h, vx_h

            # project world accel
            a_par  = ax_w*vx_h + ay_w*vy_h
            a_perp = ax_w*nx   + ay_w*ny

            # require lateral accel and ensure not dominated by tangential (brake/accel)
            if (abs(a_perp) > self.yawscale_aperp_min and
                abs(a_par) <= self.yawscale_par_over_perp_max * abs(a_perp)):

                # observation of scale (always positive)
                k_obs = abs(a_perp) / (max(v_prev,1e-3) * max(abs(gz_c),1e-6))
                # clamp and EMA
                k_obs = max(self.yawscale_k_min, min(self.yawscale_k_max, k_obs))
                self.k_yaw = (1.0 - self.yawscale_alpha)*self.k_yaw + self.yawscale_alpha*k_obs

        gz_used *= self.k_yaw
        self.x[2] = wrap(self.x[2] + gz_used*dt)

        # (2) Leak and integrate v & p in WORLD
        leak = math.exp(-self.vel_leak_hz * dt) if self.vel_leak_hz > 0.0 else 1.0
        self.x[3] = leak*(vx_prev + ax_w*dt)
        self.x[4] = leak*(vy_prev + ay_w*dt)
        self.x[0] += self.x[3]*dt
        self.x[1] += self.x[4]*dt

        # Process cov (kept minimal; defaults 0)
        A = np.array([[1,0,0,dt,0],[0,1,0,0,dt],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], float)
        sig_a = self.sigma_acc; sig_y = self.sigma_yaw
        Qd = np.diag([(0.5*dt*dt*sig_a)**2,(0.5*dt*dt*sig_a)**2,(sig_y*math.sqrt(dt))**2,(dt*sig_a)**2,(dt*sig_a)**2])
        self.P = A @ self.P @ A.T + Qd

        # Save helpers for wheel update & logs
        self.last_axw = math.hypot(ax_w, ay_w)
        self.last_yawrate = gz_used

        # Log prediction-only
        v_pred = float(math.hypot(self.x[3], self.x[4]))
        self.pred_log.append((t, self.x[0], self.x[1], self.x[2], v_pred))

    def cb_wheel(self, msg: WheelSpeedsStamped):
        if not self.bias_locked or self.last_imu_t is None:
            return
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9 if msg.header.stamp else self.last_imu_t
        if self._last_wheel_t is not None and (t - self._last_wheel_t) < (1.0 / max(1e-3, self.target_wheel_hz)):
            return
        self._last_wheel_t = t
        if t > self.last_imu_t:
            return

        rb = float(msg.speeds.rb_speed); lb = float(msg.speeds.lb_speed)
        if abs(rb) < self.rpm_min or abs(lb) < self.rpm_min:
            return

        # Wheel-speed EKF REMOVED: just publish at original cadence
        self._publish_output(self.last_imu_t)

    # ---------- CLI + helpers ----------
    def on_cli_timer(self):
        if not self.bias_locked:
            return
        now = time.time()
        if now - self.last_cli < (1.0 / max(1e-3, self.cli_hz)):
            return
        self.last_cli = now
        t = self.last_imu_t
        if t is None:
            return

        x_p, y_p, yaw_p = self.x[0], self.x[1], self.x[2]
        v_p = float(math.hypot(self.x[3], self.x[4]))
        if self.last_output is None:
            x_c,y_c,yaw_c,v_c = x_p,y_p,yaw_p,v_p
        else:
            _, x_c,y_c,yaw_c,v_c = self.last_output

        xg, yg, vg, ygaw = self._nearest_gt(t)
        epos_p = float('nan') if xg is None else math.hypot(x_p-xg, y_p-yg)
        epos_c = float('nan') if xg is None else math.hypot(x_c-xg, y_c-yg)
        eyaw_c = float('nan') if ygaw is None else math.degrees(wrap(yaw_c - ygaw))
        ev_c   = float('nan') if vg is None else (v_c - vg)

        self.get_logger().info(
            f"[t={t:7.2f}s] k_yaw={self.k_yaw:.3f} | "
            f"pred: x={x_p:6.2f} y={y_p:6.2f} yaw={math.degrees(yaw_p):6.1f}° v={v_p:4.2f} | "
            f"corr: x={x_c:6.2f} y={y_c:6.2f} yaw={math.degrees(yaw_c):6.1f}° v={v_c:4.2f} | "
            f"err_pos pred/corr={epos_p:.2f}/{epos_c:.2f} m  err_yaw corr={eyaw_c:.1f}°  dv corr={ev_c:.2f}"
        )

    def _nearest_gt(self, t):
        if not self.gt_t:
            return None, None, None, None
        idx = bisect.bisect_left(self.gt_t, t)
        if idx<=0: i=0
        elif idx>=len(self.gt_t): i=-1
        else:
            i = idx if abs(self.gt_t[idx]-t) < abs(self.gt_t[idx-1]-t) else idx-1
        return self.gt_x[i], self.gt_y[i], self.gt_v[i], self.gt_yaw[i]

    def _publish_output(self, t: float):
        from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3
        x,y,yaw = self.x[0], self.x[1], self.x[2]
        v = float(math.hypot(self.x[3], self.x[4]))
        self.corr_log.append((t, x, y, yaw, v))
        self.last_output = (t, x, y, yaw, v)

        od = Odometry()
        od.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose = PoseWithCovariance()
        od.twist = TwistWithCovariance()
        od.pose.pose = Pose(position=Point(x=x, y=y, z=0.0), orientation=Quaternion())
        od.pose.pose.orientation = quat_from_yaw(yaw)
        od.twist.twist = Twist(
            linear=Vector3(x=v*math.cos(yaw), y=v*math.sin(yaw), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=self.last_yawrate)
        )
        self.pub_out.publish(od)

        p2 = Pose2D()
        p2.x = x; p2.y = y; p2.theta = yaw
        self.pub_pose2d.publish(p2)

# ---------- main ----------
def main():
    rclpy.init()
    node = ImuWheelEKF()
    try:
        rclpy.spin(node)  # run indefinitely until externally stopped
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()