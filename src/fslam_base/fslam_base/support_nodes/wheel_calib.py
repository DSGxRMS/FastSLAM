#!/usr/bin/env python3
# calibrate_wheel_radius_cmd.py  (ROS2 / EUFSIM)
#
# Actively commands acceleration on a straight, lets speed settle, then
# estimates REAR wheel radius:
#       R ≈ v_gt / ω_wheel
# using ground-truth odometry for v_gt and rear wheel RPM for ω_wheel.
#
# Control is ACCELERATION-FIRST via ackermann_msgs/AckermannDriveStamped:
#   - drive.acceleration [m/s^2]
#   - drive.steering_angle (kept at 0 for straight)
#
# End condition: once a stable-velocity window is detected and enough samples
# are gathered, prints median and p95 wheel_radius_m, sends a stop command, exits.

import math
import time
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)

from nav_msgs.msg import Odometry
from eufs_msgs.msg import WheelSpeedsStamped
from ackermann_msgs.msg import AckermannDriveStamped


class WheelRadiusCalibrator(Node):
    def __init__(self):
        super().__init__("wheel_radius_calib", automatically_declare_parameters_from_overrides=True)

        # ---------------- Params ----------------
        # topics
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("wheels_topic", "/ros_can/wheel_speeds")
        self.declare_parameter("cmd_topic", "/cmd")

        # control
        self.declare_parameter("target_speed_mps", 6.0)     # cruise target
        self.declare_parameter("accel_gain", 1.5)           # k_p on (v_target - v_meas) -> accel
        self.declare_parameter("accel_limit_mps2", 3.0)     # clamp |accel|
        self.declare_parameter("control_hz", 50.0)          # command rate
        self.declare_parameter("brake_accel_mps2", -3.0)    # accel used to stop at the end

        # stability detection (constant-velocity window on a straight)
        self.declare_parameter("stable_window_s", 3.0)      # window length
        self.declare_parameter("speed_std_tol_mps", 0.20)   # std(v) inside window
        self.declare_parameter("speed_slope_tol_mps2", 0.10)# |Δv/Δt| inside window
        self.declare_parameter("yaw_rate_tol_radps", 0.05)  # straight-line constraint

        # sampling & validity
        self.declare_parameter("rpm_min", 5.0)              # ignore tiny/garbage RPM
        self.declare_parameter("samples_required", 80)      # stop after this many valid samples
        self.declare_parameter("timeout_s", 120.0)          # hard stop if cannot finish

        # Resolve
        P = lambda k: self.get_parameter(k).value
        self.odom_topic = str(P("odom_topic"))
        self.wheels_topic = str(P("wheels_topic"))
        self.cmd_topic = str(P("cmd_topic"))

        self.v_target = float(P("target_speed_mps"))
        self.k_accel = float(P("accel_gain"))
        self.a_limit = float(P("accel_limit_mps2"))
        self.ctrl_hz = float(P("control_hz"))
        self.a_brake = float(P("brake_accel_mps2"))

        self.win_s = float(P("stable_window_s"))
        self.v_std_tol = float(P("speed_std_tol_mps"))
        self.v_slope_tol = float(P("speed_slope_tol_mps2"))
        self.yaw_tol = float(P("yaw_rate_tol_radps"))

        self.rpm_min = float(P("rpm_min"))
        self.n_need = int(P("samples_required"))
        self.timeout_s = float(P("timeout_s"))

        # ---------------- State ----------------
        self.last_odom = None
        self.last_ws = None
        self.speed_hist = deque()   # (t, v_gt) window for stability
        self.samples_R = []         # wheel radius samples during stable window
        self.t0 = time.time()       # start wall-clock
        self._done = False

        # ---------------- QoS ----------------
        # BEST_EFFORT + shallow queues → always act on latest data (avoid lag)
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- I/O ----------------
        self.sub_o = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, sub_qos)
        self.sub_w = self.create_subscription(WheelSpeedsStamped, self.wheels_topic, self.cb_wheels, sub_qos)
        self.pub_cmd = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        # control timer
        self.timer = self.create_timer(1.0 / max(1.0, self.ctrl_hz), self.control_loop)

        self.get_logger().info(
            f"[calib] target={self.v_target:.2f} m/s, ctrl={self.ctrl_hz:.1f} Hz, window={self.win_s:.1f}s, "
            f"n_samples={self.n_need}, rpm_min={self.rpm_min:.1f}"
        )

    # ---------------- Callbacks ----------------
    def cb_odom(self, msg: Odometry):
        self.last_odom = msg
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy)
        self.speed_hist.append((t, v))
        # trim window
        tmin = t - self.win_s
        while self.speed_hist and self.speed_hist[0][0] < tmin:
            self.speed_hist.popleft()

        # collect sample if stable and we have wheels
        if self.is_stable() and self.last_ws is not None:
            rb = float(self.last_ws.rb_speed)
            lb = float(self.last_ws.lb_speed)
            if abs(rb) >= self.rpm_min and abs(lb) >= self.rpm_min:
                rpm = min(rb, lb)  # rear-only min (more robust vs slip)
                omega = rpm * (2.0 * math.pi / 60.0)  # rad/s
                if omega > 1e-3:
                    R = v / omega
                    # sanity band for FS car wheels
                    if 0.10 < R < 0.60:
                        self.samples_R.append(R)

    def cb_wheels(self, msg: WheelSpeedsStamped):
        self.last_ws = msg.speeds

    # ---------------- Control ----------------
    def control_loop(self):
        if self._done:
            return

        # timeout
        if (time.time() - self.t0) > self.timeout_s:
            self.finish("timeout")
            return

        # if no odom yet, gently nudge forward to get moving
        if self.last_odom is None:
            self.publish_cmd(accel=min(self.a_limit, 0.5 * self.a_limit), steering=0.0)
            return

        # compute accel command towards target speed
        v_now = self.current_speed()
        v_err = self.v_target - v_now
        a_cmd = max(-self.a_limit, min(self.a_limit, self.k_accel * v_err))
        self.publish_cmd(accel=a_cmd, steering=0.0)

        # exit if we have enough samples
        if len(self.samples_R) >= self.n_need:
            self.finish("enough_samples")

    def publish_cmd(self, accel: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration = float(accel)
        # leave speed field at 0.0 for accel-first interface
        msg.drive.speed = 0.0
        self.pub_cmd.publish(msg)

    # ---------------- Helpers ----------------
    def current_speed(self) -> float:
        if self.last_odom is None:
            return 0.0
        vx = float(self.last_odom.twist.twist.linear.x)
        vy = float(self.last_odom.twist.twist.linear.y)
        return math.hypot(vx, vy)

    def is_stable(self) -> bool:
        """Detect constant-velocity + straight-line over the recent window."""
        if self.last_odom is None or len(self.speed_hist) < 10:
            return False
        # speed stats
        vs = [v for (_, v) in self.speed_hist]
        std_v = statistics.pstdev(vs)
        t0, v0 = self.speed_hist[0]
        t1, v1 = self.speed_hist[-1]
        dt = max(1e-3, (t1 - t0))
        slope = abs((v1 - v0) / dt)
        # yaw rate constraint
        yaw_rate = abs(float(self.last_odom.twist.twist.angular.z))
        return (std_v <= self.v_std_tol) and (slope <= self.v_slope_tol) and (yaw_rate <= self.yaw_tol)

    def finish(self, reason: str):
        if self._done:
            return
        self._done = True

        # compute stats
        if len(self.samples_R) == 0:
            self.get_logger().warn("No valid samples collected; cannot estimate radius.")
        else:
            med = statistics.median(self.samples_R)
            p95 = statistics.quantiles(self.samples_R, n=20)[18] if len(self.samples_R) >= 20 else med
            self.get_logger().info(
                f"[{reason}] wheel_radius_m ≈ median {med:.4f} (p95 {p95:.4f})  "
                f"n={len(self.samples_R)} over ~{self.win_s:.1f}s stable windows"
            )
            print(f"\nwheel_radius_m_median={med:.6f}\nwheel_radius_m_p95={p95:.6f}\n")

        # polite stop: brake once
        try:
            self.publish_cmd(accel=float(self.a_brake), steering=0.0)
        except Exception:
            pass

        # shutdown
        rclpy.shutdown()


def main():
    rclpy.init()
    node = WheelRadiusCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.finish("ctrl_c")


if __name__ == "__main__":
    main()
