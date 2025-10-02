#!/usr/bin/env python3
# wheel_vel_compare.py  (ROS2)
#
# Subscribes to rear wheel speeds + GT odom and prints:
#   v_wheel (from RPM & calibrated radius)  vs  v_gt (from odom.twist)
#
# Notes:
# - Uses REAR-ONLY MIN RPM (robust to one-wheel slip), ignores fronts.
# - LPF on v_wheel to suppress sensor jitter.
# - BEST_EFFORT QoS + shallow queues to avoid lag.
#
# TUNE THESE (or set via ROS params):
#   wheel_radius_m        : set to your calibrated median (≈ 0.253)
#   rpm_min               : ignore junk RPM below this
#   lpf_fc_hz             : low-pass cutoff for v_wheel (8–12 Hz typical)
#   print_hz              : print rate
#
# Run:
#   ros2 run fslam_base wheel_vel_compare
#

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)

from nav_msgs.msg import Odometry
from eufs_msgs.msg import WheelSpeedsStamped


class WheelVelCompare(Node):
    def __init__(self):
        super().__init__("wheel_vel_compare", automatically_declare_parameters_from_overrides=True)

        # -------- Params (defaults are safe) --------
        self.declare_parameter("wheel_radius_m", 0.253)   # <-- set to your median
        self.declare_parameter("rpm_min", 5.0)             # ignore tiny/garbage RPM
        self.declare_parameter("lpf_fc_hz", 8.0)           # LPF cutoff for v_wheel
        self.declare_parameter("print_hz", 10.0)           # print rate (Hz)

        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("wheels_topic", "/ros_can/wheel_speeds")

        # Resolve
        P = lambda k: self.get_parameter(k).value
        self.Rw      = float(P("wheel_radius_m"))
        self.rpm_min = float(P("rpm_min"))
        self.fc      = float(P("lpf_fc_hz"))
        self.print_hz= float(P("print_hz"))
        self.odom_topic   = str(P("odom_topic"))
        self.wheels_topic = str(P("wheels_topic"))

        # -------- State --------
        self.last_ws = None       # last WheelSpeedsStamped.speeds
        self.last_ws_time = None  # sec
        self.v_wheel_lpf = 0.0
        self.last_print = 0.0
        self.v_gt = 0.0

        # -------- QoS: latest-wins to avoid lag --------
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, sub_qos)
        self.create_subscription(WheelSpeedsStamped, self.wheels_topic, self.cb_wheels, sub_qos)

        # Timer for periodic printing (decoupled from callback rates)
        self.timer = self.create_timer(1.0 / max(1.0, self.print_hz), self.on_timer)

        self.get_logger().info(
            f"[wheel_vel_compare] Rw={self.Rw:.6f} m, rpm_min={self.rpm_min:.1f}, lpf_fc={self.fc:.1f} Hz, print={self.print_hz:.1f} Hz"
        )

    # ---------- Callbacks ----------
    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.v_gt = math.hypot(vx, vy)

    def cb_wheels(self, msg: WheelSpeedsStamped):
        self.last_ws = msg.speeds
        self.last_ws_time = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9 if msg.header.stamp else time.time()

        # Compute raw wheel-based velocity from rear min RPM
        rb = float(self.last_ws.rb_speed)
        lb = float(self.last_ws.lb_speed)
        if abs(rb) < self.rpm_min or abs(lb) < self.rpm_min:
            return

        rpm = min(rb, lb)
        omega = rpm * (2.0 * math.pi / 60.0)  # rad/s
        v_raw = max(0.0, omega * self.Rw)

        # 1st-order low-pass on v_wheel
        now = self.last_ws_time if self.last_ws_time is not None else time.time()
        dt = max(1e-3, now - (self.last_print or now))
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.fc * dt)
        self.v_wheel_lpf += alpha * (v_raw - self.v_wheel_lpf)

    # ---------- Printer ----------
    def on_timer(self):
        t = time.time()
        if self.last_ws is None:
            self.get_logger().info("waiting for wheel speeds…")
            return

        v_w = self.v_wheel_lpf
        v_g = self.v_gt
        err = v_g - v_w
        rel = (err / v_g) if v_g > 0.2 else float('nan')

        # Clean, single-line print
        self.get_logger().info(
            f"v_wheel={v_w:5.2f} m/s | v_gt={v_g:5.2f} m/s | err={err:+5.2f} m/s | rel={'' if math.isnan(rel) else f'{rel:+.2%}'}"
        )
        self.last_print = t


def main():
    rclpy.init()
    node = WheelVelCompare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
