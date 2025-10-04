#!/usr/bin/env python3
import math, time
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# =============================
#      Small helpers
# =============================
def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def dot(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return a[0]*b[0] + a[1]*b[1]

def wrap_to_2pi(a: float) -> float:
    return a % (2.0*math.pi)

def unit_vec_from_yaw(yaw: float) -> Tuple[float,float]:
    return (math.cos(yaw), math.sin(yaw))

def right_normal_from_yaw(yaw: float) -> Tuple[float,float]:
    return (math.sin(yaw), -math.cos(yaw))

def left_normal_from_yaw(yaw: float) -> Tuple[float,float]:
    return (-math.sin(yaw), math.cos(yaw))


# =============================
#         Main Node
# =============================
class FixedRouteDriver(Node):
    def __init__(self):
        super().__init__("fixed_route_driver", automatically_declare_parameters_from_overrides=True)

        # -------- Params --------
        # kinematics / command
        self.declare_parameter("wheelbase_m", 1.5)
        self.declare_parameter("radius_m", 10.15)               # circle radius
        self.declare_parameter("target_speed_mps", 5.4)
        self.declare_parameter("steering_right_sign", -1.0)     # -1: right turn negative (most sims). +1 if opposite.
        self.declare_parameter("control_hz", 50.0)

        # accel controller (accel-first API)
        self.declare_parameter("accel_gain", 1.5)               # maps (v_target - v_meas) -> accel
        self.declare_parameter("accel_limit", 3.0)              # |m/s^2| clamp

        # segment lengths
        self.declare_parameter("straight1_m", 16.8)
        self.declare_parameter("straight2_m", 17.0)

        # topics
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("cmd_topic", "/cmd")             # change to /drive if your stack expects that

        # QoS
        self.declare_parameter("qos_best_effort", True)

        P = self.get_parameter
        val = lambda k: P(k).value  # ROS2 returns the correct typed value

        self.wheelbase       = float(val("wheelbase_m"))
        self.radius          = float(val("radius_m"))
        self.v_target        = float(val("target_speed_mps"))
        self.right_sign      = float(val("steering_right_sign"))
        self.control_hz      = float(val("control_hz"))
        self.accel_gain      = float(val("accel_gain"))
        self.accel_limit     = float(val("accel_limit"))

        self.seg1_len        = float(val("straight1_m"))
        self.seg2_len        = float(val("straight2_m"))

        self.odom_topic      = str(val("odom_topic"))
        self.cmd_topic       = str(val("cmd_topic"))
        self.qos_best_effort = bool(val("qos_best_effort"))

        # -------- State machine --------
        # WAIT_INIT -> STRAIGHT1 -> CIRCLE -> CIRCLE2 -> STRAIGHT2 -> DONE
        self.state = "WAIT_INIT"
        self.p0 = (0.0, 0.0)       # start pos
        self.yaw0 = 0.0
        self.fwd0 = (1.0, 0.0)

        self.s1_progress = 0.0

        # Circle 1
        self.p1 = (0.0, 0.0)       # circle1 entry
        self.yaw1 = 0.0
        self.center = (0.0, 0.0)
        self.theta_start = 0.0
        self.theta_progress = 0.0

        # Circle 2
        self.p1b = (0.0, 0.0)      # circle2 entry
        self.yaw1b = 0.0
        self.center2 = (0.0, 0.0)
        self.theta_start2 = 0.0
        self.theta_progress2 = 0.0

        # Exit to straight 2
        self.p2 = (0.0, 0.0)
        self.yaw2 = 0.0
        self.fwd2 = (1.0, 0.0)
        self.s2_progress = 0.0

        # odom
        self.have_odom = False
        self.pos = (0.0, 0.0)
        self.yaw = 0.0
        self.speed = 0.0

        # fixed circle steer (feedforward)
        self.delta_circle = math.atan(self.wheelbase / max(0.01, self.radius)) * self.right_sign

        # -------- I/O --------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.qos_best_effort
                        else QoSReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)
        self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        # control timer
        self.timer = self.create_timer(1.0 / max(1.0, self.control_hz), self._tick)

        self.get_logger().info(
            f"[route] wheelbase={self.wheelbase:.3f} m, R={self.radius:.3f} m, delta_circle={self.delta_circle:.3f} rad"
        )

    # -------- Callbacks --------
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pos = (float(p.x), float(p.y))
        self.yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        vx = float(msg.twist.twist.linear.x); vy = float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)
        self.have_odom = True

    # -------- Publish helper (ACCEL-FIRST) --------
    def _publish_cmd(self, accel: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration   = float(accel)
        msg.drive.speed          = 0.0
        self.pub_ack.publish(msg)

    # -------- State machine control --------
    def _tick(self):
        # Always publish something so some sims start moving even before odom shows up
        if not self.have_odom:
            self._publish_cmd(accel=min(self.accel_limit, 0.5*self.accel_limit), steering=0.0)
            return

        steering_cmd = 0.0  # default straight

        if self.state == "WAIT_INIT":
            self.p0 = self.pos
            self.yaw0 = self.yaw
            self.fwd0 = unit_vec_from_yaw(self.yaw0)
            self.s1_progress = 0.0
            self.state = "STRAIGHT1"
            self.get_logger().info("[state] STRAIGHT1 start")

        elif self.state == "STRAIGHT1":
            d = (self.pos[0] - self.p0[0], self.pos[1] - self.p0[1])
            self.s1_progress = dot(d, self.fwd0)
            steering_cmd = 0.0

            if self.s1_progress >= self.seg1_len:
                # enter circle 1
                self.p1 = self.pos
                self.yaw1 = self.yaw
                n_hat = right_normal_from_yaw(self.yaw1) if self.right_sign < 0 else left_normal_from_yaw(self.yaw1)
                self.center = (self.p1[0] + self.radius * n_hat[0],
                               self.p1[1] + self.radius * n_hat[1])
                self.theta_start = math.atan2(self.p1[1] - self.center[1],
                                              self.p1[0] - self.center[0])
                self.theta_progress = 0.0
                self.state = "CIRCLE"
                self.get_logger().info(f"[state] CIRCLE start center={self.center} theta0={self.theta_start:.3f}")

        elif self.state == "CIRCLE":
            steering_cmd = self.delta_circle  # fixed steer for constant-R turn

            theta_now = math.atan2(self.pos[1] - self.center[1],
                                   self.pos[0] - self.center[0])
            if self.right_sign < 0:
                self.theta_progress = wrap_to_2pi(self.theta_start - theta_now)   # clockwise for right turn
            else:
                self.theta_progress = wrap_to_2pi(theta_now - self.theta_start)   # CCW for left turn

            if self.theta_progress >= (2.0 * math.pi - 1e-2):
                # start second identical circle
                self.p1b = self.pos
                self.yaw1b = self.yaw
                n_hat = right_normal_from_yaw(self.yaw1b) if self.right_sign < 0 else left_normal_from_yaw(self.yaw1b)
                self.center2 = (self.p1b[0] + self.radius * n_hat[0],
                                self.p1b[1] + self.radius * n_hat[1])
                self.theta_start2 = math.atan2(self.p1b[1] - self.center2[1],
                                               self.p1b[0] - self.center2[0])
                self.theta_progress2 = 0.0
                self.state = "CIRCLE2"
                self.get_logger().info(f"[state] CIRCLE2 start center={self.center2} theta0={self.theta_start2:.3f}")

        elif self.state == "CIRCLE2":
            steering_cmd = self.delta_circle  # same radius & direction

            theta_now2 = math.atan2(self.pos[1] - self.center2[1],
                                    self.pos[0] - self.center2[0])
            if self.right_sign < 0:
                self.theta_progress2 = wrap_to_2pi(self.theta_start2 - theta_now2)
            else:
                self.theta_progress2 = wrap_to_2pi(theta_now2 - self.theta_start2)

            if self.theta_progress2 >= (2.0 * math.pi - 1e-2):
                # exit to straight2
                self.p2 = self.pos
                self.yaw2 = self.yaw
                self.fwd2 = unit_vec_from_yaw(self.yaw2)
                self.s2_progress = 0.0
                self.state = "STRAIGHT2"
                self.get_logger().info("[state] STRAIGHT2 start")

        elif self.state == "STRAIGHT2":
            steering_cmd = 0.0
            d = (self.pos[0] - self.p2[0], self.pos[1] - self.p2[1])
            self.s2_progress = dot(d, self.fwd2)

            if self.s2_progress >= self.seg2_len:
                self.state = "DONE"
                self.get_logger().info("[state] DONE; braking to stop")

        else:  # DONE
            # brake to zero
            v_err = 0.0 - self.speed
            accel_cmd = max(-self.accel_limit, min(self.accel_limit, self.accel_gain * v_err))
            self._publish_cmd(accel=accel_cmd, steering=0.0)
            return

        # accel control to reach/hold v_target
        v_err = self.v_target - self.speed
        accel_cmd = max(-self.accel_limit, min(self.accel_limit, self.accel_gain * v_err))
        self._publish_cmd(accel=accel_cmd, steering=steering_cmd)


def main():
    rclpy.init()
    node = FixedRouteDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
