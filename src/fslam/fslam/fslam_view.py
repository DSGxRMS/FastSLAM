#!/usr/bin/env python3
# fslam_view.py (PF-corrected vs GT; optional EKF)
#
# Subscribes to:
#   - Corrected (PF): /pf_localizer/odom_corrected  (Odometry)
#   - Ground truth:   /ground_truth/odom            (Odometry)
#   - Optional EKF:   /odometry_integration/car_state (Odometry)
#
# Plots live (single window with multiple panels):
#   - XY trajectory (corrected vs GT [+ EKF if enabled])
#   - Speed vs time (corrected vs GT [+ EKF])
#   - Yaw vs time (corrected vs GT [+ EKF], unwrapped)
#
# Usage:
#   ros2 run <your_pkg> live_plot
#
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def yaw_from_quat(qx,qy,qz,qw):
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class LivePlotNode(Node):
    def __init__(self):
        super().__init__("live_plotter", automatically_declare_parameters_from_overrides=True)

        # Topics (override via ROS params if needed)
        self.declare_parameter("topics.corr", "/fastslam/odom")     # PF corrected
        self.declare_parameter("topics.gt",   "/ground_truth/odom")               # GT
        self.declare_parameter("topics.ekf",  "/odometry_integration/car_state")  # Optional EKF. Set "" to disable.
        self.declare_parameter("plot.hz", 10.0)
        self.declare_parameter("buffer.seconds", 120.0)

        P = lambda k: self.get_parameter(k).value
        self.topic_corr = str(P("topics.corr"))
        self.topic_gt   = str(P("topics.gt"))
        self.topic_ekf  = str(P("topics.ekf"))
        self.plot_hz    = float(P("plot.hz"))
        self.buf_secs   = float(P("buffer.seconds"))

        # Buffers
        self.t0_corr = None
        self.t0_gt   = None
        self.t0_ekf  = None

        def make_buf():
            return {"t": deque(), "x": deque(), "y": deque(), "yaw": deque(), "v": deque()}
        self.buf_corr = make_buf()
        self.buf_gt   = make_buf()
        self.buf_ekf  = make_buf() if self.topic_ekf else None

        # QoS: corrected & EKF are odom; GT reliable
        qos_fast = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=100,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        qos_rel = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, self.topic_corr, self.cb_corr, qos_fast)
        self.create_subscription(Odometry, self.topic_gt,   self.cb_gt,   qos_rel)
        if self.topic_ekf:
            self.create_subscription(Odometry, self.topic_ekf,  self.cb_ekf,  qos_fast)

        # Matplotlib figure & axes
        plt.ion()
        self.fig = plt.figure(figsize=(11, 8))
        self.ax_xy   = self.fig.add_subplot(2, 2, 1)
        self.ax_v    = self.fig.add_subplot(2, 2, 3)
        self.ax_yaw  = self.fig.add_subplot(1, 2, 2)

        # Lines (initialized empty)
        (self.xy_corr_line,) = self.ax_xy.plot([], [], label="Corrected XY")
        (self.xy_gt_line,)   = self.ax_xy.plot([], [], label="GT XY")
        self.xy_ekf_line = None
        if self.buf_ekf is not None:
            (self.xy_ekf_line,) = self.ax_xy.plot([], [], label="EKF XY", linestyle=":")

        self.ax_xy.set_title("XY Trajectory")
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.grid(True, ls="--", alpha=0.3)
        self.ax_xy.legend(loc="best")

        (self.v_corr_line,) = self.ax_v.plot([], [], label="Corrected speed")
        (self.v_gt_line,)   = self.ax_v.plot([], [], label="GT speed")
        self.v_ekf_line = None
        if self.buf_ekf is not None:
            (self.v_ekf_line,) = self.ax_v.plot([], [], label="EKF speed", linestyle=":")

        self.ax_v.set_title("Speed vs Time")
        self.ax_v.set_xlabel("time (s)")
        self.ax_v.set_ylabel("m/s")
        self.ax_v.grid(True, ls="--", alpha=0.3)
        self.ax_v.legend(loc="best")

        (self.yaw_corr_line,) = self.ax_yaw.plot([], [], label="Corrected yaw (rad)")
        (self.yaw_gt_line,)   = self.ax_yaw.plot([], [], label="GT yaw (rad)")
        self.yaw_ekf_line = None
        if self.buf_ekf is not None:
            (self.yaw_ekf_line,) = self.ax_yaw.plot([], [], label="EKF yaw (rad)", linestyle=":")

        self.ax_yaw.set_title("Yaw vs Time")
        self.ax_yaw.set_xlabel("time (s)")
        self.ax_yaw.set_ylabel("rad")
        self.ax_yaw.grid(True, ls="--", alpha=0.3)
        self.ax_yaw.legend(loc="best")

        # Plot update timer
        self.create_timer(max(0.02, 1.0 / max(1e-3, self.plot_hz)), self.on_plot_timer)

        ekf_msg = f" | ekf={self.topic_ekf}" if self.topic_ekf else ""
        self.get_logger().info(f"Live plotter up: corrected={self.topic_corr} | gt={self.topic_gt}{ekf_msg} | Hz={self.plot_hz}")

    # ------------- callbacks -------------
    def cb_corr(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.t0_corr is None:
            self.t0_corr = t
        tt = t - self.t0_corr

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy)

        self._push(self.buf_corr, tt, x, y, yaw, v)

    def cb_gt(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.t0_gt is None:
            self.t0_gt = t
        tt = t - self.t0_gt

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy)

        self._push(self.buf_gt, tt, x, y, yaw, v)

    def cb_ekf(self, msg: Odometry):
        if self.buf_ekf is None:
            return
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.t0_ekf is None:
            self.t0_ekf = t
        tt = t - self.t0_ekf

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy)

        self._push(self.buf_ekf, tt, x, y, yaw, v)

    # ------------- helpers -------------
    def _push(self, buf, t, x, y, yaw, v):
        buf["t"].append(t)
        buf["x"].append(x)
        buf["y"].append(y)
        buf["yaw"].append(yaw)
        buf["v"].append(v)

        # trim by time
        while buf["t"] and (buf["t"][-1] - buf["t"][0]) > self.buf_secs:
            for k in ("t","x","y","yaw","v"):
                buf[k].popleft()

    def on_plot_timer(self):
        # XY
        if self.buf_corr["x"]:
            self.xy_corr_line.set_data(self.buf_corr["x"], self.buf_corr["y"])
        if self.buf_gt["x"]:
            self.xy_gt_line.set_data(self.buf_gt["x"], self.buf_gt["y"])
        if self.buf_ekf and self.buf_ekf["x"]:
            self.xy_ekf_line.set_data(self.buf_ekf["x"], self.buf_ekf["y"])

        # autoscale XY
        all_x = list(self.buf_corr["x"]) + list(self.buf_gt["x"])
        all_y = list(self.buf_corr["y"]) + list(self.buf_gt["y"])
        if self.buf_ekf:
            all_x += list(self.buf_ekf["x"]); all_y += list(self.buf_ekf["y"])
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            dx = max(1.0, (xmax - xmin) * 0.1)
            dy = max(1.0, (ymax - ymin) * 0.1)
            self.ax_xy.set_xlim(xmin - dx, xmax + dx)
            self.ax_xy.set_ylim(ymin - dy, ymax + dy)

        # Speed vs time
        if self.buf_corr["t"]:
            self.v_corr_line.set_data(self.buf_corr["t"], self.buf_corr["v"])
        if self.buf_gt["t"]:
            self.v_gt_line.set_data(self.buf_gt["t"], self.buf_gt["v"])
        if self.buf_ekf and self.buf_ekf["t"]:
            self.v_ekf_line.set_data(self.buf_ekf["t"], self.buf_ekf["v"])

        # Adjust x-limits for time plots
        all_t = []
        for buf in (self.buf_corr, self.buf_gt, self.buf_ekf if self.buf_ekf else []):
            if buf and buf.get("t") and len(buf["t"]) > 0:
                all_t += [buf["t"][0], buf["t"][-1]]
        if all_t:
            tmin, tmax = min(all_t), max(all_t)
            if tmax <= tmin:
                tmax = tmin + 1.0
            pad = 0.05 * (tmax - tmin)
            self.ax_v.set_xlim(tmin - pad, tmax + pad)
            self.ax_yaw.set_xlim(tmin - pad, tmax + pad)

        # Yaw vs time (unwrap for readability)
        if self.buf_corr["t"]:
            yawc = np.unwrap(np.array(self.buf_corr["yaw"]))
            self.yaw_corr_line.set_data(self.buf_corr["t"], yawc)
        if self.buf_gt["t"]:
            yawg = np.unwrap(np.array(self.buf_gt["yaw"]))
            self.yaw_gt_line.set_data(self.buf_gt["t"], yawg)
        if self.buf_ekf and self.buf_ekf["t"]:
            yawe = np.unwrap(np.array(self.buf_ekf["yaw"]))
            self.yaw_ekf_line.set_data(self.buf_ekf["t"], yawe)

        # autoscale y for v and yaw
        if self.buf_corr["v"] or self.buf_gt["v"] or (self.buf_ekf and self.buf_ekf["v"]):
            all_v = list(self.buf_corr["v"]) + list(self.buf_gt["v"])
            if self.buf_ekf:
                all_v += list(self.buf_ekf["v"])
            vmin, vmax = min(all_v), max(all_v)
            if vmax <= vmin:
                vmax = vmin + 0.5
            pad = 0.1 * (vmax - vmin)
            self.ax_v.set_ylim(vmin - pad, vmax + pad)

        yaws_all = []
        if self.buf_corr["yaw"]:
            yaws_all += list(np.unwrap(np.array(self.buf_corr["yaw"])))
        if self.buf_gt["yaw"]:
            yaws_all += list(np.unwrap(np.array(self.buf_gt["yaw"])))
        if self.buf_ekf and self.buf_ekf["yaw"]:
            yaws_all += list(np.unwrap(np.array(self.buf_ekf["yaw"])))
        if yaws_all:
            ymin, ymax = min(yaws_all), max(yaws_all)
            if ymax <= ymin:
                ymax = ymin + 0.1
            pad = 0.1 * (ymax - ymin)
            self.ax_yaw.set_ylim(ymin - pad, ymax + pad)

        # redraw
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    rclpy.init()
    node = LivePlotNode()
    try:
        # Keep matplotlib interactive window alive
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
