#!/usr/bin/env python3
# viz_online_map_matplot_node.py
#
# ROS 2 node: live 2D matplotlib view of /online_map/cones
# - Subscribes BEST_EFFORT
# - Colors: blue, yellow, orange, big-orange (red squares)
# - Runs rclpy.spin in a background thread; matplotlib owns the main thread
# - Has a proper main() for use with `ros2 run`
#
# Params:
#   topic (string)         default: "/online_map/cones"
#   fps   (double)         default: 15.0      (matplotlib update rate)
#   quiet (bool)           default: true      (suppress logs)

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use("TkAgg")  # change to "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MapViewerNode(Node):
    def __init__(self):
        super().__init__("viz_online_map_matplot", automatically_declare_parameters_from_overrides=True)

        # ---- params ----
        self.declare_parameter("topic", "/online_map/cones")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("quiet", True)

        self.topic = str(self.get_parameter("topic").value)
        self.fps   = float(self.get_parameter("fps").value)
        self.quiet = bool(self.get_parameter("quiet").value)

        # ---- QoS: BEST_EFFORT ----
        qos_best = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # shared buffers
        self._lock = threading.Lock()
        self._blue   = np.zeros((0,2), dtype=float)
        self._yellow = np.zeros((0,2), dtype=float)
        self._orange = np.zeros((0,2), dtype=float)
        self._big    = np.zeros((0,2), dtype=float)

        # sub
        self.create_subscription(ConeArrayWithCovariance, self.topic, self.cb_map, qos_best)

        if not self.quiet:
            self.get_logger().info(f"[viz] Subscribing BEST_EFFORT to {self.topic}")

        # prepare matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ax.set_title(f"Online Cone Map ({self.topic})")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal", adjustable="box")

        # scatters
        self.sc_blue = self.ax.scatter([], [], s=16, label="Blue",   c="tab:blue")
        self.sc_yel  = self.ax.scatter([], [], s=16, label="Yellow", c="gold")
        self.sc_org  = self.ax.scatter([], [], s=16, label="Orange", c="darkorange")
        self.sc_big  = self.ax.scatter([], [], s=28, label="Big-Orange", c="red", marker="s", alpha=0.9)
        self.ax.legend(loc="upper right")

        # animation
        interval_ms = max(10, int(1000.0 / max(1.0, self.fps)))
        self.ani = FuncAnimation(self.fig, self._on_anim_tick, interval=interval_ms, blit=False)

        # close hook -> shutdown ROS
        def _on_close(_evt):
            try:
                self.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()
        self.fig.canvas.mpl_connect("close_event", _on_close)

    # ---- ROS callback ----
    def cb_map(self, msg: ConeArrayWithCovariance):
        def arr(cones):
            if not cones:
                return np.zeros((0,2), dtype=float)
            xs = [float(c.point.x) for c in cones]
            ys = [float(c.point.y) for c in cones]
            return np.column_stack([xs, ys])
        with self._lock:
            self._blue   = arr(msg.blue_cones)
            self._yellow = arr(msg.yellow_cones)
            self._orange = arr(msg.orange_cones)
            self._big    = arr(msg.big_orange_cones)

    def _get_buffers(self):
        with self._lock:
            return (self._blue.copy(), self._yellow.copy(),
                    self._orange.copy(), self._big.copy())

    # ---- Matplotlib animation tick ----
    def _on_anim_tick(self, _frame):
        b, y, o, g = self._get_buffers()

        self.sc_blue.set_offsets(b if b.size else np.zeros((0,2)))
        self.sc_yel.set_offsets(y if y.size else np.zeros((0,2)))
        self.sc_org.set_offsets(o if o.size else np.zeros((0,2)))
        self.sc_big.set_offsets(g if g.size else np.zeros((0,2)))

        # autoscale with padding
        stacks = [a for a in (b,y,o,g) if a.size]
        if stacks:
            all_pts = np.vstack(stacks)
            xmin, xmax = all_pts[:,0].min(), all_pts[:,0].max()
            ymin, ymax = all_pts[:,1].min(), all_pts[:,1].max()
            dx = max(4.0, (xmax - xmin) + 2.0)
            dy = max(4.0, (ymax - ymin) + 2.0)
            cx, cy = 0.5*(xmin + xmax), 0.5*(ymin + ymax)
            self.ax.set_xlim(cx - dx/2.0, cx + dx/2.0)
            self.ax.set_ylim(cy - dy/2.0, cy + dy/2.0)
        else:
            self.ax.set_xlim(-5,5); self.ax.set_ylim(-5,5)

        return (self.sc_blue, self.sc_yel, self.sc_org, self.sc_big)

def main():
    rclpy.init()
    node = MapViewerNode()

    # spin ROS in a background thread so matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
