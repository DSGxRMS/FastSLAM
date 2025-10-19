#!/usr/bin/env python3
import threading
from collections import deque
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class FastSLAMMapTrajViz(Node):
    def __init__(self):
        super().__init__("fastslam_map_traj_viz", automatically_declare_parameters_from_overrides=True)

        self.declare_parameter("topics.odom", "/fastslam/odom")
        self.declare_parameter("topics.map", "/fastslam/map_cones")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("traj.max_points", 20000)
        self.declare_parameter("quiet", True)

        self.topic_odom = str(self.get_parameter("topics.odom").value)
        self.topic_map = str(self.get_parameter("topics.map").value)
        self.fps = float(self.get_parameter("fps").value)
        self.traj_cap = int(self.get_parameter("traj.max_points").value)
        self.quiet = bool(self.get_parameter("quiet").value)

        qos_rel = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, self.topic_odom, self.cb_odom, qos_rel)
        self.create_subscription(ConeArrayWithCovariance, self.topic_map, self.cb_map, qos_rel)

        self.traj_x = deque(maxlen=self.traj_cap)
        self.traj_y = deque(maxlen=self.traj_cap)

        self.blue = np.zeros((0, 2), float)
        self.yellow = np.zeros((0, 2), float)
        self.orange = np.zeros((0, 2), float)
        self.big = np.zeros((0, 2), float)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("FastSLAM: Trajectory + Map")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, ls="--", alpha=0.25)

        (self.traj_line,) = self.ax.plot([], [], lw=1.8, label="Trajectory")
        self.sc_blue = self.ax.scatter([], [], s=18, label="Blue", c="tab:blue")
        self.sc_yel = self.ax.scatter([], [], s=18, label="Yellow", c="gold")
        self.sc_org = self.ax.scatter([], [], s=18, label="Orange", c="darkorange")
        self.sc_big = self.ax.scatter([], [], s=36, label="Big-Orange", c="red", marker="s", alpha=0.9)

        self.ax.legend(loc="upper right")

        interval_ms = max(10, int(1000.0 / max(1.0, self.fps)))
        self.ani = FuncAnimation(self.fig, self._on_anim_tick, interval=interval_ms, blit=False)

        def _on_close(_evt):
            try:
                self.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()

        self.fig.canvas.mpl_connect("close_event", _on_close)

    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self.traj_x.append(x)
        self.traj_y.append(y)

    def cb_map(self, msg: ConeArrayWithCovariance):
        def arr(cones):
            if not cones:
                return np.zeros((0, 2), float)
            xs = [float(c.point.x) for c in cones]
            ys = [float(c.point.y) for c in cones]
            return np.column_stack([xs, ys])

        self.blue = arr(msg.blue_cones)
        self.yellow = arr(msg.yellow_cones)
        self.orange = arr(msg.orange_cones)
        self.big = arr(msg.big_orange_cones)

    def _on_anim_tick(self, _frame):
        if self.traj_x and self.traj_y:
            self.traj_line.set_data(list(self.traj_x), list(self.traj_y))
        else:
            self.traj_line.set_data([], [])

        self.sc_blue.set_offsets(self.blue if self.blue.size else np.zeros((0, 2)))
        self.sc_yel.set_offsets(self.yellow if self.yellow.size else np.zeros((0, 2)))
        self.sc_org.set_offsets(self.orange if self.orange.size else np.zeros((0, 2)))
        self.sc_big.set_offsets(self.big if self.big.size else np.zeros((0, 2)))

        stacks = []
        if self.traj_x and self.traj_y:
            stacks.append(np.column_stack([np.array(self.traj_x), np.array(self.traj_y)]))
        for a in (self.blue, self.yellow, self.orange, self.big):
            if a.size:
                stacks.append(a)

        if stacks:
            all_pts = np.vstack(stacks)
            xmin, xmax = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
            ymin, ymax = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
            dx = max(6.0, (xmax - xmin) * 1.15 + 1.0)
            dy = max(6.0, (ymax - ymin) * 1.15 + 1.0)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            self.ax.set_xlim(cx - dx / 2.0, cx + dx / 2.0)
            self.ax.set_ylim(cy - dy / 2.0, cy + dy / 2.0)
        else:
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return (self.traj_line, self.sc_blue, self.sc_yel, self.sc_org, self.sc_big)


def main():
    rclpy.init()
    node = FastSLAMMapTrajViz()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
