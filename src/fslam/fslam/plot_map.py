#!/usr/bin/env python3
# fslam_map_plotter.py
#
# Live 2D plot of cones from /mapper/known_map (eufs_msgs/ConeArrayWithCovariance).
# Colors:
#   blue_cones     -> 'b' circle
#   yellow_cones   -> 'y' circle
#   orange_cones   -> 'orange' circle
#   big_orange     -> 'r' square (bigger)
#
# Usage:
#   ros2 run your_pkg fslam_map_plotter
#   # or with args:
#   ros2 run your_pkg fslam_map_plotter --ros-args -p topic:=/mapper/known_map -p refresh_hz:=15.0

import threading, time, math
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time as RclTime

import matplotlib
matplotlib.use('TkAgg')  # or Qt5Agg if preferred
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from eufs_msgs.msg import ConeArrayWithCovariance

class MapPlotterNode(Node):
    def __init__(self):
        super().__init__('fslam_map_plotter', automatically_declare_parameters_from_overrides=True)

        # Params
        self.declare_parameter('topic', '/mapper/known_map')
        self.declare_parameter('refresh_hz', 10.0)
        self.declare_parameter('auto_limits', True)
        self.declare_parameter('limit_margin_m', 3.0)
        self.declare_parameter('point_size', 18.0)

        self.topic = str(self.get_parameter('topic').value)
        self.refresh_hz = float(self.get_parameter('refresh_hz').value)
        self.auto_limits = bool(self.get_parameter('auto_limits').value)
        self.limit_margin = float(self.get_parameter('limit_margin_m').value)
        self.pt_size = float(self.get_parameter('point_size').value)

        q = QoSProfile(depth=10,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       history=QoSHistoryPolicy.KEEP_LAST)

        self._lock = threading.Lock()
        self._data = {
            'blue':   np.zeros((0,2), float),
            'yellow': np.zeros((0,2), float),
            'orange': np.zeros((0,2), float),
            'big':    np.zeros((0,2), float),
        }

        self.create_subscription(ConeArrayWithCovariance, self.topic, self.cb_map, q)
        self.get_logger().info(f"[plotter] Subscribed to {self.topic} | refresh {self.refresh_hz:.1f} Hz")

    def cb_map(self, msg: ConeArrayWithCovariance):
        # Convert each cone array into Nx2 arrays
        def to_xy(arr):
            if not arr:
                return np.zeros((0,2), float)
            xs = [float(c.point.x) for c in arr]
            ys = [float(c.point.y) for c in arr]
            return np.stack([xs, ys], axis=1)

        blue   = to_xy(msg.blue_cones)
        yellow = to_xy(msg.yellow_cones)
        orange = to_xy(msg.orange_cones)
        big    = to_xy(msg.big_orange_cones)

        with self._lock:
            self._data['blue']   = blue
            self._data['yellow'] = yellow
            self._data['orange'] = orange
            self._data['big']    = big

    def get_snapshot(self):
        with self._lock:
            # Return copies to avoid race while plotting
            return (self._data['blue'].copy(),
                    self._data['yellow'].copy(),
                    self._data['orange'].copy(),
                    self._data['big'].copy())

def _spin_thread(node: Node):
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().warn(f"[plotter] spin exception: {e}")

def main():
    rclpy.init()
    node = MapPlotterNode()

    # Start rclpy spinning in a background thread so matplotlib can own the main thread
    t = threading.Thread(target=_spin_thread, args=(node,), daemon=True)
    t.start()

    # Matplotlib setup
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title('Cone Map')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.6)

    # Artists
    s = node.pt_size
    scat_blue   = ax.scatter([], [], s=s, c='b', marker='o', label='Blue')
    scat_yellow = ax.scatter([], [], s=s, c='y', marker='o', edgecolors='k', linewidths=0.4, label='Yellow')
    scat_orange = ax.scatter([], [], s=s, c='orange', marker='o', label='Orange')
    scat_big    = ax.scatter([], [], s=s*1.5, c='r', marker='s', label='Big Orange')

    ax.legend(loc='upper right')

    def update(_frame):
        b, y, o, big = node.get_snapshot()

        # Update offsets
        scat_blue.set_offsets(b if b.size else np.zeros((0,2)))
        scat_yellow.set_offsets(y if y.size else np.zeros((0,2)))
        scat_orange.set_offsets(o if o.size else np.zeros((0,2)))
        scat_big.set_offsets(big if big.size else np.zeros((0,2)))

        if node.auto_limits:
            # Compute bounds over all points
            pts = []
            if b.size: pts.append(b)
            if y.size: pts.append(y)
            if o.size: pts.append(o)
            if big.size: pts.append(big)
            if pts:
                allp = np.vstack(pts)
                xmin, ymin = np.min(allp, axis=0)
                xmax, ymax = np.max(allp, axis=0)
                # Add margin
                m = node.limit_margin
                ax.set_xlim(xmin - m, xmax + m)
                ax.set_ylim(ymin - m, ymax + m)

        return scat_blue, scat_yellow, scat_orange, scat_big

    interval_ms = max(20, int(1000.0 / max(1e-3, node.refresh_hz)))
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)

    def _on_close(_evt):
        try:
            if rclpy.ok():
                node.get_logger().info("[plotter] Closing… shutting down ROS.")
                rclpy.shutdown()
        except Exception:
            pass

    fig.canvas.mpl_connect('close_event', _on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
