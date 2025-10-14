#!/usr/bin/env python3
# fslam_delaunay_live.py
#
# Live Delaunay triangulation of perceived cones (SciPy ONLY).

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from scipy.spatial import Delaunay  # SciPy required

class DelaunayLive(Node):
    def __init__(self):
        super().__init__('fslam_delaunay_live', automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('cones_topic', '/ground_truth/cones')
        self.declare_parameter('refresh_hz', 15.0)
        self.declare_parameter('auto_limits', True)
        self.declare_parameter('limit_margin_m', 2.0)
        self.declare_parameter('point_size', 18.0)
        self.declare_parameter('max_points', 0)  # 0 = unlimited

        gp = self.get_parameter
        self.cones_topic  = str(gp('cones_topic').value)
        self.refresh_hz   = float(gp('refresh_hz').value)
        self.auto_limits  = bool(gp('auto_limits').value)
        self.limit_margin = float(gp('limit_margin_m').value)
        self.pt_size      = float(gp('point_size').value)
        self.max_points   = int(gp('max_points').value)

        q = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q)

        self._lock = threading.Lock()
        self._blue   = np.zeros((0,2), float)
        self._yellow = np.zeros((0,2), float)
        self._orange = np.zeros((0,2), float)
        self._big    = np.zeros((0,2), float)

        self.get_logger().info(
            f"[delaunay] Subscribed {self.cones_topic} | refresh={self.refresh_hz:.1f} Hz | SciPy Delaunay"
        )

    def cb_cones(self, msg: ConeArrayWithCovariance):
        def to_xy(arr):
            if not arr: return np.zeros((0,2), float)
            xs = [float(c.point.x) for c in arr]
            ys = [float(c.point.y) for c in arr]
            P = np.stack([xs, ys], axis=1)
            if self.max_points > 0 and P.shape[0] > self.max_points:
                med = np.median(P, axis=0)
                d2  = np.sum((P - med[None,:])**2, axis=1)
                P = P[np.argsort(d2)[:self.max_points]]
            return P

        b   = to_xy(msg.blue_cones)
        y   = to_xy(msg.yellow_cones)
        o   = to_xy(msg.orange_cones)
        big = to_xy(msg.big_orange_cones)

        with self._lock:
            self._blue, self._yellow, self._orange, self._big = b, y, o, big

        self.get_logger().info(
            f"[delaunay] cones: blue={b.shape[0]} yellow={y.shape[0]} orange={o.shape[0]} big={big.shape[0]}"
        )

    def snapshot_points(self):
        with self._lock:
            return self._blue.copy(), self._yellow.copy(), self._orange.copy(), self._big.copy()

    @staticmethod
    def segments_from_delaunay(P: np.ndarray):
        if P.shape[0] < 3:
            return np.zeros((0, 2, 2), float)
        tri = Delaunay(P)
        edges = set()
        for a, b, c in tri.simplices:
            a, b, c = int(a), int(b), int(c)
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        if not edges:
            return np.zeros((0, 2, 2), float)
        return np.array([[P[i], P[j]] for (i, j) in edges], dtype=float)

def _spin_thread(node: DelaunayLive):
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().warn(f"[delaunay] spin exception: {e}")

def main():
    rclpy.init()
    node = DelaunayLive()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("Live Delaunay of Cones")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.7)

    s = node.pt_size
    scat_blue   = ax.scatter([], [], s=s, c='b', marker='o', label='Blue')
    scat_yellow = ax.scatter([], [], s=s, c='y', marker='o', edgecolors='k', linewidths=0.4, label='Yellow')
    scat_orange = ax.scatter([], [], s=s, c='orange', marker='o', label='Orange')
    scat_big    = ax.scatter([], [], s=s*1.4, c='r', marker='s', label='Big Orange')

    tri_lines = LineCollection([], linewidths=1.1, colors='0.35', alpha=0.85)
    ax.add_collection(tri_lines)
    ax.legend(loc='upper right')

    t = threading.Thread(target=_spin_thread, args=(node,), daemon=True)
    t.start()

    def set_offsets(scat, arr):
        scat.set_offsets(arr if arr.size else np.zeros((0,2)))

    def update(_frame):
        b, y, o, big = node.snapshot_points()
        set_offsets(scat_blue, b)
        set_offsets(scat_yellow, y)
        set_offsets(scat_orange, o)
        set_offsets(scat_big, big)

        parts = [arr for arr in (b, y, o, big) if arr.size]
        if parts:
            P = np.vstack(parts)
            try:
                segs = node.segments_from_delaunay(P)
                tri_lines.set_segments(segs)
            except Exception as e:
                node.get_logger().warn(f"[delaunay] triangulation failed: {e}")
                tri_lines.set_segments([])
        else:
            tri_lines.set_segments([])

        if node.auto_limits:
            pts = [arr for arr in (b, y, o, big) if arr.size]
            segs = tri_lines.get_segments()
            if segs:
                verts = np.vstack([np.asarray(segs).reshape(-1, 2)])
                pts.append(verts)
            if pts:
                allp = np.vstack(pts)
                xmin, ymin = np.min(allp, axis=0)
                xmax, ymax = np.max(allp, axis=0)
                m = node.limit_margin
                if xmax - xmin < 1e-3: xmax = xmin + 1.0
                if ymax - ymin < 1e-3: ymax = ymin + 1.0
                ax.set_xlim(xmin - m, xmax + m)
                ax.set_ylim(ymin - m, ymax + m)

        return scat_blue, scat_yellow, scat_orange, scat_big, tri_lines

    # KEEP A REFERENCE TO THE ANIMATION — or it won't update.
    ani = FuncAnimation(fig, update, interval=max(20, int(1000.0 / max(1e-3, node.refresh_hz))), blit=False)

    # Do an initial draw to avoid an empty first frame
    update(0)

    def _on_close(_evt):
        try:
            if rclpy.ok():
                node.get_logger().info("[delaunay] Closing… shutting down ROS.")
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
