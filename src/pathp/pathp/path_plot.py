#!/usr/bin/env python3
# fslam_delaunay_live.py
#
# Live Delaunay with color constraints + pink midpoint X's +
# CYAN NN path from (0,0) that stops when a new segment would self-intersect.

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

LBL_BLUE   = 0
LBL_YELLOW = 1
LBL_ORANGE = 2
LBL_BIG    = 3  # big_orange = orange family

class DelaunayLive(Node):
    def __init__(self):
        super().__init__('fslam_delaunay_live', automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('cones_topic', '/ground_truth/cones')
        self.declare_parameter('refresh_hz', 15.0)
        self.declare_parameter('auto_limits', True)
        self.declare_parameter('limit_margin_m', 2.0)
        self.declare_parameter('point_size', 18.0)
        self.declare_parameter('max_points', 0)

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
            f"[delaunay] Subscribed {self.cones_topic} | refresh={self.refresh_hz:.1f} Hz | SciPy Delaunay enabled"
        )

    def cb_cones(self, msg: ConeArrayWithCovariance):
        def to_xy(arr):
            if not arr:
                return np.zeros((0,2), float)
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
            return (self._blue.copy(), self._yellow.copy(),
                    self._orange.copy(), self._big.copy())

    def constrained_segments_from_delaunay(self, P: np.ndarray, labels: np.ndarray):
        if P.shape[0] < 2:
            return np.zeros((0, 2, 2), float), np.zeros((0, 2, 2), float)

        edge_candidates = set()
        if P.shape[0] >= 3:
            tri = Delaunay(P)
            for a, b, c in tri.simplices:
                for i, j in ((a,b), (b,c), (c,a)):
                    if i > j: i, j = j, i
                    edge_candidates.add((i, j))
        else:
            edge_candidates.add((0, 1))

        seg_by = []   # Blue-Yellow (black)
        seg_or = []   # Orange family (red)

        for (i, j) in edge_candidates:
            li, lj = labels[i], labels[j]

            # Orange-family edges only within orange family
            if (li in (LBL_ORANGE, LBL_BIG)) or (lj in (LBL_ORANGE, LBL_BIG)):
                if (li in (LBL_ORANGE, LBL_BIG)) and (lj in (LBL_ORANGE, LBL_BIG)):
                    seg_or.append([P[i], P[j]])
                continue

            # Blue-Yellow only
            if {li, lj} == {LBL_BLUE, LBL_YELLOW}:
                seg_by.append([P[i], P[j]])
                continue

        S_by = np.array(seg_by, float) if seg_by else np.zeros((0,2,2), float)
        S_or = np.array(seg_or, float) if seg_or else np.zeros((0,2,2), float)
        return S_by, S_or


def _spin_thread(node: DelaunayLive):
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().warn(f"[delaunay] spin exception: {e}")


# --------- geometry helpers for non-self-intersecting path ---------
def _orient(a, b, c):
    # cross((b-a),(c-a))
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_seg(a, b, c, eps=1e-9):
    # c on segment ab?
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

def segments_intersect(p1, p2, q1, q2, eps=1e-9):
    # Proper segment intersection with collinearity handling
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    # General case
    if (o1*o2 < -eps) and (o3*o4 < -eps):
        return True

    # Collinear cases
    if abs(o1) <= eps and _on_seg(p1, p2, q1): return True
    if abs(o2) <= eps and _on_seg(p1, p2, q2): return True
    if abs(o3) <= eps and _on_seg(q1, q2, p1): return True
    if abs(o4) <= eps and _on_seg(q1, q2, p2): return True
    return False

def greedy_nn_path_no_self_intersections(start: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Greedy NN chain from 'start', stopping immediately if the next segment would
    intersect any already-built segment. Returns Kx2 array, including 'start'.
    """
    if pts.size == 0:
        return start.reshape(1,2)

    used = np.zeros(pts.shape[0], dtype=bool)
    path = [start.copy()]
    cur = start.copy()

    while True:
        idxs = np.where(~used)[0]
        if idxs.size == 0:
            break

        # Pick nearest remaining
        dif = pts[idxs] - cur[None,:]
        d2  = np.einsum('ij,ij->i', dif, dif)
        k   = idxs[int(np.argmin(d2))]
        nxt = pts[k]

        # Check intersection of (cur -> nxt) with any previous segment (path[i-1] -> path[i])
        intersect = False
        if len(path) >= 2:
            for i in range(1, len(path)):
                a, b = path[i-1], path[i]
                # Skip adjacency (sharing endpoint)
                if (np.allclose(a, cur) or np.allclose(b, cur)):
                    continue
                if segments_intersect(a, b, cur, nxt):
                    intersect = True
                    break

        if intersect:
            # Cut off the path as soon as a crossing would occur
            break

        # Accept this point
        path.append(nxt.copy())
        used[k] = True
        cur = nxt

    return np.stack(path, axis=0)


def main():
    rclpy.init()
    node = DelaunayLive()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Cones + Constrained Delaunay + Midpoints + NN Non-Intersecting Path")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.6)

    s = node.pt_size
    scat_blue   = ax.scatter([], [], s=s, c='b', marker='o', label='Blue')
    scat_yellow = ax.scatter([], [], s=s, c='y', marker='o', edgecolors='k', linewidths=0.4, label='Yellow')
    scat_orange = ax.scatter([], [], s=s, c='orange', marker='o', label='Orange')
    scat_big    = ax.scatter([], [], s=s*1.3, c='r', marker='s', label='Big Orange')

    lc_by = LineCollection([], linewidths=1.8, colors='k', alpha=0.75, label='Blue↔Yellow')
    lc_or = LineCollection([], linewidths=1.8, colors='r', alpha=0.65, label='Orange-family')
    ax.add_collection(lc_by)
    ax.add_collection(lc_or)

    # Pink X midpoints (including fixed origin)
    scat_mid = ax.scatter([], [], s=s*0.9, c='#ff2fa6', marker='x', linewidths=1.8, label='Midpoints + (0,0)')

    # Cyan path through midpoints (nearest-neighbor from origin, non self-intersecting)
    path_line, = ax.plot([], [], '-', color='c', linewidth=2.0, alpha=0.9, label='NN non-intersecting path')

    ax.legend(loc='upper right')

    t = threading.Thread(target=_spin_thread, args=(node,), daemon=True)
    t.start()

    def set_offsets(scat, arr):
        scat.set_offsets(arr if arr.size else np.zeros((0,2)))

    def update(_frame):
        B, Y, O, G = node.snapshot_points()
        set_offsets(scat_blue,   B)
        set_offsets(scat_yellow, Y)
        set_offsets(scat_orange, O)
        set_offsets(scat_big,    G)

        parts, labls = [], []
        if B.size: parts.append(B); labls.append(np.full(B.shape[0], LBL_BLUE,   int))
        if Y.size: parts.append(Y); labls.append(np.full(Y.shape[0], LBL_YELLOW, int))
        if O.size: parts.append(O); labls.append(np.full(O.shape[0], LBL_ORANGE, int))
        if G.size: parts.append(G); labls.append(np.full(G.shape[0], LBL_BIG,    int))

        mids = None
        if parts:
            P = np.vstack(parts)
            L = np.concatenate(labls)
            S_by, S_or = node.constrained_segments_from_delaunay(P, L)
            lc_by.set_segments(S_by)
            lc_or.set_segments(S_or)

            mids_list = []
            if S_by.shape[0]:
                mids_list.append(S_by.mean(axis=1))
            if S_or.shape[0]:
                mids_list.append(S_or.mean(axis=1))
            if mids_list:
                mids = np.vstack(mids_list)
        else:
            lc_by.set_segments([])
            lc_or.set_segments([])

        origin = np.array([[0.0, 0.0]], float)
        if mids is None or not mids.size:
            M = origin
        else:
            M = np.vstack([mids, origin])
        set_offsets(scat_mid, M)

        # NN non-self-intersecting path from origin through midpoints
        if M.shape[0] >= 2:
            is_origin = np.isclose(M[:,0], 0.0) & np.isclose(M[:,1], 0.0)
            Pm = M[~is_origin]
            path = greedy_nn_path_no_self_intersections(origin.reshape(1,2)[0], Pm)
            path_line.set_data(path[:,0], path[:,1])
        else:
            path_line.set_data([], [])

        if node.auto_limits:
            pts = []
            for arr in (B, Y, O, G, M):
                if arr.size: pts.append(arr)
            if lc_by.get_segments():
                pts.append(np.vstack(lc_by.get_segments()).reshape(-1,2))
            if lc_or.get_segments():
                pts.append(np.vstack(lc_or.get_segments()).reshape(-1,2))
            xdat = path_line.get_xdata(); ydat = path_line.get_ydata()
            if len(xdat) and len(ydat):
                pts.append(np.stack([np.asarray(xdat), np.asarray(ydat)], axis=1))
            if pts:
                allp = np.vstack(pts)
                xmin, ymin = np.min(allp, axis=0)
                xmax, ymax = np.max(allp, axis=0)
                m = node.limit_margin
                if xmax - xmin < 1e-3: xmax = xmin + 1.0
                if ymax - ymin < 1e-3: ymax = ymin + 1.0
                ax.set_xlim(xmin - m, xmax + m)
                ax.set_ylim(ymin - m, ymax + m)

        return (scat_blue, scat_yellow, scat_orange, scat_big, lc_by, lc_or, scat_mid, path_line)

    interval_ms = max(20, int(1000.0 / max(1e-3, node.refresh_hz)))
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)

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
