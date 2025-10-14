#!/usr/bin/env python3
# side_fit_minimal.py
#
# Split incoming cones (map frame) into blue / yellow, fit a curve to each,
# and publish dense discretized points as nav_msgs/Path.
#
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

class SideFitMinimal(Node):
    def __init__(self):
        super().__init__("side_fit_minimal", automatically_declare_parameters_from_overrides=True)

        # Topics
        self.declare_parameter("topics.cones_in", "/ground_truth/cones")
        self.declare_parameter("topics.left_out", "/planner/left_path")    # blue
        self.declare_parameter("topics.right_out", "/planner/right_path")  # yellow

        # Fitting options
        self.declare_parameter("samples", 1000)         # discretized samples per side
        self.declare_parameter("poly_deg_pref", 3)      # when SciPy not present: y(x) degree (3->2->1)
        self.declare_parameter("spline_smooth", 0.25)   # smoothing factor scale if SciPy present

        gp = self.get_parameter
        self.cones_topic = str(gp("topics.cones_in").value)
        self.left_out    = str(gp("topics.left_out").value)
        self.right_out   = str(gp("topics.right_out").value)
        self.Ns          = int(gp("samples").value)
        self.poly_pref   = int(gp("poly_deg_pref").value)
        self.smooth_sc   = float(gp("spline_smooth").value)

        # SciPy optional
        self._have_scipy = False
        try:
            from scipy.interpolate import splprep, splev  # noqa: F401
            self._have_scipy = True
        except Exception:
            self._have_scipy = False
            self.get_logger().warn("[side_fit] SciPy not found; using polynomial fallback.")

        q = QoSProfile(depth=60, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q)
        self.pub_left  = self.create_publisher(Path, self.left_out, 10)
        self.pub_right = self.create_publisher(Path, self.right_out, 10)

        self.get_logger().info(
            f"[side_fit] cones_in={self.cones_topic} | left_out={self.left_out} | right_out={self.right_out} | "
            f"samples={self.Ns} | SciPy={self._have_scipy}"
        )

    def cb_cones(self, msg: ConeArrayWithCovariance):
        # Grab map-frame arrays
        def to_xy(arr):
            if not arr:
                return np.zeros((0,2), float)
            xs = [float(c.point.x) for c in arr]
            ys = [float(c.point.y) for c in arr]
            return np.stack([xs, ys], axis=1)

        blue   = to_xy(msg.blue_cones)     # left
        yellow = to_xy(msg.yellow_cones)   # right

        # Fit & publish each side
        if blue.shape[0] >= 2:
            path_b = self._fit_to_path(blue, msg.header.stamp)
            self.pub_left.publish(path_b)
        else:
            self.pub_left.publish(self._empty_path(msg.header.stamp))

        if yellow.shape[0] >= 2:
            path_y = self._fit_to_path(yellow, msg.header.stamp)
            self.pub_right.publish(path_y)
        else:
            self.pub_right.publish(self._empty_path(msg.header.stamp))

    # -------- fitting --------
    def _fit_to_path(self, P: np.ndarray, stamp) -> Path:
        """
        P: Nx2 scattered points (map frame).
        Return a Path sampled along x (polyfit) or along arclength (spline).
        """
        # Sort by x for deterministic ordering
        P = P[np.argsort(P[:,0])]

        if self._have_scipy and P.shape[0] >= 4:
            try:
                from scipy.interpolate import splprep, splev
                # parameterize by cumulative arclength for stability
                dif = np.diff(P, axis=0)
                seg = np.sqrt((dif**2).sum(axis=1))
                s = np.concatenate([[0.0], np.cumsum(seg)])
                if s[-1] < 1e-6:
                    return self._empty_path(stamp)
                u = s / s[-1]
                smooth = max(0.0, self.smooth_sc) * P.shape[0]
                tck, _ = splprep([P[:,0], P[:,1]], u=u, s=smooth, k=3)
                uu = np.linspace(0.0, 1.0, max(10, self.Ns))
                xx, yy = splev(uu, tck)
                C = np.stack([xx, yy], axis=1)
                return self._points_to_path(C, stamp)
            except Exception as e:
                self.get_logger().warn(f"[side_fit] spline failed ({type(e).__name__}); fallback to polyfit.")

        # Polyfit y(x) fallback
        x = P[:,0]; y = P[:,1]
        # degree not more than N-1 and not more than pref
        deg = min(self.poly_pref, max(1, min(3, P.shape[0]-1)))
        try:
            coefs = np.polyfit(x, y, deg=deg)
            xs = np.linspace(float(x.min()), float(x.max()), max(10, self.Ns))
            ys = np.polyval(coefs, xs)
            C = np.stack([xs, ys], axis=1)
            return self._points_to_path(C, stamp)
        except Exception:
            return self._empty_path(stamp)

    # -------- helpers --------
    def _points_to_path(self, C: np.ndarray, stamp) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        for i in range(C.shape[0]):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(C[i,0])
            ps.pose.position.y = float(C[i,1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def _empty_path(self, stamp) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        return path

def main():
    rclpy.init()
    node = SideFitMinimal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
