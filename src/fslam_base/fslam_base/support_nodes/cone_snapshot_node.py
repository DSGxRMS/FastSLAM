# - Subscribes: /ground_truth/cones (eufs_msgs/msg/ConeArrayWithCovariance)
# - Aggregates ~duration seconds of messages while stationary (you start it when the car is still)
# - Deduplicates cones per color by spatial clustering (greedy, radius = cluster_eps)
# - Outputs one CSV with medians per cluster (x,y,color), plus commented meta lines

import csv
import math
import os
from statistics import median
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from eufs_msgs.msg import ConeArrayWithCovariance

# ---- Helpers ----

def extract_xy_list(msg: ConeArrayWithCovariance) -> List[Tuple[float, float, str]]:
    """Extract (x, y, color) triples from ConeArrayWithCovariance in its frame."""
    result: List[Tuple[float, float, str]] = []

    def _pull(arr, color_label: str):
        if arr is None:
            return
        for c in arr:
            # CHECK WHICH TYPE WORKS HERE
            if hasattr(c, 'point'):
                x = float(getattr(c.point, 'x', 0.0))
                y = float(getattr(c.point, 'y', 0.0))
            else:
                x = float(getattr(c, 'x', 0.0))
                y = float(getattr(c, 'y', 0.0))
            result.append((x, y, color_label))

    # Try to be tolerant with field names ####REPLACE WITH EXACTS ONCE CONFIRMED
    _pull(getattr(msg, 'blue_cones', None),   'blue')
    _pull(getattr(msg, 'yellow_cones', None), 'yellow')
    _pull(getattr(msg, 'orange_cones', None), 'orange')        # small/orange
    _pull(getattr(msg, 'big_orange_cones', None), 'orange_large')  # start-box “large” cones, if present
    _pull(getattr(msg, 'unknown_color_cones', None), 'unknown')

    return result

def greedy_cluster(points: List[Tuple[float, float]],
                   eps: float) -> List[List[Tuple[float, float]]]:
    """Simple greedy clustering by distance threshold eps. No dependencies."""
    clusters: List[List[Tuple[float, float]]] = []
    for p in points:
        px, py = p
        assigned = False
        for cl in clusters:
            # Compare to cluster centroid (mean) for speed
            cx = sum(x for x, _ in cl) / len(cl)
            cy = sum(y for _, y in cl) / len(cl)
            if math.hypot(px - cx, py - cy) <= eps:
                cl.append(p)
                assigned = True
                break
        if not assigned:
            clusters.append([p])
    return clusters

def median_of_cluster(cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [x for x, _ in cluster]
    ys = [y for _, y in cluster]
    return (median(xs), median(ys))

# ---- Node ----

class ConeSnapshotNode(Node):
    def __init__(self):
        super().__init__('cone_snapshot_node')

        # Params
        self.declare_parameter('topic', '/ground_truth/cones')
        self.declare_parameter('duration_sec', 1.0)      # how long to aggregate
        self.declare_parameter('cluster_eps', 0.30)      # meters; tweak to your cone spacing/GT jitter
        self.declare_parameter('out_csv', 'data/cone_snapshot.csv')

        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.duration_sec = float(self.get_parameter('duration_sec').value)
        self.cluster_eps = float(self.get_parameter('cluster_eps').value)
        self.out_csv = self.get_parameter('out_csv').get_parameter_value().string_value

        # QoS to match /ground_truth/cones (RELIABLE, VOLATILE)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            ConeArrayWithCovariance, topic, self.cb_cones, qos)

        # Buffers
        self.start_time_ros: Time = None
        self.end_time_ros: Time = None
        self.frame_id: str = ''
        self.buffer: Dict[str, List[Tuple[float, float]]] = {
            'blue': [], 'yellow': [], 'orange': [], 'orange_large': [], 'unknown': []
        }

        # Timer to stop after duration
        self.timer = self.create_timer(0.05, self.check_done)
        self.started_wall = self.get_clock().now()

        self.get_logger().info(
            f"[ConeSnapshot] Listening on {topic} for ~{self.duration_sec:.2f}s; "
            f"cluster_eps={self.cluster_eps:.2f} m; output={self.out_csv}"
        )

    def cb_cones(self, msg: ConeArrayWithCovariance):
        # First/last message stamps (ROS Time)
        t = Time.from_msg(msg.header.stamp) if hasattr(msg, 'header') else self.get_clock().now()
        if self.start_time_ros is None:
            self.start_time_ros = t
            # Try to record the frame_id from first message
            self.frame_id = getattr(msg.header, 'frame_id', '')
        self.end_time_ros = t

        # Extract and append
        for x, y, color in extract_xy_list(msg):
            if color not in self.buffer:
                self.buffer[color] = []
            self.buffer[color].append((x, y))

    def check_done(self):
        elapsed = (self.get_clock().now() - self.started_wall).nanoseconds * 1e-9
        if elapsed < self.duration_sec:
            return

        # Stop timer & subscription
        self.timer.cancel()
        try:
            self.destroy_subscription(self.sub)
        except Exception:
            pass

        # Cluster per color and compute medians
        snapshot_rows: List[Tuple[float, float, str]] = []
        for color, pts in self.buffer.items():
            if not pts:
                continue
            clusters = greedy_cluster(pts, self.cluster_eps)
            for cl in clusters:
                mx, my = median_of_cluster(cl)
                snapshot_rows.append((mx, my, color))

        # Save CSV with commented meta lines
        self.write_csv(snapshot_rows)

        self.get_logger().info(
            f"[ConeSnapshot] Wrote {len(snapshot_rows)} cones to {self.out_csv}. Exiting.")
        # Shutdown cleanly
        rclpy.shutdown()

    def write_csv(self, rows: List[Tuple[float, float, str]]):
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True) if os.path.dirname(self.out_csv) else None
        with open(self.out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Comment lines
            start_ns = int(self.start_time_ros.nanoseconds) if self.start_time_ros else 0
            end_ns = int(self.end_time_ros.nanoseconds) if self.end_time_ros else 0
            f.write(f"# frame_id,{self.frame_id}\n")
            f.write(f"# stamp_first_ns,{start_ns}\n")
            f.write(f"# stamp_last_ns,{end_ns}\n")
            f.write("# columns: x_m,y_m,color\n")
            # Data
            writer.writerow(['x_m', 'y_m', 'color'])
            for x, y, color in rows:
                writer.writerow([f"{x:.6f}", f"{y:.6f}", color])

def main():
    rclpy.init()
    node = ConeSnapshotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
