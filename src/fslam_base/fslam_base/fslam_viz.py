#!/usr/bin/env python3
# fslam_viz.py — real-time path plot + log-based error charts (ROS2 Galactic)

import os, math, time, argparse, glob
from pathlib import Path
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry

import matplotlib
# Use an interactive backend if available; fallback to non-interactive
try:
    import matplotlib.pyplot as plt
    _HAS_DISPLAY = True
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_DISPLAY = False

WORLD_FRAME = "world"

def yaw_from_quat(q):
    # 2D yaw from quaternion
    siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class OdomBuffer:
    def __init__(self, maxlen=20000):
        self.t = deque(maxlen=maxlen)
        self.x = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen)
        self.yaw = deque(maxlen=maxlen)

    def add(self, stamp_ns, x, y, yaw):
        self.t.append(stamp_ns * 1e-9)
        self.x.append(x); self.y.append(y); self.yaw.append(yaw)

    def last(self):
        if not self.t: return None
        return self.t[-1], self.x[-1], self.y[-1], self.yaw[-1]

    def as_arrays(self):
        return np.array(self.t), np.array(self.x), np.array(self.y), np.array(self.yaw)

class PathViz(Node):
    def __init__(self, slam_topic, gt_topic, logs_root):
        super().__init__("fslam_viz")
        self.slam = OdomBuffer()
        self.gt   = OdomBuffer()
        self.logs_root = Path(logs_root) if logs_root else None

        # Subscribers
        self.create_subscription(Odometry, slam_topic, self.cb_slam, 10)
        self.create_subscription(Odometry, gt_topic,   self.cb_gt,   10)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ax.set_title("SLAM vs Ground Truth — Real-time")
        self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("y (m)")
        self.ax.grid(True, alpha=0.4); self.ax.set_aspect('equal', adjustable='box')

        (self.slam_line,) = self.ax.plot([], [], lw=2, label="SLAM", color="tab:blue")
        (self.gt_line,)   = self.ax.plot([], [], lw=2, ls="--", label="Ground truth", color="tab:orange")

        self.status_txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                       va="top", ha="left", fontsize=9,
                                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        self.ax.legend(loc="best")

        # Keyboard hooks
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Update timer (~10 Hz)
        self.timer = self.create_timer(0.10, self.on_timer)
        self.last_redraw = 0.0

        self.get_logger().info(f"Listening: SLAM={slam_topic}, GT={gt_topic}")
        if self.logs_root:
            self.get_logger().info(f"Logs root: {self.logs_root}")

    def cb_slam(self, msg: Odometry):
        p = msg.pose.pose
        self.slam.add(Time.from_msg(msg.header.stamp).nanoseconds,
                      p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def cb_gt(self, msg: Odometry):
        p = msg.pose.pose
        self.gt.add(Time.from_msg(msg.header.stamp).nanoseconds,
                    p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def on_timer(self):
        # Update lines
        _, sx, sy, _ = self.slam.as_arrays()
        _, gx, gy, _ = self.gt.as_arrays()
        self.slam_line.set_data(sx, sy)
        self.gt_line.set_data(gx, gy)

        # Auto-zoom to data with a little margin
        all_x = np.concatenate([sx, gx]) if len(gx) else sx
        all_y = np.concatenate([sy, gy]) if len(gy) else sy
        if all_x.size >= 2:
            pad = 2.0
            self.ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
            self.ax.set_ylim(all_y.min()-pad, all_y.max()+pad)

        # Status box
        s_last = self.slam.last()
        g_last = self.gt.last()
        box = []
        if s_last:
            _, sx0, sy0, syaw0 = s_last
            box.append(f"SLAM: x={sx0:7.2f}  y={sy0:7.2f}  yaw={math.degrees(syaw0):6.2f}°")
        else:
            box.append("SLAM: (no data)")

        if g_last:
            _, gx0, gy0, gyaw0 = g_last
            box.append(f"GT  : x={gx0:7.2f}  y={gy0:7.2f}  yaw={math.degrees(gyaw0):6.2f}°")
        else:
            box.append("GT  : (no data)")

        if s_last and g_last:
            ex = gx0 - sx0; ey = gy0 - sy0
            eyaw = math.degrees(math.atan2(math.sin(gyaw0-syaw0), math.cos(gyaw0-syaw0)))
            box.append(f"Err : ex={ex:7.2f}  ey={ey:7.2f}  eyaw={eyaw:6.2f}°")

        self.status_txt.set_text("\n".join(box))

        # Draw
        now = time.time()
        if _HAS_DISPLAY and (now - self.last_redraw) > 0.05:
            plt.pause(0.001)  # process GUI events
            self.last_redraw = now

    def on_key(self, event):
        if event.key == 's':
            out = f"path_{int(time.time())}.png"
            self.fig.savefig(out, dpi=150)
            self.get_logger().info(f"Saved figure → {out}")
        elif event.key == 'e':
            self.plot_logs_errors()

    def plot_logs_errors(self):
        """Find the latest logs/fslam/<timestamp>/ and plot error charts."""
        if not self.logs_root:
            # Try to guess project root: two levels up from this file, then logs/fslam
            pkg_root = Path(__file__).resolve().parents[2]
            logs_root = pkg_root / "logs" / "fslam"
        else:
            logs_root = self.logs_root

        if not logs_root.exists():
            self.get_logger().warn(f"No logs folder at {logs_root}")
            return

        subdirs = sorted([p for p in logs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if not subdirs:
            self.get_logger().warn("No log runs found.")
            return
        latest = subdirs[-1]
        pose_est = latest / "pose_est.csv"
        pose_gt  = latest / "pose_gt.csv"
        errs     = latest / "errors.csv"

        if not (pose_est.exists() and pose_gt.exists()):
            self.get_logger().warn(f"Missing pose_est.csv or pose_gt.csv in {latest}")
            return

        # Load CSVs (they are tiny; use numpy)
        def load_csv(path, cols):
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
            return [np.array(data[c]) for c in cols]

        t_e, xe, ye, yawe, ve = load_csv(pose_est, ["t","x","y","yaw","v"])
        t_g, xg, yg, yawg     = load_csv(pose_gt,  ["t","x","y","yaw"])

        # Intersect/align times by nearest-neighbor for clarity
        def interp_to(t_src, t_dst, arr):
            return np.interp(t_dst, t_src, arr)

        t0 = max(t_e[0], t_g[0]); t1 = min(t_e[-1], t_g[-1])
        if t1 <= t0:
            self.get_logger().warn("No overlapping time range in logs.")
            return
        mask_e = (t_e>=t0) & (t_e<=t1)
        t = t_e[mask_e]
        xe_i = xe[mask_e]; ye_i = ye[mask_e]; yawe_i = yawe[mask_e]
        xg_i = interp_to(t_g, t, xg); yg_i = interp_to(t_g, t, yg); yawg_i = interp_to(t_g, t, yawg)

        ex = xg_i - xe_i; ey = yg_i - ye_i
        dyaw = np.arctan2(np.sin(yawg_i - yawe_i), np.cos(yawg_i - yawe_i))

        # Plot
        fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
        axs[0].plot(t - t[0], ex, label="ex (m)")
        axs[0].plot(t - t[0], ey, label="ey (m)")
        axs[0].set_ylabel("pos err (m)"); axs[0].grid(True); axs[0].legend()

        axs[1].plot(t - t[0], np.degrees(dyaw), label="eyaw (deg)")
        axs[1].set_ylabel("yaw err (deg)"); axs[1].grid(True); axs[1].legend()

        axs[2].plot(t - t[0], np.hypot(ex,ey), label="pos RMSE trace")
        axs[2].set_ylabel("||e|| (m)"); axs[2].set_xlabel("time since start (s)")
        axs[2].grid(True); axs[2].legend()

        fig.suptitle(f"Errors from logs: {latest.name}")
        fig.tight_layout()
        if _HAS_DISPLAY:
            plt.show(block=False)
        else:
            out = f"errors_{latest.name}.png"
            fig.savefig(out, dpi=150)
            self.get_logger().info(f"Saved errors → {out}")

def main():
    parser = argparse.ArgumentParser(description="Real-time SLAM vs GT path plotter + log error charts")
    parser.add_argument("--slam_topic", default="/fslam/odom")
    parser.add_argument("--gt_topic",   default="/ground_truth/odom")
    parser.add_argument("--logs_root",  default="", help="Path to logs/fslam root (optional)")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PathViz(args.slam_topic, args.gt_topic, args.logs_root)

    try:
        # Spin in a loop so we can update the plot
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            # plot updates are triggered by timer
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
