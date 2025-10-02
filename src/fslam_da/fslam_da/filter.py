#!/usr/bin/env python3
# dynamic_gate_filter_viz.py
# Purpose: NO association. Just:
#   1) Time-align (interpolate/extrapolate) odom ↔ perception timestamps.
#   2) Dynamic Mahalanobis gating on the KNOWN MAP to count candidates per observation.
#   3) Filter observations: pass if ≥1 candidate (green), else mark as potential NEW LANDMARK (yellow).
# Viz:
#   - LEFT pane (continuous): gate preview around car for a few ranges (purple outlines).
#   - RIGHT pane (perception-time): per-observation gates:
#       * purple ellipse if ≥1 candidate, black ellipse if 0 candidates
#       * observation point: green if ≥1 candidate, yellow if 0 candidates
#   - Known map cones are drawn faint gray in both panes.
#
# Console prints (each processed cone frame):
#   pass_dt, Δt_pose, v, yawrate, n_obs, cand histogram (0/1/2/≥3), mean_cands/obs, proc_ms
#
# Notes:
#   - No Hungarian, no JCBB, no drift estimation.
#   - No CSVs. Only console prints.
#   - “New landmarks” = observations with zero map candidates (drawn as YELLOW dots).
#   - Gates drawn in RIGHT pane centered at observation WORLD position:
#       * PURPLE if it has ≥1 candidate
#       * BLACK  if it has 0 candidates
#
# Tunables:
#   detections_frame:         "base"|"world" (default "base")
#   map_crop_m:               25.0
#   chi2_gate_2d:             11.83      # 2-DOF gate
#   meas_sigma_floor_xy:      0.10       # base measurement noise floor (m)
#   alpha_trans:              0.8        # translational jitter gain
#   beta_rot:                 0.8        # rotational jitter gain
#   odom_buffer_sec:          2.0        # seconds of odom kept for interpolation
#   extrapolation_cap_ms:     120.0      # cap for forward extrapolation
#   target_cone_hz:           25.0       # throttle cone processing target rate (Hz)
#   min_candidates:           3          # min-K fallback when nothing passes gate (Euclid)
#   map_data_dir:             "slam_utils/data" (expects alignment_transform.json & known_map.csv)
#   expected_meas_delay_ms:   50.0       # used ONLY for LEFT preview
#   gui_fps:                  20.0
#
# Topics used (EUFSIM):
#   /ground_truth/odom  (Odometry)
#   /ground_truth/cones (ConeArrayWithCovariance)
#
# Dependencies: numpy, matplotlib, rclpy, eufs_msgs, nav_msgs
# Backend: TkAgg (change to Qt5Agg if you prefer)

import math, json, time, bisect
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance


# ---------------- Paths & map helpers ----------------
BASE_DIR = Path(__file__).resolve().parents[0]

def load_known_map(csv_path: Path):
    import csv as _csv
    rows=[]
    with open(csv_path, "r", newline="") as f:
        clean = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    rdr = _csv.DictReader(clean)
    lk = [k.lower() for k in rdr.fieldnames]
    def pick(*cands):
        for c in cands:
            if c in lk: return c
        return None
    xk = pick("x","x_m","xmeter"); yk = pick("y","y_m","ymeter"); ck = pick("color","colour","tag")
    for r in rdr:
        try:
            x=float(r[xk]); y=float(r[yk]); c=str(r[ck]).strip().lower()
        except Exception:
            continue
        if c in ("bluecone","b"): c="blue"
        if c in ("yellowcone","y"): c="yellow"
        if c in ("big_orange","orange_large","large_orange"): c="orange_large"
        if c in ("small_orange","orange_small"): c="orange"
        if c not in ("blue","yellow","orange","orange_large"): c="unknown"
        rows.append((x,y,c))
    return rows


# ---------------- Math helpers ----------------
def rot2d(theta: float) -> np.ndarray:
    c,s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]], float)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def draw_ellipse(ax, center: np.ndarray, cov: np.ndarray, chi2: float,
                 edgecolor: str, linewidth: float=1.4, alpha=0.9):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    width, height = 2.0 * np.sqrt(vals * chi2)  # full axis lengths
    angle = math.degrees(math.atan2(vecs[1,0], vecs[0,0]))
    e = Ellipse(xy=(center[0], center[1]), width=width, height=height,
                angle=angle, fill=False, lw=linewidth, ec=edgecolor, alpha=alpha)
    ax.add_patch(e)
    return e


# ---------------- Node ----------------
class DynGateFilterViz(Node):
    def __init__(self):
        super().__init__("dyn_gate_filter_viz")

        # Parameters
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 25.0)
        self.declare_parameter("chi2_gate_2d", 11.83)
        self.declare_parameter("meas_sigma_floor_xy", 0.10)
        self.declare_parameter("alpha_trans", 0.8)
        self.declare_parameter("beta_rot", 0.8)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 120.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("min_candidates", 3)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("expected_meas_delay_ms", 50.0)
        self.declare_parameter("gui_fps", 20.0)

        p = self.get_parameter
        self.detections_frame = str(p("detections_frame").value).lower()
        self.map_crop_m = float(p("map_crop_m").value)
        self.chi2_gate = float(p("chi2_gate_2d").value)
        self.sigma0 = float(p("meas_sigma_floor_xy").value)
        self.alpha = float(p("alpha_trans").value)
        self.beta  = float(p("beta_rot").value)
        self.odom_buffer_sec = float(p("odom_buffer_sec").value)
        self.extrap_cap = float(p("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(p("target_cone_hz").value)
        self.min_candidates = int(p("min_candidates").value)
        self.expected_dt = float(p("expected_meas_delay_ms").value) / 1000.0
        self.gui_period = 1.0 / max(1e-3, float(p("gui_fps").value))

        # Map & alignment
        data_dir = (BASE_DIR / str(p("map_data_dir").value)).resolve()
        align_json = data_dir / "alignment_transform.json"
        known_map_csv = data_dir / "known_map.csv"
        if not align_json.exists() or not known_map_csv.exists():
            raise RuntimeError(f"Missing map assets under {data_dir}")
        with open(align_json, "r") as f:
            T = json.load(f)
        R_map = np.array(T["rotation"], float)
        t_map = np.array(T["translation"], float)
        rows = load_known_map(known_map_csv)
        M = np.array([[x,y] for (x,y,_) in rows], float)
        self.map_xy = (M @ R_map.T) + t_map

        # State buffers
        self.odom_buf: deque = deque()  # (t, x, y, yaw, v, yawrate)
        self.pose_latest = np.array([0.0,0.0,0.0], float)

        # QoS
        q = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       history=QoSHistoryPolicy.KEEP_LAST)

        # Subs
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q)

        # Throttle for cone processing
        self._last_proc_wall = 0.0
        self._last_proc_stamp = None
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)

        # Live figure
        plt.ion()
        self.fig, (self.ax_gate, self.ax_res) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.manager.set_window_title("Dynamic Gate + Filter (NO association)")

        self._setup_axes()
        self._plot_static_map()

        # LEFT dynamic gate artists
        self.ellipses_left: List[Ellipse] = []
        self.pose_gate, = self.ax_gate.plot([], [], 'k^', ms=9, label="car")
        self._last_gui = time.time()

        # RIGHT per-frame artists
        self.ellipses_right: List[Ellipse] = []
        self.obs_green = self.ax_res.scatter([], [], s=38, c='g', alpha=0.95, label="obs (≥1 cand)")
        self.obs_yellow = self.ax_res.scatter([], [], s=38, c='y', alpha=0.95, label="obs (NEW)")
        # candidates (map points within gate) shown as faint purple dots for context
        self.cand_pts = self.ax_res.scatter([], [], s=16, c='#9b59b6', alpha=0.6, label="map candidates")
        self.pose_res,  = self.ax_res.plot([], [], 'k^', ms=9, label="car")

        self.ax_gate.legend(loc='upper right', fontsize=9, framealpha=0.4)
        self.ax_res.legend(loc='upper right', fontsize=9, framealpha=0.4)

        self.frame_idx = 0
        self.get_logger().info("dyn_gate_filter_viz up. Left=gate preview; Right=per-obs gates & filtered obs (NO association).")

    # ---------------- ROS callbacks ----------------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        v, yr = 0.0, 0.0
        if self.odom_buf:
            t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[-1]
            dt = max(1e-3, t - t0)
            v = math.hypot(x-x0, y-y0) / dt
            yr = wrap_angle(yaw - yaw0) / dt
        self.pose_latest = np.array([x,y,yaw], float)
        self.odom_buf.append((t,x,y,yaw,v,yr))
        # trim
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()
        # update left panel at GUI fps
        now = time.time()
        if now - self._last_gui >= self.gui_period:
            self._last_gui = now
            self._update_left_panel()

    def _pose_at(self, t_query: float) -> Tuple[np.ndarray, float, float, float]:
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf'), 0.0, 0.0
        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)
        if idx == 0:
            t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[0]
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        if idx >= len(self.odom_buf):
            t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[-1]
            dt = t_query - t1
            dt_c = max(0.0, min(dt, self.extrap_cap))
            if abs(yr1) < 1e-3:
                x = x1 + v1*dt_c*math.cos(yaw1)
                y = y1 + v1*dt_c*math.sin(yaw1)
                yaw = wrap_angle(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt_c) - math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt_c) - math.cos(yaw1))
                yaw = wrap_angle(yaw1 + yr1*dt_c)
            return np.array([x,y,yaw], float), abs(dt), v1, yr1
        t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[idx-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[idx]
        if t1==t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap_angle(yaw1 - yaw0); yaw = wrap_angle(yaw0 + a*dyaw)
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    def cb_cones(self, msg: ConeArrayWithCovariance):
        # throttle processing to target_cone_hz
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0_proc = time.perf_counter()

        # perception timestamp
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # build obs in WORLD; keep BASE for range calc
        obs_world = []   # (pw[2], Sigma0_w[2x2])
        obs_base  = []   # pb
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma0**2, self.sigma0**2])
                if self.detections_frame == "base":
                    pb = np.array([px,py], float)
                    pw = np.array([x,y]) + Rwb @ pb
                    Sigma0_w = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y]))
                    Sigma0_w = Pw
                obs_world.append((pw, Sigma0_w))
                obs_base.append(pb)
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)
        n_obs = len(obs_world)
        if n_obs == 0:
            return

        # gating and candidate counting
        crop2 = self.map_crop_m**2
        allowed_obs_xy = []
        newland_obs_xy = []
        cand_points = []

        cand_hist = {0:0, 1:0, 2:0, 3:0}  # we’ll bucket ≥3 into key 3
        total_cands = 0

        for (pw, Sigma0_w), pb in zip(obs_world, obs_base):
            # crop by Euclid
            diffs = self.map_xy - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            near_idx = np.where(d2 <= crop2)[0]

            # per-obs inflated covariance (use this obs’ BASE range)
            r_obs = float(np.linalg.norm(pb))
            Sigma_eff = self._inflate_cov(Sigma0_w, v, yawrate, dt_pose, r_obs, yaw)

            # gate test
            cands=[]
            for mid in near_idx:
                mxy = self.map_xy[mid]
                innov = pw - mxy
                try:
                    m2 = float(innov.T @ np.linalg.inv(Sigma_eff) @ innov)
                except np.linalg.LinAlgError:
                    m2 = float(innov.T @ np.linalg.pinv(Sigma_eff) @ innov)
                if m2 <= self.chi2_gate:
                    cands.append((mid, m2))

            # fallback: if none, take min-K Euclid to avoid starving downstream DA
            if not cands and near_idx.size>0:
                by_e = sorted([(int(mid), float(d2[mid])) for mid in near_idx], key=lambda t: t[1])[:self.min_candidates]
                # keep them as "candidates" but they did not pass Mahalanobis; still count
                cands = [(mid, float('inf')) for (mid, _) in by_e]

            # book-keeping (counts + lists for viz)
            k = len(cands)
            total_cands += k
            if   k == 0: cand_hist[0]+=1
            elif k == 1: cand_hist[1]+=1
            elif k == 2: cand_hist[2]+=1
            else:        cand_hist[3]+=1

            # RIGHT panel ellipse color: purple if ≥1 cand, black if 0
            # Obs point: green if ≥1 cand (filtered), yellow if 0 (NEW landmark)
            if k >= 1:
                allowed_obs_xy.append(pw)
                # also collect candidate map points for context (purple small dots)
                for (mid, _) in cands:
                    cand_points.append(self.map_xy[mid])
                self._queue_right_ellipse(center=pw, cov=Sigma_eff, color="purple")
            else:
                newland_obs_xy.append(pw)
                self._queue_right_ellipse(center=pw, cov=Sigma_eff, color="black")

        # update RIGHT panel
        self._flush_right_panel(pose_w_b, allowed_obs_xy, newland_obs_xy, cand_points)

        # prints
        t1_proc = time.perf_counter()
        if self._last_proc_stamp is None:
            pass_dt = 0.0
        else:
            pass_dt = t_z - self._last_proc_stamp
        self._last_proc_stamp = t_z

        mean_cands = total_cands / max(1, n_obs)
        print(
            f"[gate] pass_dt={pass_dt*1000:.1f}ms  Δt_pose={dt_pose*1000:.1f}ms  "
            f"v={v:.2f}m/s  yawrate={yawrate:.2f}rad/s  obs={n_obs}  "
            f"cands_hist=[0:{cand_hist[0]}, 1:{cand_hist[1]}, 2:{cand_hist[2]}, 3+:{cand_hist[3]}]  "
            f"mean_cands/obs={mean_cands:.2f}  proc={int((t1_proc-t0_proc)*1000)}ms"
        )

        self.frame_idx += 1

    # ---------------- Internals ----------------
    def _inflate_cov(self, Sigma0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        # translational growth (isotropic) + rotational growth projected in world
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = Sigma0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    # ----------- Left panel (continuous gate preview) -----------
    def _setup_axes(self):
        for ax in (self.ax_gate, self.ax_res):
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        self.ax_gate.set_title("Gate Preview (continuous; expected delay applied)")
        self.ax_res.set_title("Perception-time Gating & Filter (NO association)")

    def _plot_static_map(self):
        xs=self.map_xy[:,0]; ys=self.map_xy[:,1]
        self.ax_gate.scatter(xs, ys, s=14, c='k', alpha=0.15, label="map cones")
        self.ax_res.scatter(xs, ys, s=14, c='k', alpha=0.15, label="map cones")

    def _update_left_panel(self):
        if not self.odom_buf:
            return
        t,x,y,yaw,v,yawrate = self.odom_buf[-1]
        self.pose_gate.set_data([x],[y])

        # clear and redraw preview ellipses (purple)
        for e in self.ellipses_left:
            e.remove()
        self.ellipses_left.clear()

        ranges = [5.0, 10.0, 15.0]
        Sigma0_w = np.diag([self.sigma0**2, self.sigma0**2])
        for r in ranges:
            S = self._inflate_cov(Sigma0_w, v, yawrate, self.expected_dt, r, yaw)
            e = draw_ellipse(self.ax_gate, np.array([x,y]), S, self.chi2_gate, edgecolor='purple', linewidth=1.4, alpha=0.85)
            self.ellipses_left.append(e)

        span = 25.0
        self.ax_gate.set_xlim(x - span, x + span)
        self.ax_gate.set_ylim(y - span, y + span)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ----------- Right panel (perception-time result) -----------
    def _queue_right_ellipse(self, center: np.ndarray, cov: np.ndarray, color: str):
        # color in {"purple","black"}
        e = draw_ellipse(self.ax_res, center, cov, self.chi2_gate,
                         edgecolor=color, linewidth=1.2, alpha=0.9)
        self.ellipses_right.append(e)

    def _flush_right_panel(self, pose_w_b: np.ndarray, allowed_obs_xy, newland_obs_xy, cand_points):
        # remove previous ellipses
        for e in self.ellipses_right:
            e.remove()
        self.ellipses_right.clear()

        # re-draw were queued already in cb_cones; so nothing more to add here

        # update scatters
        if allowed_obs_xy:
            A = np.vstack(allowed_obs_xy)
            self.obs_green.set_offsets(A)
        else:
            self.obs_green.set_offsets(np.empty((0,2)))

        if newland_obs_xy:
            Y = np.vstack(newland_obs_xy)
            self.obs_yellow.set_offsets(Y)
        else:
            self.obs_yellow.set_offsets(np.empty((0,2)))

        if cand_points:
            C = np.vstack(cand_points)
            self.cand_pts.set_offsets(C)
        else:
            self.cand_pts.set_offsets(np.empty((0,2)))

        # update car pose + view
        x,y,yaw = pose_w_b
        self.pose_res.set_data([x],[y])
        span = 25.0
        self.ax_res.set_xlim(x - span, x + span)
        self.ax_res.set_ylim(y - span, y + span)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ---------------- main ----------------
def main():
    rclpy.init()
    node = DynGateFilterViz()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
