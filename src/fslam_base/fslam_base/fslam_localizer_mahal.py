#!/usr/bin/env python3
# fslam_localizer.py  (ROS2 Galactic)
# Particle-filter localization for skidpad with fixed landmark map.
#
# TUNING CHEAT-SHEET (change params via ROS2 or edit defaults below):
#   wheel_radius_m              -> scale your v; set via calibrator (median). Too small ⇒ est too slow.
#   speed_lpf_fc_hz             -> speed smoothness. Higher follows faster; lower reduces noise.
#   proc_sigma_v                -> process noise on v (m/s/√s). + if lag in accel; - if jitter when steady.
#   proc_sigma_beta_deg         -> slip random walk (deg/√s). + if under‐turning in corners; - if drifting on straights.
#   proc_sigma_bg               -> gyro bias RW (rad/s²). + if yaw drifts between cone updates; - if yaw gets noisy.
#   meas_sigma_floor_xy         -> additive meas floor (m). ↑ if mean_mahal ≳2 and many drops; ↓ if <0.6 but bad matches.
#   gate_chi2_2d                -> χ² gate (2 dof). ↑ (e.g. 5.99→9.21) to accept more; ↓ to be stricter (e.g. 5.99→4.0).
#   map_crop_m                  -> only use map cones within this radius of particle (all colors). 25–35 is typical.
#   orange_use_within_m         -> only use orange / big_orange if within this range of particle (avoid poisoning mid-lap).
#   particles_init / particles_run / shrink_* -> start big, shrink when stable:
#        Stable when: match_frac ≥ shrink_match_frac_min  AND  p95_mahal ≤ shrink_p95_mahal_max
#        for shrink_hold_time_s seconds.
#
# Watch logs (logs/fslam/...):
#   assoc.csv  : matches, dropped, mean_mahal, p95_mahal  (sqrt(χ²))
#   errors.csv : ex, ey, eyaw  (GT − estimate)
#   pose_est.csv / pose_gt.csv : trajectories for plotting
#
# QoS: IMU BEST_EFFORT (matches /imu/data in your sim).

import os, json, math, time, csv
from pathlib import Path
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from eufs_msgs.msg import WheelSpeedsStamped, ConeArrayWithCovariance
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

# ------------------------------ Paths & constants ------------------------------
BASE_DIR = Path(__file__).resolve().parents[0]          # .../fslam_base
DATA_DIR = BASE_DIR / "slam_utils" / "data"
ALIGN_JSON = DATA_DIR / "alignment_transform.json"
KNOWN_MAP_CSV = DATA_DIR / "known_map.csv"
LOG_ROOT = BASE_DIR.parents[1] / "logs" / "fslam"       # <workspace>/logs/fslam/

WORLD_FRAME = "world"
BASE_FRAME  = "base_footprint"

# ------------------------------ Utility ------------------------------
def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw*0.5)
    q.w = math.cos(yaw*0.5)
    return q

def wrap_angle(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],[s, c]], dtype=float)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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
        rows.append({"x":x,"y":y,"color":c})
    return rows

def greedy_assign(cost_mat, gate):
    # cost_mat: (No, Nm) of Mahalanobis^2; keep pairs ≤ gate (χ²)
    No, Nm = cost_mat.shape
    flat=[]
    for i in range(No):
        for j in range(Nm):
            c = cost_mat[i,j]
            if np.isfinite(c) and c <= gate:
                flat.append((c,i,j))
    flat.sort(key=lambda x:x[0])
    used_o=set(); used_m=set(); pairs=[]
    for c,i,j in flat:
        if i in used_o or j in used_m: continue
        pairs.append((i,j)); used_o.add(i); used_m.add(j)
    return pairs

# ------------------------------ Main Node ------------------------------
class FslamLocalizer(Node):
    def __init__(self):
        super().__init__("fslam_localizer")

        # ---- Tunable params (defaults; override with ROS2 params if desired) ----
        # Particle counts (dynamic shrink)
        self.declare_parameter("particles_init", 1000)
        self.declare_parameter("particles_run",  300)
        self.declare_parameter("shrink_match_frac_min", 0.60)   # ≥60% of visible obs matched
        self.declare_parameter("shrink_p95_mahal_max",  1.80)   # p95(√χ²) ≤ 1.8
        self.declare_parameter("shrink_hold_time_s",    0.50)

        # Vehicle & motion
        self.declare_parameter("wheel_radius_m", 0.26)
        self.declare_parameter("wheelbase_m", 1.6)
        self.declare_parameter("speed_lpf_fc_hz", 8.0)
        self.declare_parameter("proc_sigma_v", 0.45)            # m/s/√s
        self.declare_parameter("proc_sigma_beta_deg", 0.8)      # deg/√s
        self.declare_parameter("proc_sigma_bg", 2e-4)           # rad/s^2 (gyro bias RW)

        # Measurement / DA
        self.declare_parameter("meas_sigma_floor_xy", 0.35)     # m
        self.declare_parameter("gate_chi2_2d", 9.21)            # χ²(2 dof)
        self.declare_parameter("map_crop_m", 35.0)              # apply to all colors
        self.declare_parameter("orange_use_within_m", 15.0)     # ONLY use orange/big_orange if within this radius

        # Init / pub
        self.declare_parameter("still_init_sec", 0.5)
        self.declare_parameter("publish_rate_hz", 60.0)

        # Resolve params
        self.N_init = int(self.get_parameter("particles_init").value)
        self.N_run  = int(self.get_parameter("particles_run").value)
        self.N_cur  = self.N_init
        self.shrink_match_frac_min = float(self.get_parameter("shrink_match_frac_min").value)
        self.shrink_p95_mahal_max  = float(self.get_parameter("shrink_p95_mahal_max").value)
        self.shrink_hold_time_s    = float(self.get_parameter("shrink_hold_time_s").value)

        self.Rw = float(self.get_parameter("wheel_radius_m").value)
        self.fc = float(self.get_parameter("speed_lpf_fc_hz").value)
        self.sv = float(self.get_parameter("proc_sigma_v").value)
        self.sb = math.radians(float(self.get_parameter("proc_sigma_beta_deg").value))
        self.sbg = float(self.get_parameter("proc_sigma_bg").value)

        self.r_floor = float(self.get_parameter("meas_sigma_floor_xy").value)
        self.chi2_gate = float(self.get_parameter("gate_chi2_2d").value)
        self.map_crop_m = float(self.get_parameter("map_crop_m").value)
        self.orange_use_within_m = float(self.get_parameter("orange_use_within_m").value)

        self.still_init_sec = float(self.get_parameter("still_init_sec").value)
        self.pub_rate = float(self.get_parameter("publish_rate_hz").value)

        # ---- Load transform & map ----
        if not ALIGN_JSON.exists():
            raise RuntimeError(f"Missing {ALIGN_JSON}")
        with open(ALIGN_JSON, "r") as f:
            T = json.load(f)
        self.R_map = np.array(T["rotation"], dtype=float)
        self.t_map = np.array(T["translation"], dtype=float)

        map_rows = load_known_map(KNOWN_MAP_CSV)
        M = np.array([[r["x"], r["y"]] for r in map_rows], float)
        C = [r["color"] for r in map_rows]
        self.landmarks_world = (M @ self.R_map.T) + self.t_map
        self.landmark_colors = C
        self.idx_by_color = {
            "blue":         np.array([i for i,c in enumerate(C) if c=="blue"], int),
            "yellow":       np.array([i for i,c in enumerate(C) if c=="yellow"], int),
            "orange":       np.array([i for i,c in enumerate(C) if c=="orange"], int),
            "orange_large": np.array([i for i,c in enumerate(C) if c=="orange_large"], int),
        }

        # ---- State / buffers ----
        self.initialized = False
        self.still_collecting = True
        self.still_t0 = None
        self.still_gyro = deque()
        self.speed_lpf_state = 0.0
        self.last_time = None

        self.particles = None
        self.weights = None

        self.last_imu = None
        self.last_wheels = None
        self.last_cones = None
        self.latest_gt_odom = None

        # Shrink bookkeeping
        self._stable_since = None
        self._shrunk = False

        # ---- Publishers & TF ----
        self.pub_odom = self.create_publisher(Odometry, "/fslam/odom", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/fslam/pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---- QoS (IMU must be BEST_EFFORT to match sbg_imu) ----
        imu_qos = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # ---- Subscribers ----
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.cb_imu, imu_qos)
        self.ws_sub  = self.create_subscription(WheelSpeedsStamped, '/ros_can/wheel_speeds', self.cb_wheels, 10)
        self.cones_sub = self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones', self.cb_cones, 10)
        self.gt_sub  = self.create_subscription(Odometry, '/ground_truth/odom', self.cb_gt, 10)

        # ---- Timers ----
        self.timer_pub = self.create_timer(1.0/self.pub_rate, self.publish_estimate)
        self.timer_heartbeat = self.create_timer(1.0, self._heartbeat)

        # ---- Logs ----
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = LOG_ROOT / ts
        ensure_dir(self.log_dir)
        self.fp_est  = open(self.log_dir / "pose_est.csv", "w", newline="")
        self.fp_gt   = open(self.log_dir / "pose_gt.csv", "w", newline="")
        self.fp_err  = open(self.log_dir / "errors.csv", "w", newline="")
        self.fp_assoc= open(self.log_dir / "assoc.csv", "w", newline="")
        self.fp_timing=open(self.log_dir / "timing.csv", "w", newline="")
        self.w_est  = csv.writer(self.fp_est);  self.w_est.writerow(["t","x","y","yaw","v"])
        self.w_gt   = csv.writer(self.fp_gt);   self.w_gt.writerow(["t","x","y","yaw"])
        self.w_err  = csv.writer(self.fp_err);  self.w_err.writerow(["t","ex","ey","eyaw"])
        self.w_assoc= csv.writer(self.fp_assoc);self.w_assoc.writerow(["t","matches","dropped","mean_mahal","p95_mahal","match_frac","N"])
        self.w_tim  = csv.writer(self.fp_timing);self.w_tim.writerow(["t","predict_ms","update_ms"])
        self._last_flush = time.time()

        # Debug heartbeat state
        self._last_rx_imu  = None
        self._last_rx_ws   = None
        self._last_rx_cone = None

        self.get_logger().info("fslam_localizer up.")

    # -------------------- Callbacks --------------------
    def cb_gt(self, msg: Odometry):
        self.latest_gt_odom = msg

    def cb_wheels(self, msg: WheelSpeedsStamped):
        self.last_wheels = msg
        self._last_rx_ws = time.time()

    def cb_cones(self, msg: ConeArrayWithCovariance):
        self.last_cones = msg
        self._last_rx_cone = time.time()

    def cb_imu(self, msg: Imu):
        self.last_imu = msg
        self._last_rx_imu = time.time()
        t_now = self.get_clock().now()
        if self.still_collecting:
            if self.still_t0 is None:
                self.still_t0 = t_now
            self.still_gyro.append(float(msg.angular_velocity.z))
            if (t_now - self.still_t0).nanoseconds * 1e-9 >= self.still_init_sec:
                bg0 = float(np.mean(self.still_gyro)) if len(self.still_gyro)>0 else 0.0
                self.init_filter(bg0)
                self.still_collecting = False
                self.get_logger().info(f"Still-init done. b_g0={bg0:.6f} rad/s")
        else:
            self.step_filter(msg)

    # -------------------- Init & steps --------------------
    def init_filter(self, bg0):
        x0 = 0.0; y0 = 0.0; yaw0 = 0.0
        v0 = 0.0; beta0 = 0.0
        self.particles = np.zeros((self.N_cur,6), float)
        self.particles[:,0] = x0 + np.random.normal(0, 0.10, size=self.N_cur)
        self.particles[:,1] = y0 + np.random.normal(0, 0.10, size=self.N_cur)
        self.particles[:,2] = yaw0 + np.random.normal(0, math.radians(2.0), size=self.N_cur)
        self.particles[:,3] = v0
        self.particles[:,4] = beta0
        self.particles[:,5] = bg0 + np.random.normal(0, 5e-5, size=self.N_cur)
        self.weights = np.ones(self.N_cur, float) / self.N_cur
        self.last_time = None
        self.initialized = True

    def _shrink_particles(self, N_new):
        # Systematic resample to new size
        N_old = self.particles.shape[0]
        w = self.weights
        positions = (np.arange(N_old) + np.random.rand()) / N_old
        indexes = np.zeros(N_old, dtype=int)
        cumulative_sum = np.cumsum(w)
        i, j = 0, 0
        while i < N_old:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        Xr = self.particles[indexes]
        if N_new < N_old:
            Xr = Xr[:N_new]
        elif N_new > N_old:
            # tile + small noise
            reps = int(np.ceil(N_new / N_old))
            Xr = np.tile(Xr, (reps,1))[:N_new]
            Xr[:,0] += np.random.normal(0, 0.02, size=N_new)
            Xr[:,1] += np.random.normal(0, 0.02, size=N_new)
            Xr[:,2] += np.random.normal(0, math.radians(0.3), size=N_new)
        self.particles = Xr
        self.weights = np.ones(N_new, float) / N_new
        self.N_cur = N_new
        self.get_logger().info(f"Shrank particles: {N_old} → {N_new}")

    def step_filter(self, imu: Imu):
        if not self.initialized or self.last_wheels is None:
            return

        # dt from IMU stamps
        t = Time.from_msg(imu.header.stamp).nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = max(1e-4, min(0.02, t - self.last_time))
        self.last_time = t

        # ------ Predict ------
        t0 = time.time()
        gyro_z = float(imu.angular_velocity.z)
        a_y = float(imu.linear_acceleration.y)

        # Wheels -> v (m/s), rear-only min (RPM->rad/s->m/s), clamp small negatives
        ws = self.last_wheels.speeds
        rb = float(ws.rb_speed); lb = float(ws.lb_speed)
        if abs(rb) < 5.0: rb = 0.0
        if abs(lb) < 5.0: lb = 0.0
        v_raw = max(0.0, min(rb, lb) * (2*math.pi/60.0) * self.Rw)

        # 1st-order LPF
        alpha = 1.0 - math.exp(-2*math.pi*self.fc*dt)
        self.speed_lpf_state += alpha * (v_raw - self.speed_lpf_state)
        v_meas = max(0.0, self.speed_lpf_state)

        X = self.particles
        bg = X[:,5]
        yaw_dot = gyro_z - bg
        X[:,2] = wrap_angle(X[:,2] + yaw_dot*dt)

        v_eps = np.maximum(0.5, v_meas)
        k_slip = 0.2
        X[:,4] = np.clip(
            X[:,4] + k_slip*(a_y/v_eps)*dt + np.random.normal(0, self.sb*math.sqrt(dt), size=self.N_cur),
            math.radians(-15.0), math.radians(15.0)
        )
        X[:,3] = np.maximum(0.0, v_meas + np.random.normal(0, self.sv*math.sqrt(dt), size=self.N_cur))
        cos_h = np.cos(X[:,2] + X[:,4]); sin_h = np.sin(X[:,2] + X[:,4])
        X[:,0] += X[:,3] * cos_h * dt
        X[:,1] += X[:,3] * sin_h * dt
        X[:,5] += np.random.normal(0, self.sbg*dt, size=self.N_cur)
        t_pred_ms = (time.time()-t0)*1000.0

        # ------ Update (colored cones, with orange gating & map crop) ------
        t1 = time.time()
        best_like = -1e18
        best_pairs = 0
        best_drops = 0
        best_mahal = []
        best_total_obs = 0

        if self.last_cones is not None:
            Yw, Pw, Col = self._extract_observed_world(self.last_cones)
            if Yw.shape[0] > 0:
                # Pre-split observed indices per color (once)
                obs_idx_by_color = {
                    "blue":   np.where(Col == "blue")[0],
                    "yellow": np.where(Col == "yellow")[0],
                    "orange": np.where(Col == "orange")[0],
                    "orange_large": np.where(Col == "orange_large")[0],
                }
                # log-likelihood per particle
                Wlog = np.zeros(self.N_cur, float)
                for k in range(self.N_cur):
                    px,py,yaw = X[k,0], X[k,1], X[k,2]
                    Rwb = rot2d(yaw); Rbw = Rwb.T
                    like = 0.0
                    pairs_count = 0
                    drops_count = 0
                    mahal_used_all = []
                    total_obs_here = 0

                    for color in ("blue","yellow","orange","orange_large"):
                        idx_map = self.idx_by_color[color]
                        if idx_map.size == 0:
                            continue
                        Mw_all = self.landmarks_world[idx_map]  # (Nm_c,2)
                        # Map crop (ALL colors)
                        d2_map = np.sum((Mw_all - np.array([px,py]))**2, axis=1)
                        keep_map = d2_map <= (self.map_crop_m**2)

                        # EXTRA gate for orange/big_orange
                        if color in ("orange","orange_large"):
                            keep_map = np.logical_and(keep_map, d2_map <= (self.orange_use_within_m**2))

                        Mw = Mw_all[keep_map]
                        Nm_c = Mw.shape[0]
                        if Nm_c == 0:
                            continue

                        # Observed indices for this color
                        idx_obs = obs_idx_by_color[color]
                        No_c = idx_obs.shape[0]
                        if No_c == 0:
                            continue
                        total_obs_here += No_c

                        # Transform observed → car frame (and cov)
                        Yw_c = Yw[idx_obs]          # (No_c,2)
                        Pw_c = Pw[idx_obs]          # (No_c,2,2)
                        Rc = Rbw
                        Yc = (Yw_c - np.array([px,py])) @ Rc.T
                        # Pc = Rc @ Pw_c @ Rc.T  (batch)
                        Pc = np.einsum('ij,njk,kl->nil', Rc, Pw_c, Rc.T)

                        # Expected map in car frame
                        Mc = (Mw - np.array([px,py])) @ Rbw.T   # (Nm_c,2)

                        # Build cost matrix (Mahalanobis^2)
                        cost = np.empty((No_c, Nm_c), float)
                        Radd = (self.r_floor**2)*np.eye(2)
                        for i in range(No_c):
                            S = Pc[i] + Radd
                            det = S[0,0]*S[1,1] - S[0,1]*S[1,0]
                            if det <= 1e-12:
                                Sinv = np.linalg.pinv(S)
                            else:
                                inv_det = 1.0/det
                                Sinv = np.array([[ S[1,1], -S[0,1]],
                                                 [-S[1,0],  S[0,0]]], float) * inv_det
                            di = Yc[i] - Mc  # (Nm_c,2)
                            tmp = di @ Sinv
                            m2 = np.sum(tmp * di, axis=1)
                            cost[i,:] = m2

                        pairs = greedy_assign(cost, gate=self.chi2_gate)
                        if pairs:
                            m2_used = [cost[i,j] for (i,j) in pairs]
                            like += -0.5 * float(np.sum(m2_used))
                            mahal_used_all.extend(m2_used)
                            pairs_count += len(pairs)
                            drops_count += (No_c - len(pairs))
                        else:
                            drops_count += No_c

                    Wlog[k] = like
                    # Track best stats online (max likelihood)
                    if like > best_like:
                        best_like = like
                        best_pairs = pairs_count
                        best_drops = drops_count
                        best_mahal = mahal_used_all[:]  # copy
                        best_total_obs = total_obs_here

                # normalize weights
                mmax = float(np.max(Wlog))
                w = np.exp(Wlog - mmax); w_sum = np.sum(w) + 1e-12
                self.weights = w / w_sum

        t_upd_ms = (time.time()-t1)*1000.0

        # ------ Resample ------
        Neff = 1.0 / (np.sum(self.weights**2) + 1e-12)
        if Neff/self.N_cur < 0.5:
            self._systematic_resample()

        # ------ Logs & shrink logic ------
        tsec = t
        mx, my, myaw, mv = self._est_mean()
        self.w_est.writerow([f"{tsec:.6f}", f"{mx:.3f}", f"{my:.3f}", f"{myaw:.5f}", f"{mv:.3f}"])
        if self.latest_gt_odom is not None:
            gx = self.latest_gt_odom.pose.pose.position.x
            gy = self.latest_gt_odom.pose.pose.position.y
            gq = self.latest_gt_odom.pose.pose.orientation
            siny_cosp = 2.0*(gq.w*gq.z + gq.x*gq.y)
            cosy_cosp = 1.0 - 2.0*(gq.y*gq.y + gq.z*gq.z)
            gyaw = math.atan2(siny_cosp, cosy_cosp)
            self.w_gt.writerow([f"{tsec:.6f}", f"{gx:.3f}", f"{gy:.3f}", f"{gyaw:.5f}"])
            self.w_err.writerow([f"{tsec:.6f}",
                                 f"{(gx-mx):.3f}", f"{(gy-my):.3f}", f"{wrap_angle(gyaw-myaw):.5f}"])
        if best_mahal:
            mahal_root = np.sqrt(np.array(best_mahal, float))
            mean_mahal = float(np.mean(mahal_root))
            p95_mahal  = float(np.percentile(mahal_root, 95))
        else:
            mean_mahal = 0.0; p95_mahal = 0.0
        total_obs = max(1, best_total_obs)
        match_frac = best_pairs / total_obs
        self.w_assoc.writerow([f"{tsec:.6f}", best_pairs, best_drops, f"{mean_mahal:.3f}", f"{p95_mahal:.3f}", f"{match_frac:.2f}", self.N_cur])
        self.w_tim.writerow([f"{tsec:.6f}", f"{t_pred_ms:.2f}", f"{t_upd_ms:.2f}"])

        # shrink condition
        if not self._shrunk:
            cond_ok = (match_frac >= self.shrink_match_frac_min) and (p95_mahal <= self.shrink_p95_mahal_max)
            now_w = time.time()
            if cond_ok:
                if self._stable_since is None:
                    self._stable_since = now_w
                elif (now_w - self._stable_since) >= self.shrink_hold_time_s:
                    self._shrink_particles(self.N_run)
                    self._shrunk = True
            else:
                self._stable_since = None

        # flush once a second so files aren’t empty until exit
        now_w = time.time()
        if now_w - self._last_flush > 1.0:
            self.fp_est.flush(); self.fp_gt.flush(); self.fp_err.flush()
            self.fp_assoc.flush(); self.fp_timing.flush()
            self._last_flush = now_w

    # Extract colored cones (world coords + cov 2x2) from ConeArrayWithCovariance
    def _extract_observed_world(self, msg: ConeArrayWithCovariance):
        pts=[]; covs=[]; cols=[]
        def add(arr, color):
            for c in arr:
                x = float(c.point.x); y=float(c.point.y)
                cov = np.array([[c.covariance[0], c.covariance[1]],
                                [c.covariance[2], c.covariance[3]]], float)
                pts.append([x,y]); covs.append(cov); cols.append(color)
        add(msg.blue_cones,        "blue")
        add(msg.yellow_cones,      "yellow")
        add(msg.orange_cones,      "orange")
        add(msg.big_orange_cones,  "orange_large")
        if len(pts)==0:
            return np.zeros((0,2)), np.zeros((0,2,2)), np.array([], dtype=object)
        return np.asarray(pts,float), np.asarray(covs,float), np.array(cols, dtype=object)

    def _systematic_resample(self):
        N = self.N_cur
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(N, float)/N

    def _est_mean(self):
        w = self.weights; X = self.particles
        mx = float(np.sum(w * X[:,0]))
        my = float(np.sum(w * X[:,1]))
        cy = float(np.sum(w * np.cos(X[:,2]))); sy = float(np.sum(w * np.sin(X[:,2])))
        myaw = math.atan2(sy, cy)
        mv = float(np.sum(w * X[:,3]))
        return mx,my,myaw,mv

    # -------------------- Publish --------------------
    def publish_estimate(self):
        if not self.initialized:
            return
        mx,my,myaw,mv = self._est_mean()
        now = self.get_clock().now().to_msg()

        # TF
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = WORLD_FRAME
        t.child_frame_id  = BASE_FRAME
        t.transform.translation.x = mx
        t.transform.translation.y = my
        t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quat(myaw)
        self.tf_broadcaster.sendTransform(t)

        # Odom
        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = WORLD_FRAME
        od.child_frame_id  = BASE_FRAME
        od.pose.pose.position.x = mx
        od.pose.pose.position.y = my
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation = yaw_to_quat(myaw)
        od.twist.twist.linear = Vector3(x=mv, y=0.0, z=0.0)
        od.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.pub_odom.publish(od)

        # Pose
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = WORLD_FRAME
        ps.pose.position.x = mx
        ps.pose.position.y = my
        ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quat(myaw)
        self.pub_pose.publish(ps)

    # -------------------- Heartbeat / diag --------------------
    def _heartbeat(self):
        def age(ts):
            return f"{(time.time()-ts):.2f}s" if ts is not None else "n/a"
        self.get_logger().info(
            f"[hb] init={self.initialized} imu_age={age(self._last_rx_imu)} "
            f"ws_age={age(self._last_rx_ws)} cones_age={age(self._last_rx_cone)} "
            f"N={self.N_cur}"
        )

    # -------------------- Shutdown --------------------
    def destroy_node(self):
        try:
            self.fp_est.close(); self.fp_gt.close(); self.fp_err.close()
            self.fp_assoc.close(); self.fp_timing.close()
        except Exception:
            pass
        super().destroy_node()

# ------------------------------ main ------------------------------
def main():
    rclpy.init()
    node = FslamLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
