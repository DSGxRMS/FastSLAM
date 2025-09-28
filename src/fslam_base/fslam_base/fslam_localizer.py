#!/usr/bin/env python3
# FastSLAM-style localizer (fixed landmark map, skidpad) + JCBB DA
# ROS2 Galactic

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

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         TUNING CONSTANTS (TOP)                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# You can edit these OR override them with ROS2 params of the same names.

TUNE = {
    # Particles
    "particles_init":            1200,     # start big
    "particles_run":             400,      # shrink to this after stable
    "shrink_match_frac_min":     0.60,     # ≥60% obs matched
    "shrink_p95_mahal_max":      1.8,      # p95 sqrt(χ²)
    "shrink_hold_time_s":        0.7,

    # Vehicle / motion
    "wheel_radius_m":            0.26,     # set this right!
    "wheelbase_m":               1.6,
    "speed_lpf_fc_hz":           8.0,
    "proc_sigma_v":              0.45,     # m/s/√s
    "proc_sigma_beta_deg":       0.8,      # deg/√s
    "proc_sigma_bg":             2e-4,     # rad/s² (gyro bias RW)
    "slip_beta_deg_limits":      15.0,     # |beta| clamp
    "imu_speed_leak_tau_s":      3.0,      # leaky v_int
    "v_zupt_rpm":                3.0,      # ZUPT threshold on min(rear)
    "zupt_yaw_rate_rad_s":       0.02,     # ZUPT if nearly static

    # Measurement model
    "meas_sigma_floor_xy":       0.35,     # additive floor (m)
    "gate_chi2_2d":              7.38,     # χ²(2): start a bit loose; can tighten later
    "gate_chi2_2d_orange":       9.21,     # χ²(2) for orange/big_orange (more tolerant near start)
    "map_crop_m":                35.0,     # consider map cones within this radius
    "orange_use_within_m":       18.0,     # only use orange/big_orange within this radius

    # JCBB limits (keep runtime sane)
    "jcbb_max_obs_per_color":    8,        # take closest K observations per color
    "jcbb_max_map_per_color":    12,       # take closest K map cones per color

    # Init/publish
    "still_init_sec":            1.0,      # collect biases when still
    "publish_rate_hz":           60.0,
}

# Frames & paths
WORLD_FRAME = "world"
BASE_FRAME  = "base_footprint"

BASE_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = BASE_DIR / "slam_utils" / "data"
ALIGN_JSON = DATA_DIR / "alignment_transform.json"
KNOWN_MAP_CSV = DATA_DIR / "known_map.csv"
LOG_ROOT = BASE_DIR.parents[1] / "logs" / "fslam"

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                             Utilities                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw*0.5); q.w = math.cos(yaw*0.5); return q

def wrap_angle(a): return (a + math.pi) % (2*math.pi) - math.pi

def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],[s, c]], float)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

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

def mahal2_batch(di, S):  # di: (N,2), S: (2,2)
    # returns (N,) of Mahalanobis^2; handle small det robustly
    det = S[0,0]*S[1,1] - S[0,1]*S[1,0]
    if det <= 1e-12:
        Sinv = np.linalg.pinv(S)
    else:
        inv = 1.0/det
        Sinv = np.array([[ S[1,1], -S[0,1]],
                         [-S[1,0],  S[0,0]]], float) * inv
    tmp = di @ Sinv
    return np.sum(tmp * di, axis=1)

# ── JCBB: joint compatibility branch & bound (simplified for 2D points) ─────────
def jcbb(Yc, Pc, Mc, chi2_gate, max_obs, max_map, r_floor):
    """
    Inputs (one color):
      Yc: (No,2) observed in CAR frame
      Pc: (No,2,2) covariances in CAR frame
      Mc: (Nm,2) map in CAR frame
      chi2_gate: scalar gate on individual pair √χ²
      max_obs, max_map: we pre-select nearest to limit branching
      r_floor: additive meas floor sqrt-variance (m)

    Returns:
      pairs: list[(i_obs, j_map)]
      used_m2: list of used m² values
    """
    No, Nm = Yc.shape[0], Mc.shape[0]
    if No == 0 or Nm == 0:
        return [], []

    # Pre-limit by nearest Euclidean to focus
    if No > max_obs:
        # pick obs nearest to origin (forward cones more useful); you can also pick by nearest map later
        d2o = np.sum(Yc**2, axis=1)
        keep_o = np.argsort(d2o)[:max_obs]
        Yc = Yc[keep_o]; Pc = Pc[keep_o]
        No = Yc.shape[0]
        obs_index_map = keep_o
    else:
        obs_index_map = np.arange(No)

    if Nm > max_map:
        d2m = np.sum(Mc**2, axis=1)
        keep_m = np.argsort(d2m)[:max_map]
        Mc = Mc[keep_m]
        Nm = Mc.shape[0]
        map_index_map = keep_m
    else:
        map_index_map = np.arange(Nm)

    # Precompute individual compatibilities & m²
    Radd = (r_floor**2)*np.eye(2)
    ind_ok = [[False]*Nm for _ in range(No)]
    m2_cache = [[np.inf]*Nm for _ in range(No)]
    for i in range(No):
        S = Pc[i] + Radd
        m2 = mahal2_batch(Mc - Yc[i], S)
        for j in range(Nm):
            if m2[j] <= chi2_gate:
                ind_ok[i][j] = True
                m2_cache[i][j] = float(m2[j])

    # Order observations (smallest nearest-map distance first) to speed up pruning
    best_near = []
    for i in range(No):
        mn = min(m2_cache[i]) if len(m2_cache[i]) else np.inf
        best_near.append((mn, i))
    order = [i for _, i in sorted(best_near)]

    best_pairs = []
    best_score = -1e18  # sum of -0.5*m2

    used_m = [False]*Nm
    cur_pairs = []
    cur_score = 0.0

    def dfs(k):
        nonlocal best_pairs, best_score, cur_score
        if k == No:
            if cur_score > best_score:
                best_score = cur_score
                best_pairs = cur_pairs.copy()
            return
        i = order[k]

        # Branch 1: try to match i with each compatible j not used
        any_branch = False
        for j in range(Nm):
            if ind_ok[i][j] and (not used_m[j]):
                any_branch = True
                used_m[j] = True
                cur_pairs.append((i, j))
                prev = cur_score
                cur_score += -0.5 * m2_cache[i][j]
                dfs(k+1)
                cur_score = prev
                cur_pairs.pop()
                used_m[j] = False

        # Branch 2: drop this observation
        dfs(k+1)

    dfs(0)

    # Map local indices back to original
    remapped_pairs = []
    used_m2 = []
    for (i_local, j_local) in best_pairs:
        i = obs_index_map[i_local]
        j = map_index_map[j_local]
        remapped_pairs.append((i, j))
        used_m2.append(m2_cache[i_local][j_local])
    return remapped_pairs, used_m2

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                              Main Node                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝
class FslamLocalizer(Node):
    def __init__(self):
        super().__init__("fslam_localizer")

        # ---- Declare params mirroring TUNE (so you can override via ROS) ----
        for k, v in TUNE.items():
            self.declare_parameter(k, v)

        # Resolve
        g = lambda k: self.get_parameter(k).value
        self.N_init = int(g("particles_init"));      self.N_run  = int(g("particles_run"))
        self.shrink_match_frac_min = float(g("shrink_match_frac_min"))
        self.shrink_p95_mahal_max  = float(g("shrink_p95_mahal_max"))
        self.shrink_hold_time_s    = float(g("shrink_hold_time_s"))

        self.Rw  = float(g("wheel_radius_m"))
        self.L   = float(g("wheelbase_m"))
        self.fc  = float(g("speed_lpf_fc_hz"))
        self.sv  = float(g("proc_sigma_v"))
        self.sb  = math.radians(float(g("proc_sigma_beta_deg")))
        self.sbg = float(g("proc_sigma_bg"))
        self.beta_lim = math.radians(float(g("slip_beta_deg_limits")))
        self.tau_v = float(g("imu_speed_leak_tau_s"))
        self.zupt_rpm = float(g("v_zupt_rpm"))
        self.zupt_yaw = float(g("zupt_yaw_rate_rad_s"))

        self.r_floor     = float(g("meas_sigma_floor_xy"))
        self.chi2_gate   = float(g("gate_chi2_2d"))
        self.chi2_orange = float(g("gate_chi2_2d_orange"))
        self.map_crop_m  = float(g("map_crop_m"))
        self.orange_within_m = float(g("orange_use_within_m"))

        self.jcbb_max_obs = int(g("jcbb_max_obs_per_color"))
        self.jcbb_max_map = int(g("jcbb_max_map_per_color"))

        self.still_init_sec = float(g("still_init_sec"))
        self.pub_rate = float(g("publish_rate_hz"))

        # ---- Load transform & map ----
        if not ALIGN_JSON.exists():
            raise RuntimeError(f"Missing {ALIGN_JSON}")
        with open(ALIGN_JSON, "r") as f:
            T = json.load(f)
        self.R_map = np.array(T["rotation"], float)
        self.t_map = np.array(T["translation"], float)

        rows = load_known_map(KNOWN_MAP_CSV)
        M = np.array([[r["x"], r["y"]] for r in rows], float)
        C = [r["color"] for r in rows]
        self.landmarks_world = (M @ self.R_map.T) + self.t_map
        self.idx_by_color = {
            "blue":         np.array([i for i,c in enumerate(C) if c=="blue"], int),
            "yellow":       np.array([i for i,c in enumerate(C) if c=="yellow"], int),
            "orange":       np.array([i for i,c in enumerate(C) if c=="orange"], int),
            "orange_large": np.array([i for i,c in enumerate(C) if c=="orange_large"], int),
        }

        # ---- State ----
        self.N_cur = self.N_init
        self.initialized = False
        self.still_collecting = True
        self.still_t0 = None

        self.bias_bg = 0.0
        self.bias_ax = 0.0
        self.bias_ay = 0.0

        self.still_gyro = deque()
        self.still_ax   = deque()
        self.still_ay   = deque()

        self.last_time = None
        self.particles = None
        self.weights = None

        # speed
        self.speed_lpf_state = 0.0
        self.v_imu = 0.0

        # Inputs
        self.last_wheels = None
        self.last_cones = None
        self.latest_gt_odom = None

        # Shrink logic
        self._stable_since = None
        self._shrunk = False

        # ---- Publishers & TF ----
        self.pub_odom = self.create_publisher(Odometry, "/fslam/odom", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/fslam/pose", 10)
        self.tf_b = TransformBroadcaster(self)

        # ---- QoS ----
        imu_qos = QoSProfile(depth=50,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE,
                             history=QoSHistoryPolicy.KEEP_LAST)

        # ---- Subs ----
        self.create_subscription(Imu, '/imu/data', self.cb_imu, imu_qos)
        self.create_subscription(WheelSpeedsStamped, '/ros_can/wheel_speeds', self.cb_wheels, 10)
        self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones', self.cb_cones, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.cb_gt, 10)

        # ---- Timers ----
        self.create_timer(1.0/self.pub_rate, self.publish_estimate)
        self.create_timer(1.0, self._heartbeat)

        # ---- Logs ----
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = LOG_ROOT / ts
        ensure_dir(self.log_dir)
        self.fp_est   = open(self.log_dir / "pose_est.csv", "w", newline="")
        self.fp_gt    = open(self.log_dir / "pose_gt.csv", "w", newline="")
        self.fp_err   = open(self.log_dir / "errors.csv", "w", newline="")
        self.fp_assoc = open(self.log_dir / "assoc.csv", "w", newline="")
        self.fp_tim   = open(self.log_dir / "timing.csv", "w", newline="")
        self.w_est  = csv.writer(self.fp_est);  self.w_est.writerow(["t","x","y","yaw","v"])
        self.w_gt   = csv.writer(self.fp_gt);   self.w_gt.writerow(["t","x","y","yaw"])
        self.w_err  = csv.writer(self.fp_err);  self.w_err.writerow(["t","ex","ey","eyaw"])
        self.w_assoc= csv.writer(self.fp_assoc);self.w_assoc.writerow(["t","matches","dropped","mean_mahal","p95_mahal","match_frac","N"])
        self.w_tim  = csv.writer(self.fp_tim);  self.w_tim.writerow(["t","predict_ms","update_ms"])
        self._last_flush = time.time()

        # Debug ages
        self._last_rx_imu = None; self._last_rx_ws = None; self._last_rx_cone = None

        self.get_logger().info("fslam_localizer up.")

    # ── Callbacks ────────────────────────────────────────────────────────
    def cb_gt(self, msg: Odometry): self.latest_gt_odom = msg
    def cb_wheels(self, msg: WheelSpeedsStamped): self.last_wheels = msg; self._last_rx_ws = time.time()
    def cb_cones(self, msg: ConeArrayWithCovariance): self.last_cones = msg; self._last_rx_cone = time.time()

    def cb_imu(self, msg: Imu):
        self._last_rx_imu = time.time()
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        gz = float(msg.angular_velocity.z)

        t_now = self.get_clock().now()
        if self.still_collecting:
            if self.still_t0 is None: self.still_t0 = t_now
            self.still_gyro.append(gz); self.still_ax.append(ax); self.still_ay.append(ay)
            if (t_now - self.still_t0).nanoseconds * 1e-9 >= TUNE["still_init_sec"]:
                self.bias_bg = float(np.mean(self.still_gyro)) if self.still_gyro else 0.0
                self.bias_ax = float(np.mean(self.still_ax))   if self.still_ax   else 0.0
                self.bias_ay = float(np.mean(self.still_ay))   if self.still_ay   else 0.0
                self.init_filter(self.bias_bg)
                self.still_collecting = False
                self.get_logger().info(f"Still-init: b_g={self.bias_bg:.6e}, b_ax={self.bias_ax:.6e}, b_ay={self.bias_ay:.6e}")
        else:
            self.step_filter(msg)

    # ── Init / Predict / Update ──────────────────────────────────────────
    def init_filter(self, bg0):
        N = self.N_cur
        self.particles = np.zeros((N,6), float)  # [x,y,yaw,v,beta,bg]
        self.particles[:,0] = np.random.normal(0.0, 0.10, size=N)
        self.particles[:,1] = np.random.normal(0.0, 0.10, size=N)
        self.particles[:,2] = np.random.normal(0.0, math.radians(2.0), size=N)
        self.particles[:,5] = bg0 + np.random.normal(0, 5e-5, size=N)
        self.weights = np.ones(N, float)/N
        self.last_time = None
        self.speed_lpf_state = 0.0
        self.v_imu = 0.0
        self.initialized = True

    def step_filter(self, imu: Imu):
        if not self.initialized or self.last_wheels is None: return

        t = Time.from_msg(imu.header.stamp).nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t; return
        dt = max(1e-4, min(0.02, t - self.last_time))
        self.last_time = t

        gz = float(imu.angular_velocity.z) - self.bias_bg
        ax = float(imu.linear_acceleration.x) - self.bias_ax
        ay = float(imu.linear_acceleration.y) - self.bias_ay

        # Rear-only wheel speed
        ws = self.last_wheels.speeds
        rb = float(ws.rb_speed); lb = float(ws.lb_speed)
        if abs(rb) < 5.0: rb = 0.0
        if abs(lb) < 5.0: lb = 0.0
        v_wheels = max(0.0, min(rb, lb) * (2*math.pi/60.0) * self.Rw)

        # IMU speed (leaky integrator + ZUPT)
        if self.tau_v > 1e-3:
            leak = math.exp(-dt/self.tau_v)
            self.v_imu = max(0.0, leak*self.v_imu + (1.0-leak)*max(0.0, self.v_imu + ax*dt))
        else:
            self.v_imu = max(0.0, self.v_imu + ax*dt)

        # ZUPT when stationary
        if (min(abs(rb), abs(lb)) < self.zupt_rpm) and (abs(gz) < self.zupt_yaw):
            self.v_imu *= 0.9

        # Blend (rear as truth when valid; IMU helps when wheels near zero)
        lam = 0.3 if v_wheels > 0.01 else 0.9
        v_raw = (1.0 - lam) * v_wheels + lam * self.v_imu

        # LPF on speed
        alpha = 1.0 - math.exp(-2*math.pi*self.fc*dt)
        self.speed_lpf_state += alpha * (v_raw - self.speed_lpf_state)
        v_meas = max(0.0, self.speed_lpf_state)

        # ─ Predict (vectorized)
        X = self.particles
        N = X.shape[0]
        X[:,2] = wrap_angle(X[:,2] + (gz - X[:,5]) * dt)                 # yaw
        v_eps = np.maximum(0.5, v_meas)
        k_slip = 0.2
        X[:,4] = np.clip(
            X[:,4] + k_slip*(ay/v_eps)*dt + np.random.normal(0, self.sb*math.sqrt(dt), size=N),
            -self.beta_lim, self.beta_lim
        )
        X[:,3] = np.maximum(0.0, v_meas + np.random.normal(0, self.sv*math.sqrt(dt), size=N))
        c = np.cos(X[:,2] + X[:,4]); s = np.sin(X[:,2] + X[:,4])
        X[:,0] += X[:,3] * c * dt
        X[:,1] += X[:,3] * s * dt
        X[:,5] += np.random.normal(0, self.sbg*dt, size=N)

        t0 = time.time()
        matches, drops, mean_mahal, p95_mahal, match_frac = 0, 0, 0.0, 0.0, 0.0

        # ─ Update with JCBB per particle
        if self.last_cones is not None:
            Yw, Pw, Col = self._extract_obs_world(self.last_cones)
            if Yw.shape[0] > 0:
                # pre-split once by color
                obs_idx_by_color = {
                    "blue":         np.where(Col=="blue")[0],
                    "yellow":       np.where(Col=="yellow")[0],
                    "orange":       np.where(Col=="orange")[0],
                    "orange_large": np.where(Col=="orange_large")[0],
                }
                Wlog = np.zeros(N, float)
                best_like = -1e18; best_m2 = []; best_pairs=0; best_drops=0; best_total_obs=0

                for k in range(N):
                    px,py,yaw = X[k,0], X[k,1], X[k,2]
                    Rwb = rot2d(yaw); Rbw = Rwb.T
                    like = 0.0; m2_all = []; pairs_all = 0; drops_all=0; total_obs=0

                    for color in ("blue","yellow","orange","orange_large"):
                        idxM = self.idx_by_color[color]
                        if idxM.size==0: continue
                        Mw_all = self.landmarks_world[idxM]
                        d2 = np.sum((Mw_all - np.array([px,py]))**2, axis=1)
                        keep = d2 <= (self.map_crop_m**2)
                        if color in ("orange","orange_large"):
                            keep = np.logical_and(keep, d2 <= (self.orange_within_m**2))
                        Mw = Mw_all[keep]
                        if Mw.shape[0]==0: continue

                        idxO = obs_idx_by_color[color]
                        if idxO.size==0: continue
                        Yw_c = Yw[idxO]; Pw_c = Pw[idxO]
                        total_obs += Yw_c.shape[0]

                        # world -> car
                        Yc = (Yw_c - np.array([px,py])) @ Rbw.T
                        Pc = np.einsum('ij,njk,kl->nil', Rbw, Pw_c, Rbw.T)
                        Mc = (Mw - np.array([px,py])) @ Rbw.T

                        gate = self.chi2_orange if color in ("orange","orange_large") else self.chi2_gate

                        # JCBB on this color
                        pairs_c, m2_c = jcbb(
                            Yc, Pc, Mc, gate,
                            TUNE["jcbb_max_obs_per_color"],
                            TUNE["jcbb_max_map_per_color"],
                            self.r_floor
                        )
                        if pairs_c:
                            like += -0.5*float(np.sum(m2_c))
                            pairs_all += len(pairs_c)
                            m2_all.extend(m2_c)
                            drops_all += (Yc.shape[0] - len(pairs_c))
                        else:
                            drops_all += Yc.shape[0]

                    Wlog[k] = like
                    if like > best_like:
                        best_like = like
                        best_m2 = m2_all[:]
                        best_pairs = pairs_all
                        best_drops = drops_all
                        best_total_obs = total_obs

                # normalize weights
                mmax = float(np.max(Wlog))
                w = np.exp(Wlog - mmax); w_sum = np.sum(w) + 1e-12
                self.weights = w / w_sum

                if best_m2:
                    r = np.sqrt(np.array(best_m2))
                    mean_mahal = float(np.mean(r))
                    p95_mahal  = float(np.percentile(r,95))
                matches = best_pairs; drops = best_drops
                tot = max(1, best_total_obs); match_frac = matches / tot

        t_upd_ms = (time.time()-t0)*1000.0

        # Resample
        Neff = 1.0 / (np.sum(self.weights**2) + 1e-12)
        if Neff/self.N_cur < 0.5:
            self._systematic_resample()

        # Logs & shrink
        mx,my,myaw,mv = self._est_mean()
        self.w_est.writerow([f"{t:.6f}", f"{mx:.3f}", f"{my:.3f}", f"{myaw:.5f}", f"{mv:.3f}"])
        if self.latest_gt_odom is not None:
            gx = self.latest_gt_odom.pose.pose.position.x
            gy = self.latest_gt_odom.pose.pose.position.y
            gq = self.latest_gt_odom.pose.pose.orientation
            siny_cosp = 2.0*(gq.w*gq.z + gq.x*gq.y)
            cosy_cosp = 1.0 - 2.0*(gq.y*gq.y + gq.z*gq.z)
            gyaw = math.atan2(siny_cosp, cosy_cosp)
            self.w_gt.writerow([f"{t:.6f}", f"{gx:.3f}", f"{gy:.3f}", f"{gyaw:.5f}"])
            self.w_err.writerow([f"{t:.6f}", f"{(gx-mx):.3f}", f"{(gy-my):.3f}", f"{wrap_angle(gyaw-myaw):.5f}"])
        self.w_assoc.writerow([f"{t:.6f}", matches, drops, f"{mean_mahal:.3f}", f"{p95_mahal:.3f}", f"{match_frac:.2f}", self.N_cur])
        self.w_tim.writerow([f"{t:.6f}", f"{(dt*1000):.2f}", f"{t_upd_ms:.2f}"])

        # Shrink logic
        if not self._shrunk:
            cond_ok = (match_frac >= self.shrink_match_frac_min) and (p95_mahal <= self.shrink_p95_mahal_max)
            now_w = time.time()
            if cond_ok:
                if self._stable_since is None: self._stable_since = now_w
                elif (now_w - self._stable_since) >= self.shrink_hold_time_s:
                    self._shrink_particles(self.N_run); self._shrunk = True
            else:
                self._stable_since = None

        # flush once a second
        if time.time() - self._last_flush > 1.0:
            for f in (self.fp_est,self.fp_gt,self.fp_err,self.fp_assoc,self.fp_tim): f.flush()
            self._last_flush = time.time()

    def _extract_obs_world(self, msg: ConeArrayWithCovariance):
        pts=[]; covs=[]; cols=[]
        def add(arr, color):
            for c in arr:
                pts.append([float(c.point.x), float(c.point.y)])
                covs.append([[c.covariance[0], c.covariance[1]],
                             [c.covariance[2], c.covariance[3]]])
                cols.append(color)
        add(msg.blue_cones,        "blue")
        add(msg.yellow_cones,      "yellow")
        add(msg.orange_cones,      "orange")
        add(msg.big_orange_cones,  "orange_large")
        if not pts:
            return np.zeros((0,2)), np.zeros((0,2,2)), np.array([], dtype=object)
        return np.asarray(pts,float), np.asarray(covs,float), np.array(cols, dtype=object)

    def _systematic_resample(self):
        N = self.N_cur
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i = j = 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(N, float)/N

    def _shrink_particles(self, N_new):
        N_old = self.particles.shape[0]
        # resample to concentrate
        self._systematic_resample()
        X = self.particles
        if N_new < N_old:
            X = X[:N_new]
        else:
            reps = int(np.ceil(N_new/N_old))
            X = np.tile(X, (reps,1))[:N_new]
            X[:,0] += np.random.normal(0,0.02,size=N_new)
            X[:,1] += np.random.normal(0,0.02,size=N_new)
            X[:,2] += np.random.normal(0,math.radians(0.3),size=N_new)
        self.particles = X
        self.weights = np.ones(N_new,float)/N_new
        self.N_cur = N_new
        self.get_logger().info(f"Particles shrunk {N_old} → {N_new}")

    def _est_mean(self):
        w = self.weights; X = self.particles
        mx = float(np.sum(w * X[:,0])); my = float(np.sum(w * X[:,1]))
        cy = float(np.sum(w * np.cos(X[:,2]))); sy = float(np.sum(w * np.sin(X[:,2])))
        myaw = math.atan2(sy, cy)
        mv = float(np.sum(w * X[:,3]))
        return mx,my,myaw,mv

    # ── Publish / Heartbeat / Shutdown ───────────────────────────────────
    def publish_estimate(self):
        if not self.initialized: return
        mx,my,myaw,mv = self._est_mean()
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now; t.header.frame_id = WORLD_FRAME; t.child_frame_id = BASE_FRAME
        t.transform.translation.x = mx; t.transform.translation.y = my; t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quat(myaw)
        self.tf_b.sendTransform(t)

        od = Odometry()
        od.header.stamp = now; od.header.frame_id = WORLD_FRAME; od.child_frame_id = BASE_FRAME
        od.pose.pose.position.x = mx; od.pose.pose.position.y = my; od.pose.pose.position.z = 0.0
        od.pose.pose.orientation = yaw_to_quat(myaw)
        od.twist.twist.linear = Vector3(x=mv, y=0.0, z=0.0)
        self.pub_odom.publish(od)

        ps = PoseStamped()
        ps.header.stamp = now; ps.header.frame_id = WORLD_FRAME
        ps.pose.position.x = mx; ps.pose.position.y = my; ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quat(myaw)
        self.pub_pose.publish(ps)

    def _heartbeat(self):
        age = lambda ts: f"{(time.time()-ts):.2f}s" if ts else "n/a"
        self.get_logger().info(
            f"[hb] init={self.initialized} imu_age={age(self._last_rx_imu)} "
            f"ws_age={age(self._last_rx_ws)} cones_age={age(self._last_rx_cone)} N={self.N_cur}"
        )

    def destroy_node(self):
        for f in (self.fp_est,self.fp_gt,self.fp_err,self.fp_assoc,self.fp_tim):
            try: f.close()
            except: pass
        super().destroy_node()

# ── main ────────────────────────────────────────────────────────────────
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
