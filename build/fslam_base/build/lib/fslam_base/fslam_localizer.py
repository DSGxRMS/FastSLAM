#!/usr/bin/env python3
# fslam_localizer.py  (ROS2 Galactic)
# Particle-filter localization for skidpad with fixed landmark map.
# - Uses precomputed alignment transform (map->world) from slam_utils/data/alignment_transform.json
# - State per particle: [x, y, yaw, v, beta, b_g]
# - Prediction: v from wheel speeds (RPM->m/s + LPF), yaw_dot from IMU z - b_g, slip beta evolves with a_y
# - Data association: Mahalanobis NN (greedy one-to-one), colored cones only, in CAR frame
# - Publishes TF world->base_footprint, /fslam/odom, /fslam/pose
# - Logs under logs/fslam/<timestamp>/

import os, json, math, time, csv
from pathlib import Path
from collections import deque, defaultdict

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

from eufs_msgs.msg import WheelSpeedsStamped, ConeArrayWithCovariance

# ------------------------------ Paths & constants ------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # .../fslam_base
DATA_DIR = BASE_DIR / "slam_utils" / "data"
ALIGN_JSON = DATA_DIR / "alignment_transform.json"
KNOWN_MAP_CSV = DATA_DIR / "known_map.csv"  # same schema we used earlier (x,y,color)
LOG_ROOT = BASE_DIR.parents[1] / "logs" / "fslam"   # <workspace>/logs/fslam/

WORLD_FRAME = "world"             # publish in "world"
BASE_FRAME  = "base_footprint"    # from your URDF

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
    # Expect at least columns: x,y,color (case-insensitive; comments allowed)
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
        # normalize colors
        if c in ("bluecone","b"): c="blue"
        if c in ("yellowcone","y"): c="yellow"
        if c in ("big_orange","orange_large","large_orange"): c="orange_large"
        if c in ("small_orange","orange_small"): c="orange"
        if c not in ("blue","yellow","orange","orange_large"): c="unknown"
        rows.append({"x":x,"y":y,"color":c})
    return rows

# Greedy one-to-one assignment from a dense cost matrix with gating
def greedy_assign(cost_mat, gate):
    # cost_mat: shape (No, Nm) [observed -> map]; returns pairs list[(i_obs, j_map)]
    No, Nm = cost_mat.shape
    flat = []
    for i in range(No):
        for j in range(Nm):
            c = cost_mat[i,j]
            if np.isfinite(c) and c <= gate:
                flat.append((c, i, j))
    flat.sort(key=lambda x:x[0])
    used_o = set(); used_m = set(); pairs=[]
    for c,i,j in flat:
        if i in used_o or j in used_m: continue
        pairs.append((i,j))
        used_o.add(i); used_m.add(j)
    return pairs

# ------------------------------ Main Node ------------------------------
class FslamLocalizer(Node):
    def __init__(self):
        super().__init__("fslam_localizer")

        # ---- Params (tunable, kept small & explicit) ----
        self.declare_parameter("particles", 200)
        self.declare_parameter("wheel_radius_m", 0.26)   # TODO: set true value
        self.declare_parameter("wheelbase_m", 1.6)       # not critical here, but kept for future
        self.declare_parameter("speed_lpf_fc_hz", 8.0)
        self.declare_parameter("proc_sigma_v", 0.3)      # m/s / sqrt(s)
        self.declare_parameter("proc_sigma_beta_deg", 0.5)  # deg / sqrt(s)
        self.declare_parameter("proc_sigma_bg", 2e-4)    # rad/s^2 (random walk on bias)
        self.declare_parameter("meas_sigma_floor_xy", 0.20) # additive floor (m)
        self.declare_parameter("gate_chi2_2d", 5.99)     # ~95% for 2 dof
        self.declare_parameter("still_init_sec", 0.5)
        self.declare_parameter("publish_rate_hz", 60.0)

        self.N = int(self.get_parameter("particles").value)
        self.Rw = float(self.get_parameter("wheel_radius_m").value)
        self.fc = float(self.get_parameter("speed_lpf_fc_hz").value)
        self.sv = float(self.get_parameter("proc_sigma_v").value)
        self.sb = math.radians(float(self.get_parameter("proc_sigma_beta_deg").value))
        self.sbg = float(self.get_parameter("proc_sigma_bg").value)
        self.r_floor = float(self.get_parameter("meas_sigma_floor_xy").value)
        self.chi2_gate = float(self.get_parameter("gate_chi2_2d").value)
        self.still_init_sec = float(self.get_parameter("still_init_sec").value)
        self.pub_rate = float(self.get_parameter("publish_rate_hz").value)

        # ---- Load transform & map ----
        if not ALIGN_JSON.exists():
            raise RuntimeError(f"Missing {ALIGN_JSON}")
        with open(ALIGN_JSON, "r") as f:
            T = json.load(f)
        R = np.array(T["rotation"], dtype=float)
        t = np.array(T["translation"], dtype=float)
        self.R_map = R
        self.t_map = t

        map_rows = load_known_map(KNOWN_MAP_CSV)
        M = np.array([[r["x"], r["y"]] for r in map_rows], float)
        C = [r["color"] for r in map_rows]
        # map (original) -> world
        self.landmarks_world = (M @ self.R_map.T) + self.t_map  # (Nm,2)
        self.landmark_colors = C
        # Pre-split by color (we only use colored cones)
        self.idx_by_color = {
            "blue":   np.array([i for i,c in enumerate(C) if c=="blue"], int),
            "yellow": np.array([i for i,c in enumerate(C) if c=="yellow"], int),
            # Optionally add 'orange' later if you want to use starts too
        }

        # ---- State / buffers ----
        self.initialized = False
        self.still_collecting = True
        self.still_t0 = None
        self.still_gyro = deque()
        self.speed_lpf_state = 0.0
        self.last_time = None

        # Particles: [x,y,yaw,v,beta,bg]
        self.particles = None
        self.weights = None

        # Latest sensors
        self.last_imu = None
        self.last_wheels = None
        self.last_cones = None
        self.latest_gt_odom = None  # dev logs

        # ---- Publishers & TF ----
        self.pub_odom = self.create_publisher(Odometry, "/fslam/odom", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/fslam/pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---- Subscribers ----
        self.create_subscription(Imu, "/imu/data", self.cb_imu, 50)
        self.create_subscription(WheelSpeedsStamped, "/ros_can/wheel_speeds", self.cb_wheels, 10)
        self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_gt, 10)  # dev logs only

        # ---- Timers ----
        self.timer_pub = self.create_timer(1.0/self.pub_rate, self.publish_estimate)

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
        self.w_assoc= csv.writer(self.fp_assoc);self.w_assoc.writerow(["t","matches","dropped","mean_mahal","p95_mahal"])
        self.w_tim  = csv.writer(self.fp_timing);self.w_tim.writerow(["t","predict_ms","update_ms"])

        self.get_logger().info("fslam_localizer up.")

    # -------------------- Callbacks --------------------
    def cb_gt(self, msg: Odometry):
        self.latest_gt_odom = msg

    def cb_wheels(self, msg: WheelSpeedsStamped):
        self.last_wheels = msg

    def cb_cones(self, msg: ConeArrayWithCovariance):
        self.last_cones = msg

    def cb_imu(self, msg: Imu):
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
        # Initial pose: set to (0,0) in world and yaw=0. This is okay because alignment pinned the map to world.
        # If you want to start at start-box centroid, you can pass that here. We'll keep it simple.
        x0 = 0.0; y0 = 0.0; yaw0 = 0.0
        v0 = 0.0; beta0 = 0.0
        self.particles = np.zeros((self.N,6), float)
        self.particles[:,0] = x0 + np.random.normal(0, 0.10, size=self.N)   # 10 cm
        self.particles[:,1] = y0 + np.random.normal(0, 0.10, size=self.N)
        self.particles[:,2] = yaw0 + np.random.normal(0, math.radians(2.0), size=self.N)
        self.particles[:,3] = v0
        self.particles[:,4] = beta0
        self.particles[:,5] = bg0 + np.random.normal(0, 5e-5, size=self.N)
        self.weights = np.ones(self.N, float) / self.N
        self.last_time = None
        self.initialized = True

    def step_filter(self, imu: Imu):
        if not self.initialized: return
        if self.last_wheels is None: return

        # dt from IMU stamps
        t = Time.from_msg(imu.header.stamp).nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = max(1e-4, min(0.02, t - self.last_time))  # clamp
        self.last_time = t

        # ------ Predict ------
        t0 = time.time()
        gyro_z = float(imu.angular_velocity.z)
        a_y = float(imu.linear_acceleration.y)

        # Wheels -> v (m/s), conservative rear min, LPF
        ws = self.last_wheels.speeds
        # speeds are RPM
        rb = float(ws.rb_speed); lb = float(ws.lb_speed)
        v_raw = min(rb, lb) * (2*math.pi/60.0) * self.Rw
        # 1st-order LPF
        alpha = 1.0 - math.exp(-2*math.pi*self.fc*dt)
        self.speed_lpf_state += alpha * (v_raw - self.speed_lpf_state)
        v_meas = max(0.0, self.speed_lpf_state)

        # Vectorized predict
        X = self.particles
        bg = X[:,5]
        yaw_dot = gyro_z - bg
        X[:,2] = wrap_angle(X[:,2] + yaw_dot*dt)           # yaw
        # Slip update: beta += k*(a_y / max(v,eps))*dt + noise
        v_eps = np.maximum(0.5, v_meas)                    # avoid crazy when near zero
        k_slip = 0.2
        X[:,4] = np.clip(X[:,4] + k_slip*(a_y/v_eps)*dt + np.random.normal(0, self.sb*math.sqrt(dt), size=self.N),
                         math.radians(-15.0), math.radians(15.0))
        # v follows measured with small process noise (shared mean + per-particle noise)
        X[:,3] = np.maximum(0.0, v_meas + np.random.normal(0, self.sv*math.sqrt(dt), size=self.N))
        # Position
        cos_h = np.cos(X[:,2] + X[:,4])
        sin_h = np.sin(X[:,2] + X[:,4])
        X[:,0] += X[:,3] * cos_h * dt
        X[:,1] += X[:,3] * sin_h * dt
        # Gyro bias random walk
        X[:,5] += np.random.normal(0, self.sbg*dt, size=self.N)
        t_pred_ms = (time.time()-t0)*1000.0

        # ------ Update (colored cones only) ------
        t1 = time.time()
        matches = 0; drops = 0; mahal_vals=[]
        if self.last_cones is not None:
            # Gather observed colored cones (world coords) + 2x2 covariances
            Yw, Pw, Col = self._extract_observed_world(self.last_cones)
            # For speed, if no colored cones, skip update
            if Yw.shape[0] > 0:
                # For each particle, compute weight via Mahalanobis NN in CAR frame
                W = np.zeros(self.N, float)
                for k in range(self.N):
                    px,py,yaw = X[k,0], X[k,1], X[k,2]
                    Rwb = rot2d(yaw)      # world-from-body
                    Rbw = Rwb.T
                    # Precompute expected positions of map cones (color-split)
                    like = 0.0
                    # Build a single big cost across colors by concatenating:
                    # (But we need one-to-one within each color; do per color.)
                    total_mahal = []
                    total_pairs = 0; total_drops = 0
                    for color in ("blue","yellow"):
                        idx = self.idx_by_color[color]
                        if idx.size == 0: continue
                        Mw = self.landmarks_world[idx]  # (Nm_c,2)
                        # Expected relative positions of map cones in CAR frame
                        Mc = (Mw - np.array([px,py])) @ Rbw.T  # (Nm_c,2)
                        # Observed for that color (in world)
                        sel = (Col == color)
                        if not np.any(sel): continue
                        Yw_c = Yw[sel]      # (No_c,2)
                        Pw_c = Pw[sel]      # (No_c,2,2)
                        # Transform observed to CAR frame, and covariances too
                        Rc = Rbw
                        Yc = (Yw_c - np.array([px,py])) @ Rc.T   # (No_c,2)
                        Pc = Rc @ Pw_c @ Rc.T                    # (No_c,2,2)
                        # Build Mahalanobis cost matrix: No_c x Nm_c
                        No_c, Nm_c = Yc.shape[0], Mc.shape[0]
                        cost = np.empty((No_c, Nm_c), float)
                        # Add small measurement floor in CAR frame
                        Radd = (self.r_floor**2)*np.eye(2)
                        for i in range(No_c):
                            S = Pc[i] + Radd
                            # Invert once (2x2)
                            det = S[0,0]*S[1,1] - S[0,1]*S[1,0]
                            if det <= 1e-12:
                                Sinv = np.linalg.pinv(S)
                            else:
                                inv_det = 1.0/det
                                Sinv = np.array([[ S[1,1], -S[0,1]],
                                                 [-S[1,0],  S[0,0]]], float) * inv_det
                            di = Yc[i] - Mc   # broadcast (Nm_c,2)
                            # mahal^2 = d^T S^-1 d
                            # Compute efficiently:
                            temp = di @ Sinv
                            m2 = np.sum(temp * di, axis=1)
                            cost[i,:] = m2
                        # Greedy one-to-one under chi2 gate
                        pairs = greedy_assign(cost, gate=self.chi2_gate)
                        # Accumulate likelihood (sum of -0.5*m2 for pairs), drop others
                        if pairs:
                            m2_used = []
                            for (i,j) in pairs:
                                m2 = cost[i,j]
                                like += -0.5*m2
                                m2_used.append(m2)
                            total_mahal.extend(m2_used)
                            total_pairs += len(pairs)
                            total_drops += (No_c - len(pairs))
                        else:
                            total_drops += No_c
                    # Weight from like
                    W[k] = like
                # Normalize weights (log-likelihoods -> exponentiate with stabilization)
                mmax = np.max(W)
                w = np.exp(W - mmax)
                w_sum = np.sum(w) + 1e-12
                self.weights = w / w_sum
                matches = total_pairs
                drops   = total_drops
                Mahal = np.array(total_mahal, float) if len(total_mahal)>0 else np.array([])
                if Mahal.size>0:
                    mahal_vals = list(np.sqrt(Mahal))  # report sqrt(m2)
                else:
                    mahal_vals = []
        t_upd_ms = (time.time()-t1)*1000.0

        # ------ Resample ------
        Neff = 1.0 / (np.sum(self.weights**2) + 1e-12)
        if Neff/self.N < 0.5:
            self._systematic_resample()

        # ------ Logs (dev) ------
        tsec = t
        mx, my, myaw, mv = self._est_mean()
        self.w_est.writerow([f"{tsec:.6f}", f"{mx:.3f}", f"{my:.3f}", f"{myaw:.5f}", f"{mv:.3f}"])
        if self.latest_gt_odom is not None:
            gx = self.latest_gt_odom.pose.pose.position.x
            gy = self.latest_gt_odom.pose.pose.position.y
            gq = self.latest_gt_odom.pose.pose.orientation
            # yaw from quaternion (2D)
            siny_cosp = 2.0*(gq.w*gq.z + gq.x*gq.y)
            cosy_cosp = 1.0 - 2.0*(gq.y*gq.y + gq.z*gq.z)
            gyaw = math.atan2(siny_cosp, cosy_cosp)
            self.w_gt.writerow([f"{tsec:.6f}", f"{gx:.3f}", f"{gy:.3f}", f"{gyaw:.5f}"])
            self.w_err.writerow([f"{tsec:.6f}",
                                 f"{(gx-mx):.3f}", f"{(gy-my):.3f}", f"{wrap_angle(gyaw-myaw):.5f}"])
        mean_mahal = (sum(mahal_vals)/len(mahal_vals)) if mahal_vals else 0.0
        p95_mahal  = (np.percentile(mahal_vals,95) if mahal_vals else 0.0)
        self.w_assoc.writerow([f"{tsec:.6f}", matches, drops, f"{mean_mahal:.3f}", f"{p95_mahal:.3f}"])
        self.w_tim.writerow([f"{tsec:.6f}", f"{t_pred_ms:.2f}", f"{t_upd_ms:.2f}"])

    # Extract colored cones (world coords + cov 2x2) from ConeArrayWithCovariance
    def _extract_observed_world(self, msg: ConeArrayWithCovariance):
        pts=[]; covs=[]; cols=[]
        def add(arr, color):
            for c in arr:
                x = float(c.point.x); y=float(c.point.y)
                cov = np.array([[c.covariance[0], c.covariance[1]],
                                [c.covariance[2], c.covariance[3]]], float)
                pts.append([x,y]); covs.append(cov); cols.append(color)
        add(msg.blue_cones,   "blue")
        add(msg.yellow_cones, "yellow")
        # we intentionally ignore oranges here for DA
        if len(pts)==0:
            return np.zeros((0,2)), np.zeros((0,2,2)), np.array([], dtype=object)
        return np.asarray(pts,float), np.asarray(covs,float), np.array(cols, dtype=object)

    def _systematic_resample(self):
        N = self.N
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(N, float)/N

    def _est_mean(self):
        w = self.weights
        X = self.particles
        mx = float(np.sum(w * X[:,0]))
        my = float(np.sum(w * X[:,1]))
        # yaw: circular mean
        cy = float(np.sum(w * np.cos(X[:,2]))); sy = float(np.sum(w * np.sin(X[:,2])))
        myaw = math.atan2(sy, cy)
        mv = float(np.sum(w * X[:,3]))
        return mx,my,myaw,mv

    # -------------------- Publish --------------------
    def publish_estimate(self):
        if not self.initialized: return
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
        od.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)  # optional: publish mean yaw_dot
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
