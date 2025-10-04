#!/usr/bin/env python3
# pf_da_corrector.py
#
# Post-filter node:
#   - Subscribes to predicted odom (your IMU+wheel node)
#   - Subscribes to cones
#   - Runs JCBB DA against known map
#   - Maintains a PF over Δ = [dx, dy, dθ] (drift)
#   - Publishes corrected odom at the **odom input rate** from cb_odom:
#       * Before first successful DA: pass-through (unchanged)
#       * After first successful DA: apply last Δ̂ to each incoming odom
#   - Logs per-frame DA metrics: obs, matches, processing time
#
# Topics (defaults):
#   in:  /odometry_integration/car_state     (nav_msgs/Odometry)
#        /ground_truth/cones                 (eufs_msgs/ConeArrayWithCovariance)
#   out: /pf_localizer/odom_corrected        (nav_msgs/Odometry)
#        /pf_localizer/pose2d                (geometry_msgs/Pose2D)
#
# Map assets expected under map_data_dir (default slam_utils/data):
#   - alignment_transform.json  -> {"rotation":[[...],[...]], "translation":[tx,ty]}
#   - known_map.csv             -> columns (x,y, color/tag optional)
#
import math, json, time, bisect
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3

# ---------- helpers ----------
def rot2d(th: float):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def quat_from_yaw(yaw):
    q = Quaternion()
    q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return  float('inf')
    a1=-3.969683028665376e+01; a2= 2.209460984245205e+02; a3=-2.759285104469687e+02
    a4= 1.383577518672690e+02; a5=-3.066479806614716e+01; a6= 2.506628277459239e+00
    b1=-5.447609879822406e+01; b2= 1.615858368580409e+02; b3=-1.556989798598866e+02
    b4= 6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00
    d4= 3.754408661907416e+00
    plow  = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p - 0.5; r = q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)

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
    xk=pick("x","x_m","xmeter"); yk=pick("y","y_m","ymeter"); ck=pick("color","colour","tag")
    for r in rdr:
        try:
            x=float(r[xk]); y=float(r[yk]); c=str(r[ck]).strip().lower()
        except Exception:
            continue
        rows.append((x,y,c))
    return rows

# -------------------------- PF + embedded JCBB --------------------------
class PFDriftCorrector(Node):
    def __init__(self):
        super().__init__("pf_da_corrector", automatically_declare_parameters_from_overrides=True)

        # ---- topics ----
        self.declare_parameter("pose_topic", "/odometry_integration/car_state")   # predicted odom IN
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("out_odom_topic", "/pf_localizer/odom_corrected")  # corrected odom OUT
        self.declare_parameter("out_pose2d_topic", "/pf_localizer/pose2d")

        # ---- JCBB params ----
        self.declare_parameter("detections_frame", "base")         # "base" or "world"
        self.declare_parameter("map_crop_m", 28.0)
        self.declare_parameter("chi2_gate_2d", 11.83)              # ~99% for 2 dof
        self.declare_parameter("joint_sig", 0.95)
        self.declare_parameter("joint_relax", 1.4)
        self.declare_parameter("min_k_for_joint", 3)
        self.declare_parameter("meas_sigma_floor_xy", 0.30)
        self.declare_parameter("alpha_trans", 0.9)
        self.declare_parameter("beta_rot", 0.9)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 60.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("min_candidates", 0)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)

        # ---- PF params ----
        self.declare_parameter("pf.num_particles", 250)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        self.declare_parameter("pf.process_std_xy_m", 0.02)      # Δ random walk per cone frame
        self.declare_parameter("pf.process_std_yaw_rad", 0.002)
        self.declare_parameter("pf.likelihood_floor", 1e-12)

        # ---- read params ----
        gp = self.get_parameter
        self.pose_topic = str(gp("pose_topic").value)
        self.cones_topic = str(gp("cones_topic").value)
        self.out_odom_topic = str(gp("out_odom_topic").value)
        self.out_pose2d_topic = str(gp("out_pose2d_topic").value)

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.map_crop_m = float(gp("map_crop_m").value)
        self.chi2_gate = float(gp("chi2_gate_2d").value)
        self.joint_sig = float(gp("joint_sig").value)
        self.joint_relax = float(gp("joint_relax").value)
        self.min_k_for_joint = int(gp("min_k_for_joint").value)
        self.sigma0 = float(gp("meas_sigma_floor_xy").value)
        self.alpha = float(gp("alpha_trans").value)
        self.beta = float(gp("beta_rot").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(gp("target_cone_hz").value)
        self.min_candidates = int(gp("min_candidates").value)
        self.max_pairs_print = int(gp("max_pairs_print").value)

        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)

        # ---- load map assets ----
        data_dir = (Path(__file__).resolve().parents[0] / str(gp("map_data_dir").value)).resolve()
        align_json = data_dir / "alignment_transform.json"
        known_map_csv = data_dir / "known_map.csv"
        if not align_json.exists() or not known_map_csv.exists():
            raise RuntimeError(f"[pf_da_corrector] Missing map assets under {data_dir}")
        with open(align_json, "r") as f:
            T = json.load(f)
        R_map = np.array(T["rotation"], float)
        t_map = np.array(T["translation"], float)
        rows = load_known_map(known_map_csv)
        M = np.array([[x, y] for (x, y, _) in rows], float)
        self.MAP = (M @ R_map.T) + t_map

        # ---- state buffers ----
        self.odom_buf: deque = deque()  # (t,x,y,yaw,v,yawrate)
        self.pose_latest = np.array([0.0, 0.0, 0.0], float)
        self._pose_feed_ready = False

        # PF state: Δ = [dx, dy, dθ] particles + weights; latched Δ̂ and flag
        self.particles = np.zeros((self.Np, 3), float)
        self.weights = np.ones(self.Np, float) / self.Np
        self.delta_hat = np.zeros(3, float)   # [dx, dy, dθ]
        self.have_delta = False               # publish pass-through until True

        self.last_cone_ts = None
        self._last_obs = 0
        self._last_match = 0

        # ROS I/O
        q_rel = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                           durability=QoSDurabilityPolicy.VOLATILE,
                           history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.pose_topic, self.cb_odom, q_rel)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_rel)
        self.pub_out = self.create_publisher(Odometry, self.out_odom_topic, 10)
        self.pub_pose2d = self.create_publisher(Pose2D, self.out_pose2d_topic, 10)

        self.get_logger().info(f"[pf_da_corrector] up. pose_in={self.pose_topic} | cones={self.cones_topic} | out={self.out_odom_topic}")
        self.get_logger().info(f"[pf_da_corrector] map cones: {self.MAP.shape[0]} | detections_frame={self.detections_frame}")

    # ---------------------- Odom callback: publish at odom rate ----------------------
    def cb_odom(self, msg: Odometry):
        # Always publish here so the output is fast (same rate as predictor).
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = math.hypot(vx, vy)
        yr = msg.twist.twist.angular.z

        if not self._pose_feed_ready:
            self._pose_feed_ready = True
            self.get_logger().info("[pf_da_corrector] Prediction feed ready.")

        # Store in buffer for DA time alignment
        self.pose_latest = np.array([x, y, yaw], float)
        self.odom_buf.append((t, x, y, yaw, v, yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

        # Compose output:
        if self.have_delta:
            dx, dy, dth = self.delta_hat
            x_c, y_c, yaw_c = x + dx, y + dy, wrap(yaw + dth)
        else:
            # pass-through until we have a valid Δ̂
            x_c, y_c, yaw_c = x, y, yaw

        self._publish_pose(t, x_c, y_c, yaw_c, v, yr)

    def _pose_at(self, t_query: float) -> Tuple[np.ndarray, float, float, float]:
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf'), 0.0, 0.0
        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)
        if idx == 0:
            t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[0]
            return np.array([x0, y0, yaw0], float), abs(t_query - t0), v0, yr0
        if idx >= len(self.odom_buf):
            t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[-1]
            dt = t_query - t1
            dt_c = max(0.0, min(dt, self.extrap_cap))
            if abs(yr1) < 1e-3:
                x = x1 + v1 * dt_c * math.cos(yaw1)
                y = y1 + v1 * dt_c * math.sin(yaw1)
                yaw = wrap(yaw1)
            else:
                x = x1 + (v1 / yr1) * (math.sin(yaw1 + yr1 * dt_c) - math.sin(yaw1))
                y = y1 - (v1 / yr1) * (math.cos(yaw1 + yr1 * dt_c) - math.cos(yaw1))
                yaw = wrap(yaw1 + yr1 * dt_c)
            return np.array([x, y, yaw], float), abs(dt), v1, yr1
        t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[idx]
        if t1 == t0:
            return np.array([x0, y0, yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a * dyaw)
        v = (1 - a) * v0 + a * v1
        yr = (1 - a) * yr0 + a * yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x, y, yaw], float), dt_near, v, yr

    # -------------------- JCBB (same behavior as before) --------------------
    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        trans_var = self.alpha * (v ** 2) * (dt ** 2)
        lateral_var = self.beta * (yawrate ** 2) * (dt ** 2) * (r ** 2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0], [0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0, 0] += 1e-9; S[1, 1] += 1e-9
        return S

    def _jcbb_match(self, t_z: float, msg: ConeArrayWithCovariance):
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        x, y, yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        obs_pw = []; obs_S0 = []; obs_pb = []
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma0 ** 2, self.sigma0 ** 2])
                if self.detections_frame == "base":
                    pb = np.array([px, py], float)
                    pw = np.array([x, y]) + Rwb @ pb
                    S0 = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px, py], float)
                    pb = Rbw @ (pw - np.array([x, y]))
                    S0 = Pw
                obs_pw.append(pw); obs_S0.append(S0); obs_pb.append(pb)
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)

        n_obs = len(obs_pw)
        if n_obs == 0:
            self._last_obs, self._last_match = 0, 0
            return [], pose_w_b, dt_pose, v, yawrate, obs_pb

        crop2 = self.map_crop_m ** 2
        MAP = self.MAP
        cands: List[List[Tuple[int, float]]] = []
        total_cands = 0
        for pw, S0, pb in zip(obs_pw, obs_S0, obs_pb):
            r = float(np.linalg.norm(pb))
            S_eff = self._inflate_cov(S0, v, yawrate, dt_pose if math.isfinite(dt_pose) else 0.0, r, yaw)
            try: S_inv = np.linalg.inv(S_eff)
            except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S_eff)
            diffs = MAP - pw
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            near = np.where(d2 <= crop2)[0]
            cc=[]
            for mid in near:
                innov = pw - MAP[mid]
                m2 = float(innov.T @ S_inv @ innov)
                if m2 <= self.chi2_gate:
                    cc.append((int(mid), m2))
            if not cc and self.min_candidates > 0 and near.size > 0:
                k = min(self.min_candidates, near.size)
                by = sorted([(int(mid), d2[mid]) for mid in near], key=lambda t: t[1])[:k]
                cc = [(mid, self.chi2_gate * 1.01) for (mid, _) in by]
            cc.sort(key=lambda t: t[1])
            cands.append(cc)
            total_cands += len(cc)

        if total_cands == 0:
            self._last_obs, self._last_match = n_obs, 0
            return [], pose_w_b, dt_pose, v, yawrate, obs_pb

        order = sorted(range(n_obs), key=lambda i: len(cands[i]) if cands[i] else 9999)
        cands_ord = [cands[i] for i in order]

        best_pairs: List[Tuple[int, int, float]] = []
        best_K = 0
        used = set()

        def dfs(idx, cur_pairs, cur_sum):
            nonlocal best_pairs, best_K
            cur_K = len(cur_pairs)
            if cur_K + (len(cands_ord) - idx) < best_K:
                return
            if cur_K >= self.min_k_for_joint:
                df = 2 * cur_K
                thresh = self.joint_relax * chi2_quantile(df, self.joint_sig)
                if cur_sum > thresh:
                    return
            if idx == len(cands_ord):
                if cur_K > best_K:
                    best_K = cur_K
                    best_pairs = cur_pairs.copy()
                return
            dfs(idx + 1, cur_pairs, cur_sum)  # skip
            for (mid, m2) in cands_ord[idx]:
                if mid in used:
                    continue
                used.add(mid)
                cur_pairs.append((order[idx], mid, m2))
                dfs(idx + 1, cur_pairs, cur_sum + m2)
                cur_pairs.pop()
                used.remove(mid)

        dfs(0, [], 0.0)
        self._last_obs, self._last_match = n_obs, best_K
        return best_pairs, pose_w_b, dt_pose, v, yawrate, obs_pb

    # -------------------- PF core over Δ = [dx, dy, dθ] --------------------
    def _pf_predict(self):
        self.particles[:, 0] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 1] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 2] += np.random.normal(0.0, self.p_std_yaw, size=self.Np)
        self.particles[:, 2] = np.array([wrap(th) for th in self.particles[:, 2]])

    def _pf_update(self, pairs, odom_pose, obs_pb):
        if len(pairs) < max(2, self.min_k_for_joint):
            return False

        idxs = [pi for (pi, mid, _) in pairs]
        mids = [mid for (pi, mid, _) in pairs]
        PB = np.stack([obs_pb[i] for i in idxs], axis=0)
        MW = self.MAP[np.array(mids, int), :]

        x_o, y_o, yaw_o = odom_pose
        new_weights = np.zeros_like(self.weights)
        K = PB.shape[0]; df_pair = 2

        for i in range(self.Np):
            dx, dy, dth = self.particles[i]
            yaw_c = wrap(yaw_o + dth)
            Rc = rot2d(yaw_c)
            pc = np.array([x_o + dx, y_o + dy], float)
            PW_pred = (pc[None, :] + (PB @ Rc.T))
            diffs = PW_pred - MW
            m2 = np.sum((diffs * diffs) / (self.sigma0 ** 2), axis=1)
            m2_sum = float(np.sum(m2))
            df = df_pair * K
            gate = self.joint_relax * chi2_quantile(df, self.joint_sig)
            like = self.like_floor if m2_sum > gate else max(math.exp(-0.5 * m2_sum), self.like_floor)
            new_weights[i] = self.weights[i] * like

        sw = float(np.sum(new_weights))
        if sw <= 0.0 or not math.isfinite(sw):
            self.weights[:] = 1.0 / self.Np
        else:
            self.weights[:] = new_weights / sw

        neff = 1.0 / float(np.sum(self.weights * self.weights))
        if neff < self.neff_ratio * self.Np:
            self._systematic_resample()
        return True

    def _systematic_resample(self):
        N = self.Np
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, 'i')
        cs = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cs[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights[:] = 1.0 / N

    def _estimate_delta(self):
        w = self.weights
        dx = float(np.sum(w * self.particles[:, 0]))
        dy = float(np.sum(w * self.particles[:, 1]))
        cs = float(np.sum(w * np.cos(self.particles[:, 2])))
        sn = float(np.sum(w * np.sin(self.particles[:, 2])))
        dth = math.atan2(sn, cs)
        return dx, dy, dth

    # -------------------- Cones callback: DA + PF update + LOGS --------------------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self._pose_feed_ready:
            return

        now_wall = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last_cone_ts is not None:
            if (t_z - self.last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        self.last_cone_ts = t_z

        t0 = time.perf_counter()
        pairs, odom_pose, dt_pose, v, yawrate, obs_pb = self._jcbb_match(t_z, msg)

        if len(pairs) < max(2, self.min_k_for_joint):
            proc_ms = int((time.perf_counter() - t0) * 1000)
            self.get_logger().info(
                f"[pf_da/jcbb] t={t_z:7.2f}s obs={self._last_obs} matched={self._last_match} proc={proc_ms}ms (no update)"
            )
            return

        # PF step
        self._pf_predict()
        _ = self._pf_update(pairs, odom_pose, obs_pb)
        dx, dy, dth = self._estimate_delta()
        self.delta_hat = np.array([dx, dy, dth], float)
        self.have_delta = True

        proc_ms = int((time.perf_counter() - t0) * 1000)
        # Pretty list of first few pairs: (obs_i -> map_j)
        show = ", ".join([f"(o{pi}->m{mi})" for (pi, mi, _) in pairs[:self.max_pairs_print]])
        self.get_logger().info(
            f"[pf_da/jcbb] t={t_z:7.2f}s obs={self._last_obs} matched={self._last_match} proc={proc_ms}ms "
            f"Δ̂=[{dx:+.3f},{dy:+.3f},{math.degrees(dth):+.2f}°] {(' | ' + show) if show else ''}"
        )

    # -------------------- Publisher --------------------
    def _publish_pose(self, t_s: float, x: float, y: float, yaw: float, v: float, yawrate: float):
        od = Odometry()
        od.header.stamp = rclpy.time.Time(seconds=t_s).to_msg()
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose = PoseWithCovariance()
        od.twist = TwistWithCovariance()
        od.pose.pose = Pose(position=Point(x=x, y=y, z=0.0), orientation=quat_from_yaw(yaw))
        od.twist.twist = Twist(
            linear=Vector3(x=v * math.cos(yaw), y=v * math.sin(yaw), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=yawrate)
        )
        self.pub_out.publish(od)

        p2 = Pose2D(); p2.x = x; p2.y = y; p2.theta = yaw
        self.pub_pose2d.publish(p2)

# -------------------------- main --------------------------
def main():
    rclpy.init()
    node = PFDriftCorrector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
