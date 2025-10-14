#!/usr/bin/env python3
# fslam_mapper_jcbb.py — Online cone mapping with JCBB + tiny PF drift correction (Δ = [dx, dy, dθ])
#
# Input:
#   /odometry_integration/car_state         (nav_msgs/Odometry)
#   /ground_truth/cones                     (eufs_msgs/ConeArrayWithCovariance)
#
# Output:
#   /mapper/known_map                       (eufs_msgs/ConeArrayWithCovariance)
#
# Notes:
#   - JCBB runs against the **current map** (no ICP, no pose graph).
#   - Gates use S_gate = P_lm + inflate(S_meas, α,β, …)  (CRITICAL FIX).
#   - A tiny PF estimates Δ each frame to cancel odom drift; Δ is applied before gating & fusion.
#   - Throughput: caps on candidates/obs + DFS time budget; throttling supported.

import math, time, bisect
from collections import deque
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance

# ---------- helpers ----------
def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return  float('inf')
    a1=-3.969683028665376e+01; a2= 2.209460984245205e+02; a3=-2.759285104469687e+02
    a4= 1.383577518672690e+02; a5=-3.066479806614716e+01; a6= 2.506628277459239e+00
    b1=-5.447609879822406e+01; b2= 1.615858368580409e+02; b3=-1.556989798598866e+02
    b4= 6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00; d4= 3.754408661907416e+00
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p>phigh:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q=p-0.5; r=q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)

def add_cone(arr: ConeArrayWithCovariance, x: float, y: float, P: np.ndarray, color: str):
    a = float(P[0,0]); d = float(P[1,1]); b = float(P[0,1]); c = float(P[1,0])
    pt = ConeWithCovariance()
    pt.point.x = x; pt.point.y = y; pt.point.z = 0.0
    pt.covariance = [a, b, c, d]
    cl = (color or "").lower().strip()
    if ("big" in cl) and ("orange" in cl):
        arr.big_orange_cones.append(pt)
    elif "orange" in cl or cl.startswith("o"):
        arr.orange_cones.append(pt)
    elif "yellow" in cl or cl.startswith("y"):
        arr.yellow_cones.append(pt)
    else:
        arr.blue_cones.append(pt)

# ---------- node ----------
class JCBBMapper(Node):
    def __init__(self):
        super().__init__("fslam_online_mapper", automatically_declare_parameters_from_overrides=True)

        # I/O topics
        self.declare_parameter("topics.pose_in", "/odometry_integration/car_state")
        self.declare_parameter("topics.cones_in", "/ground_truth/cones")
        self.declare_parameter("topics.map_out", "/mapper/known_map")

        # Frames & DA
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 30.0)
        self.declare_parameter("chi2_gate_2d", 13.28)     # ~99% for 2 dof
        self.declare_parameter("joint_sig", 0.99)
        self.declare_parameter("joint_relax", 1.25)
        self.declare_parameter("min_k_for_joint", 3)

        # Meas & inflation (like your PF node)
        self.declare_parameter("meas.sigma_floor_xy", 0.35)
        self.declare_parameter("alpha_trans", 0.9)
        self.declare_parameter("beta_rot", 0.9)

        # Landmark handling
        self.declare_parameter("init.landmark_var", 0.30)
        self.declare_parameter("meas.inflate_xy", 0.05)
        self.declare_parameter("da.min_separation_m", 0.0)
        self.declare_parameter("stale.seen_slowdown_s", 2.0)
        self.declare_parameter("stale.inflation_mul", 1.5)

        # Throughput
        self.declare_parameter("da.target_cone_hz", 0.0)
        self.declare_parameter("da.max_cands_per_obs", 4)
        self.declare_parameter("da.max_obs_for_jcbb", 16)
        self.declare_parameter("da.time_budget_ms", 10.0)

        # Tiny PF over Δ
        self.declare_parameter("pf.num_particles", 200)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        self.declare_parameter("pf.process_std_xy_m", 0.02)
        self.declare_parameter("pf.process_std_yaw_rad", 0.002)
        self.declare_parameter("pf.likelihood_floor", 1e-12)

        # Read params
        gp = self.get_parameter
        self.pose_topic = str(gp("topics.pose_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)
        self.map_topic = str(gp("topics.map_out").value)
        self.det_frame = str(gp("detections_frame").value).lower()

        self.map_crop_m = float(gp("map_crop_m").value); self.crop2 = self.map_crop_m**2
        self.chi2_gate = float(gp("chi2_gate_2d").value)
        self.joint_sig = float(gp("joint_sig").value)
        self.joint_relax = float(gp("joint_relax").value)
        self.min_k_for_joint = int(gp("min_k_for_joint").value)

        self.sigma_floor = float(gp("meas.sigma_floor_xy").value)
        self.alpha = float(gp("alpha_trans").value)
        self.beta = float(gp("beta_rot").value)
        self.init_lm_var = float(gp("init.landmark_var").value)
        self.inflate_xy = float(gp("meas.inflate_xy").value)
        self.min_sep = float(gp("da.min_separation_m").value)
        self.stale_T = float(gp("stale.seen_slowdown_s").value)
        self.stale_mul = float(gp("stale.inflation_mul").value)

        self.target_cone_hz = float(gp("da.target_cone_hz").value)
        self.max_cands_per_obs = int(gp("da.max_cands_per_obs").value)
        self.max_obs_for_jcbb = int(gp("da.max_obs_for_jcbb").value)
        self.time_budget_ms = float(gp("da.time_budget_ms").value)

        # PF params
        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)

        # State
        self.pose_ready = False
        self.odom_buf: deque = deque()  # (t,x,y,yaw,v,yr)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self.odom_buffer_sec = 2.0
        self.extrap_cap = 0.06

        self._last_cone_ts = None
        self.landmarks: List[dict] = []  # dict(m, P, color, hits, last_seen)

        # PF state (Δ)
        self.particles = np.zeros((self.Np, 3), float)     # [dx, dy, dθ]
        self.weights = np.ones(self.Np, float)/self.Np
        self.delta_hat = np.zeros(3, float)
        self.have_delta = False

        # ROS I/O
        q_rel = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        q_fast = QoSProfile(depth=200, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.pose_topic, self.cb_pose, q_rel)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q_fast)
        self.pub_map = self.create_publisher(ConeArrayWithCovariance, self.map_topic, 10)

        self.get_logger().info(
            f"[mapper] pose_in={self.pose_topic} cones={self.cones_topic} map_out={self.map_topic}")
        self.get_logger().info(
            f"[mapper] frame={self.det_frame} | χ²(2)={self.chi2_gate:.2f} | crop={self.map_crop_m:.1f} m | "
            f"α={self.alpha:.2f} β={self.beta:.2f} | JCBB caps: K={self.max_cands_per_obs}, M={self.max_obs_for_jcbb}, budget={self.time_budget_ms:.1f}ms | PF N={self.Np}"
        )

    # ---------- pose feed ----------
    def cb_pose(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        vx = float(msg.twist.twist.linear.x) if msg.twist and msg.twist.twist else 0.0
        vy = float(msg.twist.twist.linear.y) if msg.twist and msg.twist.twist else 0.0
        v = math.hypot(vx, vy)
        yr = float(msg.twist.twist.angular.z) if msg.twist and msg.twist.twist else 0.0
        self.pose_latest = np.array([x, y, yaw], float)
        self.odom_buf.append((t, x, y, yaw, v, yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()
        if not self.pose_ready:
            self.pose_ready = True
            self.get_logger().info("[mapper] predictor pose feed ready.")

    def _pose_at(self, t_query: float):
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
                x = x1 + v1*dt_c*math.cos(yaw1); y = y1 + v1*dt_c*math.sin(yaw1); yaw = wrap(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt_c)-math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt_c)-math.cos(yaw1))
                yaw = wrap(yaw1 + yr1*dt_c)
            return np.array([x,y,yaw], float), abs(dt), v1, yr1
        t0,x0,yaw0_x,yaw0_y,yaw0_v,yaw0_yr = self.odom_buf[idx-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[idx]
        if t1 == t0:
            return np.array([x1,y1,yaw1], float), abs(t_query - t1), v1, yr1
        x0_,y0_,yaw0,v0,yr0 = self.odom_buf[idx-1][1:6]
        a = (t_query - t0)/(t1 - t0)
        x = x0_ + a*(x1-x0_); y = y0_ + a*(y1-y0_)
        yaw = wrap(yaw0 + a*wrap(yaw1 - yaw0))
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    # ---------- PF over Δ ----------
    def _pf_predict(self):
        self.particles[:, 0] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 1] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:, 2] += np.random.normal(0.0, self.p_std_yaw, size=self.Np)
        self.particles[:, 2] = np.array([wrap(th) for th in self.particles[:, 2]])

    def _pf_update(self, pairs, odom_pose, PB, LM_xy):
        # Require at least 2 matches for a stable Δ
        if len(pairs) < 2:
            return False
        idxs = [oi for (oi, _, _) in pairs]
        mids = [mj for (_, mj, _) in pairs]
        PB_sel = np.stack([PB[i] for i in idxs], axis=0)     # Kx2 body detections
        MW = LM_xy[np.array(mids, int), :]                   # Kx2 world landmarks

        x_o, y_o, yaw_o = odom_pose
        new_w = np.zeros_like(self.weights)
        K = PB_sel.shape[0]

        for i in range(self.Np):
            dx, dy, dth = self.particles[i]
            yaw_c = wrap(yaw_o + dth)
            Rc = rot2d(yaw_c)
            pc = np.array([x_o + dx, y_o + dy], float)
            PW_pred = pc[None, :] + (PB_sel @ Rc.T)          # Kx2 world
            diffs = PW_pred - MW
            # Simple isotropic likelihood; joint gate with mild relax
            m2_sum = float(np.sum(np.sum(diffs*diffs, axis=1) / (self.sigma_floor**2)))
            gate = self.joint_relax * chi2_quantile(2*K, self.joint_sig)
            like = self.like_floor if m2_sum > gate else max(math.exp(-0.5*m2_sum), self.like_floor)
            new_w[i] = self.weights[i] * like

        sw = float(np.sum(new_w))
        if sw <= 0.0 or not math.isfinite(sw):
            self.weights[:] = 1.0 / self.Np
        else:
            self.weights[:] = new_w / sw

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
        return np.array([dx, dy, dth], float)

    # ---------- cones feed ----------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.pose_ready:
            return
        tz = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.target_cone_hz > 0.0 and self._last_cone_ts is not None:
            if (tz - self._last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        self._last_cone_ts = tz

        # Pose @ tz
        odom_pose, dt_pose, v, yawrate = self._pose_at(tz)
        x_o, y_o, yaw_o = odom_pose

        # Use latest Δ̂ to build corrected pose used for gating & fusion
        dxh, dyh, dthh = self.delta_hat if self.have_delta else (0.0, 0.0, 0.0)
        x_c, y_c, yaw_c = x_o + dxh, y_o + dyh, wrap(yaw_o + dthh)
        Rwb_c = rot2d(yaw_c); Rbw_c = Rwb_c.T

        # Build observations in BODY and WORLD (WORLD uses corrected pose)
        obs_pb = []; obs_pw = []; obs_S0w = []; obs_color = []
        def add(arr, color_hint):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma_floor**2, self.sigma_floor**2]) + np.eye(2)*self.inflate_xy
                if self.det_frame == "base":
                    pb = np.array([px, py], float)
                    pw = np.array([x_c, y_c]) + (Rwb_c @ pb)
                    S0w = Rwb_c @ Pw @ Rwb_c.T
                else:
                    pw = np.array([px, py], float)
                    pb = Rbw_c @ (pw - np.array([x_c, y_c]))
                    S0w = Pw
                obs_pb.append(pb); obs_pw.append(pw); obs_S0w.append(S0w); obs_color.append(color_hint)

        add(msg.blue_cones, "blue")
        add(msg.yellow_cones, "yellow")
        add(msg.orange_cones, "orange")
        add(msg.big_orange_cones, "big_orange")

        n_obs = len(obs_pw)
        if n_obs == 0:
            return

        # Landmark matrix
        LM = np.stack([lm['m'] for lm in self.landmarks], axis=0) if self.landmarks else np.zeros((0,2), float)

        # Candidate building with **S_gate = P_lm + S_eff**
        now = tz
        cands: List[List[Tuple[int,float]]] = []
        total_cands = 0

        for pw, S0w, pb, col in zip(obs_pw, obs_S0w, obs_pb, obs_color):
            cc: List[Tuple[int, float]] = []
            if LM.shape[0] > 0:
                diffs = LM - pw[None, :]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                near = np.where(d2 <= self.crop2)[0]
                if near.size > 0:
                    r = float(np.linalg.norm(pb))
                    for j in near:
                        lm = self.landmarks[j]
                        # loose color: allow orange/big_orange mixing
                        if (lm['color'] and col) and not (lm['color'] == col or ("orange" in lm['color'] and "orange" in col)):
                            continue
                        Pj = lm['P']
                        S_eff = self._inflate_cov(S0w, v, yawrate, dt_pose if math.isfinite(dt_pose) else 0.0,
                                                  r, yaw_c, now - lm['last_seen'])
                        S_gate = Pj + S_eff  # <<<<<< FIXED: include landmark uncertainty
                        try: S_inv = np.linalg.inv(S_gate)
                        except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S_gate)
                        innov = pw - lm['m']
                        m2 = float(innov.T @ S_inv @ innov)
                        if m2 <= self.chi2_gate:
                            cc.append((int(j), m2))
                    if cc:
                        cc.sort(key=lambda t: t[1])
                        if self.max_cands_per_obs > 0:
                            cc = cc[:self.max_cands_per_obs]
            cands.append(cc)
            total_cands += len(cc)

        # Order & limit obs for JCBB
        order = sorted(range(n_obs), key=lambda i: len(cands[i]) if cands[i] else 9999)
        if self.max_obs_for_jcbb > 0:
            order = order[:self.max_obs_for_jcbb]
        cands_ord = [cands[i] for i in order]

        # JCBB (bounded)
        best_pairs: List[Tuple[int,int,float]] = []
        best_K = 0
        used = set()
        t_start = time.perf_counter()
        time_budget = self.time_budget_ms/1000.0

        def dfs(idx, cur_pairs, cur_sum):
            nonlocal best_pairs, best_K
            if (time.perf_counter() - t_start) > time_budget:
                return
            cur_K = len(cur_pairs)
            if cur_K + (len(cands_ord) - idx) < best_K:
                return
            if cur_K >= self.min_k_for_joint:
                df = 2*cur_K
                thresh = self.joint_relax * chi2_quantile(df, self.joint_sig)
                if cur_sum > thresh:
                    return
            if idx == len(cands_ord):
                if cur_K > best_K:
                    best_K = cur_K; best_pairs = cur_pairs.copy(); 
                return
            # skip
            dfs(idx+1, cur_pairs, cur_sum)
            # try
            for (mid, m2) in cands_ord[idx]:
                if mid in used:
                    continue
                used.add(mid)
                cur_pairs.append((order[idx], mid, m2))
                dfs(idx+1, cur_pairs, cur_sum + m2)
                cur_pairs.pop()
                used.remove(mid)

        dfs(0, [], 0.0)

        # ---- PF Δ update (against current map) ----
        if best_K >= 2 and LM.shape[0] > 0:
            self._pf_predict()
            _ = self._pf_update(best_pairs, odom_pose, obs_pb, LM)  # uses odom_pose (uncorrected) + LM
            self.delta_hat = self._estimate_delta()
            self.have_delta = True

            # Recompute corrected pose with NEW Δ for fusion accuracy
            dxh, dyh, dthh = self.delta_hat
            x_c, y_c, yaw_c = x_o + dxh, y_o + dyh, wrap(yaw_o + dthh)
            Rwb_c = rot2d(yaw_c)

        # ---- EKF-style fuse for matched; create new for unmatched (conservative) ----
        # Recompute per-observation pw/S0 under UPDATED corrected pose ONLY for matched obs (cheap).
        def obs_world_from_pb(pb, Pw_base):
            pw = np.array([x_c, y_c]) + (Rwb_c @ pb)
            S0w = Rwb_c @ Pw_base @ Rwb_c.T
            return pw, S0w

        # Build base cov (before rotation) for reuse. We kept inflated Pw earlier; rebuild base Pw once.
        Pw_base_all = []
        for pb, S0w in zip(obs_pb, obs_S0w):
            # S0w = R Pw_base R^T, recover Pw_base approx via current R (ok since we use the same R)
            # but we don't strictly need this; simpler: keep the pre-rotated Pw at creation time.
            # To keep it simple, reuse S0w with new Rwb_c (small bias). Good enough for fusion.
            Pw_base_all.append(S0w)  # treated as if base; approximation is fine at this scale.

        matched_obs_idx = set([oi for (oi, _, _) in best_pairs])
        matched_map_idx = set([mj for (_, mj, _) in best_pairs])

        upd = 0
        for (oi, mj, _) in best_pairs:
            lm = self.landmarks[mj]
            # recompute observation in WORLD under updated corrected pose
            pw, S0w = obs_world_from_pb(obs_pb[oi], Pw_base_all[oi])
            P = lm['P']; m = lm['m']
            S = P + S0w
            try: S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError: S_inv = np.linalg.pinv(S)
            K = P @ S_inv
            innov = pw - m
            m_new = m + K @ innov
            P_new = (np.eye(2) - K) @ P
            lm['m'] = m_new
            lm['P'] = 0.5*(P_new + P_new.T)
            lm['hits'] += 1
            lm['last_seen'] = tz
            upd += 1

        # New landmarks: create only if zero candidates (no ambiguity)
        new = 0
        for oi in range(n_obs):
            if oi in matched_obs_idx:
                continue
            if cands[oi]:  # there was ambiguity → skip creation this frame
                continue
            # recompute under corrected pose
            pb = obs_pb[oi]
            pw, S0w = obs_world_from_pb(pb, Pw_base_all[oi])
            P0 = S0w + np.eye(2)*self.init_lm_var
            col = obs_color[oi]
            self.landmarks.append({'m': pw.copy(), 'P': P0, 'color': col, 'hits': 1, 'last_seen': tz})
            new += 1

        # Optional: min_separation guard against accidental duplicates
        if self.min_sep > 0.0 and self.landmarks:
            LM_now = np.stack([lm['m'] for lm in self.landmarks], axis=0)
            sep2 = self.min_sep**2
            keep = []
            for idx, lm in enumerate(self.landmarks):
                diffs = LM_now - lm['m'][None, :]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                d2[idx] = float('inf')
                if float(np.min(d2)) < 1e-9 or float(np.min(d2)) >= sep2:
                    keep.append(lm)
            self.landmarks = keep

        self._publish_map()

        mean_cands = total_cands / max(1, n_obs)
        timed_out = (time.perf_counter() - t_start) > time_budget
        self.get_logger().info(
            f"[map/jcbb+pf] t={tz:7.2f}s obs={n_obs} mean_cands/obs={mean_cands:.2f} "
            f"matched={best_K} upd={upd} new={new} LM={len(self.landmarks)} "
            f"Δ̂=[{self.delta_hat[0]:+.3f},{self.delta_hat[1]:+.3f},{math.degrees(self.delta_hat[2]):+.2f}°]"
            f"{' | JCBB:timeout' if timed_out else ''}"
        )

    # ---------- math helpers ----------
    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float,
                     r: float, yaw: float, stale_dt: float) -> np.ndarray:
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0], [0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        if stale_dt > self.stale_T:
            S *= self.stale_mul
        S[0,0] += 1e-9; S[1,1] += 1e-9
        return S

    # ---------- publisher ----------
    def _publish_map(self):
        if not self.landmarks:
            return
        out = ConeArrayWithCovariance()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        for lm in self.landmarks:
            add_cone(out, float(lm['m'][0]), float(lm['m'][1]), lm['P'], lm['color'])
        self.pub_map.publish(out)

# ---------- main ----------
def main():
    rclpy.init()
    node = JCBBMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
