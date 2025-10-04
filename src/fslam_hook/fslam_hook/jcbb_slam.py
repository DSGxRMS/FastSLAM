#!/usr/bin/env python3
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

BASE_DIR = Path(__file__).resolve().parents[0]

# ---- Chi-square helpers ----
# Wilson–Hilferty approximation for chi2 quantile: good enough for gating
def _norm_ppf(p: float) -> float:
    # Acklam rational approximation for inverse normal CDF
    # max error ~4.5e-4, fine for gating
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
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p-0.5; r = q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    # Wilson–Hilferty: X ≈ k*(1 - 2/(9k) + z*sqrt(2/(9k)))^3, z=Φ^{-1}(p)
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)

def rot2d(th):
    c,s = math.cos(th), math.sin(th)
    return np.array([[c,-s],[s,c]], float)

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x, y, z, w):
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

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
        except Exception: continue
        rows.append((x,y,c))
    return rows

class JCBBNode(Node):
    def __init__(self):
        super().__init__("fslam_jcbb")

        # params
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("map_crop_m", 28.0)
        self.declare_parameter("chi2_gate_2d", 11.83)      # per-pair gate (≈99.7% for 2 dof)
        self.declare_parameter("joint_sig", 0.95)          # joint set significance (tunable)
        self.declare_parameter("joint_relax", 1.4)        # extra slack on joint gate (to absorb shared pose error)
        self.declare_parameter("min_k_for_joint", 3)       # delay joint gate until k>=this
        self.declare_parameter("meas_sigma_floor_xy", 0.30)
        self.declare_parameter("alpha_trans", 0.9)
        self.declare_parameter("beta_rot", 0.9)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 60.0)
        self.declare_parameter("target_cone_hz", 25.0)
        self.declare_parameter("min_candidates", 0)
        self.declare_parameter("map_data_dir", "slam_utils/data")
        self.declare_parameter("max_pairs_print", 6)
        

        # topics
        self.declare_parameter("pose_topic", "/odometry_integration/car_state")
        self.declare_parameter("cones_topic", "/ground_truth/cones")

        p = self.get_parameter
        self.detections_frame = str(p("detections_frame").value).lower()
        self.map_crop_m = float(p("map_crop_m").value)
        self.chi2_gate = float(p("chi2_gate_2d").value)
        self.joint_sig  = float(p("joint_sig").value)
        self.joint_relax = float(p("joint_relax").value)
        self.min_k_for_joint = int(p("min_k_for_joint").value)
        self.sigma0 = float(p("meas_sigma_floor_xy").value)
        self.alpha = float(p("alpha_trans").value)
        self.beta  = float(p("beta_rot").value)
        self.odom_buffer_sec = float(p("odom_buffer_sec").value)
        self.extrap_cap = float(p("extrapolation_cap_ms").value) / 1000.0
        self.target_cone_hz = float(p("target_cone_hz").value)
        self.min_candidates = int(p("min_candidates").value)
        self.max_pairs_print = int(p("max_pairs_print").value)

        self.pose_topic = str(p("pose_topic").value)
        self.cones_topic = str(p("cones_topic").value)

        data_dir = (Path(__file__).resolve().parents[0] / str(p("map_data_dir").value)).resolve()
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
        self.MAP = (M @ R_map.T) + t_map

        # state
        self.odom_buf: deque = deque()  # (t,x,y,yaw,v,yawrate)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self._pose_feed_ready = False

        # logging throttle (every 0.5 s)
        self._log_period = 0.5
        self._last_log_wall = 0.0

        q = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.pose_topic, self.cb_odom, q)
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cb_cones, q)

        self._last_proc_wall = 0.0
        self._proc_period = 1.0 / max(1e-3, self.target_cone_hz)
        self._last_t = None
        self.frame_idx = 0

        self.get_logger().info(f"JCBB up. Map cones: {self.MAP.shape[0]} | pose={self.pose_topic} | cones={self.cones_topic}")

    # ------------- ROS -------------
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
            yr = wrap(yaw - yaw0) / dt
        else:
            self._pose_feed_ready = True
            self.get_logger().info("[JCBB] Pose feed ready (first fused odometry received).")
        self.pose_latest = np.array([x,y,yaw], float)
        self.odom_buf.append((t,x,y,yaw,v,yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

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
                yaw = wrap(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt_c) - math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt_c) - math.cos(yaw1))
                yaw = wrap(yaw1 + yr1*dt_c)
            return np.array([x,y,yaw], float), abs(dt), v1, yr1
        t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[idx-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[idx]
        if t1==t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a*dyaw)
        v = (1-a)*v0 + a*v1; yr = (1-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    def cb_cones(self, msg: ConeArrayWithCovariance):
        now_wall = time.time()
        if now_wall - self._last_proc_wall < self._proc_period:
            return
        self._last_proc_wall = now_wall

        t0 = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose, v, yawrate = self._pose_at(t_z)
        dt_pose_eff = dt_pose if math.isfinite(dt_pose) else 0.0

        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T

        # Build obs in WORLD
        obs_pw = []; obs_S0 = []; obs_pb = []
        def add(arr):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float)
                Pw = Pw + np.diag([self.sigma0**2, self.sigma0**2])
                if self.detections_frame == "base":
                    pb = np.array([px,py], float)
                    pw = np.array([x,y]) + Rwb @ pb
                    S0 = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y]))
                    S0 = Pw
                obs_pw.append(pw); obs_S0.append(S0); obs_pb.append(pb)
        add(msg.blue_cones); add(msg.yellow_cones); add(msg.orange_cones); add(msg.big_orange_cones)
        n_obs = len(obs_pw)
        if n_obs==0:
            return

        # Candidates
        crop2 = self.map_crop_m**2
        MAP = self.MAP
        cands: List[List[Tuple[int,float]]] = []
        total_cands=0
        for pw, S0, pb in zip(obs_pw, obs_S0, obs_pb):
            r = float(np.linalg.norm(pb))
            S_eff = self._inflate_cov(S0, v, yawrate, dt_pose_eff, r, yaw)
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
            if not cc and self.min_candidates>0 and near.size>0:
                k = min(self.min_candidates, near.size)
                by = sorted([(int(mid), d2[mid]) for mid in near], key=lambda t:t[1])[:k]
                cc = [(mid, self.chi2_gate*1.01) for (mid,_) in by]
            cc.sort(key=lambda t:t[1])
            cands.append(cc)
            total_cands += len(cc)

        if total_cands==0:
            self._print_line(t_z, n_obs, 0, int((time.perf_counter()-t0)*1000))
            self.frame_idx += 1
            return

        # Reorder obs by ascending candidate count
        order = sorted(range(n_obs), key=lambda i: len(cands[i]) if cands[i] else 9999)
        cands_ord = [cands[i] for i in order]

        # JCBB search (with delayed & relaxed joint gate)
        best_pairs: List[Tuple[int,int,float]] = []
        best_K = 0
        used = set()

        def dfs(idx, cur_pairs, cur_sum):
            nonlocal best_pairs, best_K
            cur_K = len(cur_pairs)

            # bound on max possible matches
            if cur_K + (len(cands_ord)-idx) < best_K:
                return

            # delayed joint gate (k >= min_k_for_joint)
            if cur_K >= self.min_k_for_joint:
                df = 2*cur_K
                thresh = self.joint_relax * chi2_quantile(df, self.joint_sig)
                if cur_sum > thresh:
                    return

            if idx == len(cands_ord):
                if cur_K > best_K:
                    best_K = cur_K
                    best_pairs = cur_pairs.copy()
                return

            # Option: skip this obs
            dfs(idx+1, cur_pairs, cur_sum)

            # Try candidates
            for (mid, m2) in cands_ord[idx]:
                if mid in used:
                    continue
                used.add(mid)
                cur_pairs.append((order[idx], mid, m2))
                dfs(idx+1, cur_pairs, cur_sum + m2)
                cur_pairs.pop()
                used.remove(mid)

        dfs(0, [], 0.0)

        proc_ms = int((time.perf_counter()-t0)*1000)
        self._print_line(t_z, n_obs, best_K, proc_ms)
        self.frame_idx += 1

    # ---- helpers ----
    def _inflate_cov(self, S0_w: np.ndarray, v: float, yawrate: float, dt: float, r: float, yaw: float) -> np.ndarray:
        trans_var = self.alpha * (v**2) * (dt**2)
        lateral_var = self.beta * (yawrate**2) * (dt**2) * (r**2)
        R = rot2d(yaw)
        Jrot = R @ np.array([[0.0, 0.0],[0.0, lateral_var]]) @ R.T
        S = S0_w + np.diag([trans_var, trans_var]) + Jrot
        S[0,0]+=1e-9; S[1,1]+=1e-9
        return S

    # ---- minimal, throttled logger (every 0.5s) ----
    def _print_line(self, t_z: float, n_obs: int, n_match: int, proc_ms: int):
        now = time.time()
        if (now - self._last_log_wall) < self._log_period:
            return
        self._last_log_wall = now

        pass_dt = 0.0 if self._last_t is None else (t_z - self._last_t)
        self._last_t = t_z
        pass_dt_ms_str = f"{int(pass_dt*1000)}ms" if math.isfinite(pass_dt) else "inf"
        print(f"[jcbb] obs={n_obs} matched={n_match} pass_dt={pass_dt_ms_str} proc={proc_ms}ms")

def main():
    rclpy.init()
    node = JCBBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
