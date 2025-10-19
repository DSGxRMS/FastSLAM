#!/usr/bin/env python3
# cone_mapper_pf.py  (GT-only accurate mapper, persistent map by default)
#
# Global cone mapping with dynamic gating + JCBB + birth/confirm buffer + range-weighted EMA.
# Drift is assumed ~0 because we use /ground_truth/odom; PF scaffold kept behind a switch for later.
# Map is PERSISTENT by default (no aging). Set aging.enable:=true to age out stale cones.
#
# Subscribes (BEST_EFFORT):
#   /ground_truth/odom   (nav_msgs/Odometry)
#   /ground_truth/cones  (eufs_msgs/ConeArrayWithCovariance)
#
# Publishes:
#   /online_map/cones    (eufs_msgs/ConeArrayWithCovariance)
#   /online_map/markers  (visualization_msgs/MarkerArray)
#
# Params (defaults tuned for GT):
#   topics.odom:            /ground_truth/odom
#   topics.cones:           /ground_truth/cones
#   topics.map_out:         /online_map/cones
#   topics.markers_out:     /online_map/markers
#   detections_frame:       "base" | "world"             (default "base")
#   target_cone_hz:         25.0                         (Hz)
#   chi2_gate_2d:           11.83                        (~99% for 2 dof)
#   joint_sig:              0.95
#   joint_relax:            1.35
#   min_k_for_joint:        2
#   meas_sigma_floor_xy:    0.22                         (m)
#   map_crop_m:             25.0                         (m) pre-crop radius
#   ema_alpha:              0.22                         (base α; scaled by range)
#   ema_r0_m:               8.0
#   ema_alpha_min:          0.12
#   ema_alpha_max:          0.35
#   buffer.promote_count:   2                            (sightings)
#   buffer.promote_window_s:1.0
#   aging.enable:           false                        (persistent map default)
#   aging.max_age_s:        8.0                          (only used if aging.enable=true)
#   adapt.alpha_trans:      0.9
#   adapt.beta_rot:         0.9
#   extrapolation_cap_ms:   100.0                        (pose align)
#   quiet:                  true
#
# PF scaffold (off by default; for future drift work):
#   use_pf:                 false
#   pf.num_particles:       600
#   pf.resample_neff_ratio: 0.5
#   pf.process_std_xy_m:    0.008
#   pf.process_std_yaw_rad: 0.0018
#   pf.likelihood_floor:    1e-12
#
import math, bisect
from typing import List, Tuple, Dict, Any
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion


# ---------- math helpers ----------
def wrap(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

def yaw_from_quat(x,y,z,w) -> float:
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion(); q.x = 0.0; q.y = 0.0
    q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0)
    return q

# ---- normal/chi2 quantiles (same approximation style) ----
def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return  float('inf')
    a1=-3.969683028665376e+01; a2=2.209460984245205e+02; a3=-2.759285104469687e+02
    a4=1.383577518672690e+02; a5=-3.066479806614716e+01; a6=2.506628277459239e+00
    b1=-5.447609879822406e+01; b2=1.615858368580409e+02; b3=-1.556989798598866e+02
    b4=6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5=4.374664141464968e+00; c6=2.938163982698783e+00
    d1=7.784695709041462e-03; d2=3.224671290700398e-01; d3=2.445134137142996e+00
    d4=3.754408661907416e+00
    plow=0.02425; phigh=1-plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p-0.5; r = q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0/(9.0*k) + z*math.sqrt(2.0/(9.0*k))
    return k * (t**3)


# ---------- node ----------
class ConeMapperPF(Node):
    def __init__(self):
        super().__init__("cone_mapper_pf", automatically_declare_parameters_from_overrides=True)

        # ---- params ----
        self.declare_parameter("topics.odom", "/ground_truth/odom")
        self.declare_parameter("topics.cones", "/ground_truth/cones")
        self.declare_parameter("topics.map_out", "/online_map/cones")
        self.declare_parameter("topics.markers_out", "/online_map/markers")

        self.declare_parameter("detections_frame", "base")  # "base" or "world"
        self.declare_parameter("target_cone_hz", 25.0)

        self.declare_parameter("chi2_gate_2d", 11.83)
        self.declare_parameter("joint_sig", 0.95)
        self.declare_parameter("joint_relax", 1.35)
        self.declare_parameter("min_k_for_joint", 2)

        self.declare_parameter("meas_sigma_floor_xy", 0.22)
        self.declare_parameter("map_crop_m", 25.0)

        # Range-weighted EMA
        self.declare_parameter("ema_alpha", 0.22)
        self.declare_parameter("ema_r0_m", 8.0)
        self.declare_parameter("ema_alpha_min", 0.12)
        self.declare_parameter("ema_alpha_max", 0.35)

        # Birth/confirm buffer + aging (aging disabled by default => persistent map)
        self.declare_parameter("buffer.promote_count", 2)
        self.declare_parameter("buffer.promote_window_s", 1.0)
        self.declare_parameter("aging.enable", False)
        self.declare_parameter("aging.max_age_s", 8.0)  # only used if aging.enable=true

        # Dynamic gate inflator
        self.declare_parameter("adapt.alpha_trans", 0.9)
        self.declare_parameter("adapt.beta_rot", 0.9)
        self.declare_parameter("extrapolation_cap_ms", 100.0)

        # PF scaffold (kept off for GT)
        self.declare_parameter("use_pf", False)
        self.declare_parameter("pf.num_particles", 600)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        self.declare_parameter("pf.process_std_xy_m", 0.008)
        self.declare_parameter("pf.process_std_yaw_rad", 0.0018)
        self.declare_parameter("pf.likelihood_floor", 1e-12)

        self.declare_parameter("quiet", True)

        gp = self.get_parameter
        self.topic_odom = str(gp("topics.odom").value)
        self.topic_cones = str(gp("topics.cones").value)
        self.topic_map_out = str(gp("topics.map_out").value)
        self.topic_markers_out = str(gp("topics.markers_out").value)

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.target_cone_hz = float(gp("target_cone_hz").value)

        self.chi2_gate = float(gp("chi2_gate_2d").value)
        self.joint_sig = float(gp("joint_sig").value)
        self.joint_relax = float(gp("joint_relax").value)
        self.min_k_for_joint = int(gp("min_k_for_joint").value)

        self.sigma0 = float(gp("meas_sigma_floor_xy").value)
        self.crop2 = float(gp("map_crop_m").value) ** 2

        self.ema_alpha0 = float(gp("ema_alpha").value)
        self.ema_r0 = float(gp("ema_r0_m").value)
        self.ema_alpha_min = float(gp("ema_alpha_min").value)
        self.ema_alpha_max = float(gp("ema_alpha_max").value)

        self.promote_count = int(gp("buffer.promote_count").value)
        self.promote_window = float(gp("buffer.promote_window_s").value)
        self.aging_enable = bool(gp("aging.enable").value)
        self.max_age = float(gp("aging.max_age_s").value)

        self.alpha_trans = float(gp("adapt.alpha_trans").value)
        self.beta_rot = float(gp("adapt.beta_rot").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0

        # PF scaffold
        self.use_pf = bool(gp("use_pf").value)
        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)

        self.quiet = bool(gp("quiet").value)

        # ---- map state per color ----
        self.map_blue = {"mu": np.zeros((0,2), float), "seen": np.zeros((0,), int), "t": np.zeros((0,), float)}
        self.map_yel  = {"mu": np.zeros((0,2), float), "seen": np.zeros((0,), int), "t": np.zeros((0,), float)}
        self.map_org  = {"mu": np.zeros((0,2), float), "seen": np.zeros((0,), int), "t": np.zeros((0,), float)}
        self.map_big  = {"mu": np.zeros((0,2), float), "seen": np.zeros((0,), int), "t": np.zeros((0,), float)}

        # tentative buffers
        self.tentative: Dict[str, list] = {"blue": [], "yellow": [], "orange": [], "big": []}
        # entries: (xy(np.array shape(2,)), first_ts, last_ts, count)

        # ---- odom buffer ----
        self.odom_buf: List[Tuple[float,float,float,float,float,float]] = []  # (t,x,y,yaw,v,yr)
        self.pose_latest = np.array([0.0,0.0,0.0], float)
        self.pose_ready = False

        # PF skeleton
        self.particles = np.zeros((self.Np, 3), float)
        self.weights = np.ones((self.Np,), float) / max(1, self.Np)
        self.delta_hat = np.zeros(3, float)

        # ---- QoS: BEST_EFFORT ----
        qos_best = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=200,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, self.topic_odom, self.cb_odom, qos_best)
        self.create_subscription(ConeArrayWithCovariance, self.topic_cones, self.cb_cones, qos_best)

        from eufs_msgs.msg import ConeArrayWithCovariance as _CAC
        self.pub_map = self.create_publisher(_CAC, self.topic_map_out, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.topic_markers_out, 10)

        # publish heartbeat (even if no new cones)
        self.create_timer(1.0 / 25.0, self.publish_map_and_markers)

        if not self.quiet:
            self.get_logger().info(
                f"[mapper] GT mode persistent map (aging={'on' if self.aging_enable else 'off'})"
            )

    # ------------- odom -------------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x); y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x,q.y,q.z,q.w)
        vx = float(msg.twist.twist.linear.x); vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy); yr = float(msg.twist.twist.angular.z)
        self.pose_latest[:] = [x,y,yaw]
        self.odom_buf.append((t,x,y,yaw,v,yr))
        # keep ~2s
        tmin = t - 2.0
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.pop(0)
        self.pose_ready = True

    def pose_at(self, t_query: float):
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf'), 0.0, 0.0
        times = [it[0] for it in self.odom_buf]
        i = bisect.bisect_left(times, t_query)
        if i == 0:
            t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[0]
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        if i >= len(self.odom_buf):
            t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[-1]
            dt = max(0.0, min(t_query - t1, 0.100))  # cap 100ms
            if abs(yr1) < 1e-3:
                x = x1 + v1*dt*math.cos(yaw1); y = y1 + v1*dt*math.sin(yaw1); yaw = wrap(yaw1)
            else:
                x = x1 + (v1/yr1)*(math.sin(yaw1+yr1*dt)-math.sin(yaw1))
                y = y1 - (v1/yr1)*(math.cos(yaw1+yr1*dt)-math.cos(yaw1))
                yaw = wrap(yaw1+yr1*dt)
            return np.array([x,y,yaw], float), abs(t_query - t1), v1, yr1
        t0,x0,y0,yaw0,v0,yr0 = self.odom_buf[i-1]
        t1,x1,y1,yaw1,v1,yr1 = self.odom_buf[i]
        if t1 == t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0), v0, yr0
        a = (t_query - t0)/(t1 - t0)
        x = x0 + a*(x1-x0); y = y0 + a*(y1-y0)
        dyaw = wrap(yaw1 - yaw0); yaw = wrap(yaw0 + a*dyaw)
        v  = (1.0-a)*v0  + a*v1
        yr = (1.0-a)*yr0 + a*yr1
        dt_near = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_near, v, yr

    # ------------- dynamic gate inflator (scalar variance add) -------------
    def inflate_sigma2(self, v: float, yawrate: float, dt: float, r: float) -> float:
        trans_var   = 0.9 * (v ** 2) * (dt ** 2)     # tuned defaults
        lateral_var = 0.9 * (yawrate ** 2) * (dt ** 2) * (r ** 2)
        return (self.sigma0 ** 2) + trans_var + lateral_var

    # ------------- cones callback -------------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.pose_ready:
            return
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if hasattr(self, "last_cone_ts") and self.last_cone_ts is not None:
            if (t_z - self.last_cone_ts) < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        self.last_cone_ts = t_z

        (xw,yw,yaw), dt_pose, v_local, yr_local = self.pose_at(t_z)
        Rwb = rot2d(yaw)

        # parse detections into WORLD coords
        banks = [
            ("blue",   msg.blue_cones),
            ("yellow", msg.yellow_cones),
            ("orange", msg.orange_cones),
            ("big",    msg.big_orange_cones),
        ]
        obs_by_bank = {"blue":[], "yellow":[], "orange":[], "big":[]}

        def to_world(px,py):
            if self.detections_frame == "base":
                return (np.array([xw,yw]) + Rwb @ np.array([px,py], float))
            else:
                return np.array([px,py], float)

        for name, arr in banks:
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                obs_by_bank[name].append(to_world(px,py))

        # seed if first time
        if (self.map_blue["mu"].shape[0]==0 and self.map_yel["mu"].shape[0]==0
            and self.map_org["mu"].shape[0]==0 and self.map_big["mu"].shape[0]==0):
            for name in ["blue","yellow","orange","big"]:
                obs = obs_by_bank[name]
                if obs:
                    mu = np.stack(obs, axis=0)
                    N = mu.shape[0]
                    m = self._bank(name)
                    m["mu"] = mu
                    m["seen"] = np.zeros((N,), int)
                    m["t"] = np.full((N,), t_z, float)
            self.publish_map_and_markers()
            return

        # associate & update each bank
        for name in ["blue","yellow","orange","big"]:
            obs = obs_by_bank[name]
            if not obs:
                continue
            self._associate_update_bank(
                name,
                np.stack(obs, axis=0),
                xw, yw, yaw,
                dt_pose, v_local, yr_local
            )

        self.publish_map_and_markers()

    # ------------- association & update (per bank) -------------
    def _bank(self, name: str) -> Dict[str, Any]:
        return {"blue": self.map_blue, "yellow": self.map_yel, "orange": self.map_org, "big": self.map_big}[name]

    def _associate_update_bank(self, name: str, OBS_W: np.ndarray,
                               xw: float, yw: float, yaw: float,
                               dt_pose: float, v_local: float, yr_local: float):
        x_c, y_c = xw, yw  # GT pose (drift ~0)
        bank = self._bank(name)
        MU = bank["mu"]
        Nmap = MU.shape[0]

        cands: List[List[Tuple[int,float,float]]] = []  # (mid, m2, eff_sigma2) per obs
        crop2 = self.crop2
        pose_xy = np.array([x_c, y_c])

        for pw in OBS_W:
            if Nmap == 0:
                cands.append([])
                continue
            d2 = np.einsum('ij,ij->i', (MU - pw), (MU - pw))
            near = np.where(d2 <= crop2)[0]
            cc=[]
            r = float(np.linalg.norm(pw - pose_xy))
            eff_sigma2 = self.inflate_sigma2(v_local, yr_local, dt_pose, r)
            if near.size>0:
                for mid in near:
                    innov = pw - MU[mid]
                    m2 = float((innov[0]*innov[0] + innov[1]*innov[1]) / eff_sigma2)
                    if m2 <= self.chi2_gate:
                        cc.append((int(mid), m2, eff_sigma2))
                if not cc:
                    # soft fallback: nearest in crop
                    best = int(near[np.argmin(d2[near])])
                    cc = [(best, self.chi2_gate*1.01, eff_sigma2)]
                cc.sort(key=lambda t: t[1])
            cands.append(cc)

        # JCBB
        order = sorted(range(OBS_W.shape[0]), key=lambda i: len(cands[i]) if cands[i] else 9999)
        cands_ord = [cands[i] for i in order]

        best_pairs: List[Tuple[int,int,float,float]] = []  # (obs_i, map_j, m2, eff_sigma2)
        best_K = 0
        used = set()
        def dfs(idx, cur_pairs, cur_sum):
            nonlocal best_pairs, best_K
            cur_K = len(cur_pairs)
            if cur_K + (len(cands_ord) - idx) < best_K:
                return
            if cur_K >= self.min_k_for_joint:
                df = 2*cur_K
                if cur_sum > self.joint_relax * chi2_quantile(df, self.joint_sig):
                    return
            if idx == len(cands_ord):
                if cur_K > best_K:
                    best_K = cur_K
                    best_pairs = cur_pairs.copy()
                return
            dfs(idx+1, cur_pairs, cur_sum)  # skip
            for (mid, m2, effs2) in cands_ord[idx]:
                if mid in used: continue
                used.add(mid); cur_pairs.append((order[idx], mid, m2, effs2))
                dfs(idx+1, cur_pairs, cur_sum + m2)
                cur_pairs.pop(); used.remove(mid)
        dfs(0, [], 0.0)

        # Update matched landmarks with RANGE-WEIGHTED EMA
        matched_obs = set()
        if best_pairs:
            for (pi, mi, _m2, _effs2) in best_pairs:
                z = OBS_W[pi]
                m = MU[mi]
                r = float(np.linalg.norm(z - np.array([x_c, y_c])))
                # α(r): nearer -> larger α
                alpha = max(self.ema_alpha_min,
                            min(self.ema_alpha_max,
                                self.ema_alpha0 * (self.ema_r0 / max(r, self.ema_r0))))
                MU[mi] = (1.0 - alpha)*m + alpha*z
                bank["seen"][mi] += 1
                bank["t"][mi] = self.last_cone_ts
                matched_obs.add(pi)

        # Unmatched observations -> tentative buffer (birth/confirm)
        now = self.last_cone_ts
        new_idx = [i for i in range(OBS_W.shape[0]) if i not in matched_obs]
        if new_idx:
            T = self.tentative[name]
            for i in new_idx:
                z = OBS_W[i]
                merged = False
                for k,(p, t0, tlast, cnt) in enumerate(T):
                    if np.linalg.norm(z - p) < 1.0:  # merge radius
                        p_new = 0.5*p + 0.5*z
                        T[k] = (p_new, t0, now, cnt+1)
                        merged = True
                        break
                if not merged:
                    T.append((z, now, now, 1))

            # promote & keep tentatives fresh
            keep = []
            for (p, t0, tlast, cnt) in T:
                if (now - t0) <= self.promote_window and cnt >= self.promote_count:
                    # promote to map
                    if MU.shape[0] == 0:
                        bank["mu"] = p.reshape(1,2)
                        bank["seen"] = np.array([1], int)
                        bank["t"] = np.array([now], float)
                        MU = bank["mu"]
                    else:
                        bank["mu"]  = np.vstack([bank["mu"], p.reshape(1,2)])
                        bank["seen"]= np.concatenate([bank["seen"], np.array([1], int)])
                        bank["t"]   = np.concatenate([bank["t"],  np.array([now], float)])
                        MU = bank["mu"]
                else:
                    # keep tentative if still within window
                    if (now - tlast) < self.promote_window:
                        keep.append((p, t0, tlast, cnt))
            self.tentative[name] = keep

        # OPTIONAL aging (disabled by default). If you want permanent map, leave aging.enable=false.
        if self.aging_enable and MU.shape[0] > 0:
            alive = (now - bank["t"]) < self.max_age
            if not np.all(alive):
                bank["mu"]   = MU[alive]
                bank["seen"] = bank["seen"][alive]
                bank["t"]    = bank["t"][alive]

    # ------------- publishers -------------
    def publish_map_and_markers(self):
        # map (as ConeArrayWithCovariance)
        msg = ConeArrayWithCovariance()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        def bank_to_list(mu):
            from eufs_msgs.msg import ConeWithCovariance
            out=[]
            for i in range(mu.shape[0]):
                c = ConeWithCovariance()
                c.point.x = float(mu[i,0]); c.point.y = float(mu[i,1]); c.point.z = 0.0
                s2 = self.sigma0**2
                c.covariance = [s2,0.0,0.0,s2]
                out.append(c)
            return out

        msg.blue_cones       = bank_to_list(self.map_blue["mu"])
        msg.yellow_cones     = bank_to_list(self.map_yel["mu"])
        msg.orange_cones     = bank_to_list(self.map_org["mu"])
        msg.big_orange_cones = bank_to_list(self.map_big["mu"])
        self.pub_map.publish(msg)

        # markers (RViz) — persistent lists; lifetime = 0 (forever) until updated
        ma = MarkerArray()
        def add_marker(ns, mu, rgba, mid):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = msg.header.stamp
            m.ns = ns
            m.id = mid
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.25; m.scale.y = 0.25; m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()  # persist
            from geometry_msgs.msg import Point
            m.points = []
            for i in range(mu.shape[0]):
                p = Point(); p.x = float(mu[i,0]); p.y = float(mu[i,1]); p.z = 0.0
                m.points.append(p)
            ma.markers.append(m)

        add_marker("blue",   self.map_blue["mu"],   (0.1,0.4,1.0,1.0), 1)
        add_marker("yellow", self.map_yel["mu"],    (1.0,0.9,0.1,1.0), 2)
        add_marker("orange", self.map_org["mu"],    (1.0,0.55,0.1,1.0),3)
        add_marker("big",    self.map_big["mu"],    (1.0,0.1,0.1,1.0), 4)
        self.pub_markers.publish(ma)

    # ---------- (PF scaffold; not used in GT mode) ----------
    def pf_predict(self):
        if not self.use_pf: return
        self.particles[:,0] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:,1] += np.random.normal(0.0, self.p_std_xy, size=self.Np)
        self.particles[:,2] += np.random.normal(0.0, self.p_std_yaw, size=self.Np)
        self.particles[:,2] = np.array([wrap(th) for th in self.particles[:,2]])

    def estimate_delta(self):
        if not self.use_pf: return np.zeros(3, float)
        w = self.weights
        dx = float(np.sum(w * self.particles[:,0]))
        dy = float(np.sum(w * self.particles[:,1]))
        cs = float(np.sum(w * np.cos(self.particles[:,2])))
        sn = float(np.sum(w * np.sin(self.particles[:,2])))
        dth = math.atan2(sn, cs)
        return np.array([dx,dy,dth], float)

    def systematic_resample(self):
        if not self.use_pf: return
        N = self.Np
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, dtype=int)
        cs = np.cumsum(self.weights)
        i = j = 0
        while i < N:
            if positions[i] < cs[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights[:] = 1.0 / N


def main():
    rclpy.init()
    node = ConeMapperPF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
