#!/usr/bin/env python3
# fslam_da_node.py (ROS2 Galactic)
# Data Association (DA-only) for skidpad with fixed landmark map.
# - FIX: use time-aligned GT odom for BASE<->WORLD transforms (buffer + interpolation)
# - Detections frame switch: detections_frame: 'base' (default) or 'world'
# - Mahalanobis χ² gating + iJCBB
# - RViz sanity markers + CSV logs (assoc.csv, metrics.csv, timing.csv)

import os, json, math, time, csv, bisect
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from eufs_msgs.msg import ConeArrayWithCovariance

# ------------------------------ Paths & constants ------------------------------
BASE_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = BASE_DIR / "slam_utils" / "data"
ALIGN_JSON = DATA_DIR / "alignment_transform.json"
KNOWN_MAP_CSV = DATA_DIR / "known_map.csv"
LOG_ROOT = BASE_DIR.parents[1] / "logs" / "fslam"

WORLD_FRAME = "world"
BASE_FRAME  = "base_footprint"

COLORS = ("blue","yellow","orange","orange_large")

# ------------------------------ Utility ------------------------------
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],[s, c]], dtype=float)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

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

# ------------------------------ χ² utilities ------------------------------
CHI2_APPROX = {
    2: {0.95: 5.9915, 0.99: 9.2103, 0.997: 11.829},
    4: {0.95: 9.4877, 0.99: 13.276, 0.997: 16.266},
}
def chi2_threshold(dof:int, sig:float)->float:
    table = CHI2_APPROX.get(dof)
    if table and sig in table:
        return table[sig]
    z = {0.95:1.64485, 0.99:2.32635, 0.997:2.96774}.get(sig, 2.32635)
    return dof * (1 - 2/(9*dof) + z*math.sqrt(2/(9*dof)))**3

def mahal2(v: np.ndarray, S: np.ndarray)->float:
    try:
        return float(v.T @ np.linalg.solve(S, v))
    except np.linalg.LinAlgError:
        return 1e12

# ------------------------------ Color penalties ------------------------------
INF = 1e12
COLOR_COST = {
    ('blue','blue'):0.0, ('yellow','yellow'):0.0, ('orange','orange'):0.0, ('orange_large','orange_large'):0.0,
    ('blue','yellow'):1.0, ('yellow','blue'):1.0,
    ('orange','orange_large'):0.5, ('orange_large','orange'):0.5,
}
def color_penalty(obs_c:str, map_c:str)->float:
    if (obs_c,map_c) in COLOR_COST:
        return COLOR_COST[(obs_c,map_c)]
    if ({obs_c,map_c} & {'orange','orange_large'}) and ({obs_c,map_c} & {'blue','yellow'}):
        return INF
    return 2.0

# ------------------------------ Data types ------------------------------
@dataclass
class ConeObs:
    pos_world: np.ndarray     # (2,), WORLD frame (DA + viz)
    pos_base: np.ndarray      # (2,), BASE frame (for side-rule)
    color: str
    Sigma_world: np.ndarray   # (2,2)

@dataclass
class ConeMap:
    pos_world: np.ndarray
    color: str
    id: int

@dataclass
class AssocResult:
    obs_to_map: List[int]
    mahalanobis: List[float]
    metrics: Dict[str, float]
    per_obs: List[Dict]

# ------------------------------ Main Node ------------------------------
class FslamDANode(Node):
    def __init__(self):
        super().__init__("fslam_da")

        # ---- Tunables ----
        self.declare_parameter("map_crop_m", 15.0)
        self.declare_parameter("gate_chi2_2d", 9.2103)          # 2 dof @99%
        self.declare_parameter("joint_sig", 0.99)
        self.declare_parameter("meas_sigma_floor_xy", 0.05)     # try 0.06 if brittle
        self.declare_parameter("color_lambda", 0.30)
        self.declare_parameter("allow_unmatched", True)
        self.declare_parameter("enforce_side_rule", True)       # blue y<0, yellow y>0 in BASE
        self.declare_parameter("orange_use_within_m", 12.0)     # use orange only near start
        self.declare_parameter("start_center_world", [0.0, 0.0])

        # Frame + sync
        self.declare_parameter("detections_frame", "base")      # 'base' or 'world'
        self.declare_parameter("odom_buffer_sec", 2.0)          # how much GT odom to retain
        self.declare_parameter("sync_max_slop_sec", 0.05)       # warn if |Δt|>slop

        # Resolve
        self.map_crop_m = float(self.get_parameter("map_crop_m").value)
        self.gate_chi2_2d = float(self.get_parameter("gate_chi2_2d").value)
        self.joint_sig = float(self.get_parameter("joint_sig").value)
        self.meas_floor_xy = float(self.get_parameter("meas_sigma_floor_xy").value)
        self.color_lambda = float(self.get_parameter("color_lambda").value)
        self.allow_unmatched = bool(self.get_parameter("allow_unmatched").value)
        self.enforce_side_rule = bool(self.get_parameter("enforce_side_rule").value)
        self.orange_use_within_m = float(self.get_parameter("orange_use_within_m").value)
        self.start_center_world = np.array(self.get_parameter("start_center_world").value, float)

        self.detections_frame = str(self.get_parameter("detections_frame").value).lower()
        self.odom_buffer_sec = float(self.get_parameter("odom_buffer_sec").value)
        self.sync_max_slop_sec = float(self.get_parameter("sync_max_slop_sec").value)

        # ---- Load alignment + map ----
        if not ALIGN_JSON.exists():
            raise RuntimeError(f"Missing {ALIGN_JSON}")
        with open(ALIGN_JSON, "r") as f:
            T = json.load(f)
        self.R_map = np.array(T["rotation"], float)
        self.t_map = np.array(T["translation"], float)

        map_rows = load_known_map(KNOWN_MAP_CSV)
        M = np.array([[r["x"], r["y"]] for r in map_rows], float)
        C = [r["color"] for r in map_rows]
        self.map_cones: List[ConeMap] = []
        for idx, (xy, c) in enumerate(zip((M @ self.R_map.T) + self.t_map, C)):
            if c not in COLORS:
                continue
            self.map_cones.append(ConeMap(pos_world=np.array(xy, float), color=c, id=idx))

        # ---- State ----
        self.pose_latest = np.array([0.0,0.0,0.0], float)  # fallback
        self.pose_ready = False
        self.odom_times: List[float] = []
        self.odom_buf: deque = deque()  # each item: (t, x, y, yaw)

        # ---- QoS ----
        q_default = QoSProfile(depth=50,
                               reliability=QoSReliabilityPolicy.RELIABLE,
                               durability=QoSDurabilityPolicy.VOLATILE,
                               history=QoSHistoryPolicy.KEEP_LAST)

        # ---- Subs ----
        self.sub_odom = self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, q_default)
        self.sub_cones = self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_cones, q_default)

        # ---- Pubs ----
        self.pub_markers = self.create_publisher(MarkerArray, "/fslam/markers", 10)

        # ---- Logs ----
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = LOG_ROOT / ts
        ensure_dir(self.log_dir)
        self.fp_assoc  = open(self.log_dir / "assoc.csv", "w", newline="")
        self.fp_met    = open(self.log_dir / "metrics.csv", "w", newline="")
        self.fp_timing = open(self.log_dir / "timing.csv", "w", newline="")
        self.w_assoc = csv.writer(self.fp_assoc); self.w_assoc.writerow(
            ["frame","t","obs_idx","obs_x","obs_y","obs_color","map_id","map_x","map_y","map_color","m2","side_ok","color_penalty"]
        )
        self.w_met = csv.writer(self.fp_met); self.w_met.writerow(
            ["frame","t","match_frac","mean_mahal_sqrt","p95_mahal_sqrt","wrong_side_rate","color_confusions","n_obs"]
        )
        self.w_tim = csv.writer(self.fp_timing); self.w_tim.writerow(
            ["frame","t","n_obs","n_cands_total","ordering_ms","jcbb_ms"]  # unchanged
        )
        self._last_flush = time.time()

        # ---- Heartbeat ----
        self.frame_idx = 0
        self.last_hb = time.time()
        self.get_logger().info(f"fslam_da up. Map cones: {len(self.map_cones)}. Logs: {self.log_dir}")

        self.publish_static_map_markers()

    # -------------------- Odom buffer & interpolation --------------------
    def cb_odom(self, msg: Odometry):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        self.pose_latest = np.array([x,y,yaw], float)
        self.pose_ready = True

        # push into time-ordered buffer
        self.odom_buf.append((t, x, y, yaw))
        # trim by time horizon
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

    def _pose_at(self, t_query: float) -> Tuple[np.ndarray, float]:
        """
        Return pose [x,y,yaw] interpolated to t_query (sec),
        plus |Δt| to the nearest odom sample.
        If buffer empty, fall back to latest.
        """
        if not self.odom_buf:
            return self.pose_latest.copy(), float('inf')

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        # Nearest only
        if idx == 0:
            t0,x0,y0,yaw0 = self.odom_buf[0]
            return np.array([x0,y0,yaw0], float), abs(t_query - t0)
        if idx >= len(self.odom_buf):
            t1,x1,y1,yaw1 = self.odom_buf[-1]
            return np.array([x1,y1,yaw1], float), abs(t_query - t1)

        # Interpolate between idx-1 and idx
        t0,x0,y0,yaw0 = self.odom_buf[idx-1]
        t1,x1,y1,yaw1 = self.odom_buf[idx]
        # guard division
        if t1 == t0:
            return np.array([x0,y0,yaw0], float), abs(t_query - t0)
        a = (t_query - t0) / (t1 - t0)
        x = x0 + a*(x1 - x0)
        y = y0 + a*(y1 - y0)
        # yaw: shortest arc
        dyaw = wrap_angle(yaw1 - yaw0)
        yaw = wrap_angle(yaw0 + a*dyaw)
        dt_nearest = min(abs(t_query - t0), abs(t_query - t1))
        return np.array([x,y,yaw], float), dt_nearest

    # -------------------- Cones --------------------
    def cb_cones(self, msg: ConeArrayWithCovariance):
        if not self.pose_ready:
            return

        # use pose aligned to this detections timestamp
        tsec = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose_w_b, dt_pose = self._pose_at(tsec)
        if dt_pose > self.sync_max_slop_sec:
            # Warn once per second max
            now = time.time()
            if not hasattr(self, "_last_sync_warn") or (now - getattr(self, "_last_sync_warn")) > 1.0:
                self.get_logger().warn(f"DA sync slop {dt_pose*1000:.1f} ms exceeds {self.sync_max_slop_sec*1000:.0f} ms")
                self._last_sync_warn = now

        t0 = time.perf_counter()
        obs_list = self._extract_obs(msg, pose_w_b)  # time-consistent pose
        if len(obs_list)==0:
            return

        # Precompute candidates & counts
        n_cands_total = 0
        cand_per_obs = []
        for o in obs_list:
            cands = self._gen_candidates(o)
            cand_per_obs.append(cands)
            n_cands_total += len(cands)

        # iJCBB
        t_order0 = time.perf_counter()
        assoc = self._jcbb(obs_list, cand_per_obs)
        t1 = time.perf_counter()

        # Logs (keep format)
        self._write_assoc_logs(self.frame_idx, tsec, obs_list, assoc)
        mean_sqrt, p95_sqrt = self._sqrt_mahal_stats(assoc.mahalanobis)
        self.w_met.writerow([
            self.frame_idx, f"{tsec:.6f}",
            f"{assoc.metrics.get('match_frac',0.0):.3f}",
            _blank_if_nan(mean_sqrt,3), _blank_if_nan(p95_sqrt,3),
            f"{assoc.metrics.get('wrong_side_rate',0.0):.3f}",
            int(assoc.metrics.get('color_confusions',0)),
            len(obs_list)
        ])
        self.w_tim.writerow([
            self.frame_idx, f"{tsec:.6f}",
            len(obs_list), n_cands_total,
            int((t_order0 - t0)*1000), int((t1 - t_order0)*1000)
        ])

        # Flush occasionally
        now_w = time.time()
        if now_w - self._last_flush > 0.5:
            self.fp_assoc.flush(); self.fp_met.flush(); self.fp_timing.flush()
            self._last_flush = now_w

        # RViz
        self._publish_frame_markers(self.frame_idx, obs_list, assoc)

        # Heartbeat
        if now_w - self.last_hb > 1.0:
            self.get_logger().info(
                f"frame {self.frame_idx}: match_frac={assoc.metrics.get('match_frac',0.0):.2f}, "
                f"p95√m²={_blank_if_nan(p95_sqrt,2)}"
            )
            self.last_hb = now_w

        self.frame_idx += 1

    def _extract_obs(self, msg: ConeArrayWithCovariance, pose_w_b: np.ndarray)->List[ConeObs]:
        """
        Build observations in BOTH frames using the time-aligned pose:
          pos_world: WORLD (for DA + viz)
          pos_base : BASE (for side-rule)
        If detections_frame == 'base', convert base→world with pose_w_b.
        If detections_frame == 'world', also compute base via world→base with pose_w_b.
        """
        obs=[]
        x,y,yaw = pose_w_b
        Rwb = rot2d(yaw); Rbw = Rwb.T
        Radd = np.diag([self.meas_floor_xy**2, self.meas_floor_xy**2])

        def add(arr, color):
            for c in arr:
                px = float(c.point.x); py = float(c.point.y)
                Pw = np.array([[c.covariance[0], c.covariance[1]],
                               [c.covariance[2], c.covariance[3]]], float) + Radd

                if self.detections_frame == 'base':
                    pb = np.array([px,py], float)
                    pw = np.array([x,y], float) + Rwb @ pb
                    Sigma_w = Rwb @ Pw @ Rwb.T
                else:
                    pw = np.array([px,py], float)
                    pb = Rbw @ (pw - np.array([x,y], float))
                    Sigma_w = Pw

                obs.append(ConeObs(pos_world=pw, pos_base=pb, color=color, Sigma_world=Sigma_w))

        add(msg.blue_cones,       "blue")
        add(msg.yellow_cones,     "yellow")
        add(msg.orange_cones,     "orange")
        add(msg.big_orange_cones, "orange_large")
        return obs

    # -------------------- Candidates + iJCBB --------------------
    def _gen_candidates(self, obs: ConeObs)->List[Tuple[int,float,np.ndarray,np.ndarray]]:
        out=[]
        crop2 = self.map_crop_m**2
        start_c = self.start_center_world

        # side rule in BASE
        if self.enforce_side_rule:
            if obs.color=='blue' and obs.pos_base[1] > 0: return []
            if obs.color=='yellow' and obs.pos_base[1] < 0: return []

        for m in self.map_cones:
            if np.sum((m.pos_world - obs.pos_world)**2) > crop2:
                continue
            if m.color in ('orange','orange_large'):
                if np.linalg.norm(m.pos_world - start_c) > self.orange_use_within_m:
                    continue
            if color_penalty(obs.color, m.color) >= INF:
                continue

            innov = obs.pos_world - m.pos_world
            m2 = mahal2(innov, obs.Sigma_world)
            if m2 <= self.gate_chi2_2d:
                m2_aug = m2 + self.color_lambda * color_penalty(obs.color, m.color)
                out.append((m.id, m2_aug, innov, obs.Sigma_world))
        out.sort(key=lambda t: t[1])
        return out

    def _jcbb(self, obs_list: List[ConeObs], cand_per_obs: List[List[Tuple[int,float,np.ndarray,np.ndarray]]]) -> AssocResult:
        order = np.argsort([ (c[0][1] if c else INF) for c in cand_per_obs ])
        best = {'score': INF, 'assign': [-1]*len(obs_list), 'mahal':[math.nan]*len(obs_list)}
        used = set()

        def dfs(k:int, assign:List[int], score:float, innovs:List[np.ndarray], Sigmas:List[np.ndarray]):
            nonlocal best
            if k == len(order):
                matched = sum(1 for a in assign if a!=-1)
                best_matched = sum(1 for a in best['assign'] if a!=-1)
                if (matched > best_matched) or (matched==best_matched and score < best['score']):
                    best = {'score':score, 'assign':assign.copy(), 'mahal':[math.nan]*len(assign)}
                    idx=0
                    for oi in range(len(assign)):
                        if assign[oi]!=-1:
                            best['mahal'][oi] = mahal2(innovs[idx], Sigmas[idx])
                            idx+=1
                return
            oi = order[k]
            cands = cand_per_obs[oi]
            for (mid, m2_aug, innov, S) in cands:
                if mid in used: continue
                sum_m2 = 0.0
                for v,Si in zip(innovs+[innov], Sigmas+[S]):
                    sum_m2 += mahal2(v, Si)
                dof = 2*(len(innovs)+1)
                if sum_m2 > chi2_threshold(dof, self.joint_sig):
                    continue
                used.add(mid); assign[oi]=mid
                dfs(k+1, assign, score + m2_aug, innovs+[innov], Sigmas+[S])
                assign[oi]=-1; used.remove(mid)
            if self.allow_unmatched:
                dfs(k+1, assign, score, innovs, Sigmas)

        dfs(0, [-1]*len(obs_list), 0.0, [], [])

        # metrics
        id2map = {m.id:m for m in self.map_cones}
        per_obs=[]
        wrong_side = 0
        color_confusions = 0
        matched = 0
        mahals=[]
        for i,o in enumerate(obs_list):
            mid = best['assign'][i]
            row = {'obs_idx':i, 'obs_x':float(o.pos_world[0]), 'obs_y':float(o.pos_world[1]), 'obs_color':o.color,
                   'map_id':mid, 'mahal':best['mahal'][i]}
            if mid!=-1:
                m = id2map[mid]
                matched += 1
                if not math.isnan(best['mahal'][i]): mahals.append(best['mahal'][i])
                row.update({'map_x':float(m.pos_world[0]), 'map_y':float(m.pos_world[1]), 'map_color':m.color})
                if (o.color=='blue' and o.pos_base[1]>0) or (o.color=='yellow' and o.pos_base[1]<0):
                    wrong_side += 1
                if m.color!=o.color and color_penalty(o.color, m.color)<INF:
                    color_confusions += 1
            per_obs.append(row)

        N = len(obs_list)
        metrics = {
            'match_frac': matched/max(1,N),
            'mean_mahal': float(np.nanmean(mahals)) if mahals else float('nan'),
            'p95_mahal': float(np.nanpercentile(mahals,95)) if mahals else float('nan'),
            'wrong_side_rate': wrong_side/max(1,matched) if matched>0 else 0.0,
            'color_confusions': color_confusions
        }
        return AssocResult(best['assign'], best['mahal'], metrics, per_obs)

    # -------------------- Logging --------------------
    def _write_assoc_logs(self, frame:int, tsec:float, obs_list:List[ConeObs], assoc:AssocResult):
        id2map = {m.id:m for m in self.map_cones}
        for i,o in enumerate(obs_list):
            mid = assoc.obs_to_map[i]
            m2 = assoc.mahalanobis[i]
            side_ok = 1
            if self.enforce_side_rule and ((o.color=='blue' and o.pos_base[1]>0) or (o.color=='yellow' and o.pos_base[1]<0)):
                side_ok = 0
            if mid == -1:
                self.w_assoc.writerow([frame, f"{tsec:.6f}", i,
                    f"{o.pos_world[0]:.3f}", f"{o.pos_world[1]:.3f}", o.color,
                    -1, "", "", "", _blank_if_nan(m2,4), side_ok, ""
                ])
            else:
                m = id2map[mid]
                self.w_assoc.writerow([frame, f"{tsec:.6f}", i,
                    f"{o.pos_world[0]:.3f}", f"{o.pos_world[1]:.3f}", o.color,
                    mid, f"{m.pos_world[0]:.3f}", f"{m.pos_world[1]:.3f}", m.color,
                    _blank_if_nan(m2,4), side_ok, color_penalty(o.color, m.color)
                ])

    def _sqrt_mahal_stats(self, m2_list: List[float])->Tuple[float,float]:
        vals = [math.sqrt(v) for v in m2_list if v is not None and not math.isnan(v)]
        if not vals: return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.percentile(vals,95))

    # -------------------- RViz markers --------------------
    def publish_static_map_markers(self):
        ma = MarkerArray()
        delm = Marker(); delm.action = Marker.DELETEALL
        ma.markers.append(delm)

        now = rclpy.clock.Clock().now().to_msg()
        for k,m in enumerate(self.map_cones):
            mk = Marker()
            mk.header.frame_id = WORLD_FRAME
            mk.header.stamp = now
            mk.ns = "map_cones"; mk.id = 10000 + k
            mk.type = Marker.SPHERE; mk.action = Marker.ADD
            mk.pose.position.x = float(m.pos_world[0])
            mk.pose.position.y = float(m.pos_world[1])
            mk.pose.position.z = 0.0
            s = 0.35 if m.color!='orange_large' else 0.55
            mk.scale.x = mk.scale.y = mk.scale.z = s
            mk.color = _rgba_by_color(m.color)
            mk.lifetime = Duration(sec=0)
            ma.markers.append(mk)
        self.pub_markers.publish(ma)

    def _publish_frame_markers(self, frame:int, obs_list:List[ConeObs], assoc:AssocResult):
        ma = MarkerArray()
        now = rclpy.clock.Clock().now().to_msg()

        # Observations (WORLD)
        for i,o in enumerate(obs_list):
            mk = Marker()
            mk.header.frame_id = WORLD_FRAME
            mk.header.stamp = now
            mk.ns = "observations"; mk.id = 20000 + i
            mk.type = Marker.SPHERE; mk.action = Marker.ADD
            mk.pose.position.x = float(o.pos_world[0])
            mk.pose.position.y = float(o.pos_world[1])
            mk.pose.position.z = 0.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.25
            mk.color = _rgba_by_color(o.color)
            mk.lifetime = Duration(sec=1)
            ma.markers.append(mk)

        # Matches (lines)
        line = Marker()
        line.header.frame_id = WORLD_FRAME
        line.header.stamp = now
        line.ns = "matches"; line.id = 30000
        line.type = Marker.LINE_LIST; line.action = Marker.ADD
        line.scale.x = 0.06
        for i,o in enumerate(obs_list):
            mid = assoc.obs_to_map[i]
            if mid == -1: continue
            m = next((mm for mm in self.map_cones if mm.id==mid), None)
            if m is None: continue
            m2 = assoc.mahalanobis[i]
            rgba = _color_for_mahal(m2)
            p1 = Point(x=float(o.pos_world[0]), y=float(o.pos_world[1]), z=0.0)
            p2 = Point(x=float(m.pos_world[0]), y=float(m.pos_world[1]), z=0.0)
            line.points.append(p1); line.points.append(p2)
            line.colors.append(rgba); line.colors.append(rgba)
        line.lifetime = Duration(sec=1)
        ma.markers.append(line)

        # Unmatched crosses
        for i,o in enumerate(obs_list):
            if assoc.obs_to_map[i] != -1: continue
            for k in range(2):
                ln = Marker()
                ln.header.frame_id = WORLD_FRAME
                ln.header.stamp = now
                ln.ns = "unmatched"; ln.id = 40000 + i*2 + k
                ln.type = Marker.LINE_LIST; ln.action = Marker.ADD
                ln.scale.x = 0.04
                ln.color = _rgba_by_name("white")
                dx, dy = (0.25,0.0) if k==0 else (0.0,0.25)
                p1 = Point(x=float(o.pos_world[0]-dx), y=float(o.pos_world[1]-dy), z=0.0)
                p2 = Point(x=float(o.pos_world[0]+dx), y=float(o.pos_world[1]+dy), z=0.0)
                ln.points.append(p1); ln.points.append(p2)
                ln.lifetime = Duration(sec=1)
                ma.markers.append(ln)

        # HUD
        badge = Marker()
        badge.header.frame_id = WORLD_FRAME
        badge.header.stamp = now
        badge.ns = "hud"; badge.id = 50000
        badge.type = Marker.TEXT_VIEW_FACING; badge.action = Marker.ADD
        badge.pose.position.x = float(self.pose_latest[0] + 2.0)
        badge.pose.position.y = float(self.pose_latest[1] + 2.0)
        badge.pose.position.z = 0.0
        badge.scale.z = 0.6
        mean_sqrt, p95_sqrt = self._sqrt_mahal_stats(assoc.mahalanobis)
        badge.text = (f"frame {frame}\n"
                      f"match_frac: {assoc.metrics.get('match_frac',0.0):.2f}\n"
                      f"p95 √m²: {_blank_if_nan(p95_sqrt,2)}\n"
                      f"wrong_side: {assoc.metrics.get('wrong_side_rate',0.0):.2f}\n"
                      f"confusions: {int(assoc.metrics.get('color_confusions',0))}")
        badge.color = _rgba_by_name("white")
        badge.lifetime = Duration(sec=1)
        ma.markers.append(badge)

        self.pub_markers.publish(ma)

    # -------------------- Shutdown --------------------
    def destroy_node(self):
        try:
            self.fp_assoc.close(); self.fp_met.close(); self.fp_timing.close()
        except Exception:
            pass
        super().destroy_node()

# ------------------------------ Helpers ------------------------------
def _rgba_by_color(c:str) -> ColorRGBA:
    return {
        'blue': _rgba(0.1,0.4,1.0), 'yellow': _rgba(1.0,0.9,0.1),
        'orange': _rgba(1.0,0.5,0.0), 'orange_large': _rgba(1.0,0.25,0.0),
        'white': _rgba(1,1,1)
    }.get(c, _rgba(0.8,0.8,0.8))

def _rgba_by_name(n:str) -> ColorRGBA:
    return {
        'green': _rgba(0.2,1.0,0.2), 'amber': _rgba(1.0,0.8,0.0),
        'red': _rgba(1.0,0.2,0.2), 'white': _rgba(1,1,1),
        'grey': _rgba(0.7,0.7,0.7)
    }[n]

def _rgba(r,g,b,a=1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))

def _color_for_mahal(m2: float) -> ColorRGBA:
    if m2 is None or math.isnan(m2): return _rgba_by_name('grey')
    if m2 <= 4.0:    return _rgba_by_name('green')
    if m2 <= 9.21:   return _rgba_by_name('amber')
    return _rgba_by_name('red')

def _blank_if_nan(v, ndigits=3):
    try:
        if v is None or (isinstance(v,float) and math.isnan(v)): return ""
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return ""

# ------------------------------ main ------------------------------
def main():
    rclpy.init()
    node = FslamDANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
