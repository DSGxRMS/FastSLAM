#!/usr/bin/env python3
import math, bisect
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance
from geometry_msgs.msg import Pose2D, Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3

# -------------------- math utils --------------------
def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi
def rot2d(th):
    c,s=math.cos(th),math.sin(th)
    return np.array([[c,-s],[s,c]],float)
def yaw_from_quat(x,y,z,w):
    siny_cosp=2*(w*z+x*y); cosy_cosp=1-2*(y*y+z*z)
    return math.atan2(siny_cosp,cosy_cosp)
def quat_from_yaw(yaw):
    q=Quaternion(); q.x=0.0; q.y=0.0; q.z=math.sin(yaw/2.0); q.w=math.cos(yaw/2.0)
    return q

def _norm_ppf(p):
    if p<=0.0: return -float('inf')
    if p>=1.0: return float('inf')
    a1=-3.969683028665376e+01; a2=2.209460984245205e+02; a3=-2.759285104469687e+02
    a4=1.383577518672690e+02; a5=-3.066479806614716e+01; a6=2.506628277459239e+00
    b1=-5.447609879822406e+01; b2=1.615858368580409e+02; b3=-1.556989798598866e+02
    b4=6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5=4.374664141464968e+00; c6=2.938163982698783e+00
    d1=7.784695709041462e-03; d2=3.224671290700398e-01; d3=2.445134137142996e+00
    d4=3.754408661907416e+00
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p>phigh:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q=p-0.5; r=q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
def chi2_quantile(dof,p):
    k=max(1,int(dof)); z=_norm_ppf(p)
    t=1.0-2.0/(9.0*k)+z*math.sqrt(2.0/(9.0*k))
    return k*(t**3)

def spd_joseph_update(P, H, R, K):
    I = np.eye(P.shape[0]); A = (I - K @ H)
    Pn = A @ P @ A.T + K @ R @ K.T
    Pn = 0.5*(Pn + Pn.T)
    w,v = np.linalg.eigh(Pn); w = np.clip(w, 1e-9, None)
    return (v * w) @ v.T

def safe_inv_and_logdet(S):
    try:
        Sinv = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0: raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
        u,s,vt = np.linalg.svd(S)
        s = np.clip(s, 1e-12, None)
        logdet = float(np.sum(np.log(s)))
    return Sinv, logdet

def mahal2(mu_a, S_a, mu_b, S_b):
    d = (mu_a - mu_b).reshape(2,1)
    S = S_a + S_b
    Sinv, _ = safe_inv_and_logdet(S)
    return float(d.T @ Sinv @ d)

# -------------------- time-synced odom buffer --------------------
class OdomBuffer:
    def __init__(self, cap_s=3.0, extrap_cap=0.25):
        self.buf=deque(); self.cap_s=cap_s; self.extrap_cap=extrap_cap
        self.latest=np.array([0.0,0.0,0.0],float)
    def push(self,t,x,y,yaw,v,yr):
        self.buf.append((t,x,y,yaw,v,yr))
        tmin=t-self.cap_s
        while self.buf and self.buf[0][0]<tmin: self.buf.popleft()
        self.latest[:]=[x,y,yaw]
    def pose_at(self,tq:float):
        if not self.buf: return self.latest.copy(), float('inf'), 0.0, 0.0
        times=[it[0] for it in self.buf]
        i=bisect.bisect_left(times,tq)
        if i==0:
            t0,x0,y0,yaw0,v0,yr0=self.buf[0]
            return np.array([x0,y0,yaw0],float), abs(tq-t0), v0, yr0
        if i>=len(self.buf):
            t1,x1,y1,yaw1,v1,yr1=self.buf[-1]
            dt=max(0.0,min(tq-t1,self.extrap_cap))
            if abs(yr1)<1e-6:
                x=x1+v1*dt*math.cos(yaw1); y=y1+v1*dt*math.sin(yaw1); yaw=yaw1
            else:
                x=x1+(v1/yr1)*(math.sin(yaw1+yr1*dt)-math.sin(yaw1))
                y=y1-(v1/yr1)*(math.cos(yaw1+yr1*dt)-math.cos(yaw1))
                yaw=wrap(yaw1+yr1*dt)
            return np.array([x,y,yaw],float), abs(tq-t1), v1, yr1
        t0,x0,y0,yaw0,v0,yr0=self.buf[i-1]
        t1,x1,y1,yaw1,v1,yr1=self.buf[i]
        if t1==t0:
            return np.array([x0,y0,yaw0],float), abs(tq-t0), v0, yr0
        a=(tq-t0)/(t1-t0)
        x=x0+a*(x1-x0); y=y0+a*(y1-y0)
        dyaw=wrap(yaw1-yaw0); yaw=wrap(yaw0+a*dyaw)
        v=(1-a)*v0+a*v1; yr=(1-a)*yr0+a*yr1
        dt_near=min(abs(tq-t0),abs(tq-t1))
        return np.array([x,y,yaw],float), dt_near, v, yr

# -------------------- FastSLAM Core --------------------
class FastSLAMCore(Node):
    def __init__(self):
        super().__init__("fastslam_core", automatically_declare_parameters_from_overrides=True)

        # topics
        self.declare_parameter("topics.odom_in","/odometry_integration/car_state")
        self.declare_parameter("topics.cones","/ground_truth/cones")
        self.declare_parameter("topics.odom_out","/fastslam/odom")
        self.declare_parameter("topics.map_out","/fastslam/map_cones")

        # frames & rate
        self.declare_parameter("detections_frame","base")   # your source says base
        self.declare_parameter("target_cone_hz",25.0)       # informative only

        # gates (looser)
        self.declare_parameter("chi2_gate_2d",9.210)        # ~99% df=2
        self.declare_parameter("joint_sig",0.98)
        self.declare_parameter("joint_relax",1.30)
        self.declare_parameter("min_k_for_joint",2)

        # noise/search
        self.declare_parameter("map_crop_m",22.0)
        self.declare_parameter("meas_sigma_floor_xy",0.45)
        self.declare_parameter("alpha_trans",0.9)
        self.declare_parameter("beta_rot",0.9)

        # PF
        self.declare_parameter("pf.num_particles",300)
        self.declare_parameter("pf.resample_neff_ratio",0.5)
        self.declare_parameter("pf.process_std_xy_m",0.02)
        self.declare_parameter("pf.process_std_yaw_rad",0.002)

        # birth/merge
        self.declare_parameter("birth.promote_count",1)
        self.declare_parameter("birth.promote_window_s",1.0)  # gets adapted to rate
        self.declare_parameter("birth.merge_radius_m",0.8)
        self.declare_parameter("merge.mahal_gate2",6.0)
        self.declare_parameter("jcbb.topk",4)

        # persistence / aging
        self.declare_parameter("landmark.max_age_s",6.0)       # only used if persist=False
        self.declare_parameter("landmark.persist", True)       # keep credible LM forever
        self.declare_parameter("landmark.persist_seen_threshold", 3)

        self.declare_parameter("like_floor",1e-12)

        # resolve
        gp=self.get_parameter
        self.topic_odom      = str(gp("topics.odom_in").value)
        self.topic_cones     = str(gp("topics.cones").value)
        self.topic_odom_out  = str(gp("topics.odom_out").value)
        self.topic_map_out   = str(gp("topics.map_out").value)

        self.frame_mode      = str(gp("detections_frame").value).lower()
        if self.frame_mode not in ("world","base"): self.frame_mode="base"

        self.t_cone          = float(gp("target_cone_hz").value)
        self.gate            = float(gp("chi2_gate_2d").value)
        self.joint_sig       = float(gp("joint_sig").value)
        self.joint_relax     = float(gp("joint_relax").value)
        self.kmin            = int(gp("min_k_for_joint").value)
        self.crop2           = float(gp("map_crop_m").value)**2
        self.sigma0          = float(gp("meas_sigma_floor_xy").value)
        self.a_trans         = float(gp("alpha_trans").value)
        self.b_rot           = float(gp("beta_rot").value)

        self.Np              = int(gp("pf.num_particles").value)
        self.neff_ratio      = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy        = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw       = float(gp("pf.process_std_yaw_rad").value)

        self.promote_count   = int(gp("birth.promote_count").value)
        self.promote_window  = float(gp("birth.promote_window_s").value)
        self.merge_radius    = float(gp("birth.merge_radius_m").value)
        self.merge_m2_gate   = float(gp("merge.mahal_gate2").value)
        self.jcbb_topk       = int(gp("jcbb.topk").value)

        self.max_age         = float(gp("landmark.max_age_s").value)
        self.persist         = bool(gp("landmark.persist").value)
        self.persist_seen_th = int(gp("landmark.persist_seen_threshold").value)
        self.like_floor      = float(gp("like_floor").value)

        # qos/io
        qos_in=QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=200,
                          reliability=QoSReliabilityPolicy.BEST_EFFORT,
                          durability=QoSDurabilityPolicy.VOLATILE)
        qos_out=10
        self.create_subscription(Odometry,self.topic_odom,self.cb_odom,qos_in)
        self.create_subscription(ConeArrayWithCovariance,self.topic_cones,self.cb_cones,qos_in)
        self.pub_odom=self.create_publisher(Odometry,self.topic_odom_out,qos_out)
        self.pub_map =self.create_publisher(ConeArrayWithCovariance,self.topic_map_out,qos_out)
        self.pub_pose2d=self.create_publisher(Pose2D,"/fastslam/pose2d",qos_out)

        # state
        self.odom_buf=OdomBuffer(3.0,0.25)
        self.pose_ready=False
        self.last_cone_ts=None
        self.have_delta=False
        self.delta_hat=np.zeros(3,float)
        self.particles=[self._make_particle() for _ in range(self.Np)]
        self.weights=np.ones(self.Np,float)/self.Np
        self._LOG2PI = math.log(2.0*math.pi)

        # rate adaptation
        self._rx_dt_ema = None
        self._ema_alpha = 0.2
        self._now_s = lambda: self.get_clock().now().nanoseconds * 1e-9

        # persistent fused map (world frame)
        self.global_map = {
            "blue"  : {"mu":np.zeros((0,2)), "S":np.zeros((0,2,2)), "seen":np.zeros((0,),int), "t":np.zeros((0,))},
            "yellow": {"mu":np.zeros((0,2)), "S":np.zeros((0,2,2)), "seen":np.zeros((0,),int), "t":np.zeros((0,))},
            "orange": {"mu":np.zeros((0,2)), "S":np.zeros((0,2,2)), "seen":np.zeros((0,),int), "t":np.zeros((0,))},
            "big"   : {"mu":np.zeros((0,2)), "S":np.zeros((0,2,2)), "seen":np.zeros((0,),int), "t":np.zeros((0,))},
        }

    def _make_particle(self):
        z2=(0,2,2)
        return {
            "pose":np.zeros(3,float),
            "maps":{
                "blue":{"mu":np.zeros((0,2),float),"Sigma":np.zeros(z2,float),"seen":np.zeros((0,),int),"t":np.zeros((0,),float)},
                "yellow":{"mu":np.zeros((0,2),float),"Sigma":np.zeros(z2,float),"seen":np.zeros((0,),int),"t":np.zeros((0,),float)},
                "orange":{"mu":np.zeros((0,2),float),"Sigma":np.zeros(z2,float),"seen":np.zeros((0,),int),"t":np.zeros((0,),float)},
                "big":{"mu":np.zeros((0,2),float),"Sigma":np.zeros(z2,float),"seen":np.zeros((0,),int),"t":np.zeros((0,),float)},
            },
            "tentative":{"blue":[], "yellow":[], "orange":[], "big":[]}
        }

    # -------------------- callbacks --------------------
    def cb_odom(self,msg:Odometry):
        t=RclTime.from_msg(msg.header.stamp).nanoseconds*1e-9
        x=float(msg.pose.pose.position.x); y=float(msg.pose.pose.position.y)
        q=msg.pose.pose.orientation; yaw=yaw_from_quat(q.x,q.y,q.z,q.w)
        vx=float(msg.twist.twist.linear.x); vy=float(msg.twist.twist.linear.y)
        v=math.hypot(vx,vy); yr=float(msg.twist.twist.angular.z)
        self.odom_buf.push(t,x,y,yaw,v,yr)
        self.pose_ready=True

        if self.have_delta:
            dx,dy,dth=self.delta_hat; x_c=x+dx; y_c=y+dy; yaw_c=wrap(yaw+dth)
        else:
            x_c=x; y_c=y; yaw_c=yaw
        od=Odometry()
        od.header.stamp=msg.header.stamp
        od.header.frame_id="map"; od.child_frame_id="base_link"
        od.pose=PoseWithCovariance(); od.twist=TwistWithCovariance()
        od.pose.pose=Pose(position=Point(x=x_c,y=y_c,z=0.0),orientation=quat_from_yaw(yaw_c))
        od.twist.twist=Twist(linear=Vector3(x=v*math.cos(yaw_c),y=v*math.sin(yaw_c),z=0.0),
                             angular=Vector3(x=0.0,y=0.0,z=yr))
        self.pub_odom.publish(od)
        p2=Pose2D(); p2.x=x_c; p2.y=y_c; p2.theta=yaw_c
        self.pub_pose2d.publish(p2)

    def cb_cones(self,msg:ConeArrayWithCovariance):
        if not self.pose_ready: return
        t_z=RclTime.from_msg(msg.header.stamp).nanoseconds*1e-9

        # adapt to observed rate (no throttling)
        now = self._now_s()
        if hasattr(self, "_last_rx"):
            dt_rx = max(1e-3, now - self._last_rx)
            self._rx_dt_ema = dt_rx if self._rx_dt_ema is None else \
                (1 - self._ema_alpha)*self._rx_dt_ema + self._ema_alpha*dt_rx
        self._last_rx = now
        self.promote_window = float(np.clip(3.0*(self._rx_dt_ema if self._rx_dt_ema else 1.0), 0.6, 4.0))

        self.last_cone_ts=t_z

        od_pose,dt_pose,v_local,yr_local=self.odom_buf.pose_at(t_z)
        dt_pose = 0.0 if not math.isfinite(dt_pose) else dt_pose

        for i in range(self.Np):
            self.particles[i]["pose"]=self._propose_pose(od_pose, v_local, yr_local, dt_pose)

        obs=self._parse_obs(msg)
        if not any(len(obs[k])>0 for k in obs.keys()): return

        ll=np.zeros(self.Np,float)
        for i in range(self.Np):
            ll[i]=self._update_particle(self.particles[i],obs,t_z,dt_pose,v_local,yr_local)

        w_new=self.weights*np.exp(ll-np.max(ll))
        sw=float(np.sum(w_new))
        self.weights[:] = (1.0/self.Np) if (not math.isfinite(sw) or sw<=0.0) else (w_new/sw)

        neff=1.0/float(np.sum(self.weights*self.weights))
        if neff<self.neff_ratio*self.Np:
            self._resample_systematic_deep()

        bi=int(np.argmax(self.weights))
        best=self.particles[bi]
        self.delta_hat[:]=self._compute_delta(best["pose"], od_pose=self.odom_buf.latest.copy())
        self.have_delta=True

        # update persistent global map from best
        self._accumulate_global_map(best, t_z)

        # publish persistent map
        self._publish_map_persistent(t_z)

    # -------------------- core ops --------------------
    def _compute_delta(self,pose_particle, od_pose):
        x_p,y_p,yaw_p=pose_particle; x_o,y_o,yaw_o=od_pose
        return np.array([x_p-x_o, y_p-y_o, wrap(yaw_p-yaw_o)],float)

    def _propose_pose(self,od_pose, v, yr, dt):
        x,y,yaw=od_pose
        s_xy = self.p_std_xy + 0.1*abs(v)*max(dt,1e-3)
        s_yaw= self.p_std_yaw + 0.05*abs(yr)*max(dt,1e-3)
        x+=np.random.normal(0.0,s_xy)
        y+=np.random.normal(0.0,s_xy)
        yaw=wrap(yaw+np.random.normal(0.0,s_yaw))
        return np.array([x,y,yaw],float)

    def _parse_obs(self,msg):
        out={"blue":[], "yellow":[], "orange":[], "big":[]}
        def arr_append(name, c):
            R=np.array([[c.covariance[0],c.covariance[1]],[c.covariance[2],c.covariance[3]]],float)
            p=np.array([float(c.point.x), float(c.point.y)],float)
            if not np.isfinite(p).all(): return
            if not np.isfinite(R).all(): return
            R = 0.5*(R+R.T)
            w,v = np.linalg.eigh(R); w = np.clip(w, 1e-6, None); R = (v*w)@v.T
            out[name].append((p, R))
        for c in msg.blue_cones: arr_append("blue", c)
        for c in msg.yellow_cones: arr_append("yellow", c)
        for c in msg.orange_cones: arr_append("orange", c)
        for c in msg.big_orange_cones: arr_append("big", c)
        return out

    def _inflate_R(self, R_meas_world, v, yr, dt, r):
        add = self.a_trans*(v**2)*(dt**2) + self.b_rot*(yr**2)*(dt**2)*(r**2)
        R_eff = R_meas_world + np.diag([self.sigma0**2, self.sigma0**2]) + add*np.eye(2)
        R_eff = 0.5*(R_eff + R_eff.T)
        w,vv = np.linalg.eigh(R_eff); w = np.clip(w, 1e-9, None)
        return (vv * w) @ vv.T

    def _jcbb(self, OBS, MU, S, pose_p, dt_pose, v, yr, _crop2_unused):
        if MU.shape[0]==0 or len(OBS)==0: return []
        x_p,y_p,yaw_p=pose_p
        pose_xy=np.array([x_p,y_p])
        if self.frame_mode == "base": Rwb=rot2d(yaw_p)

        obs_world=[]
        for (z, Rm) in OBS:
            if self.frame_mode=="base":
                pw = pose_xy + Rwb @ z; Rw = Rwb @ Rm @ Rwb.T
            else:
                pw = z.copy(); Rw = Rm.copy()
            r = float(np.linalg.norm(pw - pose_xy))
            R_eff = self._inflate_R(Rw, v, yr, dt_pose, r)
            obs_world.append((pw,R_eff))

        cands=[]
        for (pw, R_eff) in obs_world:
            cc=[]
            for mid in range(MU.shape[0]):
                St = S[mid] + R_eff
                if not np.isfinite(St).all(): continue
                Sinv, _ = safe_inv_and_logdet(St)
                innov = pw - MU[mid]
                if not np.isfinite(innov).all(): continue
                m2 = float(innov.T @ Sinv @ innov)
                if m2 <= self.gate: cc.append((mid, m2, R_eff))
            if len(cc) > self.jcbb_topk:
                cc = sorted(cc, key=lambda t: t[1])[:self.jcbb_topk]
            cands.append(cc)

        order = sorted(range(len(obs_world)), key=lambda i: len(cands[i]) if cands[i] else float("inf"))
        cands_ord = [cands[i] for i in order]
        best_pairs=[]; best_K=0; best_sum=1e18; used=set()
        def dfs(idx,cur,sum_m2):
            nonlocal best_pairs, best_K, best_sum
            if idx==len(cands_ord):
                K=len(cur)
                if K>best_K or (K==best_K and sum_m2<best_sum):
                    best_pairs=cur.copy(); best_K=K; best_sum=sum_m2
                return
            dfs(idx+1,cur,sum_m2)
            for (mid,m2,R_eff) in cands_ord[idx]:
                if mid in used: continue
                new_sum=sum_m2+m2; K_new=len(cur)+1
                if K_new>=self.kmin:
                    df=2*K_new
                    if new_sum > self.joint_relax*chi2_quantile(df,self.joint_sig): continue
                used.add(mid); cur.append((order[idx], mid, m2, R_eff))
                dfs(idx+1,cur,new_sum)
                cur.pop(); used.remove(mid)
        dfs(0,[],0.0)
        return best_pairs

    def _update_particle(self, part, obs_by_color, t_z, dt_pose, v, yr):
        pose_p = part["pose"]
        x_p,y_p,yaw_p = pose_p
        if self.frame_mode=="base": Rwb=rot2d(yaw_p)
        ll_acc=0.0
        matched_total=0

        for name in ("blue","yellow","orange","big"):
            bank=part["maps"][name]
            MU=bank["mu"]; S=bank["Sigma"]; seen=bank["seen"]; tt=bank["t"]
            OBS=obs_by_color[name]
            if len(OBS)==0: continue

            # seed when empty
            if MU.shape[0] == 0:
                pts_world=[]
                if self.frame_mode == "base":
                    for (z, Rm) in OBS:
                        pw = np.array([x_p, y_p]) + Rwb @ z
                        Rw = Rwb @ Rm @ Rwb.T
                        pts_world.append((pw, Rw))
                else:
                    pts_world = OBS
                if pts_world:
                    mu_list=[]; S_list=[]; t_list=[]; s_list=[]
                    for (pw, Rw) in pts_world:
                        r = float(np.linalg.norm(pw - np.array([x_p, y_p])))
                        R_init = self._inflate_R(Rw, v, yr, dt_pose, r)
                        mu_list.append(pw.reshape(1,2))
                        S_list.append(R_init.reshape(1,2,2))
                        t_list.append(t_z); s_list.append(1)
                    bank["mu"]   = np.vstack(mu_list)
                    bank["Sigma"]= np.vstack(S_list)
                    bank["t"]    = np.array(t_list, float)
                    bank["seen"] = np.array(s_list, int)
                    MU=bank["mu"]; S=bank["Sigma"]; seen=bank["seen"]; tt=bank["t"]

            pairs = self._jcbb(OBS, MU, S, pose_p, dt_pose, v, yr, self.crop2)

            # prevent double-punching the same landmark
            if len(pairs) > 1:
                used_landmarks=set(); filtered=[]
                for (pi, mi, m2_gate, R_eff) in sorted(pairs, key=lambda t: t[2]):
                    if mi in used_landmarks: continue
                    used_landmarks.add(mi); filtered.append((pi, mi, m2_gate, R_eff))
                pairs=filtered

            used_obs=set([pi for (pi,_,_,_) in pairs])
            matched_total += len(pairs)

            # EKF updates
            for (pi, mi, _m2_gate, R_eff) in pairs:
                z, Rm = OBS[pi]
                if self.frame_mode=="base":
                    z_w = np.array([x_p,y_p]) + Rwb @ z; zhat = MU[mi]; innov = z_w - zhat
                else:
                    zhat = MU[mi]; innov = z - zhat
                St = S[mi] + R_eff
                if not np.isfinite(St).all() or not np.isfinite(innov).all(): continue
                Sinv, logdet = safe_inv_and_logdet(St)
                K = S[mi] @ Sinv
                MU[mi] = (MU[mi] + K @ innov).reshape(2)
                H = np.eye(2)
                S[mi] = spd_joseph_update(S[mi], H, R_eff, K)
                seen[mi] += 1; tt[mi] = t_z
                m2_ll = float(innov.T @ Sinv @ innov)
                ll_acc += -0.5 * (m2_ll + logdet + 2*math.log(2.0*math.pi))

            # births / tentatives
            new_idx=[i for i in range(len(OBS)) if i not in used_obs]
            if new_idx:
                T=part["tentative"][name]
                for i in new_idx:
                    z, Rm = OBS[i]
                    if self.frame_mode=="base":
                        p_world = np.array([x_p,y_p]) + Rwb @ z; Rw = Rwb @ Rm @ Rwb.T
                    else:
                        p_world = z; Rw = Rm
                    merged=False
                    for k,(p,t0,tlast,cnt,SR) in enumerate(T):
                        if np.linalg.norm(p_world-p)<self.merge_radius:
                            T[k]=(0.5*p+0.5*p_world, t0, t_z, cnt+1, SR); merged=True; break
                    if not merged:
                        T.append((p_world, t_z, t_z, 1, Rw))
                keep=[]
                stale_horizon = 5.0*(self._rx_dt_ema if self._rx_dt_ema else 0.6)
                for (p,t0,tlast,cnt,SR) in T:
                    if (t_z-t0)<=self.promote_window and cnt>=self.promote_count:
                        MU,S,seen,tt=self._promote(MU,S,seen,tt,p,SR, v, yr, dt_pose, pose_p, t_z)
                    else:
                        if (t_z-tlast)<stale_horizon: keep.append((p,t0,tlast,cnt,SR))
                part["tentative"][name]=keep

            # write back
            bank["mu"]=MU; bank["Sigma"]=S; bank["seen"]=seen; bank["t"]=tt
            self._dedup_bank(bank)

            # aging: only if not persistent OR not yet credible
            if bank["mu"].shape[0]>0 and (not self.persist):
                alive = (t_z - bank["t"]) <= self.max_age
                bank["mu"]    = bank["mu"][alive]
                bank["Sigma"] = bank["Sigma"][alive]
                bank["seen"]  = bank["seen"][alive]
                bank["t"]     = bank["t"][alive]
            elif bank["mu"].shape[0]>0:
                # if persistent: never drop if seen >= threshold
                keep = (bank["seen"] >= self.persist_seen_th)
                # still allow dropping very flaky singles older than max_age
                maybe = (t_z - bank["t"]) <= self.max_age
                alive = np.logical_or(keep, maybe)
                bank["mu"]    = bank["mu"][alive]
                bank["Sigma"] = bank["Sigma"][alive]
                bank["seen"]  = bank["seen"][alive]
                bank["t"]     = bank["t"][alive]

        if matched_total==0:
            return math.log(self.like_floor)
        return ll_acc

    def _promote(self, MU, S, seen, tt, p_world, Rw_meas, v, yr, dt, pose_p, t_now):
        x_p,y_p,_ = pose_p
        r = float(np.linalg.norm(p_world - np.array([x_p,y_p])))
        R_init = self._inflate_R(Rw_meas, v, yr, dt, r)

        if MU.shape[0] > 0:
            m2_best, idx_best = 1e18, -1
            for i in range(MU.shape[0]):
                m2i = mahal2(MU[i], S[i], p_world, R_init)
                if m2i < m2_best: m2_best, idx_best = m2i, i
            if m2_best < self.merge_m2_gate:
                St = S[idx_best] + R_init
                if not np.isfinite(St).all(): return MU,S,seen,tt
                Sinv, _ = safe_inv_and_logdet(St)
                K = S[idx_best] @ Sinv
                innov = (p_world - MU[idx_best]).reshape(2)
                MU[idx_best] = (MU[idx_best] + K @ innov).reshape(2)
                H = np.eye(2)
                S[idx_best] = spd_joseph_update(S[idx_best], H, R_init, K)
                seen[idx_best] += 1
                tt[idx_best] = t_now
                return MU,S,seen,tt
            # block births too close to existing
            for i in range(MU.shape[0]):
                if mahal2(MU[i], S[i], p_world, R_init) < (0.75 * self.merge_m2_gate):
                    return MU,S,seen,tt

        if MU.shape[0]==0:
            MU = p_world.reshape(1,2)
            S  = R_init.reshape(1,2,2)
            seen = np.array([1],int)
            tt = np.array([t_now],float)
            return MU,S,seen,tt

        MU = np.vstack([MU, p_world.reshape(1,2)])
        S  = np.vstack([S, R_init.reshape(1,2,2)])
        seen = np.concatenate([seen, np.array([1],int)])
        tt = np.concatenate([tt, np.array([t_now],float)])
        return MU,S,seen,tt

    def _dedup_bank(self, bank):
        MU = bank["mu"]; S = bank["Sigma"]; seen = bank["seen"]; tt = bank["t"]
        n = MU.shape[0]
        if n <= 1: return
        idxs = list(range(n))
        merged_mu=[]; merged_S=[]; merged_seen=[]; merged_tt=[]
        visited = set()
        for i in idxs:
            if i in visited: continue
            cluster=[i]; visited.add(i)
            for j in idxs:
                if j in visited: continue
                if mahal2(MU[i], S[i], MU[j], S[j]) < self.merge_m2_gate:
                    cluster.append(j); visited.add(j)
            Info = np.zeros((2,2)); Iu = np.zeros(2); ssum=0; tmax=0.0
            for k in cluster:
                try: Sk_inv = np.linalg.inv(S[k])
                except np.linalg.LinAlgError: Sk_inv = np.linalg.pinv(S[k])
                Info += Sk_inv; Iu += Sk_inv @ MU[k]
                ssum += seen[k]; tmax = max(tmax, tt[k])
            try: Pm = np.linalg.inv(Info)
            except np.linalg.LinAlgError: Pm = np.linalg.pinv(Info)
            mum = Pm @ Iu
            merged_mu.append(mum.reshape(1,2))
            merged_S.append(Pm.reshape(1,2,2))
            merged_seen.append(int(ssum))
            merged_tt.append(float(tmax))
        bank["mu"]   = np.vstack(merged_mu) if merged_mu else np.zeros((0,2))
        bank["Sigma"]= np.vstack(merged_S)  if merged_S  else np.zeros((0,2,2))
        bank["seen"] = np.array(merged_seen, dtype=int) if merged_seen else np.zeros((0,), dtype=int)
        bank["t"]    = np.array(merged_tt, dtype=float) if merged_tt else np.zeros((0,), dtype=float)

    # -------- persistent global map accumulation --------
    def _accumulate_global_map(self, best, t_now):
        for name in ("blue","yellow","orange","big"):
            gm = self.global_map[name]
            MU = best["maps"][name]["mu"]; S = best["maps"][name]["Sigma"]
            if MU.shape[0]==0: continue
            if gm["mu"].shape[0]==0:
                gm["mu"]=MU.copy(); gm["S"]=S.copy()
                gm["seen"]=np.maximum(1, best["maps"][name]["seen"].copy())
                gm["t"]=np.full((MU.shape[0],), t_now, float)
                continue
            # fuse each best LM into global set via information fusion
            mu_g, S_g, seen_g, t_g = gm["mu"], gm["S"], gm["seen"], gm["t"]
            for i in range(MU.shape[0]):
                mu_i = MU[i]; Si = S[i]
                # find closest global by Mahalanobis
                d_best, j_best = 1e18, -1
                for j in range(mu_g.shape[0]):
                    d = mahal2(mu_g[j], S_g[j], mu_i, Si)
                    if d < d_best: d_best, j_best = d, j
                if d_best < self.merge_m2_gate:
                    # info fusion
                    try: Sg_inv = np.linalg.inv(S_g[j_best])
                    except np.linalg.LinAlgError: Sg_inv = np.linalg.pinv(S_g[j_best])
                    try: Si_inv = np.linalg.inv(Si)
                    except np.linalg.LinAlgError: Si_inv = np.linalg.pinv(Si)
                    Info = Sg_inv + Si_inv
                    try: Pm = np.linalg.inv(Info)
                    except np.linalg.LinAlgError: Pm = np.linalg.pinv(Info)
                    mu = Pm @ (Sg_inv @ mu_g[j_best] + Si_inv @ mu_i)
                    mu_g[j_best] = mu.reshape(2); S_g[j_best] = Pm
                    seen_g[j_best] = min(32767, seen_g[j_best] + 1)
                    t_g[j_best] = t_now
                else:
                    mu_g = np.vstack([mu_g, mu_i.reshape(1,2)])
                    S_g  = np.vstack([S_g,  Si.reshape(1,2,2)])
                    seen_g = np.concatenate([seen_g, np.array([1],int)])
                    t_g = np.concatenate([t_g, np.array([t_now],float)])
            gm["mu"], gm["S"], gm["seen"], gm["t"] = mu_g, S_g, seen_g, t_g

    def _publish_map_persistent(self, t_s):
        msg = ConeArrayWithCovariance()
        msg.header.stamp = rclpy.time.Time(seconds=t_s).to_msg()
        msg.header.frame_id = "map"
        def bank(mu, Sigma):
            out = []
            for i in range(mu.shape[0]):
                cov = 0.5*(Sigma[i] + Sigma[i].T)
                c = ConeWithCovariance()
                c.point = Point(x=float(mu[i,0]), y=float(mu[i,1]), z=0.0)
                c.covariance = [float(cov[0,0]), float(cov[0,1]),
                                float(cov[1,0]), float(cov[1,1])]
                out.append(c)
            return out
        gm = self.global_map
        msg.blue_cones       = bank(gm["blue"]["mu"],   gm["blue"]["S"])
        msg.yellow_cones     = bank(gm["yellow"]["mu"], gm["yellow"]["S"])
        msg.orange_cones     = bank(gm["orange"]["mu"], gm["orange"]["S"])
        msg.big_orange_cones = bank(gm["big"]["mu"],    gm["big"]["S"])
        self.pub_map.publish(msg)

    # -------------------- PF boilerplate --------------------
    def _resample_systematic_deep(self):
        N=self.Np
        positions=(np.arange(N)+np.random.uniform())/N
        indexes=np.zeros(N,int)
        cs=np.cumsum(self.weights); i=j=0
        while i<N:
            if positions[i]<cs[j]: indexes[i]=j; i+=1
            else: j+=1
        new_particles=[]
        for k in indexes:
            src=self.particles[k]
            dst={"pose":src["pose"].copy(),"maps":{},"tentative":{}}
            for name in ("blue","yellow","orange","big"):
                m=src["maps"][name]
                dst["maps"][name]={
                    "mu":m["mu"].copy(),
                    "Sigma":m["Sigma"].copy(),
                    "seen":m["seen"].copy(),
                    "t":m["t"].copy()
                }
                Tsrc=src["tentative"][name]; Tdst=[]
                for (p,t0,tlast,cnt,SR) in Tsrc:
                    Tdst.append((p.copy(), float(t0), float(tlast), int(cnt), SR.copy()))
                dst["tentative"][name]=Tdst
            new_particles.append(dst)
        self.particles=new_particles
        self.weights[:]=1.0/N

# -------------------- main --------------------
def main():
    rclpy.init()
    node=FastSLAMCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__=="__main__":
    main()
