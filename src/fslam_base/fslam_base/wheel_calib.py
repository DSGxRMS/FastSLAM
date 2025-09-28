#!/usr/bin/env python3
# calibrate_wheel_radius.py
import rclpy, math, statistics
from rclpy.node import Node
from eufs_msgs.msg import WheelSpeedsStamped
from nav_msgs.msg import Odometry

class Calib(Node):
    def __init__(self):
        super().__init__("calib_Rw")
        self.ws = None; self.samples=[]
        self.sub_w = self.create_subscription(WheelSpeedsStamped, "/ros_can/wheel_speeds", self.cb_w, 10)
        self.sub_o = self.create_subscription(Odometry, "/ground_truth/odom", self.cb_o, 10)
        self.timer = self.create_timer(0.5, self.report)
    def cb_w(self, m): self.ws = m.speeds
    def cb_o(self, m):
        if not self.ws: return
        vx = m.twist.twist.linear.x; vy = m.twist.twist.linear.y
        v = (vx*vx+vy*vy)**0.5
        rb = float(self.ws.rb_speed); lb = float(self.ws.lb_speed)
        # use rear min; ignore nonsense & turning
        if abs(rb)<5 or abs(lb)<5: return
        omega = (min(rb,lb) * 2*math.pi/60.0)  # rad/s
        if omega < 1e-3: return
        R = v/omega
        # only keep calm segments (low yaw rate/accel would be better; simple here)
        if 0.5 < v < 25.0:
            self.samples.append(R)
    def report(self):
        if len(self.samples)<30: 
            self.get_logger().info(f"collecting… n={len(self.samples)}")
            return
        med = statistics.median(self.samples)
        p95 = statistics.quantiles(self.samples, n=20)[18]  # ~p95
        self.get_logger().info(f"Estimated wheel_radius_m ≈ {med:.3f} (p95 {p95:.3f}); set param wheel_radius_m to median")
        rclpy.shutdown()

def main():
    rclpy.init(); n=Calib()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    rclpy.shutdown()
if __name__=="__main__": main()
