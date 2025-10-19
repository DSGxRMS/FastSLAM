#!/usr/bin/env python3
# teleop_ackermann_tk.py
#
# Arrow-key teleop for EUFSIM (Ackermann accel-first) using a tiny Tk window.
# Focus the window titled "Ackermann Teleop" and press:
#   ↑ : accel + (throttle)         |  ↓ : accel − (brake)
#   ← : steer left                 |  → : steer right
#   SPACE : accel = 0              |  Z : steering = 0
#   X : full brake (−brake_limit)  |  R : reset accel & steer to 0
#   Q / Esc : quit
#
# Publishes: /cmd (ackermann_msgs/AckermannDriveStamped)
# QoS: BEST_EFFORT by default (flip qos_best_effort:=false for RELIABLE)

import math
import threading
import time
import tkinter as tk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped


class TeleopAckermannTk(Node):
    def __init__(self):
        super().__init__("teleop_ackermann_tk", automatically_declare_parameters_from_overrides=True)

        # -------- Params --------
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("qos_best_effort", True)
        self.declare_parameter("publish_hz", 40.0)

        # steering & accel behavior
        self.declare_parameter("steer_step_rad", 0.04)       # per key press
        self.declare_parameter("max_steer_rad", 0.55)
        self.declare_parameter("steer_rate_limit_radps", 1.2)
        self.declare_parameter("steering_right_sign", -1.0)  # -1.0: right turn negative (common)

        self.declare_parameter("accel_step_mps2", 0.1)
        self.declare_parameter("accel_limit_pos", 0.3)       # +0.3 (throttle)
        self.declare_parameter("brake_limit_neg", 1.0)       # -1.0 (brake)

        # Optional idle decay back to zero (feels safer)
        self.declare_parameter("idle_decay_enable", True)
        self.declare_parameter("idle_decay_after_s", 0.8)
        self.declare_parameter("idle_decay_rate_accel", 0.5)  # m/s^2 per second toward 0
        self.declare_parameter("idle_decay_rate_steer", 1.5)  # rad per second toward 0

        P = self.get_parameter
        self.cmd_topic   = str(P("cmd_topic").value)
        self.qos_best    = bool(P("qos_best_effort").value)
        self.pub_hz      = float(P("publish_hz").value)

        self.steer_step  = float(P("steer_step_rad").value)
        self.max_steer   = float(P("max_steer_rad").value)
        self.steer_rate_limit = float(P("steer_rate_limit_radps").value)
        self.right_sign  = float(P("steering_right_sign").value)

        self.accel_step  = float(P("accel_step_mps2").value)
        self.a_pos_lim   = float(P("accel_limit_pos").value)
        self.a_neg_lim   = float(P("brake_limit_neg").value)

        self.decay_en    = bool(P("idle_decay_enable").value)
        self.decay_after = float(P("idle_decay_after_s").value)
        self.decay_accel = float(P("idle_decay_rate_accel").value)
        self.decay_steer = float(P("idle_decay_rate_steer").value)

        # -------- State --------
        self.delta_phys = 0.0           # left-positive (physical), converted with right_sign on publish
        self.accel_cmd = 0.0
        self._last_pub_delta = 0.0
        self._last_key_time = time.time()

        # -------- ROS I/O --------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.qos_best
                        else QoSReliabilityPolicy.RELIABLE
        )
        self.pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self.create_timer(1.0 / max(1.0, self.pub_hz), self._on_timer)

        # -------- Tk window --------
        self._make_window()

    # ---------- Tk UI ----------
    def _make_window(self):
        self.root = tk.Tk()
        self.root.title("Ackermann Teleop")
        self.root.geometry("360x180")

        self.lbl = tk.Label(
            self.root,
            text=self._status_text(),
            font=("DejaVu Sans Mono", 12),
            justify="left"
        )
        self.lbl.pack(padx=10, pady=8, anchor="w")

        help_text = (
            "↑ accel+   ↓ accel-   ← steer L   → steer R\n"
            "SPACE accel=0    Z steer=0    X full brake\n"
            "R reset both     Q or Esc quit\n"
            f"Topic: {self.cmd_topic}  QoS: {'BEST_EFFORT' if self.qos_best else 'RELIABLE'}"
        )
        tk.Label(self.root, text=help_text, justify="left").pack(padx=10, anchor="w")

        # Bind keys
        self.root.bind("<Up>",    lambda e: self._accel_plus())
        self.root.bind("<Down>",  lambda e: self._accel_minus())
        self.root.bind("<Left>",  lambda e: self._steer_left())
        self.root.bind("<Right>", lambda e: self._steer_right())
        self.root.bind("<space>", lambda e: self._accel_zero())
        self.root.bind("z",       lambda e: self._steer_zero())
        self.root.bind("Z",       lambda e: self._steer_zero())
        self.root.bind("x",       lambda e: self._full_brake())
        self.root.bind("X",       lambda e: self._full_brake())
        self.root.bind("r",       lambda e: self._reset_both())
        self.root.bind("R",       lambda e: self._reset_both())
        self.root.bind("q",       lambda e: self._quit())
        self.root.bind("Q",       lambda e: self._quit())
        self.root.bind("<Escape>",lambda e: self._quit())

        # Drive Tk in main thread; spin ROS in background
        t = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        t.start()

        # Periodically refresh status text
        self._ui_tick()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()

    def _ui_tick(self):
        self.lbl.config(text=self._status_text())
        # Re-run every ~60 ms
        if rclpy.ok():
            self.root.after(60, self._ui_tick)

    def _status_text(self):
        return (f"steer (rad): {self.delta_phys:+.3f}  "
                f"accel (m/s²): {self.accel_cmd:+.3f}\n"
                f"steer_step={self.steer_step:.3f}  accel_step={self.accel_step:.2f}\n"
                f"limits: steer±{self.max_steer:.2f}  +{self.a_pos_lim:.1f}/-{self.a_neg_lim:.1f} m/s²")

    # ---------- Key actions ----------
    def _mark_key(self):
        self._last_key_time = time.time()

    def _steer_left(self):
        self._mark_key()
        self.delta_phys = self._clip(self.delta_phys + self.steer_step, -self.max_steer, self.max_steer)

    def _steer_right(self):
        self._mark_key()
        self.delta_phys = self._clip(self.delta_phys - self.steer_step, -self.max_steer, self.max_steer)

    def _steer_zero(self):
        self._mark_key()
        self.delta_phys = 0.0

    def _accel_plus(self):
        self._mark_key()
        self.accel_cmd = self._clip(self.accel_cmd + self.accel_step, -self.a_neg_lim, self.a_pos_lim)

    def _accel_minus(self):
        self._mark_key()
        self.accel_cmd = self._clip(self.accel_cmd - self.accel_step, -self.a_neg_lim, self.a_pos_lim)

    def _accel_zero(self):
        self._mark_key()
        self.accel_cmd = 0.0

    def _full_brake(self):
        self._mark_key()
        self.accel_cmd = -self.a_neg_lim

    def _reset_both(self):
        self._mark_key()
        self.delta_phys = 0.0
        self.accel_cmd = 0.0

    def _quit(self):
        try:
            self.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            try:
                self.root.destroy()
            except Exception:
                pass

    # ---------- ROS timer ----------
    def _on_timer(self):
        # Idle decay back to zero
        dt = 1.0 / max(1.0, self.pub_hz)
        if self.decay_en and (time.time() - self._last_key_time) > self.decay_after:
            self.accel_cmd = self._decay_to_zero(self.accel_cmd, self.decay_accel * dt)
            self.delta_phys = self._decay_to_zero(self.delta_phys, self.decay_steer * dt)

        # Steering slew-limit
        dmax = self.steer_rate_limit * dt
        delta_phys = self._clamp_slew(self._last_pub_delta, self.delta_phys, dmax)
        self._last_pub_delta = delta_phys

        # Convert to sim sign convention and clamp accel asymmetrically
        delta_cmd = self.right_sign * delta_phys
        accel_cmd = self._clip(self.accel_cmd, -self.a_neg_lim, self.a_pos_lim)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(delta_cmd)
        msg.drive.acceleration   = float(accel_cmd)
        msg.drive.speed          = 0.0
        self.pub.publish(msg)

    # ---------- utils ----------
    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _decay_to_zero(x, step):
        if x > 0.0: x = max(0.0, x - step)
        elif x < 0.0: x = min(0.0, x + step)
        return x

    @staticmethod
    def _clamp_slew(prev, target, dmax):
        if target > prev + dmax: return prev + dmax
        if target < prev - dmax: return prev - dmax
        return target


def main():
    rclpy.init()
    TeleopAckermannTk()  # Tk mainloop runs inside __init__


if __name__ == "__main__":
    main()
