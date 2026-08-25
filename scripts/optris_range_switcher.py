#!/usr/bin/env python3
"""Automatic temperature-range switching for the Optris Xi80 (optris_drivers2).

The Xi80 has no single calibrated range that covers a wok: 0..250 C saturates
at 274.9 C, while 150..900 C reports a constant "under range" value for every
pixel below 150 C. This node watches the thermal image and flips the imager
between a LOW and a HIGH range with hysteresis, blanking the frames captured
after the switch.

MEASURED 2026-08-25 (Xi80 21034332, libirimager 8.9.3): after a runtime range
switch the radiometric output is wrong for ~50-60 s - it jumps by about +30 C,
decays slowly and only snaps back to the true value at the next automatic flag
cycle (~50 s later). Forcing a flag right after the switch does NOT help. So
this node blanks for at least `blank_min_s` (45 s) and then until the next
flag cycle completes (capped at `blank_max_s`), and refuses to switch again
for `min_dwell_s` (120 s). Every switch therefore costs about a minute of
data; switch only at phase boundaries (pre-heat -> cooking), not continuously.
If you need both ranges at once, use two cameras (one fixed per range).

It also publishes a merged temperature image in which every pixel the current
range cannot measure is NaN, so downstream code never sees the 124.9 C
sentinel or a saturated 274.9 C as a real temperature.

Topics (relative to --ns):
  sub  thermal_image            mono16, T[C] = (v - 1000) / 10          (driver)
  sub  flag_state               optris_drivers2/Flag                      (driver)
  pub  thermal_image_valid      32FC1 C, NaN where invalid / while switching
  pub  thermal_range            std_msgs/String, JSON:
         {"min":0,"max":250,"state":"steady|blanking|dwell","p_hi":123.4,"valid_frac":0.98,"sat_frac":0.0}

Range control goes through the imager node's parameters
(temperature_range_min/max, live-settable since 2026-08-25), so the active
range is always visible with `ros2 param get <ns>/optris_imager temperature_range_max`
and an imager restart (supervisor USB reset) is detected by polling them.

Decision rule (evaluated per frame on the ROI, needs `hold_frames` consecutive
frames, then respects `min_dwell_s`):
  LOW  -> HIGH  when  percentile(p_hi) of valid ROI pixels >= up_temp_c
                 or   saturated fraction of ROI          >= up_sat_frac
  HIGH -> LOW   when  percentile(p_hi) of valid ROI pixels <  down_temp_c
                 or   no valid pixel at all (everything < 150 C)

Run:
  python3 optris_range_switcher.py --ns /emily01/head
  python3 optris_range_switcher.py --ns /emily01/head --roi 40,40,25   # cx,cy,r in pixels
  python3 optris_range_switcher.py --ns /emily01/head --up-temp 30 --down-temp 25  # bench test
Or from the launch file: ros2 launch optris_drivers2 optris_node.launch.py ... auto_range:=true
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from sensor_msgs.msg import Image
from std_msgs.msg import String

from optris_drivers2.msg import Flag

FLAG_OPEN = 0  # evo::EnumFlagState: Open, Close, Opening, Closing, Error, Initializing

# Xi80 factory ranges. Saturation shows up as range_max + 24.9 (274.9 for 0..250,
# 124.9 for -20..100); under-range in 150..900 is a constant 124.9 sentinel.
SAT_MARGIN_C = 24.5


class RangeSwitcher(Node):
    def __init__(self, a):
        super().__init__("optris_range_switcher")
        self.a = a
        self.low = (a.low_min, a.low_max)
        self.high = (a.high_min, a.high_max)
        self.imager = f"{a.ns}/optris_imager"
        self.cur = None            # (min, max) currently active on the imager
        self.state = "init"        # init | steady | blanking | dwell
        self.blank_until = 0.0
        self.dwell_until = 0.0
        self.saw_flag_closed = False
        self.hold = 0
        self.last_stats = {}
        self.frames = 0
        self._get_pending = None
        self._switch_t0 = 0.0
        self.blank_min_until = 0.0

        self.pub_img = self.create_publisher(Image, f"{a.ns}/thermal_image_valid", 5)
        self.pub_rng = self.create_publisher(String, f"{a.ns}/thermal_range", 5)
        self.create_subscription(Image, f"{a.ns}/thermal_image", self.on_image, 5)
        self.create_subscription(Flag, f"{a.ns}/flag_state", self.on_flag, 5)
        self.cli_get = self.create_client(GetParameters, f"{self.imager}/get_parameters")
        self.cli_set = self.create_client(SetParametersAtomically, f"{self.imager}/set_parameters_atomically")
        self.create_timer(a.poll_s, self.poll_range)
        self.create_timer(1.0, self.publish_state)
        self.get_logger().info(
            f"ranges LOW={self.low} HIGH={self.high}; up>={a.up_temp}C or sat>={a.up_sat_frac:.0%}, "
            f"down<{a.down_temp}C; p{a.percentile}; hold {a.hold_frames} frames; dwell {a.min_dwell_s}s; "
            f"ROI={'full frame' if a.roi is None else a.roi}")

    # ---------------- imager parameter access (all async: never spin inside a callback) ----------------
    def _param(self, name, v):
        return Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(v)))

    def poll_range(self):
        """Read the imager's active range; detects imager restarts (range falls
        back to its startup default) and does the initial sync."""
        now = time.monotonic()
        if self._get_pending is not None:
            if now - self._get_pending > 3.0:
                self.get_logger().warn("imager get_parameters timed out")
                self._get_pending = None
            return
        if not self.cli_get.service_is_ready():
            if self.state != "init":
                self.get_logger().warn("imager parameter service gone (restarting?) - waiting")
                self.state, self.cur = "init", None
            return
        req = GetParameters.Request(names=["temperature_range_min", "temperature_range_max"])
        self._get_pending = now
        self.cli_get.call_async(req).add_done_callback(self._on_range_read)

    def _on_range_read(self, fut):
        self._get_pending = None
        r = fut.result()
        if r is None or len(r.values) != 2 or r.values[0].type != ParameterType.PARAMETER_INTEGER:
            self.get_logger().warn("could not read imager range")
            return
        rng = (int(r.values[0].integer_value), int(r.values[1].integer_value))
        if self.state in ("switching", "blanking"):
            return  # our own change is in flight
        if self.cur is None:
            self.cur = rng
            if rng not in (self.low, self.high):
                self.get_logger().warn(f"imager range {rng} is neither LOW nor HIGH - switching to LOW")
                self.switch(self.low)
            else:
                self.state = "steady"
                self.get_logger().info(f"synced: imager range is {rng}")
        elif rng != self.cur:
            self.get_logger().warn(f"imager range changed externally {self.cur} -> {rng} (restart?) - resyncing")
            self.cur, self.state, self.hold = rng, "steady", 0

    # ---------------- switching ----------------
    def switch(self, rng):
        if self.state == "switching":
            return
        self.get_logger().info(f"SWITCH {self.cur} -> {rng}  (stats {self.last_stats})")
        if not self.cli_set.service_is_ready():
            self.get_logger().error("imager set_parameters service unavailable")
            return
        self.state = "switching"
        self.saw_flag_closed = False
        self._switch_t0 = time.monotonic()
        req = SetParametersAtomically.Request(
            parameters=[self._param("temperature_range_min", rng[0]), self._param("temperature_range_max", rng[1])])
        self.cli_set.call_async(req).add_done_callback(lambda f, rng=rng: self._on_switch_done(f, rng))

    def _on_switch_done(self, fut, rng):
        now = time.monotonic()
        r = fut.result()
        self.dwell_until = now + self.a.min_dwell_s
        self.hold = 0
        if r is None or not r.result.successful:
            self.get_logger().error(f"range switch to {rng} rejected: {r.result.reason if r else 'no response'}")
            self.state = "dwell"
            return
        self.cur = rng
        self.state = "blanking"
        self.saw_flag_closed = False
        self.blank_min_until = now + self.a.blank_min_s
        self.blank_until = now + self.a.blank_max_s
        self.get_logger().info(f"range is now {rng}; blanking >= {self.a.blank_min_s}s then until the next flag cycle (max {self.a.blank_max_s}s)")

    def on_flag(self, msg: Flag):
        if self.state != "blanking":
            return
        now = time.monotonic()
        if now < self.blank_min_until:
            return  # the flag the SDK forces 1 s after the switch does not fix the readings
        if msg.flag_state != FLAG_OPEN:
            self.saw_flag_closed = True
        elif self.saw_flag_closed:
            # first natural flag cycle after the minimum blanking: readings snap
            # back to normal right after it
            self.blank_until = min(self.blank_until, now + self.a.blank_after_flag_s)

    # ---------------- per frame ----------------
    def on_image(self, msg: Image):
        if msg.encoding != "mono16":
            self.get_logger().error(f"unexpected encoding {msg.encoding}")
            return
        self.frames += 1
        now = time.monotonic()
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        t = (raw.astype(np.float32) - 1000.0) / 10.0

        if self.cur is None:
            return  # not synced yet; publish nothing rather than something wrong
        if self.state == "switching":
            if now - self._switch_t0 > 5.0:
                self.get_logger().error("range switch got no response in 5 s - resyncing")
                self.state, self.cur = "init", None
            return

        rmin, rmax = self.cur
        invalid = (t < rmin - 0.05) | (t >= rmax + SAT_MARGIN_C) | (raw == 0)
        sat = t >= rmax + SAT_MARGIN_C
        blanking = self.state == "blanking" and now < self.blank_until
        if self.state == "blanking" and not blanking:
            self.state = "dwell" if now < self.dwell_until else "steady"

        out = t.copy()
        out[invalid] = np.nan
        if blanking:
            out[:] = np.nan
        self.publish_image(out, msg)

        # ROI statistics
        roi = self.roi_mask(msg.height, msg.width)
        v = t[roi & ~invalid]
        n_roi = int(roi.sum())
        valid_frac = float(v.size) / n_roi if n_roi else 0.0
        sat_frac = float((sat & roi).sum()) / n_roi if n_roi else 0.0
        p_hi = float(np.percentile(v, self.a.percentile)) if v.size else float("nan")
        self.last_stats = {"p_hi": round(p_hi, 1) if not math.isnan(p_hi) else None,
                           "valid_frac": round(valid_frac, 3), "sat_frac": round(sat_frac, 3)}

        if blanking:
            return
        if self.state == "dwell":
            if now < self.dwell_until:
                return
            self.state = "steady"

        want = None
        if self.cur == self.low:
            if sat_frac >= self.a.up_sat_frac or (not math.isnan(p_hi) and p_hi >= self.a.up_temp):
                want = self.high
        elif self.cur == self.high:
            if math.isnan(p_hi) or p_hi < self.a.down_temp:
                want = self.low
        if want is None:
            self.hold = 0
            return
        self.hold += 1
        if self.hold >= self.a.hold_frames:
            self.switch(want)

    def roi_mask(self, h, w):
        if self.a.roi is None:
            return np.ones((h, w), dtype=bool)
        cx, cy, r = self.a.roi
        yy, xx = np.ogrid[:h, :w]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    def publish_image(self, arr: np.ndarray, src: Image):
        m = Image()
        m.header = src.header
        m.height, m.width = arr.shape
        m.encoding = "32FC1"
        m.is_bigendian = 0
        m.step = m.width * 4
        m.data = arr.astype(np.float32).tobytes()
        self.pub_img.publish(m)

    def publish_state(self):
        d = {"min": self.cur[0] if self.cur else None, "max": self.cur[1] if self.cur else None,
             "state": self.state, **self.last_stats}
        self.pub_rng.publish(String(data=json.dumps(d)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/emily01/head")
    ap.add_argument("--low-min", type=int, default=0)
    ap.add_argument("--low-max", type=int, default=250)
    ap.add_argument("--high-min", type=int, default=150)
    ap.add_argument("--high-max", type=int, default=900)
    ap.add_argument("--up-temp", type=float, default=240.0, help="LOW->HIGH when ROI percentile >= this (C)")
    ap.add_argument("--up-sat-frac", type=float, default=0.02, help="LOW->HIGH when this fraction of ROI is saturated")
    ap.add_argument("--down-temp", type=float, default=200.0, help="HIGH->LOW when ROI percentile < this (C)")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--roi", default=None, help="cx,cy,r in pixels (default: whole frame)")
    ap.add_argument("--hold-frames", type=int, default=20, help="consecutive frames the condition must hold (~1 s at 18 Hz)")
    ap.add_argument("--min-dwell-s", type=float, default=120.0, help="no new switch for this long after a switch")
    ap.add_argument("--blank-min-s", type=float, default=45.0, help="minimum blanking after a switch (measured transient ~50 s)")
    ap.add_argument("--blank-max-s", type=float, default=75.0, help="max blanking if no flag cycle is seen after blank-min")
    ap.add_argument("--blank-after-flag-s", type=float, default=1.0, help="extra blanking after the flag re-opens")
    ap.add_argument("--poll-s", type=float, default=3.0, help="how often to re-read the imager range")
    a = ap.parse_args()
    if a.roi:
        a.roi = tuple(int(x) for x in a.roi.split(","))
        if len(a.roi) != 3:
            ap.error("--roi needs cx,cy,r")
    if a.down_temp >= a.up_temp:
        ap.error("--down-temp must be below --up-temp (hysteresis)")
    if a.blank_max_s < a.blank_min_s:
        ap.error("--blank-max-s must be >= --blank-min-s")
    rclpy.init()
    node = RangeSwitcher(a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
