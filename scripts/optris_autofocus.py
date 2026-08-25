#!/usr/bin/env python3
"""Focus helper for the Optris Xi80 (motorised focus) and any other thermal image topic.

Two modes:

  sweep (default) - drive the Xi80 focus motor through its range via the
    <ns>/focus_motor_pos service, score image sharpness at each position
    (median Tenengrad gradient energy over N frames), refine around the best
    coarse position and leave the motor at the sharpest one.

      python3 optris_autofocus.py --ns /emily01/head
      python3 optris_autofocus.py --ns /emily01/head --coarse-step 10 --fine-step 2 --frames 8

  meter - just print the sharpness score of an image topic continuously, for
    lenses that are focused by hand (e.g. the CSI Mars 320): turn the lens ring
    until the number peaks.

      python3 optris_autofocus.py --meter --topic /thermal/image_gray

Point the camera at a static scene with thermal edges (a warm object on a
cooler background, e.g. a hand, a mug, the wok rim) while running. Scores are
only comparable within one run/scene.
"""
import argparse
import csv
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from optris_drivers2.srv import FocusMotorPos
except ImportError:  # meter mode does not need the service type
    FocusMotorPos = None


def to_gray(msg: Image) -> np.ndarray:
    """Return the image as float32 2-D array (temperature or intensity)."""
    buf = bytes(msg.data)
    h, w = msg.height, msg.width
    enc = msg.encoding
    if enc in ("mono16", "16UC1"):
        a = np.frombuffer(buf, dtype=np.uint16).reshape(h, w).astype(np.float32)
    elif enc == "32FC1":
        a = np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
        a[~np.isfinite(a)] = np.nanmedian(a) if np.isfinite(a).any() else 0.0
    elif enc in ("mono8", "8UC1"):
        a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).astype(np.float32)
    elif enc in ("rgb8", "bgr8"):
        a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.float32).mean(axis=2)
    else:
        raise ValueError(f"unsupported encoding {enc}")
    return a


def sharpness(a: np.ndarray) -> float:
    """Tenengrad-style gradient energy, normalised by image contrast so it is
    insensitive to global gain/temperature changes between frames."""
    gx = a[:, 2:] - a[:, :-2]
    gy = a[2:, :] - a[:-2, :]
    g = (gx[1:-1, :] ** 2 + gy[:, 1:-1] ** 2)
    contrast = np.percentile(a, 99) - np.percentile(a, 1)
    if contrast <= 0:
        return 0.0
    return float(np.mean(g)) / float(contrast ** 2) * 1e3


class FocusNode(Node):
    def __init__(self, topic: str, ns: str):
        super().__init__("optris_autofocus")
        self._frames = []
        self._collect = False
        self.create_subscription(Image, topic, self._cb, 5)
        self._cli = None
        if ns is not None and FocusMotorPos is not None:
            self._cli = self.create_client(FocusMotorPos, f"{ns}/focus_motor_pos")

    def _cb(self, msg: Image):
        if self._collect:
            try:
                self._frames.append(to_gray(msg))
            except ValueError as e:
                self.get_logger().error(str(e))
                raise SystemExit(2)

    def grab(self, n: int, timeout: float = 10.0):
        self._frames = []
        self._collect = True
        t0 = time.time()
        while len(self._frames) < n and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._collect = False
        return list(self._frames)

    def score(self, n: int):
        frames = self.grab(n)
        if not frames:
            return None, 0
        s = np.array([sharpness(f) for f in frames])
        return float(np.median(s)), len(frames)

    def set_focus(self, pos: float, timeout: float = 5.0) -> bool:
        if not self._cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {self._cli.srv_name} not available")
        req = FocusMotorPos.Request()
        req.pos = float(pos)
        fut = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        if fut.result() is None:
            raise RuntimeError("focus service call timed out")
        return bool(fut.result().success)


def sweep(node: FocusNode, positions, frames: int, settle: float, log):
    rows = []
    for pos in positions:
        ok = node.set_focus(pos)
        if not ok:
            print(f"focus {pos:6.1f} %: service reported no focus motor", file=sys.stderr)
            return None
        time.sleep(settle)
        node.grab(2)  # flush frames captured while the motor was moving
        s, n = node.score(frames)
        rows.append((pos, s if s is not None else float("nan"), n))
        log(f"  focus {pos:6.1f} %  sharpness {s if s is not None else float('nan'):8.3f}  ({n} frames)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/emily01/head", help="Optris node namespace (default /emily01/head)")
    ap.add_argument("--topic", default=None, help="image topic to score (default <ns>/thermal_image)")
    ap.add_argument("--meter", action="store_true", help="only print sharpness continuously (manual lenses)")
    ap.add_argument("--min", type=float, default=0.0)
    ap.add_argument("--max", type=float, default=100.0)
    ap.add_argument("--coarse-step", type=float, default=10.0)
    ap.add_argument("--fine-step", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=8, help="frames scored per position (median)")
    ap.add_argument("--settle", type=float, default=1.2, help="seconds to wait after moving the motor")
    ap.add_argument("--out", default=None, help="write the sweep table to this CSV")
    args = ap.parse_args()

    topic = args.topic or f"{args.ns}/thermal_image"
    rclpy.init()
    node = FocusNode(topic, None if args.meter else args.ns)
    log = lambda m: print(m, flush=True)

    try:
        if args.meter:
            log(f"sharpness meter on {topic} (Ctrl-C to stop) - turn the lens until the value peaks")
            while rclpy.ok():
                s, n = node.score(max(2, args.frames // 2))
                if s is None:
                    log("  no frames...")
                else:
                    log(f"  sharpness {s:8.3f}   ({n} frames)")
            return 0

        if FocusMotorPos is None:
            print("optris_drivers2 srv not importable - source the workspace", file=sys.stderr)
            return 2
        log(f"scoring {topic}; coarse sweep {args.min}..{args.max} step {args.coarse_step}")
        coarse = list(np.arange(args.min, args.max + 1e-6, args.coarse_step))
        rows = sweep(node, coarse, args.frames, args.settle, log)
        if rows is None:
            return 1
        best = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else -1))
        lo, hi = max(args.min, best[0] - args.coarse_step), min(args.max, best[0] + args.coarse_step)
        log(f"best coarse {best[0]:.1f} % -> fine sweep {lo:.1f}..{hi:.1f} step {args.fine_step}")
        fine = [p for p in np.arange(lo, hi + 1e-6, args.fine_step) if p not in coarse]
        rows2 = sweep(node, fine, args.frames, args.settle, log)
        if rows2 is None:
            return 1
        allrows = sorted(rows + rows2)
        best = max(allrows, key=lambda r: (r[1] if np.isfinite(r[1]) else -1))
        node.set_focus(best[0])
        time.sleep(args.settle)
        s, n = node.score(args.frames)
        log(f"\nBEST focus = {best[0]:.1f} %  (sharpness {best[1]:.3f}; re-check {s if s is not None else float('nan'):.3f})")
        log(f"persist it: set <focus>{best[0]:.0f}</focus> in the xi80 xml template used by the launch")
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["focus_pct", "sharpness", "frames"])
                w.writerows(allrows)
            log(f"table written to {args.out}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
