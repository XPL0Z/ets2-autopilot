"""ETS2 autopilot: capture the screen, predict steering with PilotNet, and send it to the game
through a virtual Xbox controller. Windows only.

Run it from a terminal opened ON the Windows PC (not over SSH: an SSH session cannot
see the desktop and would capture a black image).

Modes:
    python drive.py --check       grab one frame and save check_full.jpg / check_crop.jpg
                                  to verify that the crop lands on the road
    python drive.py --pad         plug in the virtual controller and sweep the stick, so it can
                                  be selected in ETS2 under Options > Controls
    python drive.py --calibrate   measure how the in-game steering reacts to the stick, then test
                                  the closed-loop controller (needs telemetry; truck stopped)
    python drive.py               drive. F8 = toggle the autopilot, F10 = quit

Expected game settings (the same as when the dataset was recorded):
    interior camera (key 1), FOV 71, HUD mirrors shown, 16:9 resolution,
    fullscreen on the primary monitor, LANE ASSIST OFF (otherwise the game drives,
    not the model), cruise control for the throttle.
"""

import argparse
import csv
import ctypes
import json
import mmap
import struct
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import torch

from train import PilotNet

# Same preprocessing as prepare.py: frame resized to 1920x1080, road band, 256x64, RGB
CROP_Y = (290, 635)
CROP_X = (240, 1680)
OUT_W, OUT_H = 256, 64

# SCS telemetry (RenCloud/scs-sdk-plugin, PLUGIN_REVID 12): same offsets as prepare.py
TELEMETRY_NAME = "Local\\SCSTelemetry"
TELEMETRY_SIZE = 32 * 1024
OFF_SDK_ACTIVE, OFF_PAUSED, OFF_REVID, OFF_TRUCK_F = 0, 4, 40, 948

VK_F8, VK_F10 = 0x77, 0x79
CALIB_FILE = Path("calibration.json")


# ---------------------------------------------------------------- capture / model

def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if (w, h) != (1920, 1080):
        frame_bgr = cv2.resize(frame_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
    crop = frame_bgr[CROP_Y[0]:CROP_Y[1], CROP_X[0]:CROP_X[1]]
    small = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class Screen:
    def __init__(self, monitor_index: int):
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self.mon = self.sct.monitors[monitor_index]
        w, h = self.mon["width"], self.mon["height"]
        if abs(w / h - 16 / 9) > 0.01:
            print(f"! screen {w}x{h} is not 16:9: set the game to a 16:9 resolution "
                  "(e.g. 1920x1080), otherwise the crop will not land on the road")

    def grab(self) -> np.ndarray:
        return np.asarray(self.sct.grab(self.mon))[:, :, :3]  # BGRA -> BGR


def load_model(path: Path, device: torch.device) -> PilotNet:
    ckpt = torch.load(path, map_location=device)
    model = PilotNet().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"model {path} (epoch {ckpt.get('epoch')}, turn MAE {ckpt.get('mae_turns', float('nan')):.3f}) on {device}")
    return model


@torch.no_grad()
def predict(model, img_rgb: np.ndarray, device) -> float:
    x = torch.from_numpy(img_rgb).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(model(x).item())


# ---------------------------------------------------------------- telemetry

class Telemetry:
    """Read the SCS plugin shared memory. available=False if the plugin is not loaded."""

    def __init__(self):
        try:
            self.mm = mmap.mmap(-1, TELEMETRY_SIZE, tagname=TELEMETRY_NAME, access=mmap.ACCESS_READ)
        except OSError:
            self.mm = None

    @property
    def available(self) -> bool:
        if self.mm is None:
            return False
        revid, = struct.unpack_from("<I", self.mm, OFF_REVID)
        return self.mm[OFF_SDK_ACTIVE] != 0 and revid == 12

    def read(self) -> dict:
        speed, = struct.unpack_from("<f", self.mm, OFF_TRUCK_F)
        steer, = struct.unpack_from("<f", self.mm, OFF_TRUCK_F + 24)  # gameSteer
        return {"speed": speed, "steer": steer, "paused": self.mm[OFF_PAUSED] != 0}


# ---------------------------------------------------------------- virtual controller

class Pad:
    def __init__(self):
        try:
            import vgamepad as vg
        except ImportError:
            sys.exit("vgamepad missing: pip install vgamepad (accept the ViGEmBus driver install)")
        self.pad = vg.VX360Gamepad()
        # stick "rest" position: 0 by default, or the "wheel straight" position
        # measured by the calibration when ETS2 has shifted the axis
        self.neutral = 0.0

    def steer(self, x: float):
        self.pad.left_joystick_float(x_value_float=float(np.clip(x, -1, 1)), y_value_float=0.0)
        self.pad.update()

    def release(self):
        self.steer(self.neutral)


class SteerController:
    """Closed-loop steering controller.

    At each step: error = target steering (gameSteer units, the same as the labels) - actual
    steering read from telemetry, then stick = neutral + sign x (dead_zone + kp x error).
    This works whether the game treats the stick as a steering ANGLE (regular gamepad) or as a
    rotation SPEED (stick handled like a mouse, which is what was measured on this PC):
    it keeps pushing in the right direction until the error is gone.

    Parameters measured by --calibrate (calibration.json):
        neutral  stick position where the wheel does not move
        sign     +1 if raising the stick increases gameSteer, -1 otherwise
        dz       half-width of the dead zone around neutral (the wheel does not move there)
        kp       gain: stick deviation per unit of error
        span     maximum allowed deviation (beyond it the wheel turns at full speed)"""

    TOLERANCE = 0.01  # error below which no correction is made (avoids jitter)

    def __init__(self):
        if not CALIB_FILE.exists():
            sys.exit(f"no {CALIB_FILE}: run python drive.py --calibrate first")
        data = json.loads(CALIB_FILE.read_text())
        if data.get("mode") != "closed_loop":
            sys.exit(f"{CALIB_FILE} uses an old format: run python drive.py --calibrate again")
        self.neutral = float(data["neutral"])
        self.sign = float(data["sign"])
        self.dz = float(data["dz"])
        self.kp = float(data["kp"])
        self.span = float(data["span"])
        print(f"closed-loop steering: neutral {self.neutral:+.3f}, sign {self.sign:+.0f}, "
              f"dead zone ±{self.dz:.3f}, kp {self.kp:.2f}, max deviation {self.span:.2f}")

    def __call__(self, target: float, current: float) -> float:
        err = target - current
        if not np.isfinite(err) or abs(err) < self.TOLERANCE:
            return self.neutral
        dev = min(self.dz + self.kp * abs(err), self.span)
        return float(np.clip(self.neutral + self.sign * np.sign(err) * dev, -1, 1))


def key_pressed(vk: int) -> bool:
    # low bit: key pressed since the last call
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x1)


# ---------------------------------------------------------------- modes

def run_check(args):
    screen = Screen(args.monitor)
    print("capturing in 3 s: switch to the game window...")
    time.sleep(3)
    frame = screen.grab()
    full = cv2.resize(frame, (1920, 1080)) if frame.shape[:2] != (1080, 1920) else frame.copy()
    cv2.rectangle(full, (CROP_X[0], CROP_Y[0]), (CROP_X[1], CROP_Y[1]), (0, 0, 255), 4)
    cv2.imwrite("check_full.jpg", cv2.resize(full, (960, 540)))
    cv2.imwrite("check_crop.jpg", cv2.cvtColor(preprocess(frame), cv2.COLOR_RGB2BGR))
    print("wrote check_full.jpg (red rectangle = area seen by the model) and check_crop.jpg")
    print("the rectangle must cover the road, ABOVE the steering wheel and BELOW the mirrors")


def run_pad(args):
    """Keep the virtual controller plugged in and sweep the left stick, so ETS2 detects it
    and it can be selected in Options > Controls (the "move the stick" wizard)."""
    pad = Pad()
    print("Virtual Xbox controller plugged in. The left stick sweeps left/right.")
    print("In ETS2: Options > Controls > pick the Xbox 360 controller (not the DualSense),")
    print("then check that the wheel moves on its own. Ctrl+C to stop.")
    t0 = time.perf_counter()
    try:
        while True:
            x = 0.8 * np.sin((time.perf_counter() - t0) * 2 * np.pi / 4)  # 4 s period
            pad.steer(x)
            print(f"\rstick {x:+.2f} ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pad.release()


def calibrate_core(steer, read, sleep, now, log=print) -> dict:
    """Measure how the game reacts to the stick. The I/O functions are injected so this can be
    tested in simulation: steer(x) moves the stick, read() -> gameSteer, sleep(s), now() -> seconds."""

    def settle(s, timeout=8.0):
        steer(s)
        t0, last, stable_since = now(), None, None
        while now() - t0 < timeout:
            g = read()
            if last is not None and abs(g - last) < 0.002:
                stable_since = stable_since if stable_since is not None else now()
                if now() - stable_since > 0.6:
                    return g
            else:
                stable_since = None
            last = g
            sleep(0.05)
        return last

    # 1) coarse sweep: response direction and useful stick range
    coarse = [round(float(x), 2) for x in np.linspace(-1, 1, 21)]
    cg = []
    log("coarse sweep:")
    for s in coarse:
        cg.append(settle(s))
        log(f"  stick {s:+.2f} -> gameSteer {cg[-1]:+.3f}")
    cg = np.array(cg)
    if np.ptp(cg) < 0.05:
        return {"error": "the wheel (almost) did not move: the game is not listening to the virtual controller",
                "coarse": list(zip(coarse, cg.tolist()))}
    sign = 1.0 if cg[-1] > cg[0] else -1.0
    # useful range: between the last saturated reading on the low stick side and the first on the high side
    low_side, high_side = cg[0], cg[-1]
    lo_i = max(i for i in range(len(cg)) if abs(cg[i] - low_side) < 0.05)
    hi_i = min(i for i in range(len(cg)) if abs(cg[i] - high_side) < 0.05)
    if hi_i <= lo_i:
        lo_i, hi_i = max(hi_i - 1, 0), min(lo_i + 1, len(cg) - 1)
    # widen by 0.2 on each side: if the stick controls a SPEED, the wheel stays saturated inside
    # the dead zone and the transition seen in the coarse sweep is not centered on neutral
    lo, hi = coarse[max(lo_i - 2, 0)], coarse[min(hi_i + 2, len(coarse) - 1)]
    push_up = 1.0 if sign > 0 else -1.0     # stick that surely RAISES gameSteer
    push_down = -push_up                    # stick that surely LOWERS gameSteer

    def recenter(timeout=10.0):
        """Bring the wheel back near center (|gameSteer| < 0.3), then leave the stick halfway."""
        t0 = now()
        while now() - t0 < timeout:
            g = read()
            if abs(g) < 0.3:
                steer((lo + hi) / 2)
                return True
            steer(push_up if g < 0 else push_down)
            sleep(0.02)
        return False

    # 2) wheel rotation speed for each stick position in the useful range
    grid = [round(float(x), 3) for x in np.linspace(lo, hi, 15)]
    rates = []
    log(f"wheel speed between {lo:+.2f} and {hi:+.2f}:")
    for s in grid:
        recenter()
        steer(s)
        sleep(0.15)
        t0, g0 = now(), read()
        sleep(0.6)  # short: measure the speed before the wheel hits its end stop
        rate = (read() - g0) / max(now() - t0, 1e-3)
        rates.append(rate)
        log(f"  stick {s:+.3f} -> {rate:+.3f} /s")
    recenter()
    grid, rates = np.array(grid), np.array(rates)

    # 3) analysis: neutral (wheel does not move), dead zone, slope
    still = np.abs(rates) < 0.03
    if still.any():
        neutral = float((grid[still].min() + grid[still].max()) / 2)
        dz = float((grid[still].max() - grid[still].min()) / 2)
    else:
        cross = np.flatnonzero(np.sign(rates[:-1]) != np.sign(rates[1:]))
        if not cross.size:
            return {"error": "could not find the neutral stick position",
                    "grid": grid.tolist(), "rates": rates.tolist()}
        i = int(cross[0])
        neutral = float(np.interp(0.0, [rates[i], rates[i + 1]], [grid[i], grid[i + 1]])
                        if rates[i] < rates[i + 1] else
                        np.interp(0.0, [rates[i + 1], rates[i]], [grid[i + 1], grid[i]]))
        dz = 0.0
    moving = ~still
    if moving.sum() < 2:
        return {"error": "not enough readings where the wheel moves", "grid": grid.tolist(), "rates": rates.tolist()}
    eff = np.abs(grid[moving] - neutral) - dz          # deviation beyond the dead zone
    c = float(np.sum(eff * np.abs(rates[moving])) / max(np.sum(eff ** 2), 1e-9))  # |speed| per unit
    tau = 0.3  # target time constant (s) to close an error
    kp = 1.0 / (tau * max(c, 1e-3))
    span = float(max(abs(lo - neutral), abs(hi - neutral)))
    return {"mode": "closed_loop", "neutral": neutral, "sign": sign, "dz": dz, "kp": kp, "span": span,
            "rate_per_unit": c, "coarse": list(zip(coarse, cg.tolist())),
            "grid": grid.tolist(), "rates": rates.tolist()}


def closed_loop_test(ctrl, steer, read, sleep, now, log=print) -> float:
    """Check the controller: ask for 0, +0.1, -0.1 then 0 and measure the final error."""
    errors = []
    for target in (0.0, 0.1, -0.1, 0.0):
        t0 = now()
        while now() - t0 < 3.0:
            steer(ctrl(target, read()))
            sleep(0.1)
        err = read() - target
        errors.append(abs(err))
        log(f"  target {target:+.2f} -> wheel {read():+.3f}  (error {err:+.3f})")
    steer(ctrl.neutral)
    return max(errors)


def run_calibrate(args):
    tel = Telemetry()
    if not tel.available:
        sys.exit("telemetry unavailable: install the scs-telemetry.dll plugin and start a game")
    pad = Pad()
    print("calibrating in 5 s: switch to the game, truck STOPPED, parking brake on, touch nothing")
    time.sleep(5)
    io = dict(steer=pad.steer, read=lambda: tel.read()["steer"], sleep=time.sleep, now=time.perf_counter)
    result = calibrate_core(**io)
    Path("calibration_debug.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        pad.release()
        sys.exit(result["error"] + " (readings in calibration_debug.json)")
    print(f"\nneutral {result['neutral']:+.3f}, sign {result['sign']:+.0f}, dead zone ±{result['dz']:.3f}, "
          f"speed {result['rate_per_unit']:.2f}/s per stick unit, kp {result['kp']:.2f}")
    CALIB_FILE.write_text(json.dumps(result, indent=1))
    ctrl = SteerController()
    pad.neutral = ctrl.neutral
    print("\ncontroller test:")
    worst = closed_loop_test(ctrl, **io)
    pad.release()
    if worst > 0.05:
        print(f"! final error up to {worst:.3f}: the controller is inaccurate, check it before driving")
    else:
        print(f"controller OK (max error {worst:.3f}). Saved to {CALIB_FILE}")


def run_drive(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device)
    screen = Screen(args.monitor)
    tel = Telemetry()
    if not tel.available:
        sys.exit("telemetry unavailable: it is required for closed-loop steering "
                 "(scs-telemetry.dll plugin, game running)")
    pad = Pad()
    ctrl = SteerController()
    if args.kp_scale != 1.0:
        ctrl.kp *= args.kp_scale
        print(f"kp scaled x{args.kp_scale} -> {ctrl.kp:.2f}")
    pad.neutral = ctrl.neutral
    pad.release()

    log_path = Path(f"drive_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["t", "active", "pred", "smoothed", "stick", "game_steer", "speed_kmh", "loop_ms"])

    print("\nReady. Switch to the game, turn on cruise control, then:")
    print("  F8  = toggle the autopilot     F10 = quit\n")
    key_pressed(VK_F8), key_pressed(VK_F10)  # clear the key state
    active, smoothed, period = False, 0.0, 1.0 / args.hz
    t_start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            if key_pressed(VK_F10):
                break
            if key_pressed(VK_F8):
                active = not active
                smoothed = 0.0
                if not active:
                    pad.release()
                print(f"\nautopilot {'ENABLED' if active else 'disabled'}")

            pred = predict(model, preprocess(screen.grab()), device)
            smoothed = args.smooth * smoothed + (1 - args.smooth) * pred
            t = tel.read()
            stick = ctrl(smoothed, t["steer"])
            if active and not t["paused"]:
                pad.steer(stick)

            dt = (time.perf_counter() - t0) * 1000
            log.writerow([round(t0 - t_start, 3), int(active), round(pred, 4), round(smoothed, 4),
                          round(stick, 4), round(t["steer"], 4), round(t["speed"] * 3.6, 1), round(dt, 1)])
            print(f"\r{'ON ' if active else 'off'} pred {pred:+.3f} smoothed {smoothed:+.3f} stick {stick:+.2f} "
                  f"| game {t['steer']:+.3f} | {t['speed'] * 3.6:5.1f} km/h | {dt:4.0f} ms ", end="", flush=True)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    finally:
        pad.release()
        log_file.close()
        print(f"\nlog: {log_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check the crop on one screen capture")
    ap.add_argument("--calibrate", action="store_true", help="measure stick -> in-game steering")
    ap.add_argument("--pad", action="store_true",
                    help="plug in the virtual controller and move the stick, to select it in ETS2")
    ap.add_argument("--model", type=Path, default=Path("models/pilotnet.pt"))
    ap.add_argument("--monitor", type=int, default=1, help="monitor to capture (1 = primary)")
    ap.add_argument("--hz", type=float, default=10.0, help="loop rate (dataset: 10 Hz)")
    ap.add_argument("--smooth", type=float, default=0.5, help="exponential smoothing 0..1 (0 = none)")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="steering responsiveness: <1 if the wheel oscillates, >1 if it is too sluggish")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if sys.platform != "win32":
        sys.exit("drive.py only works on Windows (screen capture + ViGEm controller)")
    if args.check:
        run_check(args)
    elif args.pad:
        run_pad(args)
    elif args.calibrate:
        run_calibrate(args)
    else:
        run_drive(args)


if __name__ == "__main__":
    main()
