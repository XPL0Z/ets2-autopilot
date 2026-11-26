"""Extract the ETS2 dataset (WebDataset .tar shards) into .npz files ready for training.

For each session in the .tar (= one {key}.mp4 + {key}.bin.zst + {key}.json), writes
<out>/<split>/<key>.npz containing:
    frames    uint8   (N, H, W, 3)  RGB images, cropped to the road and downscaled
    steer     float32 (N,)  truck_f.gameSteer    (effective steering, ~[-1, 1])
    throttle  float32 (N,)  truck_f.gameThrottle [0, 1]
    brake     float32 (N,)  truck_f.gameBrake    [0, 1]
    speed     float32 (N,)  truck_f.speed in m/s
    paused    bool    (N,)  game paused

Usage:
    .venv/bin/python prepare.py ets2-dataset/ets2-train-000000.tar --split train
    .venv/bin/python prepare.py ets2-dataset/ets2-val-000000.tar --split val
"""

import argparse
import json
import struct
import tarfile
import tempfile
from compression import zstd  # stdlib since Python 3.14
from pathlib import Path

import cv2
import numpy as np

# --- SCS telemetry: offsets in the 32 KiB block (scs-sdk-plugin, PLUGIN_REVID 12) ---
BLOCK = 32768
OFF_REVID = 40        # scs_values.telemetry_plugin_revision (uint32)
OFF_PAUSED = 4        # paused (bool)
OFF_TRUCK_F = 948     # truck_f: speed, engineRpm, userSteer, userThrottle, userBrake,
                      #          userClutch, gameSteer, gameThrottle, gameBrake, ...
EXPECTED_REVID = 12

# --- Crop (coordinates in the 1920x1080 frame) ---
# Road band below the mirrors and ABOVE the steering wheel: the wheel turns with the
# steering, so leaving it visible would let the model "cheat".
CROP_Y = (290, 635)
CROP_X = (240, 1680)
OUT_W, OUT_H = 256, 64


def parse_telemetry(raw: bytes) -> dict[str, np.ndarray]:
    n = len(raw) // BLOCK
    out = {k: np.empty(n, np.float32) for k in ("steer", "throttle", "brake", "speed")}
    out["paused"] = np.empty(n, bool)
    for i in range(n):
        b = raw[i * BLOCK:(i + 1) * BLOCK]
        revid, = struct.unpack_from("<I", b, OFF_REVID)
        if revid != EXPECTED_REVID:
            raise ValueError(f"unexpected plugin_revid {revid} at frame {i} (expected {EXPECTED_REVID})")
        speed, _rpm, _us, _ut, _ub, _uc, steer, thr, brk = struct.unpack_from("<9f", b, OFF_TRUCK_F)
        out["speed"][i], out["steer"][i], out["throttle"][i], out["brake"][i] = speed, steer, thr, brk
        out["paused"][i] = b[OFF_PAUSED] != 0
    return out


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if (w, h) != (1920, 1080):
        frame_bgr = cv2.resize(frame_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
    crop = frame_bgr[CROP_Y[0]:CROP_Y[1], CROP_X[0]:CROP_X[1]]
    small = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def decode_video(mp4_bytes: bytes) -> np.ndarray:
    # OpenCV needs a file path, so go through a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(mp4_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(preprocess(f))
        cap.release()
    return np.stack(frames) if frames else np.empty((0, OUT_H, OUT_W, 3), np.uint8)


def process_session(key: str, parts: dict[str, bytes], out_dir: Path) -> str:
    meta = json.loads(parts["json"])
    if meta.get("plugin_revid") != EXPECTED_REVID:
        return f"skipped (plugin_revid={meta.get('plugin_revid')})"
    tel = parse_telemetry(zstd.decompress(parts["bin.zst"]))
    frames = decode_video(parts["mp4"])
    n = min(len(frames), len(tel["steer"]))
    if len(frames) != len(tel["steer"]):
        print(f"  ! {key}: {len(frames)} frames vs {len(tel['steer'])} telemetry samples, truncated to {n}")
    np.savez_compressed(out_dir / f"{key}.npz", frames=frames[:n], **{k: v[:n] for k, v in tel.items()})
    return f"{n} frames"


def iter_sessions(tar_path: Path):
    """Stream the .tar and yield (key, {ext: bytes}) as soon as a session is complete.
    Tolerates a truncated .tar (partial download): the last incomplete session is skipped."""
    needed = {"mp4", "bin.zst", "json"}
    pending: dict[str, dict[str, bytes]] = {}
    try:
        with tarfile.open(tar_path, "r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = Path(m.name).name  # never reuse the path stored in the .tar on disk
                key, _, ext = name.partition(".")
                if ext not in needed:
                    continue
                pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
                if pending[key].keys() == needed:
                    yield key, pending.pop(key)
    except (tarfile.ReadError, EOFError) as e:
        print(f"  ! end of {tar_path.name} unreadable ({e}): truncated archive, stopping here")
    for key in pending:
        print(f"  ! incomplete session skipped: {key}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tars", nargs="+", type=Path)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for tar_path in args.tars:
        print(f"== {tar_path}")
        for key, parts in iter_sessions(tar_path):
            if (out_dir / f"{key}.npz").exists():
                print(f"  {key}: already done")
                continue
            print(f"  {key}: {process_session(key, parts, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
