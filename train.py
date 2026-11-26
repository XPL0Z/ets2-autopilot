"""Train a small CNN (NVIDIA PilotNet style) to predict steering from the camera image.

Data:   the .npz files written by prepare.py (data/processed/{train,val})
Output: models/pilotnet.pt (best model on validation) + models/history.csv

Runs on an NVIDIA GPU (CUDA), Intel GPU (XPU) or CPU, picked automatically.
All images are loaded once into device memory (~1 GB as uint8) and batches are built
there directly: no DataLoader, so nothing slow on Windows.

Usage:
    python train.py                    # default training, auto device
    python train.py --epochs 5         # quick test
    python train.py --device cpu       # force CPU
"""

import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MIN_SPEED = 1.0  # m/s: skip frames where the truck is almost stopped


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_split(folder: Path):
    frames, steer = [], []
    for f in sorted(glob.glob(str(folder / "*.npz"))):
        d = np.load(f)
        keep = (d["speed"] >= MIN_SPEED) & ~d["paused"]
        frames.append(d["frames"][keep])
        steer.append(d["steer"][keep])
    if not frames:
        raise SystemExit(f"No .npz in {folder}: run prepare.py first")
    return np.concatenate(frames), np.concatenate(steer).astype(np.float32)


def to_batch(frames_u8: torch.Tensor, augment: bool) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1], with optional augmentation."""
    x = frames_u8.permute(0, 3, 1, 2).float().div_(255.0)
    if augment:
        b = x.shape[0]
        # brightness / contrast jitter (day, night, weather), different for each image
        contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
        brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
        x = ((x - 0.5) * contrast + 0.5 + brightness).clamp_(0, 1)
        # No horizontal flip: traffic drives on the right, a mirrored image would teach left-hand driving.
    return x


class PilotNet(nn.Module):
    """PilotNet (Bojarski et al., NVIDIA 2016) adapted to 64x256 images."""

    def __init__(self):
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv2d(cin, cout, k, s), nn.BatchNorm2d(cout), nn.ELU())

        self.features = nn.Sequential(
            block(3, 24, 5, 2),   # -> 30 x 126
            block(24, 36, 5, 2),  # -> 13 x 61
            block(36, 48, 5, 2),  # -> 5 x 29
            block(48, 64, 3, 1),  # -> 3 x 27
            block(64, 64, 3, 1),  # -> 1 x 25
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 1 * 25, 100), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.25  # simple normalization
        return self.head(self.features(x))


def balanced_weights(steer, bins=(0, 0.02, 0.05, 0.1, 0.2, 0.4, 10)):
    """Sampling weights inversely proportional to the frequency of each |steer| bin:
    without this, 75 % of frames are straight road and the model always predicts ~0."""
    idx = np.digitize(np.abs(steer), bins) - 1
    counts = np.bincount(idx, minlength=len(bins) - 1).astype(np.float64)
    w = 1.0 / np.maximum(counts[idx], 1)
    w = np.sqrt(w)  # partial rebalancing: keep some of the real distribution
    return torch.as_tensor(w / w.sum(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, frames, steer, batch=1024):
    model.eval()
    preds = torch.cat([model(to_batch(frames[i:i + batch], augment=False))
                       for i in range(0, len(steer), batch)]).squeeze(1)
    err = (preds - steer).abs()
    turning = steer.abs() > 0.05
    return {
        "mae": err.mean().item(),
        "mae_turns": err[turning].mean().item() if turning.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", help="auto | cuda | xpu | cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="PyTorch CPU threads; on a laptop CPU (Core Ultra U), 4 is faster than 12")
    args = ap.parse_args()

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: {device} ({torch.cuda.get_device_name()})")
    else:
        print(f"device: {device}")
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    xtr, ytr = load_split(args.data / "train")
    xva, yva = load_split(args.data / "val")
    print(f"train: {len(ytr)} images | val: {len(yva)} images")

    # Baseline: a "model" that always answers straight ahead (0)
    print(f"'always 0' baseline -> val MAE {np.abs(yva).mean():.4f} | "
          f"turn MAE {np.abs(yva[np.abs(yva) > 0.05]).mean():.4f}")

    weights = balanced_weights(ytr).to(device)
    xtr, ytr = torch.from_numpy(xtr).to(device), torch.from_numpy(ytr).to(device)
    xva, yva = torch.from_numpy(xva).to(device), torch.from_numpy(yva).to(device)

    model = PilotNet().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    best = float("inf")
    with open(args.out / "history.csv", "w", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_mae", "val_mae_turns", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            # weighted sampling (with replacement): turns come up more often than straight road
            order = torch.multinomial(weights, len(ytr), replacement=True)
            total = torch.zeros((), device=device)
            for i in range(0, len(order), args.batch):
                idx = order[i:i + args.batch]
                x, y = to_batch(xtr[idx], augment=True), ytr[idx].unsqueeze(1)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += loss.detach() * len(idx)
            sched.step()
            train_loss = total.item() / len(order)
            m = evaluate(model, xva, yva)
            dt = time.time() - t0
            flag = ""
            # selection criterion = error in turns: that is what matters for driving
            # (the overall error is dominated by straight road)
            if m["mae_turns"] < best:
                best = m["mae_turns"]
                state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({"model": state, "epoch": epoch, **m}, args.out / "pilotnet.pt")
                flag = "  <- best, saved"
            print(f"epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | val MAE {m['mae']:.4f} | "
                  f"turn MAE {m['mae_turns']:.4f} | {dt:.0f}s{flag}", flush=True)
            log.writerow([epoch, train_loss, m["mae"], m["mae_turns"], round(dt, 1)])
            fh.flush()


if __name__ == "__main__":
    main()
