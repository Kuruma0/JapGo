"""Phase 0.5 GPU spike.

Answers one question before Phase 4 depends on it: **does a training stack actually come up on
this machine, and does the 16 GB budget close?**

Deliberately backend-agnostic. The research doc's concern is specific to gfx1030 on Windows, but
the spike should give the same verdict on CUDA, ROCm, DirectML or CPU so the result is comparable
across whatever hardware this ends up running on.

Run:

    python scripts/gpu_spike.py
    python scripts/gpu_spike.py --crop 512 --batch 8 --channels 15

Exit code is 0 if the stack trained, 1 if it did not. The report is what matters, not the speed —
this is a go/no-go, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Report:
    verdict: str = "unknown"
    backend: str = "none"
    device_name: str = ""
    vram_gb: float | None = None
    torch_version: str = ""
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    python: str = field(default_factory=lambda: platform.python_version())

    fp16_ok: bool = False
    peak_memory_gb: float | None = None
    step_ms: float | None = None
    samples_per_s: float | None = None
    est_epoch_minutes: float | None = None
    notes: list[str] = field(default_factory=list)


def detect(report: Report):
    """Identify the best available backend and record what it is."""
    try:
        import torch
    except ImportError:
        report.notes.append(
            "PyTorch is not installed. On an AMD RDNA2 card (gfx1030) the official Windows "
            "wheels do not cover it — see docs/phase0-research.md §20.1 for the routes."
        )
        report.verdict = "no-torch"
        return None, None

    report.torch_version = torch.__version__

    if torch.cuda.is_available():
        # ROCm builds also report through the cuda namespace; torch.version.hip disambiguates.
        report.backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        report.device_name = props.name
        report.vram_gb = round(props.total_memory / 1e9, 2)
        return torch, device

    try:
        import torch_directml  # noqa: F401

        report.backend = "directml"
        report.notes.append(
            "DirectML works but has thin op coverage and poor training throughput; treat a pass "
            "here as a fallback, not a result."
        )
        return torch, torch_directml.device()
    except ImportError:
        pass

    report.backend = "cpu"
    report.device_name = platform.processor() or "cpu"
    report.notes.append("No GPU backend found; timings below are CPU and not representative.")
    return torch, torch.device("cpu")


def build_unet(torch, channels: int, width: int = 32):
    """The dull baseline of research doc §12, at roughly the real shape.

    Not the final architecture — just enough encoder/decoder depth and activation volume that the
    memory figure means something.
    """
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1, self.d2, self.d3, self.d4 = (
                block(channels, width), block(width, width * 2),
                block(width * 2, width * 4), block(width * 4, width * 8),
            )
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.u3 = block(width * 8 + width * 4, width * 4)
            self.u2 = block(width * 4 + width * 2, width * 2)
            self.u1 = block(width * 2 + width, width)
            self.head = nn.Conv2d(width, 4, 1)  # 4 targets: mask, class, orient sin/cos

        def forward(self, x):
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            c4 = self.d4(self.pool(c3))
            x = self.u3(torch.cat([self.up(c4), c3], 1))
            x = self.u2(torch.cat([self.up(x), c2], 1))
            x = self.u1(torch.cat([self.up(x), c1], 1))
            return self.head(x)

    return UNet()


def run(args) -> Report:
    report = Report()
    torch, device = detect(report)
    if torch is None:
        return report

    import torch.nn as nn

    try:
        model = build_unet(torch, args.channels).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        use_amp = report.backend in {"cuda", "rocm"}
        # fp16, not bf16: RDNA2 has no bf16 hardware (research doc invariant 9).
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        x = torch.randn(args.batch, args.channels, args.crop, args.crop, device=device)
        y = torch.randn(args.batch, 4, args.crop, args.crop, device=device)

        if report.backend in {"cuda", "rocm"}:
            torch.cuda.reset_peak_memory_stats()

        timings = []
        for step in range(args.steps):
            start = time.perf_counter()
            optimiser.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                optimiser.step()

            if report.backend in {"cuda", "rocm"}:
                torch.cuda.synchronize()
            if step >= args.warmup:
                timings.append(time.perf_counter() - start)

        report.fp16_ok = use_amp
        if timings:
            step_s = sum(timings) / len(timings)
            report.step_ms = round(step_s * 1000, 1)
            report.samples_per_s = round(args.batch / step_s, 1)
            # A 100-tile site at 512² crops is roughly 900 crops per epoch.
            report.est_epoch_minutes = round(900 / (args.batch / step_s) / 60, 1)

        if report.backend in {"cuda", "rocm"}:
            report.peak_memory_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if report.vram_gb and report.peak_memory_gb > report.vram_gb * 0.9:
                report.notes.append(
                    "Peak memory is within 10% of VRAM; reduce batch or enable gradient "
                    "checkpointing before adding channels."
                )

        report.verdict = "pass" if report.backend in {"cuda", "rocm"} else "degraded"

    except Exception as exc:  # noqa: BLE001 - the failure mode IS the result here
        report.verdict = "fail"
        report.notes.append(f"{type(exc).__name__}: {exc}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", type=int, default=512, help="Patch size (research doc §20.2).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--channels", type=int, default=15, help="Current raster stack depth.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    report = run(args)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print("\n=== Phase 0.5 GPU spike ===")
        print(f"verdict        : {report.verdict.upper()}")
        print(f"backend        : {report.backend}")
        print(f"device         : {report.device_name}")
        print(f"vram           : {report.vram_gb} GB" if report.vram_gb else "vram           : n/a")
        print(f"torch          : {report.torch_version}")
        print(f"platform       : {report.platform}  py{report.python}")
        print(f"fp16 autocast  : {report.fp16_ok}")
        print(f"peak memory    : {report.peak_memory_gb} GB" if report.peak_memory_gb else "")
        print(f"step time      : {report.step_ms} ms")
        print(f"throughput     : {report.samples_per_s} samples/s")
        print(f"~epoch (900)   : {report.est_epoch_minutes} min")
        for note in report.notes:
            print(f"  ! {note}")
        print()

    return 0 if report.verdict in {"pass", "degraded"} else 1


if __name__ == "__main__":
    sys.exit(main())
