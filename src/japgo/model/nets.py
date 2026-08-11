"""The dull baseline of spec §51 and research doc §12.

A plain multi-channel U-Net predicting a road probability raster. It is meant to be unremarkable:
its whole job is to establish the floor that anything more interesting has to clear on held-out
cities. Choosing it is not a bet on architecture, it is a refusal to make one before there is a
number to argue with.

Sized against the 16 GB budget rather than against the literature. Width 32 with four encoder
stages measured 4.03 GB at 512² × 15 channels batch 8 in the Phase 0.5 spike, which leaves
gradient checkpointing unspent for the SAM-class route later (§20.2).
"""

from __future__ import annotations


def build_unet(channels: int, *, width: int = 32, outputs: int = 1):
    """Encoder/decoder with skip connections, returning logits.

    Logits rather than probabilities: the loss applies the sigmoid itself, which is numerically
    stabler under fp16 than sigmoid-then-BCE. On a target where roads are a few percent of pixels,
    that stability is not academic.
    """
    import torch
    import torch.nn as nn

    def block(a: int, b: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d1 = block(channels, width)
            self.d2 = block(width, width * 2)
            self.d3 = block(width * 2, width * 4)
            self.d4 = block(width * 4, width * 8)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.u3 = block(width * 8 + width * 4, width * 4)
            self.u2 = block(width * 4 + width * 2, width * 2)
            self.u1 = block(width * 2 + width, width)
            self.head = nn.Conv2d(width, outputs, 1)

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


def masked_bce(logits, target, valid, *, positive_weight: float):
    """Binary cross-entropy over observed pixels only, weighted toward the positive class.

    Two corrections that matter more than the architecture:

    ``valid`` excludes voids. ``nodata_fill`` is 0.0 and the background class is also 0, so an
    unmasked loss trains the model to confidently predict "no road" over sea and LiDAR shadow —
    and rewards it for doing so.

    ``positive_weight`` counteracts the class imbalance. Roads are a few percent of pixels, and
    the minimiser of an unweighted BCE on that target is a model that predicts road nowhere. It
    would score well on accuracy and be worthless.
    """
    import torch.nn.functional as F

    mask = valid > 0.5
    if not mask.any():
        return logits.sum() * 0.0

    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=logits.new_tensor(positive_weight),
    )
    return (loss * mask).sum() / mask.sum()
