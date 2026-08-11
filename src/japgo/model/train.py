"""Phase 4: train the baseline, evaluate it on ground it has never seen.

The exit criterion is *beats a non-learned prior on a held-out city*, so a run is not a training
loop with a loss curve — it is a comparison. Every run reports the model and both priors on the
same held-out tiles at each one's own best threshold, because a comparison won by tuning one
side's cutoff is not a comparison.

Reproducibility is structural, not aspirational (invariant 8): a :class:`RunConfig` pins the
corpus, the fold, the seed, the crop, the channel spec version and the registry hash, and it is
written beside the checkpoint. A run that cannot be re-created from its config is a failed run
whatever its numbers say.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..pipeline.channels import load_stack_spec
from ..pipeline.splits import SplitDefinition
from ..provenance import registry_hash
from .baseline import ROAD_TARGET, best_threshold, built_proximity_prior, constant_prior, road_rate
from .dataset import (
    DEFAULT_CROP,
    Fold,
    PatchLoader,
    assert_no_overlap,
    configured_fold,
    index_patches,
    leave_one_site_out,
)

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Everything needed to re-create a run."""

    root: str
    fold: str
    train_tiles: list[str]
    eval_tiles: list[str]
    crop: int = DEFAULT_CROP
    batch: int = 8
    epochs: int = 8
    learning_rate: float = 1e-3
    width: int = 32
    seed: int = 0
    stack_version: int | None = None
    registry: str | None = None
    channels: list[str] = field(default_factory=list)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path


@dataclass
class Result:
    fold: str
    held_out: str
    model: dict
    constant: dict
    built: dict
    train_patches: int
    eval_tiles: int
    epochs: int
    seconds: float
    topology: dict | None = None
    """APLS/TOPO on the extracted graph. None when no held-out tile carried a road graph."""

    def verdict(self) -> str:
        """Whether the model cleared the floor. The Phase 4 exit criterion, in one line."""
        beat_constant = self.model["f1"] > self.constant["f1"]
        beat_built = self.model["f1"] > self.built["f1"]
        if beat_constant and beat_built:
            return "PASS - beats both priors"
        if beat_constant:
            return "PARTIAL - beats the constant prior but not built-proximity"
        return "FAIL - does not beat the constant prior"


def folds_for(split: SplitDefinition, *, scheme: str) -> list[Fold]:
    """``configured`` for the split as written, ``loso`` to rotate the held-out site."""
    if scheme == "configured":
        return [configured_fold(split)]
    if scheme == "loso":
        return leave_one_site_out(split)
    raise ValueError(f"unknown scheme {scheme!r}; use 'configured' or 'loso'")


def train_fold(root: Path, fold: Fold, config: RunConfig, *, out_dir: Path) -> Result:
    """Train on the fold's training sites and evaluate on the site it holds out."""
    import torch

    from .nets import build_unet, masked_bce

    assert_no_overlap(fold)

    spec = load_stack_spec()
    target_index = spec.target_index_of(ROAD_TARGET)
    config.stack_version = spec.stack_version
    config.registry = registry_hash()
    config.channels = spec.names

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    loader = PatchLoader(root)
    train_patches = index_patches(root, fold.train_tiles, crop=config.crop)
    if not train_patches:
        raise ValueError(f"{fold.name}: no usable training patches")

    rate = road_rate(loader, train_patches, target_index=target_index, crop=config.crop)
    # Weight the positive class by its own rarity. Measured on the training split only: a weight
    # taken from the held-out site would be a small but real leak of its statistics.
    positive_weight = float(min(max((1 - rate) / rate if rate else 1.0, 1.0), 50.0))
    log.info(
        "%s: %d train patches, road rate %.3f%%, pos_weight %.1f",
        fold.name, len(train_patches), 100 * rate, positive_weight,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(spec.depth, width=config.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(train_patches))
        running, steps = 0.0, 0

        for start in range(0, len(order) - config.batch + 1, config.batch):
            picks = [train_patches[i] for i in order[start : start + config.batch]]
            xs, ys = zip(*(loader.read(p, config.crop) for p in picks), strict=True)
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)[:, target_index : target_index + 1]).to(device)
            valid = x[:, -1:, :, :]

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = masked_bce(model(x), y, valid, positive_weight=positive_weight)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running += float(loss.detach())
            steps += 1

        log.info("%s epoch %d/%d  loss %.4f", fold.name, epoch + 1, config.epochs,
                 running / max(steps, 1))

    seconds = time.perf_counter() - started
    scores = evaluate_fold(root, fold, model, config, rate=rate, target_index=target_index)
    topology = topology_of(
        root, fold, scores.pop("_rasters"),
        threshold=scores["model"].threshold, seed=config.seed,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{fold.name}.pt")
    config.write(out_dir / f"{fold.name}.config.json")

    return Result(
        fold=fold.name,
        held_out=fold.held_out,
        model=asdict(scores["model"]),
        constant=asdict(scores["constant"]),
        built=asdict(scores["built"]),
        train_patches=len(train_patches),
        eval_tiles=len(fold.eval_tiles),
        epochs=config.epochs,
        seconds=seconds,
        topology=topology,
    )


def evaluate_fold(root: Path, fold: Fold, model, config: RunConfig, *, rate: float, target_index: int):
    """Score the model and both priors over whole held-out tiles.

    Whole tiles rather than patches: the question is whether the road layer of an unseen place is
    recovered, and a patch average hides what happens at the seams between them.
    """
    import torch

    from ..pipeline.store import read_tile

    spec = load_stack_spec()
    built_index = spec.index_of("building_mask")
    device = next(model.parameters()).device
    model.eval()

    probs, truths, valids, priors = [], [], [], []
    for tile_id in fold.eval_tiles:
        bundle = read_tile(root, tile_id)
        if bundle.targets is None:
            continue
        stack = bundle.stack
        with torch.no_grad():
            x = torch.from_numpy(stack[None]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
            p = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

        probs.append(p)
        truths.append(bundle.target(ROAD_TARGET))
        valids.append(stack[spec.index_of("valid")])
        priors.append(built_proximity_prior(stack, built_index=built_index))

    probability = np.concatenate([p.ravel() for p in probs])
    truth = np.concatenate([t.ravel() for t in truths])
    valid = np.concatenate([v.ravel() for v in valids])
    built = np.concatenate([b.ravel() for b in priors])

    return {
        "model": best_threshold(probability, truth, valid=valid),
        "constant": best_threshold(constant_prior(truth.shape, rate), truth, valid=valid),
        "built": best_threshold(built, truth, valid=valid),
        "_rasters": (probs, fold.eval_tiles),
    }


def topology_of(root: Path, fold: Fold, rasters, *, threshold: float, seed: int = 0):
    """APLS and TOPO per held-out tile, against the tile's own OSM graph.

    Run at the model's best pixel threshold rather than a fixed 0.5: extraction is downstream of
    thresholding, and scoring a well-calibrated model and a badly-calibrated one at the same cutoff
    compares calibration rather than topology.

    Per tile and then averaged, not over a merged region graph — a tile's road network is the unit
    the model predicts, and merging them would let one dense tile dominate.
    """
    from ..pipeline.store import read_tile
    from .extract import ExtractionSpec, extract_graph
    from .topology import compare

    probs, tile_ids = rasters
    spec = ExtractionSpec(threshold=threshold)
    scores = []

    for probability, tile_id in zip(probs, tile_ids, strict=True):
        bundle = read_tile(root, tile_id)
        if bundle.roads is None or not bundle.roads.edges:
            continue
        predicted = extract_graph(
            probability, bundle.tile.read, bundle.manifest.crs,
            spec=spec, tile_id=tile_id,
        )
        scores.append(compare(bundle.roads, predicted, seed=seed))

    if not scores:
        return None
    return {
        "apls": float(np.mean([s.apls for s in scores])),
        "topo_f1": float(np.mean([s.topo_f1 for s in scores])),
        "topo_precision": float(np.mean([s.topo_precision for s in scores])),
        "topo_recall": float(np.mean([s.topo_recall for s in scores])),
        "tiles": len(scores),
        "predicted_nodes": int(np.mean([s.proposal_nodes for s in scores])),
        "truth_nodes": int(np.mean([s.truth_nodes for s in scores])),
    }
