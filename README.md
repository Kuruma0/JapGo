# JapGo

An engine-independent **environmental understanding and procedural reconstruction system**, with an
initial focus on Japan.

The system's job is not to draw roads. It is to learn the spatial relationships that cause roads,
parcels, buildings and settlements to appear where they do — and then to reconstruct or generate an
environment that is *geographically, structurally and architecturally plausible for that specific
place*.

The first shipped output is a **road network graph**. The architecture is designed so that parcels,
buildings, building types and street environments are added later without redesign.

## Status

**Phase 1 in progress.** Phase 0 is complete with no open questions. The provenance gate, CRS/tiling
layer and tile manifest are implemented and tested; source adapters are next.

Linux / macOS:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[geo,dev]"
```

Windows (PowerShell):

```bash
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[geo,dev]"
```

Then, once per machine — this recreates the local untracked files a clone does not carry:

```bash
python scripts/bootstrap.py
```

Verify the environment before doing anything else:

```bash
japgo provenance check && pytest -q
```

The GPU go/no-go for Phase 4 (see [docs/phase0-research.md](docs/phase0-research.md) §20.1):

```bash
python scripts/gpu_spike.py
```

Settled: MVP region is **Shizuoka Prefecture**; terrain is **VIRTUAL SHIZUOKA at 0.5 m** under
CC BY 4.0; commercial use is in scope, weighted toward **generated** environments with reconstruction
kept legally clean alongside; target hardware is a **16 GB AMD RX 6800** (Linux dual-boot approved as
fallback); **reconstruction before generation**.

| Document | Purpose |
| --- | --- |
| [docs/phase0-research.md](docs/phase0-research.md) | The Phase 0 technical research document (datasets, licensing, ML/procedural approaches, architecture, MVP, roadmap, risks) |
| [docs/site-selection.md](docs/site-selection.md) | The three MVP sites and why they were chosen |
| [docs/data-provenance.md](docs/data-provenance.md) | How the provenance registry works and the rules it enforces |
| [data/provenance/registry.yaml](data/provenance/registry.yaml) | Machine-readable registry of every candidate data source and its license status |
| [AGENTS.md](AGENTS.md) | Working agreement and invariants for anyone working in this repo |

## Core invariants

1. **Engine-agnostic core.** No Unity or Unreal type ever enters the core model. Engines are exporters.
2. **Environment-first.** Roads are conditioned on terrain, water, land use, rail, population and
   buildings — never generated in isolation.
3. **Hybrid by default.** ML proposes (where roads *want* to be); deterministic procedural code
   disposes (grade limits, turning radii, intersection validity, hierarchy consistency).
4. **Licensing is a build-time concern, not a footnote.** Every source has a registry entry before
   a single byte of it reaches a training set.
5. **Geographic splits only.** Never randomly split adjacent tiles between train and test.

## Read this first

Two findings shape everything else, both in
[docs/phase0-research.md §6](docs/phase0-research.md#6-licensing-the-decisive-constraint):

**Training on OSM is fine; shipping OSM geometry is not.** A substantial OSM extraction used as a
training set is a Derivative Database under ODbL — but *predictions made by the resulting model are
not implicated*. Reconstruction output *transformed from* OSM geometry, however, is. Since this
project does reconstruction first and may go commercial, shipped geometry comes from the
**redistributable core** (PLATEAU, 国土数値情報, VIRTUAL SHIZUOKA, AW3D30 — all attribution-only),
with OSM confined to training and evaluation.

**Shizuoka has 0.5 m open terrain.** VIRTUAL SHIZUOKA publishes prefecture-wide LiDAR under CC BY
4.0 / ODbL dual license, already gridded at 0.5 m in the right CRS. That is ten times finer than the
GSI product the project is avoiding for licensing reasons, and it is why the MVP region is Shizuoka:
**the region was chosen by where the best openly-licensed data exists**, not the other way round.
