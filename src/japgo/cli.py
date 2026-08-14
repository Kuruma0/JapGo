"""``japgo`` command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .geo import SCALE_TIERS, Bounds, TileGrid, from_wgs84, zone
from .provenance import (
    ProvenanceViolation,
    Severity,
    SourceGate,
    check_registry,
    find_registry,
    load_registry,
    registry_hash,
)


@click.group()
@click.version_option(__version__, prog_name="japgo")
def main() -> None:
    """Engine-independent environmental understanding system, focused on Japan."""


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------


@main.group()
def provenance() -> None:
    """Inspect and enforce the dataset provenance registry."""


@provenance.command("check")
@click.option("--registry", "registry_path", type=click.Path(path_type=Path), default=None)
def provenance_check(registry_path: Path | None) -> None:
    """Audit the registry against project policy."""
    path = registry_path or find_registry()
    reg = load_registry(path)

    click.echo(f"registry : {path}")
    click.echo(f"version  : {reg.registry_version}   hash: {registry_hash(path)}")
    click.echo(f"reviewed : {reg.last_reviewed} by {reg.reviewed_by}")
    click.echo(
        f"sources  : {len(reg.sources)} "
        f"({len(reg.ingestible)} ingestible, "
        f"{len(reg.sources) - len(reg.ingestible)} quarantined)"
    )
    click.echo(f"policy   : commercial_intent={reg.policy.commercial_intent}")
    click.echo("")

    findings = check_registry(reg)
    if not findings:
        click.secho("all checks passed", fg="green")
        return

    for f in findings:
        colour = "red" if f.severity is Severity.ERROR else "yellow"
        click.secho(str(f), fg=colour)

    if any(f.severity is Severity.ERROR for f in findings):
        sys.exit(1)


@provenance.command("list")
@click.option("--tier", type=str, default=None, help="Filter by usage tier.")
@click.option("--role", type=str, default=None, help="Filter by output role.")
def provenance_list(tier: str | None, role: str | None) -> None:
    """List registered sources and their disposition."""
    reg = load_registry()
    rows = reg.sources
    if tier:
        rows = [s for s in rows if s.usage_tier.value == tier]
    if role:
        rows = [s for s in rows if s.output_role and s.output_role.value == role]

    width = max((len(s.id) for s in rows), default=10)
    for s in rows:
        role_txt = s.output_role.value if s.output_role else "-"
        colour = {"public": "green", "quarantined": "red"}.get(s.usage_tier.value, "yellow")
        click.echo(
            f"{s.id:<{width}}  "
            + click.style(f"{s.usage_tier.value:<22}", fg=colour)
            + f"{role_txt:<22}{s.license}"
        )


@provenance.command("attribution")
@click.argument("source_ids", nargs=-1, required=True)
def provenance_attribution(source_ids: tuple[str, ...]) -> None:
    """Emit the attribution block for a set of sources."""
    gate = SourceGate(load_registry())
    try:
        for line in gate.attribution_for(source_ids):
            click.echo(line)
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc


@provenance.command("can-export")
@click.argument("source_ids", nargs=-1, required=True)
@click.option(
    "--redistribution-class",
    type=click.Choice(["attribution-only", "share-alike"]),
    default="attribution-only",
)
def provenance_can_export(source_ids: tuple[str, ...], redistribution_class: str) -> None:
    """Test whether an artifact from these sources may be redistributed."""
    gate = SourceGate(load_registry())
    try:
        gate.assert_exportable(source_ids, redistribution_class=redistribution_class)
    except (ProvenanceViolation, LookupError) as exc:
        click.secho(f"REFUSED as {redistribution_class}", fg="red", bold=True)
        click.echo(str(exc))
        sys.exit(1)
    click.secho(f"OK as {redistribution_class}", fg="green", bold=True)
    for line in gate.attribution_for(source_ids):
        click.echo(f"  {line}")


# ---------------------------------------------------------------------------------------------
# tiles
# ---------------------------------------------------------------------------------------------


@main.group()
def tiles() -> None:
    """Plan and inspect the tile grid."""


@tiles.command("plan")
@click.option("--zone", "zone_number", type=int, default=8, show_default=True)
@click.option("--bbox", nargs=4, type=float, help="minlon minlat maxlon maxlat (WGS84)")
@click.option("--limit", type=int, default=10, show_default=True, help="Tile ids to print.")
def tiles_plan(zone_number: int, bbox: tuple[float, ...] | None, limit: int) -> None:
    """Report the tiles covering a WGS84 bounding box."""
    if not bbox:
        raise click.UsageError("--bbox is required, e.g. --bbox 138.3 34.9 138.5 35.1")

    z = zone(zone_number)
    grid = TileGrid(z)
    minlon, minlat, maxlon, maxlat = bbox
    x0, y0 = from_wgs84(minlon, minlat, z.crs)
    x1, y1 = from_wgs84(maxlon, maxlat, z.crs)
    area = Bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    total = grid.count_covering(area)
    click.echo(f"zone      : {z.zone} (EPSG:{z.epsg})")
    click.echo(f"extent    : {area.width / 1000:.2f} x {area.height / 1000:.2f} km")
    click.echo(f"tiles     : {total}")
    click.echo(f"core/halo : {grid.core_size_m:.0f} m / {grid.halo_m:.0f} m")
    click.echo("")
    click.echo("scale tiers:")
    for tier in SCALE_TIERS:
        click.echo(
            f"  {tier.name:<14} {tier.resolution_m:>5.1f} m/px  "
            f"footprint {tier.footprint_m / 1000:>5.1f} km"
        )
    click.echo("")
    for i, tile in enumerate(grid.tiles_covering(area)):
        if i >= limit:
            click.echo(f"  ... {total - limit} more")
            break
        b = tile.core
        click.echo(f"  {tile.id}  core=({b.minx:.0f},{b.miny:.0f})-({b.maxx:.0f},{b.maxy:.0f})")


@tiles.command("inspect")
@click.argument("tile_id")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
def tiles_inspect(tile_id: str, root: Path) -> None:
    """Show a built tile's channels, provenance and redistribution class."""
    from .pipeline import channel_summary, read_tile

    gate = SourceGate(load_registry())
    try:
        bundle = read_tile(root, tile_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"tile      : {bundle.tile.id}")
    click.echo(f"shape     : {bundle.shape}  (channels, rows, cols)")
    click.echo(f"coverage  : {bundle.coverage:.1%}")
    click.echo(f"crs       : {bundle.manifest.crs}")
    click.echo(f"registry  : {bundle.manifest.registry_hash}")
    click.echo(f"preproc   : v{bundle.manifest.preprocessing_version}")
    click.echo(f"buildings : {len(bundle.buildings)}")

    klass = bundle.redistribution_class(gate)
    click.secho(
        f"redistrib : {klass}",
        fg="green" if klass == "attribution-only" else "yellow",
        bold=True,
    )
    click.echo("")
    click.echo("sources:")
    for record in bundle.manifest.sources:
        version = f" @{record.version}" if record.version else ""
        click.echo(f"  {record.source_id}{version}  {', '.join(record.layers)}")

    click.echo("")
    click.echo(f"{'channel':<32}{'min':>10}{'mean':>10}{'max':>10}")
    for name, lo, mean, hi in channel_summary(bundle):
        click.echo(f"{name:<32}{lo:>10.3f}{mean:>10.3f}{hi:>10.3f}")

    click.echo("")
    click.echo("attribution:")
    for line in bundle.attribution(gate):
        click.echo(f"  {line}")


@tiles.command("channels")
def tiles_channels() -> None:
    """Show the raster stack specification the model consumes."""
    from .pipeline import load_stack_spec

    spec = load_stack_spec()
    click.echo(f"stack version : {spec.stack_version}")
    click.echo(f"depth         : {spec.depth} channels")
    click.echo(f"sources       : {', '.join(spec.required_sources)}")
    click.echo("")
    def _rows(items):
        for i, c in enumerate(items):
            norm = c.normalise.value
            if c.scale:
                norm += f" /{c.scale:g}"
            click.echo(f"{i:>3}  {c.name:<32}{c.source or '-':<20}{c.units:<10}{norm}")

    click.echo(f"{'#':>3}  {'channel':<32}{'source':<20}{'units':<10}norm")
    _rows(spec.channels)

    if spec.targets:
        click.echo("")
        click.secho(
            f"targets ({spec.target_depth}) — from {', '.join(spec.target_sources)}", bold=True
        )
        click.secho(
            "  training-only: predictions are unencumbered, but this geometry must not ship",
            fg="yellow",
        )
        _rows(spec.targets)


@tiles.command("build")
@click.argument("site", required=False)
@click.option("--data-root", type=click.Path(path_type=Path), default="data/raw", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--resolution", type=float, default=1.0, show_default=True)
@click.option("--limit", type=int, default=None, help="Build at most N tiles (for a smoke run).")
@click.option("--all-sites", is_flag=True, help="Build every configured site.")
@click.option(
    "--min-coverage",
    type=float,
    default=0.5,
    show_default=True,
    help="Skip tiles with less than this fraction of real observations.",
)
@click.option(
    "--remote",
    is_flag=True,
    help="Fetch from the published sources instead of staged files. Terrain is streamed and "
    "cached as raster; PLATEAU members are range-read; roads come from Overpass.",
)
@click.option(
    "--cache",
    type=click.Path(path_type=Path),
    default="data/cache",
    show_default=True,
    help="Where --remote keeps fetched terrain rasters, GML members and Overpass responses.",
)
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=None,
    help="Override the site extent for this run: minlon minlat maxlon maxlat (WGS84).",
)
def tiles_build(
    site: str | None,
    data_root: Path,
    out: Path,
    resolution: float,
    limit: int | None,
    all_sites: bool,
    min_coverage: float,
    remote: bool,
    cache: Path,
    bbox: tuple[float, ...] | None,
) -> None:
    """Build a manifest-carrying tile set for a site.

    By default source files are found by convention under
    DATA_ROOT/<site>/{terrain,plateau,landuse,roads}/. With --remote nothing needs staging: the
    published endpoints are read directly and only the derived rasters are cached.
    """
    from .geo.crs import from_wgs84
    from .geo.tiling import Bounds
    from .pipeline import RegionBuilder, SourceFiles, build_default_split, load_sites

    sites = load_sites()
    if all_sites:
        names = list(sites.sites)
    elif site:
        names = [site]
    else:
        raise click.UsageError(
            f"give a site or --all-sites. Configured: {', '.join(sites.sites)}"
        )

    gate = SourceGate(load_registry())
    builder = RegionBuilder(
        gate, sites=sites, resolution=resolution, min_coverage=min_coverage
    )
    written: dict[str, list[str]] = {}

    extent = None
    if bbox:
        if all_sites:
            raise click.UsageError("--bbox overrides one site's extent; it cannot apply to all")
        crs = builder.zone.crs
        x0, y0 = from_wgs84(bbox[0], bbox[1], crs)
        x1, y1 = from_wgs84(bbox[2], bbox[3], crs)
        extent = Bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    for name in names:
        if name not in sites.sites:
            raise click.ClickException(f"unknown site {name!r}; have {', '.join(sites.sites)}")

        spec = sites.sites[name]
        planned = builder.tiles_for(name, bounds=extent)

        click.secho(f"\n{name}  ({spec.archetype})", bold=True)
        click.echo(f"  tiles planned : {len(planned)}{'  (bbox override)' if extent else ''}")

        if remote:
            import logging

            from .pipeline.remote import RemoteSources

            # A remote build is minutes of network per tile. Silence for that long is
            # indistinguishable from a hang, and the INFO lines are where the counts live.
            logging.basicConfig(
                level=logging.INFO, format="  %(message)s", force=True
            )

            plateau_url = getattr(spec, "plateau_url", None)
            if not plateau_url:
                click.secho(
                    f"  no plateau_url in config/sites.yaml for {name}; "
                    "building without buildings",
                    fg="yellow",
                )
            source = RemoteSources(
                gate,
                builder.zone.crs,
                cache_dir=cache / name,
                plateau_url=plateau_url,
                resolution=resolution,
            )
            click.echo(f"  mode          : remote (cache {cache / name})")
            report = builder.build_from(name, source, out, limit=limit, bounds=extent)
            click.echo(f"  fetched       : {source.describe()}")
        else:
            files = SourceFiles.discover(data_root, name)
            click.echo(f"  sources       : {files.describe()}")
            report = builder.build(name, files, out, limit=limit)

        written[name] = report.tiles_written

        colour = "green" if report.ok else "red"
        click.secho(f"  written       : {len(report.tiles_written)}", fg=colour)
        if report.tiles_skipped:
            click.secho(f"  skipped       : {len(report.tiles_skipped)}", fg="yellow")
        for warning in report.warnings[:5]:
            click.secho(f"    {warning}", fg="yellow")
        if len(report.warnings) > 5:
            click.secho(f"    ... {len(report.warnings) - 5} more", fg="yellow")
        for line in report.attribution:
            click.echo(f"    {line}")

    populated = {k: v for k, v in written.items() if v}
    if len(populated) > 1:
        definition = build_default_split(builder, populated)
        path = definition.write(out / "split.json")
        click.echo("")
        click.secho(f"split written: {path}", bold=True)
        click.echo(f"  {definition.counts}")


@main.group()
def splits() -> None:
    """Define and validate geographic train/test splits."""


@splits.command("build")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Default ROOT/split.json.")
def splits_build(root: Path, out: Path | None) -> None:
    """Assign the tiles already on disk to their sites and write a split.

    ``tiles build`` writes a split only when one invocation produced several sites. A corpus grown
    one site at a time — which is how it actually gets built, because each site has its own extent
    and its own municipality package — never gets one. Without it every tile counts as a single
    group and no association can be given an interval, since the bootstrap resamples sites.

    Membership comes from each site's configured extent in ``config/sites.yaml``, which is the only
    thing that knows where a site is; a tile manifest does not record its site.
    """
    from .pipeline import RegionBuilder, build_default_split, load_sites
    from .pipeline.store import list_tiles

    sites = load_sites()
    builder = RegionBuilder(SourceGate(load_registry()), sites=sites)

    on_disk = set(list_tiles(root))
    if not on_disk:
        raise click.ClickException(f"no tiles under {root}")

    written: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for name in sites.sites:
        ids = [t.id for t in builder.tiles_for(name) if t.id in on_disk]
        if ids:
            written[name] = ids
            claimed |= set(ids)
        click.echo(f"  {name:<20} {len(ids)} tile(s)")

    orphans = on_disk - claimed
    if orphans:
        click.secho(
            f"  {len(orphans)} tile(s) fall outside every configured site extent and are left "
            f"out: {', '.join(sorted(orphans)[:4])}{' ...' if len(orphans) > 4 else ''}",
            fg="yellow",
        )
    if len(written) < 2:
        click.secho(
            "  only one site has tiles — a split will not give the study anything to resample",
            fg="yellow",
        )

    definition = build_default_split(builder, written)
    path = definition.write(out or root / "split.json")
    click.secho(f"\nsplit written: {path}", bold=True)
    click.echo(f"  {definition.counts}")


@splits.command("show")
@click.option("--path", type=click.Path(path_type=Path), default="data/tiles/split.json")
def splits_show(path: Path) -> None:
    """Show a split and validate it for geographic leakage."""
    from .pipeline import Split, SplitDefinition, validate_split

    if not path.is_file():
        raise click.ClickException(f"no split at {path}. Run `japgo tiles build --all-sites` first.")

    definition = SplitDefinition.read(path)
    click.echo(f"buffer      : {definition.buffer_tiles} tile(s)")
    click.echo(f"counts      : {definition.counts}")
    click.echo("")
    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        archetypes = definition.archetypes_in(split)
        click.echo(
            f"  {split.value:<6} {len(definition.tiles_in(split)):>5} tiles  "
            f"{', '.join(sorted(archetypes)) or '-'}"
        )
    click.echo(f"  {'buffer':<6} {len(definition.tiles_in(Split.BUFFER)):>5} tiles  (discarded)")

    overlap = definition.archetypes_in(Split.TEST) & definition.archetypes_in(Split.TRAIN)
    if overlap:
        click.secho(
            f"\nwarning: test shares archetype(s) {sorted(overlap)} with train; this measures "
            "transfer between similar places, not generalisation",
            fg="yellow",
        )

    problems = validate_split(definition)
    click.echo("")
    if problems:
        click.secho("INVALID — geography leaks between folds:", fg="red", bold=True)
        for problem in problems:
            click.echo(f"  {problem}")
        sys.exit(1)
    click.secho("valid: no geographic leakage", fg="green", bold=True)


@main.group()
def viz() -> None:
    """Phase 2: visual inspection. No modelling until alignment is confirmed."""


@viz.command("tiles")
@click.argument("tile_ids", nargs=-1)
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default="reports", show_default=True)
@click.option("--decimate", type=int, default=2, show_default=True, help="Pixel decimation.")
@click.option("--limit", type=int, default=None, help="Render at most N tiles.")
def viz_tiles(
    tile_ids: tuple[str, ...], root: Path, out: Path, decimate: int, limit: int | None
) -> None:
    """Render self-contained inspection pages for built tiles.

    With no TILE_IDS, renders every tile in ROOT and writes a contact-sheet index.
    """
    from .pipeline import list_tiles, read_tile
    from .viz import summarise, write_index_page, write_report

    gate = SourceGate(load_registry())
    names = list(tile_ids) or list_tiles(root)
    if not names:
        raise click.ClickException(
            f"no tiles under {root}. Run `japgo tiles build` first."
        )
    if limit is not None:
        names = names[:limit]

    out.mkdir(parents=True, exist_ok=True)
    paths, entries = [], []

    for tile_id in names:
        try:
            bundle = read_tile(root, tile_id)
        except (FileNotFoundError, ValueError) as exc:
            click.secho(f"  {tile_id}: {exc}", fg="red")
            continue

        path = write_report(bundle, gate, out / f"{tile_id}.html", decimate=decimate)
        paths.append(path)
        entries.append(summarise(bundle, gate))
        click.echo(
            f"  {tile_id}  coverage {bundle.coverage:>5.1%}  "
            f"{len(bundle.buildings):>4} buildings  "
            f"{len(bundle.roads.edges) if bundle.roads else 0:>4} edges  -> {path.name}"
        )

    if not paths:
        raise click.ClickException("nothing rendered")

    index = write_index_page(paths, out / "index.html", entries)
    click.echo("")
    click.secho(f"{len(paths)} report(s) written", fg="green", bold=True)
    click.echo(f"open: {index}")
    click.secho(
        "\nPhase 2 gate: confirm building outlines sit on their raster masks and roads "
        "continue across the core boundary before any modelling begins.",
        fg="yellow",
    )


@main.group()
def roads() -> None:
    """Read and measure road networks."""


@roads.command("analyse")
@click.argument("osm_file", type=click.Path(exists=True, path_type=Path))
@click.option("--zone", "zone_number", type=int, default=8, show_default=True)
@click.option("--lod", type=int, default=None, help="Filter to a level of detail (0-4).")
@click.option("--area-km2", type=float, default=1.0, show_default=True)
def roads_analyse(osm_file: Path, zone_number: int, lod: int | None, area_km2: float) -> None:
    """Measure a road network from an OSM extract.

    OSM is training-only: this reads for analysis, never for export.
    """
    from .core import load_hierarchy
    from .sources import OsmAdapter

    gate = SourceGate(load_registry())
    adapter = OsmAdapter(gate, target_crs=zone(zone_number).crs)

    try:
        result = adapter.read(osm_file, purpose="analysis")
    except (ProvenanceViolation, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc

    graph = result.layers["roads"][0]
    if lod is not None:
        graph = graph.at_lod(lod)

    click.secho("source: osm — TRAINING/ANALYSIS ONLY, not for export", fg="yellow")
    click.echo("")
    click.echo(f"edges                : {len(graph.edges)}")
    click.echo(f"nodes                : {len(graph.nodes)}")
    click.echo(f"total length         : {graph.total_length_m / 1000:.2f} km")
    click.echo(f"road density         : {graph.road_density_km_per_km2(area_km2):.2f} km/km²")
    click.echo(f"intersection density : {graph.intersection_density_per_km2(area_km2):.2f} /km²")
    click.echo(f"dead-end ratio       : {graph.dead_end_ratio:.1%}")
    click.echo(f"orientation entropy  : {graph.orientation_entropy():.3f}  (0 = grid, 1 = organic)")
    click.echo(f"components           : {len(graph.connected_components())}")

    click.echo("")
    click.echo("degree histogram:")
    for degree, count in sorted(graph.degree_histogram().items()):
        click.echo(f"  {degree}: {'#' * min(count, 40)} {count}")

    hierarchy = load_hierarchy()
    click.echo("")
    click.echo("by class:")
    by_class: dict[str, float] = {}
    for edge in graph.edges.values():
        by_class[edge.road_class] = by_class.get(edge.road_class, 0.0) + edge.length_m
    for road_class, length in sorted(
        by_class.items(), key=lambda kv: hierarchy.spec(kv[0]).rank
    ):
        click.echo(f"  {road_class:<16}{length / 1000:>8.2f} km")

    for w in result.warnings:
        click.secho(f"warning: {w}", fg="yellow")


@main.command("train")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--split", "split_path", type=click.Path(path_type=Path),
              default="data/tiles/split.json", show_default=True)
@click.option("--scheme", type=click.Choice(["loso", "configured"]), default="loso",
              show_default=True,
              help="loso rotates the held-out site so every fold trains on both flat and steep "
                   "ground; configured uses the single train/val/test assignment in sites.yaml.")
@click.option("--epochs", type=int, default=8, show_default=True)
@click.option("--batch", type=int, default=8, show_default=True)
@click.option("--crop", type=int, default=512, show_default=True)
@click.option("--width", type=int, default=32, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--dice-weight", type=float, default=1.0, show_default=True,
              help="Weight on the soft-Dice term; 0 falls back to BCE only.")
@click.option("--max-pos-weight", type=float, default=5.0, show_default=True,
              help="Ceiling on the BCE positive weight. 5.0 is the reference value; raise "
                   "toward the true inverse frequency only for a sparser target.")
@click.option("--distance-tolerance-px", type=float, default=0.0, show_default=True,
              help="Soften false positives within this distance of a road. For thin targets; off by default.")
@click.option("--fold", "only_fold", default=None,
              help="Run one fold by name or held-out site. A run that dies partway should cost the remaining folds, not the finished ones.")
@click.option("--out", type=click.Path(path_type=Path), default="runs", show_default=True)
def train(root, split_path, scheme, epochs, batch, crop, width, seed, dice_weight,
          max_pos_weight, distance_tolerance_px, only_fold, out):
    """Phase 4: train the baseline and compare it against the non-learned priors.

    The exit criterion is not that it trains — it is that it beats a prior on a site it has never
    seen. Every fold reports the model and both priors on the same held-out tiles, each at its own
    best threshold.
    """
    import logging

    from .model.train import RunConfig, folds_for, train_fold
    from .pipeline.splits import SplitDefinition

    logging.basicConfig(level=logging.INFO, format="  %(message)s", force=True)

    if not Path(split_path).is_file():
        raise click.ClickException(
            f"no split at {split_path}. Run `japgo splits build --root {root}` first."
        )
    split = SplitDefinition.read(split_path)
    folds = folds_for(split, scheme=scheme)
    if only_fold:
        folds = [f for f in folds if only_fold in (f.name, f.held_out)]
        if not folds:
            raise click.ClickException(f"no fold matching {only_fold!r}")

    results = []
    for fold in folds:
        click.secho(f"\n{fold.describe()}", bold=True)
        if not fold.train_tiles or not fold.eval_tiles:
            click.secho("  skipped: a side of this fold is empty", fg="yellow")
            continue
        config = RunConfig(
            root=str(root), fold=fold.name,
            train_tiles=fold.train_tiles, eval_tiles=fold.eval_tiles,
            crop=crop, batch=batch, epochs=epochs, width=width, seed=seed,
            dice_weight=dice_weight, max_positive_weight=max_pos_weight,
            distance_tolerance_px=distance_tolerance_px,
        )
        try:
            result = train_fold(Path(root), fold, config, out_dir=Path(out))
        except (ValueError, RuntimeError) as exc:
            raise click.ClickException(f"{fold.name}: {exc}") from exc

        results.append(result)
        click.echo(f"  model            {result.model['f1']:.3f} F1  "
                   f"(P {result.model['precision']:.3f} R {result.model['recall']:.3f})")
        click.echo(f"  prior: constant  {result.constant['f1']:.3f} F1")
        click.echo(f"  prior: built     {result.built['f1']:.3f} F1")
        if result.topology:
            tp = result.topology
            click.echo(f"  graph            APLS {tp['apls']:.3f}  TOPO F1 {tp['topo_f1']:.3f} "
                       f"(P {tp['topo_precision']:.3f} R {tp['topo_recall']:.3f})")
            click.echo(f"  nodes            {tp['predicted_nodes']} predicted vs "
                       f"{tp['truth_nodes']} real, per tile")
            if tp.get("prior"):
                pp = tp["prior"]
                click.echo(f"  graph prior      APLS {pp['apls']:.3f}  "
                           f"TOPO F1 {pp['topo_f1']:.3f}  ({pp['predicted_nodes']} nodes)")
                won = tp["apls"] > pp["apls"] and tp["topo_f1"] > pp["topo_f1"]
                click.secho(
                    "  topology         " + ("beats the prior on APLS and TOPO"
                                             if won else "DOES NOT beat the prior"),
                    fg="green" if won else "red",
                )
        colour = {"P": "green", "F": "red"}.get(result.verdict()[0], "yellow")
        click.secho(f"  {result.verdict()}", fg=colour)

    if results:
        cleared = sum(1 for r in results if r.verdict().startswith("PASS"))
        click.secho(f"\n{cleared}/{len(results)} fold(s) beat both priors", bold=True)


@main.command("evaluate")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--runs", "runs_dir", type=click.Path(path_type=Path), default="runs",
              show_default=True, help="Directory of *.config.json / *.pt from `japgo train`.")
def evaluate(root, runs_dir):
    """Score saved checkpoints without retraining them.

    Evaluation is deterministic given a fixed checkpoint, so this recovers exactly the figures a
    run printed — a lost log no longer costs a retrain.
    """
    import logging

    from .model.train import evaluate_checkpoint

    logging.basicConfig(level=logging.INFO, format="  %(message)s", force=True)

    configs = sorted(Path(runs_dir).glob("*.config.json"))
    if not configs:
        raise click.ClickException(f"no run configs under {runs_dir}")

    cleared = 0
    for config in configs:
        click.secho(f"\n{config.stem.replace('.config','')}", bold=True)
        try:
            r = evaluate_checkpoint(Path(root), config)
        except (ValueError, FileNotFoundError) as exc:
            click.secho(f"  skipped: {exc}", fg="yellow")
            continue

        click.echo(f"  model            {r.model['f1']:.3f} F1  "
                   f"(P {r.model['precision']:.3f} R {r.model['recall']:.3f})")
        click.echo(f"  prior: built     {r.built['f1']:.3f} F1")
        if r.topology:
            tp = r.topology
            click.echo(f"  graph            APLS {tp['apls']:.3f}  TOPO F1 {tp['topo_f1']:.3f}")
            click.echo(f"  nodes            {tp['predicted_nodes']} predicted vs "
                       f"{tp['truth_nodes']} real, per tile")
            if tp.get("prior"):
                pp = tp["prior"]
                click.echo(f"  graph prior      APLS {pp['apls']:.3f}  "
                           f"TOPO F1 {pp['topo_f1']:.3f}  ({pp['predicted_nodes']} nodes)")
                won = tp["apls"] > pp["apls"] and tp["topo_f1"] > pp["topo_f1"]
                cleared += won
                click.secho("  topology         " + ("beats the prior on APLS and TOPO"
                                                     if won else "DOES NOT beat the prior"),
                            fg="green" if won else "red")

    click.secho(f"\n{cleared}/{len(configs)} checkpoint(s) beat the prior on topology", bold=True)


@main.command("plausibility")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--runs", "runs_dir", type=click.Path(path_type=Path), default="runs",
              show_default=True)
@click.option("--limit", type=int, default=0, show_default=True,
              help="Held-out tiles per fold; 0 uses every one. Sampling a fixed count makes the real reference move when the corpus grows, which invalidates comparison between runs.")
def plausibility(root, runs_dir, limit):
    """Is the output plausible for its environment, even where it is not correct?

    APLS asks whether a specific real network was reproduced. For a generator that is the wrong
    question — what matters is whether the network has the right character for its terrain, and
    whether the archetypes stay distinguishable from one another.
    """
    import json
    import logging

    import numpy as np
    import torch

    from .model.baseline import ROAD_TARGET, best_threshold
    from .model.extract import ExtractionSpec, extract_graph
    from .model.nets import build_unet
    from .model.plausibility import PLAUSIBILITY_METRICS, compare, ordering_preserved
    from .pipeline.channels import load_stack_spec
    from .pipeline.store import read_tile

    logging.basicConfig(level=logging.WARNING, format="  %(message)s", force=True)
    spec = load_stack_spec()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sites = []
    for config_path in sorted(Path(runs_dir).glob("*.config.json")):
        cfg = json.loads(config_path.read_text())
        checkpoint = config_path.with_name(config_path.name.replace(".config.json", ".pt"))
        if not checkpoint.is_file():
            continue

        model = build_unet(spec.depth, width=cfg.get("width", 32))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = model.to(device).eval()

        # All tiles by default. Taking the first N made the ground-truth reference itself
        # shift between runs -- Kawanehon's real road density moved 2.129 -> 3.028 purely
        # because the site grew and a different first ten were sampled, flipping an ordering
        # the model had nothing to do with.
        chosen = cfg["eval_tiles"][:limit] if limit else cfg["eval_tiles"]
        bundles = [read_tile(Path(root), t) for t in chosen]
        bundles = [b for b in bundles if b.roads is not None and b.roads.edges]
        if not bundles:
            continue

        probs = []
        for b in bundles:
            with torch.no_grad():
                x = torch.from_numpy(b.stack[None]).to(device)
                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=device == "cuda"):
                    probs.append(torch.sigmoid(model(x).float())[0, 0].cpu().numpy())

        # The model's own operating point, as everywhere else in this project.
        flat_p = np.concatenate([p.ravel() for p in probs])
        flat_t = np.concatenate([b.target(ROAD_TARGET).ravel() for b in bundles])
        flat_v = np.concatenate([b.channel("valid").ravel() for b in bundles])
        threshold = best_threshold(flat_p, flat_t, valid=flat_v).threshold

        graphs = [
            extract_graph(p, b.tile.read, b.manifest.crs,
                          spec=ExtractionSpec(threshold=threshold), tile_id=b.tile.id)
            for p, b in zip(probs, bundles, strict=True)
        ]
        site = compare(graphs, [b.roads for b in bundles], [b.tile for b in bundles],
                       cfg["fold"].removeprefix("holdout_"))
        sites.append(site)

        click.secho(f"\n{site.site}  ({site.tiles} tiles, threshold {threshold:.2f})", bold=True)
        for a in site.agreements:
            click.secho(f"  {a.describe()}", fg="green" if a.plausible else "yellow")
        click.echo(f"  plausible on {site.score:.0%} of measures")

    if len(sites) < 2:
        return
    click.secho("\narchetype ordering — does the output stay environment-specific?", bold=True)
    kept = 0
    for metric in PLAUSIBILITY_METRICS:
        agrees, real_order, pred_order = ordering_preserved(sites, metric)
        kept += agrees
        click.secho(f"  {'ok ' if agrees else 'NO '} {metric:<32}"
                    f" real {' < '.join(s[:9] for s in real_order)}"
                    f"   predicted {' < '.join(s[:9] for s in pred_order)}",
                    fg="green" if agrees else "red")
    click.secho(f"\n{kept}/{len(PLAUSIBILITY_METRICS)} measures keep the archetypes in the "
                "right order", bold=True)


@main.command("sweep")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True,
              help="A .pt from `japgo train`; its .config.json is read alongside it.")
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--limit", type=int, default=6, show_default=True,
              help="Held-out tiles to sweep. Each perturbation re-predicts every one of them.")
@click.option("--mode", type=click.Choice(["quantile", "scale"]), default="quantile",
              show_default=True,
              help="quantile swaps a real site's slope distribution in; scale multiplies by a "
                   "factor, which invents terrain the model never saw.")
@click.option("--split", "split_path", type=click.Path(path_type=Path),
              default="data/tiles/split.json", show_default=True)
def sweep(root, checkpoint, threshold, limit, mode, split_path):
    """Phase 5: does changing the environment change the road network?

    The project's actual thesis. Holds the model fixed, perturbs one environmental channel at a
    time on real tiles, and reports whether the predicted network moves the way a geographer would
    predict — directions fixed in advance, not read off afterwards.
    """
    import json
    import logging

    import torch

    from .model.nets import build_unet
    from .model.sweep import DEFAULT_SWEEP, RESPONSES, quantile_sweep, run_sweep
    from .pipeline.channels import load_stack_spec
    from .pipeline.splits import SplitDefinition

    logging.basicConfig(level=logging.INFO, format="  %(message)s", force=True)

    config_path = Path(str(checkpoint).replace(".pt", ".config.json"))
    if not config_path.is_file():
        raise click.ClickException(f"no run config beside the checkpoint at {config_path}")
    cfg = json.loads(config_path.read_text())

    spec = load_stack_spec()
    if cfg.get("stack_version") not in (None, spec.stack_version):
        raise click.ClickException(
            f"checkpoint was trained on stack v{cfg['stack_version']} but the corpus is now "
            f"v{spec.stack_version}. Retrain, or the channels do not mean the same thing."
        )

    model = build_unet(spec.depth, width=cfg.get("width", 32))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    site_tiles = None
    perturbations = DEFAULT_SWEEP
    if mode == "quantile":
        if not Path(split_path).is_file():
            raise click.ClickException(
                f"quantile mode needs the split at {split_path} to find each site's tiles"
            )
        split = SplitDefinition.read(split_path)
        site_tiles = {name: sorted(s.tiles) for name, s in split.sites.items()}
        held = cfg["fold"].removeprefix("holdout_")
        others = [s for s in sorted(site_tiles) if s != held]
        if len(others) < 2:
            raise click.ClickException("quantile mode needs two other sites to swap between")
        # Order by median slope so the labels mean what they say.
        import numpy as np

        from .model.sweep import reference_values

        index = spec.index_of("slope")
        medians = {
            s: float(np.median(reference_values(Path(root), site_tiles[s][:4], index)))
            for s in others
        }
        # Expectations are set against the held-out site's own slope, not against the other
        # reference: sweeping the flattest site, *both* references are steeper than home.
        home = float(np.median(reference_values(Path(root), cfg["eval_tiles"][:4], index)))
        click.echo(f"  held out slope p50 {home:.3f}; references " + ", ".join(
            f"{s} {m:.3f} ({'flatter' if m < home else 'steeper'})" for s, m in medians.items()
        ))
        perturbations = quantile_sweep(medians, home)

    click.secho(f"\nsweep on {cfg['fold']} (held out: {cfg['eval_tiles'][0]} ...)", bold=True)
    results = run_sweep(Path(root), model, cfg["eval_tiles"], threshold=threshold, limit=limit,
                        perturbations=perturbations, site_tiles=site_tiles)

    agreed = total = 0
    for r in results:
        note = f"  [{r.clamped:.0%} clamped]" if r.clamped > 0.01 else ""
        how = "quantile-mapped" if r.factor != r.factor else f"x{r.factor}"
        click.secho(f"\n  {r.perturbation}: {r.channel} {how}  ({r.tiles} tiles)"
                    + note, bold=True)
        for response in RESPONSES:
            got, want = r.direction(response), r.expect.get(response, "?")
            ok = r.agrees(response)
            if r.perturbation != "null":
                agreed += ok
                total += 1
            click.secho(
                f"    {response:<32} {r.baseline[response]:8.3f} -> {r.perturbed[response]:8.3f}"
                f"   {got:<5} (expected {want})",
                fg="green" if ok else "red",
            )

    click.echo("")
    if total:
        colour = "green" if agreed / total >= 0.6 else "red"
        click.secho(f"{agreed}/{total} responses moved as predicted", fg=colour, bold=True)
        click.echo("The null row must be flat; a response there is inference noise, not signal.")


@main.command("study")
@click.option("--root", type=click.Path(path_type=Path), default="data/tiles", show_default=True)
@click.option(
    "--split",
    "split_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Split definition, for site grouping. Without it every tile counts as one site and no "
    "interval can be estimated.",
)
@click.option("--iterations", type=int, default=2000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True, help="Pinned for reproducibility.")
@click.option("--limit", type=int, default=None, help="Show only the top N in each section.")
def study(root: Path, split_path: Path | None, iterations: int, seed: int, limit: int | None):
    """Phase 3: which environmental features predict road structure?

    Reports the null results alongside the supported ones, because the phase's exit criterion
    asks for both. An association whose interval spans zero is a finding, not a gap.
    """
    from .analysis.study import run_study

    try:
        result, skipped = run_study(
            root, split_path=split_path, iterations=iterations, seed=seed
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc}\nBuild a corpus first: japgo tiles build --all-sites"
        ) from exc

    if not result.tiles:
        raise click.ClickException(
            f"no usable tiles under {root}. One tile is an anecdote and none is not a study — "
            "see docs/decision-log.md."
        )

    click.echo(result.report(limit=limit))

    for note in skipped:
        click.secho(f"skipped {note}", fg="yellow")


if __name__ == "__main__":  # pragma: no cover
    main()
