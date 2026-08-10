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
def tiles_build(
    site: str | None,
    data_root: Path,
    out: Path,
    resolution: float,
    limit: int | None,
    all_sites: bool,
    min_coverage: float,
) -> None:
    """Build a manifest-carrying tile set for a site.

    Source files are found by convention under DATA_ROOT/<site>/{terrain,plateau,landuse,roads}/.
    """
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

    for name in names:
        if name not in sites.sites:
            raise click.ClickException(f"unknown site {name!r}; have {', '.join(sites.sites)}")

        spec = sites.sites[name]
        files = SourceFiles.discover(data_root, name)
        planned = builder.tiles_for(name)

        click.secho(f"\n{name}  ({spec.archetype})", bold=True)
        click.echo(f"  tiles planned : {len(planned)}")
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


if __name__ == "__main__":  # pragma: no cover
    main()
