"""piepy command-line interface: ``piepy <session|hub|training-report|dashboard>``.

Thin argparse wrappers over the analysis API (Session / Hub / Mouse). Shared flags
(``--verbose``, ``--output``) live on a parent parser inherited by every subcommand.
Heavy imports are done inside each handler so ``piepy --help`` stays fast.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime as dt

from .core.config import config as cfg
from .core.io import display

# the session/cohort viz plots to render+save (skipped individually if not yet drawable)
_PLOTS = ("psychometric", "reaction_time_cloud", "reaction_time_dist")


def _common() -> argparse.ArgumentParser:
    """Parent parser holding the flags every subcommand shares."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=cfg.verbose,
        help="terminal verbosity (default: from config)",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="override the default save location for outputs",
    )
    return p


def _save_plots(viz, outdir: str, prefix: str) -> None:
    """Render each ``_PLOTS`` entry off ``viz`` and save its figure as a PNG."""
    os.makedirs(outdir, exist_ok=True)
    for name in _PLOTS:
        try:
            res = getattr(viz, name)()
        except Exception as exc:  # noqa: BLE001 - a bad plot shouldn't sink the others
            display(f" >> skip plot {name}: {exc}", color="yellow")
            continue
        fig = res.figure[0] if isinstance(res.figure, tuple) else res.figure
        if fig is None:  # plot's behaviz drawing not filled in yet
            continue
        path = os.path.join(outdir, f"{prefix}_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        display(f"saved {path}", color="green")


def cmd_session(opts) -> None:
    from .core.registry import get_session_class

    session = get_session_class(opts.paradigm)(opts.sessiondir)
    session.analyze(load_flag=opts.load)
    if opts.plot:
        outdir = opts.output or os.path.join(
            cfg.paths["analysis"][0], opts.sessiondir, "plots"
        )
        _save_plots(session.viz, outdir, opts.sessiondir)


def cmd_hub(opts) -> None:
    import polars as pl

    from .core.hub import Hub
    from .core.mouse import list_animal_sessions

    frames = [list_animal_sessions(a) for a in opts.animalids]
    sessions = pl.concat(frames) if frames else pl.DataFrame()
    if not sessions.is_empty():
        sessions = sessions.filter(pl.col("paradigm") == opts.paradigm)
    session_list = sessions["sessiondir"].to_list() if not sessions.is_empty() else []
    if not session_list:
        display("No sessions found for the given animals/paradigm.", color="red")
        return

    hub = Hub(opts.paradigm)
    hub.initialize(session_list, load_sessions=opts.load)
    hub.save(opts.output)  # None -> Hub's default (analysis dir)
    if opts.plot:
        outdir = os.path.join(opts.output or cfg.paths["analysis"][0], "hub_plots")
        _save_plots(hub.viz, outdir, "hub")


def cmd_training_report(opts) -> None:
    from tabulate import tabulate

    from .core.mouse import Mouse

    rows = []
    for animal in opts.animalids:
        m = Mouse(animal, paradigm=opts.paradigm)
        if m.session_list.is_empty():
            display(f"No {opts.paradigm} sessions for {animal}", color="yellow")
            continue
        # ponytail: no_load reanalyzes every session each run (correct, not fast);
        # switch to "load_and_add" once per-session parquet caching (Phase 5) lands.
        m.gather_data(load_type="no_load")
        s = m.data.summary_data
        if s is None or s.is_empty():
            continue
        first, last = s[0, "date"], s[-1, "date"]
        days = (dt.strptime(last, "%y%m%d") - dt.strptime(first, "%y%m%d")).days
        rows.append(
            [
                animal,
                first,
                days,
                s.height,
                s[-1, "level"],
                s[-1, "hit_rate"] if "hit_rate" in s.columns else None,
            ]
        )

    headers = [
        "animal id",
        "date started",
        "days in training",
        "sessions trained",
        "current level",
        "latest performance",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def cmd_widefield(opts) -> None:
    from .core.registry import get_session_class
    from .imaging.widefield import save_averages, widefield_from_run

    session = get_session_class(opts.paradigm)(opts.sessiondir)
    session.analyze(load_flag=opts.load)  # populate each run's trial table + frame ids

    outroot = opts.output or os.path.join(
        cfg.paths["analysis"][0], opts.sessiondir, "widefield"
    )
    # tpre/tpost are seconds; widefield_from_run's pre_t/post_t are ms
    kwargs = dict(
        pre_t=opts.tpre * 1000.0,
        post_t=opts.tpost * 1000.0,
        downsample=opts.downsample,
        timestamp_precision=opts.precision,
    )
    for i, run in enumerate(session.runs):
        try:
            results = widefield_from_run(run, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad run shouldn't sink the rest
            display(f" >> skip run {i}: {exc}", color="yellow")
            continue
        paths = save_averages(results, os.path.join(outroot, f"run{i}"))
        display(
            f"run {i}: saved {len(paths)} movie(s) under {outroot}/run{i}", color="green"
        )


def cmd_dashboard(opts) -> None:
    display("dashboard: not implemented yet (placeholder).", color="cyan")


def build_parser() -> argparse.ArgumentParser:
    """The full ``piepy`` argument parser (separate from ``main`` so it's testable)."""
    common = _common()
    parser = argparse.ArgumentParser(prog="piepy", description="piepy analysis CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("session", parents=[common], help="parse one session")
    ps.add_argument("sessiondir", help="e.g. 240810_KC150_detect__no_cam_KC")
    ps.add_argument("-p", "--paradigm", required=True, help="e.g. wheel_detection")
    ps.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)
    ps.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ps.set_defaults(func=cmd_session)

    ph = sub.add_parser("hub", parents=[common], help="parse animals in a paradigm")
    ph.add_argument("animalids", nargs="+", help="one or more animal ids (e.g. KC150)")
    ph.add_argument("-p", "--paradigm", required=True, help="e.g. wheel_detection")
    ph.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)
    ph.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ph.set_defaults(func=cmd_hub)

    pt = sub.add_parser(
        "training-report", parents=[common], help="training table for animals"
    )
    pt.add_argument("animalids", nargs="+", help="one or more animal ids (e.g. KC150)")
    pt.add_argument("-p", "--paradigm", required=True, help="e.g. wheel_detection")
    pt.set_defaults(func=cmd_training_report)

    pw = sub.add_parser(
        "widefield", parents=[common], help="analyze a widefield imaging session"
    )
    pw.add_argument("sessiondir", help="e.g. 240810_KC150_detect_1P__onepcam_KC")
    pw.add_argument("-p", "--paradigm", required=True, help="e.g. wheel_detection")
    pw.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)
    pw.add_argument(
        "--tpre", type=float, default=0.0, help="pre-stimulus time to include (s)"
    )
    pw.add_argument(
        "--tpost", type=float, default=0.0, help="post-stimulus time to include (s)"
    )
    pw.add_argument("--downsample", type=int, default=1, help="spatial downsample factor")
    pw.add_argument(
        "--precision",
        type=float,
        default=1e-6,
        help="timestamp_precision (numeric, seconds)",
    )
    pw.set_defaults(func=cmd_widefield)

    pd = sub.add_parser(
        "dashboard", parents=[common], help="launch the dashboard (placeholder)"
    )
    pd.set_defaults(func=cmd_dashboard)

    return parser


def main(argv=None) -> None:
    opts = build_parser().parse_args(argv)
    cfg.set_verbosity(opts.verbose)
    opts.func(opts)


if __name__ == "__main__":
    main()
