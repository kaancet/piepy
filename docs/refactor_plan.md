# piepy refactor roadmap

A staged plan to make piepy experiment-agnostic, robust, and cluster/dashboard-ready
without breaking the working pipeline. Each phase is independently shippable.

## Guiding principles

1. **Stable core + experiment plugins.** A small, well-tested core (paths, IO, schema,
   concat, parallel, stats, fitting, viz) that knows nothing about wheel/detection.
   Experiments are plugins that register a Trial schema + handler and (optionally) a
   Session/Hub. Adding an experiment should never require touching core.
2. **Safety net before surgery.** No automated tests exist today. Land golden-master
   tests on real fixture sessions first so refactors are provably behavior-preserving.
3. **Parallel tracks, never break `main`.** Build new modules alongside the old, port one
   consumer at a time behind a feature flag / import switch, delete the old only at parity.
4. **One canonical data contract.** Most fragility (hand-rolled dtype reconciliation in
   `hub.py`/`mouse.py`, run/session/mouse concat) stems from there being no single schema.
   Fix that once and concat, aggregation, cluster sharding, and the dashboard all simplify.

---

## Current-state assessment (what I read)

- **Pathfinding** (`core/pathfinder.py`): `glob.glob(sessiondir + "*")` + `os.walk` to guess
  runs, hardcoded log types, "presentation XOR training" assertion, "analysis" special-cased,
  ambiguous list-vs-str `Paths` attributes resolved by `path_idx`. Brittle and positional.
- **Session-name parsing** (`core/run.py` `RunMeta`): positional `sessiondir.split("_")`
  (`baredate, animalid, ..., imaging_mode, user_id`). Any naming deviation silently misparses.
- **Data contract**: `Trial` is a `patito` model extended at runtime in `TrialHandler`.
  Schemas are reconciled by hand in `hub.gather_sessions`, `MouseData.append`
  (loops casting dtypes, bare `except`, `print("jlsdiobjsdf")`). This is the core debt.
- **Run/Session**: `read_combine_logs` already stitches multiple stim/rig logs, but `Session`
  treats each run as separate (`run_count = len(stimlog)`); there is no run *concatenation*.
- **Parallelism** (`core/hub.py`): `multiprocessing.Pool(spawn)` + manual concat. No
  resume, no per-session cache reuse during the run, not cluster-aware.
- **Stats/Fitting**: `core/statistics.py` (628 LOC of free functions) and
  `psychophysics/fit_funcs.py` + `models/` are loose function bags, not a composable API.
- **Plotting**: ~12k LOC across duplicated matplotlib **and** bokeh trees, mostly
  experiment-specific (e.g. `wheelSessionPlotter.py` 1488 LOC). A WIP `plotters/core/`
  re-implements behaviz's `PlotSpec` by hand but does not yet depend on behaviz.
- **No tests.** `requires-python >=3.10`; pinned `numpy==2.2.6`, `matplotlib==3.10.0`,
  `polars~=1.19`, `patito~=0.8.3`.

---

## Target package layout

```
piepy/
  core/
    paths/        # locator, layout templates, session-name parsers (pluggable)
    schema/       # canonical base trial schema + registry + concat/align helpers
    io/           # parquet/mat/hdf5 read-write, lazy scan
    parsing/      # stimpy/riglog parsers, log repair (today's parsers.py + repairs)
    parallel/     # pluggable executor: local / joblib / submitit(SLURM) / dask
    registry.py   # @register_paradigm decorator + entry-point discovery
  experiments/    # was psychophysics/* — each is a plugin
    wheel_detection/   (trial schema, handler, session, hub)
    wheel_discrimination/
  stats/          # consolidated statistics API
  fitting/        # models (sigmoid/weibull/erf/lapse) + Fit objects
  viz/            # behaviz-backed plots + PlotSpec presets (replaces plotters/)
  imaging/        # widefield/2p (optionally temporaldata-backed raw layer)
```

`.tach` is already present but unconfigured — add a `tach.toml` to enforce that
`experiments/*`, `stats`, `fitting`, `viz` may import `core` but not each other, and that
`core` imports none of them.

---

## Phases

Each phase lists: **Goal · Changes · Deliverable · Effort · Risk · Depends on**.

### Phase 0 — Safety net & env (prerequisite)
- **Goal:** Make the overhaul safe and reproducible.
- **Changes:** Add `pytest`; commit 2–3 *small* real fixture sessions (single-run,
  multi-run, with-imaging). Golden-master tests: parse fixture → snapshot `runData.parquet`
  → assert byte/df-equality. Add `ruff` + pre-commit. Add a `uv` lockfile. As part of this,
  *attempt* a Python 3.12 venv and record what breaks (don't block on it).
- **Deliverable:** `tests/`, CI (GitHub Actions), green baseline.
- **Effort:** M · **Risk:** Low · **Depends on:** —

### Phase 1 — Canonical data contract + run concatenation
- **Goal:** One schema, one concat path; deliver multi-run concatenation.
- **Changes:**
  - Promote `Trial` into an explicit base schema with reserved core columns
    (`trial_no`, `t_trialstart`, `t_trialend`, `animalid`, `baredate`, `date`, `run_no`,
    `run_uid`, `session_uid`, `paradigm`). Experiments extend it.
  - Single `align_and_concat(frames)` in `core/schema` that reconciles dtypes/missing
    columns *once* (supersedes the bespoke loops in `hub.py` and `mouse.py`).
  - `Session.concatenate_runs(policy=...)`: stacks run trial-tables with a `run_no`/`run_uid`
    column, cumulative `total_trial_no`, and an explicit time policy (per-run reset vs.
    monotonic session clock — important for later event alignment).
- **Deliverable:** `session.concatenate_runs()`; hub/mouse aggregation reuse the shared
  concat; golden tests extended to a concatenated multi-run session.
- **Effort:** M–L · **Risk:** Med · **Depends on:** Phase 0

### Phase 2 — Robust pathfinding
- **Goal:** Replace glob/positional discovery with declarative, testable resolution.
- **Changes:**
  - `SessionLocator` that resolves a session against **configurable layout templates**
    (e.g. `{root}/presentation/{session}/{run}`) using `pathlib`, returning a structured
    `SessionManifest` (runs + their artifacts) instead of parallel lists.
  - Pluggable **session-name parser** (regex/format spec per lab) replacing positional
    `split("_")`; clear errors on mismatch.
  - Drop the presentation-XOR-training assumption; make "kinds" of source dirs declarative.
- **Deliverable:** `core/paths`; old `PathFinder` shimmed to it, then removed. Tested with
  synthetic directory trees (no real data needed).
- **Effort:** M · **Risk:** Med (touches every entry point) · **Depends on:** Phase 0
  (can run in parallel with Phase 1 — different files).

### Phase 3 — Statistics + Fitting modules
- **Goal:** Composable, reusable analysis primitives.
- **Changes:**
  - `piepy.stats`: estimators returning `(value, ci)`, tests returning a small result
    object (`statistic`, `pvalue`, `effect_size`, `n`). Wrap today's functions; keep names.
  - `piepy.fitting`: a `Model` protocol (`predict`, `n_params`, `bounds`) with
    sigmoid/weibull/erf/lapse registered; `fit(model, x, y) -> Fit` exposing `params`,
    bootstrap `param_ci`, `predict()`, and goodness-of-fit. Plays directly into viz.
- **Deliverable:** `piepy.stats`, `piepy.fitting` with unit tests; old call sites redirected.
- **Effort:** M · **Risk:** Low · **Depends on:** Phase 0 (loosely Phase 1 for grouped aggregation).

### Phase 4 — Plotting on behaviz
- **Goal:** Collapse the duplicated mpl/bokeh trees into thin, declarative plots.
- **Changes:**
  - Add `behaviz` as a dependency; delete the hand-rolled `plotters/core/PlotSpec` in favor
    of behaviz's `PlotSpec`/`AxisSpec`/`FigureSpec`.
  - Build `piepy.viz` as thin functions over `plot_line/plot_errorbar/...` + saved presets
    (`~/.behaviz`). Backend switch (`set_renderer`) makes the separate bokeh tree obsolete.
  - Port plot-by-plot (psychometric first — already started), keep legacy plotters until
    parity, then delete `plotters/` (~12k LOC retired).
- **Deliverable:** `piepy.viz` with psychometric/progress/reaction-time/wheel ports.
- **Effort:** L · **Risk:** Med · **Depends on:** Phase 3 (plots render fits/CIs).

### Phase 5 — Cluster / parallel execution
- **Goal:** Run swiftly on the cluster from the terminal.
- **Changes:**
  - Abstract `gather_sessions` behind a `core/parallel` executor interface with backends:
    local `Pool`, `joblib`, `submitit` (SLURM array jobs), optionally `dask`.
  - Make per-session analysis idempotent: write **one parquet per session**, reuse on rerun,
    support `--resume`. Final aggregate is `pl.scan_parquet(dir)` (lazy, cluster- and
    dashboard-friendly) rather than in-memory concat of every session.
  - Structured logging + a CLI that runs headless on a node.
- **Deliverable:** `piepy run --paradigm ... --backend slurm`; resumable, sharded output.
- **Effort:** M–L · **Risk:** Med · **Depends on:** Phase 1 (canonical schema/per-session parquet).

### Phase 6 — temporaldata raw layer (spike first; see brainstorm below)
- **Goal:** Easier, downstream-compatible event/imaging alignment.
- **Effort:** M (prototype S) · **Risk:** Med (young dep) · **Depends on:** Phase 1.

### Phase 7 — Python 3.12
- **Goal:** Modern runtime.
- **Changes:** Resolve whatever Phase 0 flagged (watch `patito`/`polars`/`numpy`); bump
  `requires-python`, add 3.12 to the test matrix.
- **Effort:** S–M · **Risk:** Low–Med · **Depends on:** Phase 0; easiest once tests are green.

---

## Recommended priority order

1. **Phase 0** — safety net (do first, always).
2. **Phase 1** — data contract + run concatenation (highest leverage; unblocks 5, 6, dashboard;
   delivers a feature you explicitly asked for).
3. **Phase 2** — pathfinding (your #1 ask; biggest daily-friction win; parallelizable with 1).
4. **Phase 3** — stats + fitting (prerequisite content for plotting).
5. **Phase 4** — behaviz plotting (largest LOC win; needs 3).
6. **Phase 5** — cluster (slot in opportunistically once 1 lands).
7. **Phase 6** — temporaldata spike (low-commitment, high-optionality; anytime after 1).
8. **Phase 7** — Python 3.12 (cheap once tests exist).

Parallelizable: **1 ∥ 2**, then **3 ∥ (start 5)**, then **4**. **6** as an independent spike.

---

## temporaldata brainstorm

**Why it fits.** piepy's `rawdata` is already a dict of polars frames keyed by `duinotime`
(`vstim`, `screen`, `lick`, `reward`, `opto`, `statemachine`, cam logs). That is exactly
`IrregularTimeSeries` (events) + `Interval` (trials/stim epochs) + `RegularTimeSeries`
(widefield/2p frames). The trial parser's manual
`filter(duinotime.is_between(start, end))` becomes `data.slice(trialstart, trialend)`.

**Strategic win.** temporaldata comes from the neuro-galaxy / POYO stack. If the lab's
downstream neural packages use (or could use) it, adopting it at piepy's raw layer gives
**shared, aligned time representations** across behavior and neural analysis — exactly where
piepy serves as a baseline.

**Where it belongs — and where it does NOT.** Use it for the **raw/intermediate layer**
(continuous + event streams + trial intervals per run). Keep the **trial table (polars,
one row per trial)** as the analysis layer for psychophysics aggregation/stats/plotting.
Don't try to replace the trial table with temporaldata — different jobs.

**Concrete shape.** `Run.to_temporaldata() -> temporaldata.Data` holding trial `Interval`s +
event `IrregularTimeSeries` + imaging `RegularTimeSeries`. Trial parsing can be reimplemented
on top of it; widefield/2p alignment to trials becomes trivial and consistent downstream.

**Caveats / decision.** Younger library, pandas-based (not polars), h5py serialization
(parquet stays for the trial table). Wrap it behind a thin adapter so API churn can't reach
the rest of piepy. **Recommendation:** a 1–2 day spike on one widefield session to prove the
imaging-alignment win before committing — medium-low priority, high optionality.

---

## How this sets up the far-future dashboard

Nothing here is dashboard work, but the phases line it up: a **canonical schema** (1) +
**one-parquet-per-session with lazy `scan_parquet`** (5) + **behaviz's bokeh backend** (4)
means the multi-animal training dashboard is mostly a thin Panel/Bokeh app over a lazy frame,
not a fresh plotting stack.
