# piepy architecture review and Opus handoff

Branch reviewed: `claude/codebase-architecture-review-d8114c` at `6116f70` (worktree, clean). Date: 2026-09-06.
Evidence base: full read of `piepy/` (113 files, ~12.8k LOC) and `tests/`, the Graphify map (`graphify-out/`, 1528 nodes, no import cycles), the data-free suite run against this tree (194 passed), ruff over the package, dependency-API probes in the project venv (polars 1.19.0, pandas 2.3.3, tifffile 2026.6, patito 0.8.4, py3.12), and one read-only timing/profile run on a local detection session.

---

## 1. Executive assessment

**Health: good, and clearly post-refactor.** The core that the roadmap in `docs/refactor_plan.md` set out to build exists and is the strongest part of the codebase: `core/schema.py` (identity + one concat path), `core/paths/` (locator + pluggable name parser, structured errors), `core/registry.py` (paradigm registry incl. wiring-only synthesis and drop-in discovery), `core/errors.py`, `stats/`, `fitting/`, `viz/` (plots as functions over behaviz), `imaging/{windows,average,executor,widefield}` (pure, order-independent reductions), `simulations/`, `temporal/`. These modules are small, documented, tested data-free, and dependency direction is coherent (plugins -> core; core never imports plugins except the lazy builtin table in the registry; `viz` is reached from core only through lazy imports on the `.viz` property).

**Where the debt actually is.** Almost all real problems sit in the older "glue": `core/run.py`, `core/mouse.py`, `core/hub.py` save/error paths, the two paradigm `*Session.py` modules' `get_run_stats`, `core/parsers.py`, `core/log_repair_functions.py`, and a ring of dead legacy modules (`dbinterface`, `experimenter`, `session_launcher`, `logger`, `psychophysicalRunData`, `multiSenseTrial`, `imaging/scripts.py`, `models/*`). These are concrete, verifiable defects (crashes on edge cases, removed-API calls, a Python-3.12-only syntax in a `>=3.10` package, a raw-data-dir write side effect), not style.

**One measured hotspot.** Trial extraction costs ~24 ms/trial (12.06 s for 499 trials). 49% of that is patito rebuilding the trial model class on every trial (`TrialHandler._update_model`), 14% is per-trial polars filtering of every raw channel, 13% is wheel-trace processing. The first is pure accidental cost and fixable without touching output.

**Verdict.** Architecture is stable; no wholesale refactor is warranted. The right work is a short list of localized correctness fixes, one perf fix, one dead-code sweep, and two small API alignments. Everything else should be left alone. Confidence: high on findings verified by execution (marked "verified"), medium on the ones established by code reading only.

---

## 2. Architecture map

```
                 cli.py  (argparse; imports handlers lazily)
                    |
   Session (core/session.py) ----> Run (core/run.py) ----> TrialHandler (core/trial.py)
        |  uses SessionLocator          |  uses parsers.py, log_repair_functions.py
        |  uses schema.concat_session_runs / attach_run_identity
   Hub (core/hub.py) --- registry.get_paradigm ---> ParadigmSpec(session_cls)
   Mouse (core/mouse.py) -- registry.get_session_class
        |  both stack sessions with schema.align_and_concat
   stats/  fitting/  viz/ (plots read Run/Session/Hub/DataFrame via viz.base._resolve)
   temporal/ (SessionStreams over the session-clock trial table) <- tasks/wheel_detection/wheelDetectionStreams
   imaging/ (windows -> average -> dff; widefield_from_run reads run.data.data + run.paths.onepcam)
   plugins: psychophysics/tasks/{wheel_detection,wheel_discrimination}, sensory/visual
            (Run subclass = trial_handler_cls + state_transitions + augment/enrich/compute_stats hooks)
```

Data flow: `.stimlog/.riglog` -> `parse_stimpy_log` -> `rawdata: dict[str, pl.DataFrame]` (per channel, keyed on `duinotime`) -> `translate_state_changes` -> repairs -> per-trial `TrialHandler.get_trial` (dict of lists) -> `pt.DataFrame(...).validate()` -> `to_plain_polars` -> `augment_data` (pure df transforms) -> `compute_stats` -> `enrich_data` -> provenance hash column -> `runData.parquet` + `sessionStats.json` + `runProvenance.json` under `analysis/<session>/[run<NN>]`. Session-level: `concatenate_runs` (session clock, keep-both policy) -> Hub/Mouse `align_and_concat`.

Configuration: `~/.piepy/config.json`, loaded at import of `core/config.py` into a module singleton (with a one-time backfill write). Paradigm drop-ins in `~/.piepy/paradigms/<name>/{scheme.json,handler.py}`.

Error strategy: `PiepyError(problem, where, fix, hint)` and category subclasses; adopted in paths/opto/viz, not yet in run/parsers/trial handlers.

Concurrency: `Hub.gather_sessions` uses `multiprocessing.Pool` (spawn) with a module-level worker that rebuilds the paradigm from its name; `imaging.executor` has Local/Process executors with order-independent partial sums.

Tests: data-free suite (CI, 194 tests) covers schema, paths, registry, stats, fitting, viz accessor/base, simulations, temporal, imaging windows/average, CLI arg parsing, hub/mouse concat mechanics. `needs_data` golden-master tests (local only) guard byte-identical parse output for 4 real sessions. Not covered anywhere: Hub pool path, Mouse load modes, `get_run_stats` edge cases, `translate_state_changes` error path, `parse_stimpygithub_log`, legacy imaging classes.

Graphify signals checked against source: god nodes `display()` (a logger, fine), `Run` (44 edges, the pipeline hub, fine), `TrialHandler`/`WheelTrace` (algorithmic cores), `_resolve()` (the viz seam, intentional). No cycles. Isolated components (`dbinterface`, `experimenter`, `logger`, `session_launcher`, `models/*`, `eyecam/facecam`) are dead code, confirmed by grep: no importers outside themselves and the smoke-import list.

---

## 3. What should NOT change

Defend these. A later agent should not "improve" them.

1. **`core/schema.py` contract.** SHA-256 `session_uid`/`run_uid`, `align_and_concat` via `diagonal_relaxed` with scalar-to-list promotion, `concat_session_runs` keep-both time policy (`run_time_offset` + `*_session` copies, `session_trial_no`). Consumers: Hub, Mouse, temporal streams, viz. The hard-coded default time-list column names (`wheel_t`, `lick`, `opto_pulse`, `reward`) look like core knowing about wheel data, but they are already parameters with defaults; leave them.
2. **`core/paths/`**: exact-match resolution, "flat only for a single run" layout rule, prefix `^run\d+` run dirs, `RunArtifacts` exposing the old `Paths` attribute surface, three-layer name parsing (override > scheme regex > default). This is the design the user asked for and it is well tested.
3. **`core/registry.py`**: class-attribute wiring on `Run` (`trial_handler_cls`, `rundata_cls`, `state_transitions`), synthesized named classes, lazy builtin import, drop-in discovery, `_validate_transitions`. Small, tested, and it is what lets spawn workers rebuild a paradigm from a string.
4. **patito dynamic model expansion in `TrialHandler._update_model`.** vstim columns genuinely vary per rig/StimPy version, so the model must grow at parse time. Make it cheaper (Plan 7); do not replace it.
5. **TrialHandler mixin layering** (`WheelDetectionTrialHandler(VisualTrialHandler, PsychophysicalTrialHandler)`) and the per-paradigm `get_trial` sequences. Order of calls encodes rig timing semantics (sync before state events, wheel reset preference rig > state). Golden-guarded. Leave.
6. **`to_plain_polars`** (strip patito subclass so Hub workers can pickle results). Regression-tested; required.
7. **Log-repair heuristics**: `add_total_iStim` scenarios, `extrapolate_time`, `stitch_logs` value offsets, the visual `+300/+200 ms` offsets in `VisualRun.repair_rawdata`, `fix_first_line_state_logging`. Rig quirks with no spec; only golden tests can validate them. Do not "clean up".
8. **`WheelTrace.get_movements`** Hankel-window algorithm and `match_response_movement` two-pass matching. Domain algorithm, deliberate constants.
9. **stats/fitting/viz design**: pure functions, tidy output schema `[*group, metric, value, sem, ci_low, ci_high, n]`, `PlotResult`, `_resolve` duck typing, lazy `import behaviz` inside each plot, `Viz` accessor scope defaults. This is the user's explicit design decision (functions, not classes).
10. **imaging/{windows,average,executor}**: integer partial sums, dF/F only on the finished mean, `FrameWindows` front alignment. Correct and tested.
11. **`config.py` import-time singleton** (reads/creates `~/.piepy`, backfills `paradigms`). It is a smell, but every entry point, the CLI, the tests' monkeypatching and spawn workers depend on it. Changing it is a cross-cutting change with no current pain. Leave.
12. **`display()` as the logger.** Migrating to `logging` touches ~50 call sites for no user-visible gain. Leave.
13. **`enrich_data` 1-row join-on-`run_no` broadcast** in the detection/discrimination Runs. Odd but golden-stable and it handles list-valued columns for free. Only fix the guard (Plan 2), do not restructure.
14. **`generate_unique_session_id` semantics** (`hash(date, animal)`, 7 digits) as the `session_id` column. Consumers exist; do not unify with `session_uid` without a coordinated plan.

---

## 4. Findings

Legend: P0 critical, P1 high value, P2 opportunistic, P3 cosmetic. "verified" = reproduced by execution in this review; "read" = established from source.

| ID | Pri | Conf | Category | Area | Observation | Assessment | Recommendation |
|---|---|---|---|---|---|---|---|
| F01 | P1 | verified | portability | `core/log_repair_functions.py:149` | `f"{save_path}/{dt.today().strftime("%Y%m%d")}_pseudo.camlog"` reuses `"` inside an f-string. `python3.9 -m py_compile` -> `SyntaxError: f-string: unmatched '('`; same on 3.10/3.11. | `pyproject` declares `>=3.10` and 3.10/3.11 classifiers; `import piepy.core.run` fails on those interpreters. CI only runs 3.12 so it is invisible. | Plan 1: use `'%Y%m%d'`; decide declared support. |
| F02 | P1 | read | correctness / side effect | `convert_riglog_to_camloglike` (`log_repair_functions.py:142`), `Run.read_run_data:452`, `imaging/widefield.run_frame_period_ms`, `paths/locator._attach_aux` | On every parse of a mesorig session (onepcam dir, no camlog) a `<today>_pseudo.camlog` is **written into the raw camera data dir**. | Raw data dir mutated by analysis; file accumulates per day; `run_frame_period_ms` raises when `!= 1` log file; on the next parse the locator finds the pseudo file as `onepcamlog` and takes the labcams branch instead. Comment in code admits it. | Plan 8: derive frame period in memory from `rawdata["onepcam_log"]`, stop writing. Needs user confirmation (behavior change). |
| F03 | P1 | verified | correctness | `wheelDetectionSession.get_run_stats`, `wheelDiscriminationSession.get_run_stats`, both `Run.__repr__` | Detection: `round(nonopto hits median())` -> `TypeError: NoneType doesn't define __round__` when a run has no non-opto hit (reproduced); `ZeroDivisionError` when `stim_trial_count`/`len(nonopto_data)` is 0; `d_prime` -> inf/nan at 0%/100%. Discrimination: `nonopto_hit_rate` filters `outcome == "hit"` (outcomes are correct/incorrect) so it is always 0 (reproduced); key `"median_response_latency "` has a trailing space; `__repr__` reads `stats['hit_rate']`/`['false_alarm_rate']` which discrimination never produces -> `KeyError` on `repr(run)`. | `compute_stats` is on the `analyze_run` path, so early-training or short runs abort the whole session parse; Hub/Mouse swallow it as a "faulty session". Discrimination stats are partly wrong and its Run repr crashes in notebooks. | Plan 2. |
| F04 | P1 | read | error handling | `core/hub.py:43-60,149-157` | `_analyze_one` wraps only `spec.session_cls(name)` in try; `session.analyze()` runs outside it, so one failing session raises inside `pool.map` and kills the gather (docstring promises "empty frame on failure"). `Hub.save(None)` reads `self.data[-1, "session_path"]`, a column that no longer exists (identity columns are `session_uid`...), so `piepy hub ...` without `--output` crashes after the full gather. `_filter_session_list` is never called. | Cohort gathers are long; losing them to one bad session or at the save step is the worst failure mode. | Plan 3. |
| F05 | P1 | read | correctness | `core/mouse.py:252-258, 141-157, 394` | `gather_data(load_type=None)` validates `load_type in load_modes` before applying the `None -> "last_saved"` default -> `ValueError` on the documented default call. `MouseData.load` uses `pl.Expr.apply` (removed in polars 1.0; verified absent) -> `AttributeError`, so `load_and_add` and `last_saved` modes cannot work. `isSaved` error path references `self.cumul_file_loc`/`self.summary_file_loc` (never set). | Only `no_load`/`reanalyze` work, which is why the CLI training report re-parses everything. | Plan 4. |
| F06 | P1 | verified | error handling + perf | `core/run.py:477-524` | Unmapped transitions raise `polars.exceptions.ComputeError: KeyError: '1->7'` from `map_elements`; the `except WrongSessionTypeError` and the `is_null().any()` -> `StateMachineError` check are unreachable. `map_elements` also runs a Python lambda per row. | Users hit an opaque polars error instead of the structured message the code intends; this is exactly the "clear errors" priority in the roadmap. | Plan 5: vectorized `replace_strict(..., default=None)`; report the unmapped keys in a `StateMachineError`. |
| F07 | P1 | read | correctness | `core/trial.py:160-185` | In `set_trial`, when a channel has `presentTime` but is not `vstim`, `temp_v` is not assigned and the **previous iteration's slice is stored under the new key** (silent wrong data), or `NameError` on the first such channel. A `None` channel -> `AttributeError: 'NoneType' object has no attribute 'columns'` (the open item in the user's notes). | Silent data corruption path plus an opaque crash on real discrimination data. | Plan 6. |
| F08 | P1 | verified | performance | `core/trial.py:52-77` | Profile (499 trials, 12.06 s): `_update_model` 5.85 s (49%) = `Model.with_fields` 3.0 s + `Model.dtypes` 2.8 s (pydantic schema regeneration), called every trial because `_retry_none_type_cols` re-derives the model whenever any column still has a `Null` dtype. `set_trial` 1.6 s, `set_wheel_traces` 1.5 s. | Pure accidental cost; the final model is what matters and it is unchanged if the rebuild happens only when a Null column gains a value. Roughly halves parse time for detection sessions. | Plan 7 (golden tests are the safety net). |
| F09 | P2 | verified | performance | `core/session.py:50-54` | `Session.__init__` calls `run.get_rawdata()` for every run before `analyze()` decides to load. Measured: raw parse 0.18 s vs parquet load 0.003 s for one run (190k vstim rows). | Every `load_flag=True` session (Hub/Mouse/notebooks) pays the full raw read for nothing. Small per session, large over thousands of sessions. Behavior risk: `run.rawdata` is public and may be read after a load. | Plan 11: make `rawdata` a lazily-populated property (same attribute surface). |
| F10 | P2 | verified | dead code | see list in Plan 9 | `core/dbinterface.py` + `core/experimenter.py` (pandas `DataFrame.append`, removed in 2.0; verified absent; no importers), `core/session_launcher.py` (imports `piepy.psychophysics.<paradigm>.<Class>` paths that do not exist; superseded by `cli.py`), `core/logger.py` (unused), `psychophysics/psychophysicalRunData.py` (duplicates `transforms.py`; only the smoke test imports it), `tasks/multisensory/multiSenseTrial.py` (calls `Trial.__init__(trial_no, meta, logger)` on a patito model; invalid), `imaging/scripts.py` (imports deleted `..plotters.plotting_utils`), `imaging/onep/{eyecam,facecam}` (stubs; call `super().__init__(runpath, data)` with swapped args), `models/glmhmmModel.py` (2 lines), `models/lapseModel.py`/`glmModel.py` (import `autograd`, `ssm`: not declared deps, not installed), `imaging/onep/stacks.py` (`BinaryStack`, `ImagerStack`, `VideoStack` undefined: ruff F821 x4; `tf.imsave` removed in tifffile), `imaging/onep/filters.py` (`plt` undefined). | Dead or import-broken code inflates the surface Opus/readers must understand and hides which paths work. Nothing in `piepy/` or `tests/` (other than smoke imports) uses them; notebooks use only `OnePAnalysis`, `regions`, `load_stack`. | Plan 9. |
| F11 | P2 | read | duplication / latent bug | `psychophysics/transforms.set_outcome`, `wheelDetectionSession.augment_data:71` | `set_outcome` returns `df.with_columns(pl.col(col).alias(col))`: a no-op (aliases `state_outcome` to itself). The intended `alias("outcome")` would overwrite the string `outcome` with the int `state_outcome` and break `get_run_stats`. | The bug is masking a design mistake; "fixing" it would break parity. | Remove the call and the function (Plan 9). Do not fix the alias. |
| F12 | P2 | read | duplication / API | `core/session.py:56`, `wheelDetectionSession.py:117`, `wheelDiscriminationSession.py:199`, `visualSession.py:76` | Base `analyze(paradigm, load_flag, save_mat)`; three subclasses override with `analyze(load_flag, save_mat)` only to pass a literal paradigm that `register_paradigm` already stamps as `cls.paradigm` (and `concatenate_runs` already reads). Positional meaning of the first argument differs between base and subclasses. | Three identical 15-line copies; a synthesized (wiring-only) Session behaves differently positionally from a builtin one. | Plan 10: base `analyze(load_flag=False, save_mat=False, *, paradigm=None)`; delete overrides. |
| F13 | P2 | read | registry gap | `core/registry.py:57-60`, `tests/conftest.py:32` | `_BUILTIN_MODULES` lacks `"visual"` although `sensory/visual/visualSession.py` registers it; `get_paradigm("visual")` / `piepy session -p visual` fail unless the module was imported first. `conftest.PARADIGMS` is a hand-kept parallel map. | Inconsistent discovery; the tests duplicate registry knowledge. | Plan 10. |
| F14 | P2 | read | concurrency hygiene | `core/hub.py:126-139` | `set_start_method("spawn")` mutates process-global state (silently ignored if already set); `cfg.multiprocess["enable"]` is never consulted; a Pool is spawned even for one session. | Global side effect from a library call; needless process spawn for small gathers; hard to debug (no in-process path). | Plan 3: `get_context("spawn").Pool`, sequential path when disabled or `cores <= 1`. |
| F15 | P2 | read | broken feature | `core/run.py:218-228`, `core/session.py:121-131` | `RunData.save_as_mat` reads `self.data.stim_data` (no such attribute on a polars frame) so `save_mat=True` crashes; `Session.save_session/load_session` reference `self.save_mat` (never set) and have no callers. | Either the .mat export matters (fix: `{c: df[c].to_numpy() for c in df.columns}`) or it does not (delete the flag through 5 signatures). | Ask (Q1). Default in Plan 9: delete `save_session/load_session`; leave `save_mat` until answered. |
| F16 | P2 | verified | docs drift | `notebooks/` | Notebooks 02, 02.5, 03, 06, 07 import `piepy.tasks.*` (moved in `6116f70` to `piepy.psychophysics.tasks` / `piepy.sensory`); `example_*` notebooks import deleted `piepy.plotters.*`, `piepy.psychophysics.wheel.*`, `piepy.viz.colors`, `piepy.detection.*`. | First-run experience is broken for the tutorial notebooks. | Opportunistic: fix the 5 numbered notebooks' imports; delete or archive the `example_*` ones (user call). |
| F17 | P2 | read | error quality | `wheelDetectionTrial.py:57,111,185` | `check_early()` runs before the `_is_trial_set` check, so an incomplete first trial hits `self.data["state"]` -> `KeyError` (or reads the previous trial's stale slice); its fallthrough raises `ValueError("ijbasdjsdobwdfibwdefiubweiubwef")`; discarded-trial notice uses `print`. | Contradicts the errors design; the first-trial case is realistic. | Plan 6. |
| F18 | P2 | read | dependency direction | `psychophysics/tasks/*Session.py` import `piepy.core.hub.generate_unique_session_id` | Plugins depend on the aggregation module (`hub.py`) for a pure hashing helper. | Mild inversion; harmless today, but hub imports registry which lazily imports the plugins. | Opportunistic: move to `core/schema.py`, re-export from hub. |
| F19 | P2 | verified | removed APIs | `core/parsers.py:8-14, 95, 406` | `parse_stimpygithub_log` uses `Expr.map_dict` (removed in polars 1.0; verified) and `parse_protocol` catches `pd.io.common.EmptyDataError` (verified absent in pandas 2.3; correct is `pandas.errors.EmptyDataError`). The `except` clause itself raises `AttributeError` when triggered. Py2 `cStringIO` bare-except import relic; `faulty` flag in `parse_stimpy_log` is never set True (dead fix block). | The github-log fallback in `read_combine_logs` (bare `except Exception`) therefore cannot succeed, and a real read error is replaced by an unrelated `AttributeError`. | Plan 9 (API fixes are 3 lines; relics deleted). |
| F20 | P3 | verified | hygiene | `core/errors.py:139,183`; unused placeholder classes | `ScreenPulseError` defined twice (ruff F811); `WrongSessionTypeError`, `LogTypeMissingError`, `FrameLoggingError`, `PrefProtMismatchError`, `NoRigReactionTimeError`, `PathSettingError`, `MissingConfigKeyError`, `ConfigError` have no raise sites. | Noise only. | Plan 9: drop the duplicate; keep the placeholders (cheap, public-ish). |
| F21 | P3 | verified | hygiene | `core/utils.timeit` | No `functools.wraps`; decorated methods lose name/doc/source (it broke introspection during this review). | Trivial. | Plan 9. |
| F22 | P3 | read | latent bug / docs | `wheelDetectionSession.enrich_data:88`, discrimination twin | `if d is not None and d is not d.is_empty()` is always true (identity vs bool); `_enrich` would be unbound if `d` were None; `_detection_per_run(self, d, self)` passes the run as `session`; docstring describes a Session-level hook that never existed. | Cannot fail today because `analyze_run` always sets data; misleading to readers. | Plan 2 (guard + docstring only). |
| F23 | P3 | read | edge cases | `RunMeta.get_prot:118-135`, `VisualTrialHandler.sync_timeframes:175-181`, `set_vstim_properties` `k.strip("_l")` | `lvl` unbound if "level" appears without digits; `create_epoch` unbound on other platforms; `vstim_diff` unbound if `_vstim_onset` is a non-empty Series; `str.strip("_l")` strips characters, not a suffix (a column named e.g. `level_l` would become `leve`). | Not hit by current data. | Leave; note for whoever touches these functions. |

Other P3s noted, not recommended for work: `IDENTITY_COLUMNS` comment-outs in `schema.py` (intentional parity choice; docstring slightly stale), `mantel_haenzsel` misspelling plus a 100-line commented block in `stats/tests.py`, `NakaRushton` missing from `fitting.__all__`, `CONFIG["paths"]["colors"]` default pointing at a deleted `plotters/colors`, `ks_2d`/`energy_2d` using unseeded global numpy RNG, redundant `self._trial = {...}` in three handler `__init__`s.

---

## 5. Recommended implementation roadmap

Ordered by expected value (benefit x confidence / risk). Plans 1 to 7 are independent of each other and can be given to separate agents; each leaves the tree green. Validation baseline for every plan: `pytest -m "not needs_data"` (194 pass today), `ruff check piepy tests`, `black --check`, and, where noted, the golden-master tests on the user's machine (`pytest -m needs_data`, snapshots byte-identical).

### Plan 1: Python-version syntax and declared support

- **Objective**: package imports on every interpreter it claims to support.
- **Current state**: `piepy/core/log_repair_functions.py:149` nested-quote f-string (3.12-only). `pyproject.toml` says `requires-python = ">=3.10"`, classifiers 3.10/3.11/3.12, `ruff`/`black` `target-version = py312`.
- **Problem**: F01.
- **Desired state**: either the file is 3.10-compatible, or the metadata says 3.12 only. Do both halves that apply.
- **Scope**: that one line; `pyproject` `requires-python`/classifiers per Q5.
- **Non-scope**: any other change to `convert_riglog_to_camloglike` (that is Plan 8).
- **Steps**: (1) change the inner quotes to `'%Y%m%d'`; (2) run `python3.10 -m py_compile` over `piepy/**/*.py` if such an interpreter is available, else `/usr/bin/python3 -m py_compile` (3.9 parses the same f-string rules); (3) if the user confirms 3.12-only, set `requires-python = ">=3.12"` and drop the two classifiers; otherwise add a 3.10 job to `.github/workflows/ci.yml`.
- **Invariants**: written file name and contents unchanged.
- **Tests**: none needed beyond compile check; optional CI matrix.
- **Risk**: none. **Rollback**: revert one line.

### Plan 2: Robust per-run stats for detection and discrimination

- **Objective**: `compute_stats` never aborts a parse; discrimination stats are correct; `repr(run)` never raises.
- **Current state**: `wheelDetectionSession.get_run_stats` (l.133-218), `wheelDiscriminationSession.get_run_stats` (l.217-252), both `Run.__repr__`, both `enrich_data` guards.
- **Problem**: F03, F22.
- **Desired state**: every ratio uses a safe divide (`None` or `nan` when the denominator is 0; pick one and document it in the function docstring); every median is `None`-safe; `d_prime` uses a clipped rate (e.g. `1/(2N)` correction) or is `nan` at 0/1 and never `inf`; discrimination `nonopto_correct_rate` counts `"correct"`; the trailing-space key becomes `median_response_time`; discrimination `__repr__` reads keys it produces (`correct_rate`); `enrich_data` guard is `if d is None or d.is_empty(): return`.
- **Scope**: the two `get_run_stats`, the two `__repr__`, the two `enrich_data` guards and docstrings.
- **Non-scope**: the set of stats keys for detection (they are persisted in `sessionStats.json` and read by Mouse summaries: `hit_rate`, `level`... keep names); the join-on-`run_no` mechanism; anything in `augment_data`.
- **Dependencies**: `Mouse.gather_data` builds summaries from `runs[0].stats` and `cli.cmd_training_report` reads `hit_rate`. Discrimination key rename `"median_response_latency "` -> check no consumer (grep shows none).
- **Steps**: (1) add data-free tests reproducing each crash (no-hit run, zero stim trials, all-opto run, discrimination frame) and the wrong discrimination rate; (2) implement guards with a small local `_ratio(num, den)` helper per module (no shared abstraction needed); (3) fix repr keys; (4) fix guards/docstrings; (5) run golden tests: detection snapshots must stay byte-identical (stats are not in the parquet, only `stat_*` columns from `enrich_data` are; those values are unchanged for the golden sessions because none of them hits the zero-denominator branches; confirm by diff).
- **Behavior intentionally changing**: discrimination `nonopto_hit_rate` value and key name; detection stats for degenerate runs (previously a crash).
- **Ambiguous**: whether degenerate ratios should be `None` or `-1` (the code already uses `-1` for `easy_hit_rate`). Recommend `None`, note in docstring.
- **Risk**: low. **Rollback**: revert files.

### Plan 3: Hub gather resilience, save default, multiprocessing context

- **Objective**: one bad session never sinks a cohort; `Hub.save()` works without an explicit path; no process-global side effects.
- **Current state**: `core/hub.py` `_analyze_one`, `gather_sessions`, `save`.
- **Problem**: F04, F14.
- **Desired state**: `_analyze_one` wraps construction and `analyze()` in one try; returns `pl.DataFrame()` and prints the structured warning (include the session name and `type(exc).__name__`). `gather_sessions` uses `multiprocessing.get_context("spawn").Pool(...)` and runs sequentially (`map(_analyze_one, ...)`) when `cfg.multiprocess.get("enable") is False` or `cores <= 1`. `save(None)` defaults to `cfg.paths["analysis"][0]` and `os.makedirs(saveloc, exist_ok=True)`. Delete `_filter_session_list` or call it from `initialize` (recommend delete: `cmd_hub` already filters by paradigm).
- **Non-scope**: `_combine_session_data`; result schema; `generate_unique_session_id`; per-session parquet caching (roadmap Phase 5).
- **Steps**: (1) test: monkeypatch `get_paradigm` to a fake whose `analyze` raises; assert `_analyze_one` returns an empty frame; (2) test: `Hub.save` with `saveloc=None` writes under a monkeypatched analysis path; (3) implement; (4) test sequential path by setting `config.multiprocess = {"enable": False, "cores": 1}` and asserting `gather_sessions` calls `_analyze_one` in-process (monkeypatch the pool).
- **Invariants**: worker argument tuple `(paradigm, load_flag, sessiondir)`; cohort column order; `total_trial_no` first.
- **Risk**: low. **Rollback**: revert file.

### Plan 4: Repair the Mouse load path

- **Objective**: all four `load_type` modes work.
- **Current state**: `core/mouse.py` `gather_data`, `MouseData.load`, `isSaved`, `save`.
- **Problem**: F05.
- **Desired state**: default applied before validation; `MouseData.load` parses `sf`/`tf` back with polars expressions (`str.strip_chars("[]").str.split(",").list.eval(pl.element().str.strip_chars().cast(pl.Float64))`); `isSaved` error message references real values; `save()` tolerates `saved_dir` unset (`getattr(self, "saved_dir", None)`).
- **Non-scope**: the summary CSV format (round-trip must stay compatible with files already on disk: `[a, b]` strings for `sf`/`tf`, `dt_date` as ISO date string); the `cumul_trial_no` sort keys; the session-numbering logic.
- **Steps**: (1) data-free round-trip test: build a `MouseData`, `save()` to `tmp_path`, `load()` back, assert `sf`/`tf` lists and `dt_date` equal; (2) test `gather_data()` default does not raise validation (monkeypatch `get_unanalyzed_sessions`/`isSaved`); (3) implement.
- **Invariants**: file names `<paradigm>BehaviorData.parquet` / `...Summary.csv`; CSV encoding of lists.
- **Risk**: low-medium (touches persisted format reading). **Rollback**: revert file.

### Plan 5: Vectorized state-transition translation with a structured error

- **Objective**: unmapped transitions raise `StateMachineError` naming the offending `old->new` keys; translation is vectorized.
- **Current state**: `Run.translate_state_changes` (`core/run.py:477-524`).
- **Problem**: F06.
- **Desired state**: build the key column as `pl.col("oldState").cast(pl.Int64).cast(pl.Utf8) + "->" + pl.col("newState").cast(pl.Int64).cast(pl.Utf8)`, then `.replace_strict(transform_dict, default=None, return_dtype=pl.Utf8).alias("transition")`; if any null, raise `StateMachineError(problem=f"{n} state transition(s) have no name in the map.", where=<sessiondir or run_dir>, fix=f"Add {sorted(unmapped)} to the paradigm's state_transitions (scheme.json or the Run class).")`. Remove the dead `except WrongSessionTypeError`. Keep the generic-fallback warning branch as is.
- **Non-scope**: the `cycle -> trialNo` rename; the fallback dict; `extract_trial_count`.
- **Steps**: (1) data-free test: tiny statemachine frame with an unmapped pair -> `StateMachineError` whose message contains `'1->7'`; a fully mapped frame -> identical `transition` column to the old implementation (compute the expected list by hand); (2) implement; (3) run golden tests: `transition` values feed everything downstream; snapshots must be byte-identical.
- **Invariants**: `transition` dtype `String`, same values, same row order; `trialNo` rename.
- **Risk**: low (golden-guarded). **Rollback**: revert function.

### Plan 6: Channel guards in `TrialHandler.set_trial` and detection early-check ordering

- **Objective**: no silent stale slices, no opaque `AttributeError`/`KeyError` in per-trial parsing.
- **Current state**: `core/trial.py:160-185`; `wheelDetectionTrial.py:54-60, 94-118, 185`.
- **Problem**: F07, F17.
- **Desired state**: in `set_trial`, loop body: skip `None` or empty channels (`if v is None or v.is_empty(): continue`); `presentTime` channels other than `vstim` are explicitly skipped (`continue`) rather than falling through; `vstim` keeps the exact `presentTime * 1000` window filter. In `WheelDetectionTrialHandler.get_trial`, move `self.is_early = self.check_early()` after the `if not _is_trial_set: return None`. `check_early` fallthrough raises `StateMachineError(f"Trial {trial_no}: no outcome transition (hit/miss/catch/early) in the state table.", where=..., fix="Check the state_transitions map names hit/miss/catch/early.")`. Replace the `print` in `set_state_events` with `display(..., color="yellow")`.
- **Non-scope**: any change to which channels are sliced for vstim/rig events; the 150 ms early threshold; `set_screen_events` logic.
- **Steps**: (1) tests in `tests/test_stimpy_primitives.py`: a rawdata with a `None` channel, and with a non-vstim `presentTime` channel (e.g. `photo`) -> no error, channel absent from `handler.data`; (2) test that an incomplete first trial returns `None` from `WheelDetectionTrialHandler.get_trial` without raising (statemachine-only rawdata is enough); (3) implement; (4) golden tests byte-identical (the golden sessions have no `None`/extra-presentTime channels; confirm by inspection of `handler.data.keys()` during a run).
- **Invariants**: `self.data["state"]` contents; the vstim window; return value semantics of `get_trial`.
- **Risk**: low. **Rollback**: revert two files.

### Plan 7: Stop rebuilding the trial model every trial

- **Objective**: cut trial-extraction time roughly in half with byte-identical output.
- **Current state**: `TrialHandler._update_model` (`core/trial.py:52-77`) calls `self.trial_model.with_fields(...)` and reads `self.trial_model.dtypes` (pydantic schema regeneration) on every `get_trial`. Profile: 5.85 s of 12.06 s for 499 trials.
- **Problem**: F08.
- **Desired state**: (a) cache `self._model_dtypes` / `self._model_columns` alongside `trial_model`, refreshed only inside `set_model`/after a `with_fields`; (b) compute `_retry_none_type_cols` only for Null-typed columns whose current trial value is not `None` (a `None` value would re-derive `NoneType` again, which is the wasted rebuild); (c) merge the "new columns" and "retry Null" updates into one `with_fields` call per trial when either is non-empty.
- **Non-scope**: the `list_field_fixer` typing rules; `Run.get_trials`' final `.set_model().derive().drop().cast().fill_null().validate()` chain; anything in `set_trial`.
- **Steps**: (1) capture a baseline: golden snapshots up to date on the user's machine (`pytest --update-golden -m needs_data` on the unchanged tree if they are stale); (2) implement; (3) `pytest -m needs_data`: byte-identical; (4) re-run the profile snippet used in this review on one session and record before/after in the PR (expected: `_update_model` drops from ~5.8 s to well under 1 s).
- **Invariants**: final `trial_model` fields and dtypes per run; parquet schema and values; column order.
- **Risk**: low-medium (touches the model growth logic), fully guarded by golden tests. **Rollback**: revert one function.

### Plan 8: Remove the pseudo-camlog write to the raw data directory (needs confirmation, Q2)

- **Objective**: analysis never writes into raw camera data dirs; widefield frame timing works on every re-parse.
- **Current state**: `Run.read_run_data` mesorig branch -> `convert_riglog_to_camloglike(..., save_path=self.paths.onepcam)` writes `<YYYYMMDD>_pseudo.camlog` into the onepcam folder; `imaging.widefield.run_frame_period_ms(folder)` re-reads a camlog from disk; `SessionLocator._attach_aux` picks any `*camlog` as `onepcamlog`.
- **Problem**: F02.
- **Desired state**: `convert_riglog_to_camloglike(cam_data, timecol)` is pure (no `save_path`); `read_run_data` stores the frame table in `self.rawdata["onepcam_log"]` as today; `widefield_from_run(run, ...)` computes `frame_t` from `run.rawdata["onepcam_log"]["timestamp"]` when present (via `frame_period_ms`), falling back to `run_frame_period_ms(folder)` only when the run has no in-memory log; also `read_run_data` re-parses the riglog a second time for this branch: reuse `self.rawdata["onepcam"]` instead (the riglog channel already parsed under the code-8 key).
- **Non-scope**: `frame_period_ms` wrap-around patch; `load_stack`; `analyze_widefield`.
- **Dependencies**: Plan 11 (lazy rawdata) if adopted: `widefield_from_run` must trigger the lazy load; fine with a property.
- **Steps**: (1) test `frame_period_ms` unchanged; test `widefield_from_run` with a fake run carrying `rawdata["onepcam_log"]` and a stub stack (no disk camlog) -> works; (2) implement; (3) manual check on one mesorig session (user).
- **Behavior intentionally changing**: no file is written into the camera dir. Existing pseudo camlogs on disk: leave them; the locator will still find them, and the labcams branch parses them, so old sessions keep working. Document this in the PR.
- **Risk**: medium (imaging path has no golden test). **Rollback**: revert two files.

### Plan 9: Dead-code and removed-API sweep

- **Objective**: every module in `piepy/` imports and is reachable from some entry point, notebook, or test.
- **Delete** (verified no importers outside themselves; smoke list adjusted): `core/dbinterface.py`, `core/experimenter.py`, `core/session_launcher.py`, `core/logger.py`, `psychophysics/psychophysicalRunData.py`, `psychophysics/tasks/multisensory/` (whole dir), `imaging/scripts.py`, `imaging/onep/eyecam/`, `imaging/onep/facecam/`, `models/glmhmmModel.py`. Also delete `Session.save_session`/`load_session`, `transforms.set_outcome` and its call in `WheelDetectionRun.augment_data`, the duplicate `ScreenPulseError` (keep the first, docstringed one), the dead `faulty` block and Py2 `StringIO` import in `parsers.py`, and `Hub._filter_session_list` if Plan 3 did not already.
- **Fix in place**: `parsers.py`: `map_dict` -> `replace_strict({"true": True, "false": False}, default=None)`; `pd.io.common.EmptyDataError` -> `pandas.errors.EmptyDataError`. `imaging/onep/stacks.py`: remove the `BinaryStack`/`ImagerStack`/`VideoStack` branches from `load_stack` (raise `NotImplementedError` naming the format), replace `tf.imsave` with `tf.imwrite`. `imaging/onep/filters.py`: import matplotlib inside the `if plot:` block. `core/utils.timeit`: add `functools.wraps`.
- **Decide with user (Q3, Q4)**: `models/lapseModel.py` + `glmModel.py` + `transformer.py` (need `autograd`/`ssm`: either add an optional extra `models = ["autograd", "ssm"]` and guard imports, or remove); `imaging/onep/widefield/onepAnalysis.py` + `imaging/onep/cameraAnalysis.py` (still used by notebooks 08 and `example_onep_analysis`; superseded by `imaging/widefield.py`). Default: keep both untouched, add a module docstring pointing to the replacement.
- **Non-scope**: renaming files; touching `retinoutils.py`/`myio.py` beyond the above; `data_functions.py` (small, harmless; keep).
- **Steps**: (1) grep for each symbol once more before deleting (`grep -rn <name> piepy tests notebooks docs`); (2) delete/fix; (3) update `tests/test_smoke.py` `CORE_MODULES` (remove deleted modules; add `piepy.imaging.widefield`, `piepy.viz`, `piepy.temporal`, `piepy.cli`); (4) `ruff check piepy` should report zero F821/F811/E722.
- **Invariants**: no parse output changes (the `set_outcome` removal is a no-op by construction; confirm with golden tests).
- **Risk**: low. **Rollback**: `git revert` the single commit.

### Plan 10: Align `Session.analyze` and complete the builtin registry

- **Objective**: one `analyze` signature for builtin and synthesized sessions; every builtin paradigm resolvable by name.
- **Current state**: F12, F13.
- **Desired state**: `Session.analyze(self, load_flag: bool = False, save_mat: bool = False, *, paradigm: str | None = None)`; the three overrides deleted (their docstrings folded into the base); `_BUILTIN_MODULES["visual"] = "piepy.sensory.visual.visualSession"`; `tests/conftest.build_session` uses `get_session_class(paradigm)` and `PARADIGMS` shrinks to the list of names used by `golden_sessions.toml`.
- **Non-scope**: `concatenate_runs` signature; the `paradigm` stamping in `register_paradigm`.
- **Compatibility**: any caller passing `analyze("wheel_detection", ...)` positionally to the base class would break; grep shows none (`Hub`, `Mouse`, `cli`, tests, notebooks all use `load_flag=`).
- **Steps**: (1) test: `get_paradigm("visual")` resolves without prior import; (2) test: synthesized session (`register_paradigm("x", trial_handler_cls=...)`) and `WheelDetectionSession` both accept `analyze(load_flag=True)` and expose the same `inspect.signature`; (3) implement; (4) golden tests.
- **Risk**: low. **Rollback**: revert.

### Plan 11: Lazy raw-data loading (needs confirmation, Q6)

- **Objective**: loading a saved session does not parse the raw logs.
- **Current state**: `Session.init_session_runs` calls `run.get_rawdata()` eagerly; `Run.rawdata` is a plain attribute read by `get_trials`, `repair_rawdata`, `translate_state_changes`, imaging.
- **Desired state**: `Run.rawdata` becomes a property backed by `self._rawdata`; first access runs today's `get_rawdata()` body; `init_session_runs` only does `set_meta()`; `analyze_run` and `Session.analyze`'s parse branch keep working unchanged because they access `self.rawdata`. Attribute surface identical, so notebooks reading `run.rawdata` after a load still work (they just pay then).
- **Non-scope**: `get_rawdata` contents; save/load formats; Hub/Mouse.
- **Steps**: (1) test: construct a `Session` subclass with a fake `run_cls` whose `get_rawdata` records calls; assert zero calls after `__init__`, one after `analyze(load_flag=False)`, zero after `analyze(load_flag=True)` with a saved parquet in `tmp_path`; (2) implement; (3) golden tests; (4) time `Hub.gather_sessions(load_sessions=True)` on a handful of sessions before/after.
- **Behavior intentionally changing**: pathfinding/log errors that used to surface at `Session(...)` construction now surface at `analyze()` or on first `rawdata` access. Flag in the PR.
- **Risk**: medium (timing of errors moves). **Rollback**: revert.

---

## 6. Opportunistic improvements (not in the core roadmap)

- Fix imports in notebooks 02, 02.5, 03, 06, 07 (`piepy.tasks.*` -> new paths); archive or delete the `example_*` notebooks that import deleted modules (F16). User owns the notebooks; do not execute them.
- Move `generate_unique_session_id` to `core/schema.py`, re-export from `core/hub.py` (F18).
- `fitting/__init__.py`: export `NakaRushton`.
- `stats/tests.py`: rename `mantel_haenzsel` -> `mantel_haenszel` with an alias; delete the commented `kase == 1` block; accept a `seed`/`rng` in `ks_2d`/`energy_2d`.
- `core/config.py`: drop the `colors` default (points at a deleted directory).
- `core/errors.py`: give `StateMachineError`/`ScreenPulseError` raise sites `where=`/`fix=` as they are touched (Plans 5, 6 do two of them).
- Add a `tach.toml` (roadmap idea) only if the team wants tooling to enforce `core` not importing plugins; today the direction is already clean, so this is optional.

---

## 7. Rejected or not recommended

- **Splitting `core/run.py`** (597 LOC, 44-edge node). It is the pipeline spine; splitting IO/meta/parse into modules adds indirection with no reader benefit. Rejected.
- **Replacing patito with a static schema or pydantic-free validation.** The dynamic vstim columns are essential domain variability; the cost is fixable in place (Plan 7). Rejected.
- **Moving `rawdata` onto temporaldata now.** Roadmap Phase 6 is a spike, not a migration; the trial-table layer is the analysis contract and works. Rejected for now.
- **Unifying `session_id` (hash of date+animal) with `session_uid`.** Consumers exist in enrich columns and downstream files; needs a coordinated data migration. Deferred, as the roadmap already says.
- **Migrating `display()` to `logging`.** No pain, ~50 call sites, spawn-safe today. Rejected.
- **Introducing a `core/parallel` executor abstraction for Hub.** `imaging.executor` already shows the pattern; Hub has one call site. Plan 3's sequential/pool switch is enough until Phase 5 (per-session parquet + `scan_parquet`) is actually built.
- **Renaming camelCase modules (`wheelDetectionSession.py`) to snake_case.** Broad rename, breaks notebooks and drop-in handlers, zero behavior value. Rejected.
- **Making `Config` a lazily-constructed object.** Cross-cutting; tests monkeypatch the singleton; workers rely on import-time load. Rejected.
- **Restructuring `enrich_data`'s 1-row join into `with_columns` literals.** Golden-visible risk (list columns, dtypes) for a cosmetic gain. Rejected.
- **Parameterizing the time-list columns in `schema.py` per paradigm.** Already keyword-parameterized; no second paradigm needs different defaults yet. Rejected (YAGNI).

---

## 8. Questions for the user

1. **Q1 `.mat` export**: is `save_mat` still used by anyone (MATLAB consumers)? If yes, Plan 9 fixes `save_as_mat` (`{c: df[c].to_numpy() for c in df.columns}`); if no, delete the flag through `Run.save_run`, `RunData.save_data`, `Session.analyze`, and the three subclass docstrings.
2. **Q2 pseudo camlog**: was writing `<date>_pseudo.camlog` into the onepcam raw dir intentional as a persistent artifact (e.g. other tools read it)? If not, Plan 8 proceeds.
3. **Q3 `models/`**: keep `lapseModel`/`glmModel`/`transformer` as an optional extra (`autograd`, `ssm`), or drop them from the package?
4. **Q4 `OnePAnalysis`/`CamDataAnalysis`**: still needed alongside `imaging.widefield`, or can notebook 08 move to `analyze_widefield` and the legacy classes go?
5. **Q5 Python versions**: is 3.12-only acceptable to declare, or must 3.10/3.11 keep working (then CI needs a 3.10 job)?
6. **Q6 `run.rawdata` after load**: do any workflows read `session.runs[i].rawdata` after `analyze(load_flag=True)`? Plan 11 keeps the attribute working lazily either way, but errors move from construction to first use.

---

## 9. Final Opus handoff

Each block is self-contained. Common rules for all: do not modify `core/schema.py`, `core/paths/`, `core/registry.py` beyond what a plan names; do not change any parquet column name, dtype or order; run `pytest -m "not needs_data"`, `ruff check piepy tests`, `black --check piepy tests` before finishing; when a plan says "golden", the user runs `pytest -m needs_data` locally and snapshots must be byte-identical. Write commit messages normally.

**H1 (Plan 1) Python syntax + declared version.** Goal: package imports on declared interpreters. Files: `piepy/core/log_repair_functions.py:149`, `pyproject.toml`, `.github/workflows/ci.yml`. Steps: fix inner quotes; `py_compile` all files with a `<3.12` interpreter; adjust `requires-python`/classifiers or add a 3.10 CI job per Q5. Non-goals: anything else in that function. Invariants: identical output string. Tests: compile check. Acceptance: `py_compile` clean; CI green. Risk: none. Deps: none.

**H2 (Plan 2) Run stats robustness.** Goal: `compute_stats` never raises; discrimination stats correct; `repr(run)` safe. Files: `piepy/psychophysics/tasks/wheel_detection/wheelDetectionSession.py` (`get_run_stats`, `WheelDetectionRun.__repr__`, `enrich_data` guard/docstring), `.../wheel_discrimination/wheelDiscriminationSession.py` (same three + rename key `"median_response_latency "` -> `"median_response_time"`, `nonopto_hit_rate` -> counts `"correct"` and is named `nonopto_correct_rate`), new `tests/test_run_stats.py`. Steps: write failing tests (no-hit run, zero stim trials, all-opto, discrimination frame, repr); implement safe divide / None-safe median / finite d_prime; fix guards. Non-goals: detection stat key names, `augment_data`, the run_no join. Invariants: detection keys unchanged; golden snapshots identical. Acceptance: new tests pass; golden identical. Risk: low. Deps: none.

**H3 (Plan 3) Hub resilience.** Goal: bad session -> empty frame; `save()` default path works; no global start-method mutation; sequential path when disabled. Files: `piepy/core/hub.py`, `tests/test_hub.py`. Steps: tests first (fake paradigm whose `analyze` raises; `save(None)` under monkeypatched `config.paths["analysis"]`; sequential path with `multiprocess={"enable": False}`), then implement `get_context("spawn")`, try around `analyze()`, `saveloc` default `cfg.paths["analysis"][0]` + makedirs, delete `_filter_session_list` and its test. Non-goals: `_combine_session_data`, output schema. Invariants: worker args tuple; `total_trial_no` first column. Acceptance: tests pass; `piepy hub KC150 -p wheel_detection` saves without `--output` (user check). Risk: low. Deps: none.

**H4 (Plan 4) Mouse load path.** Goal: all four `load_type` modes work. Files: `piepy/core/mouse.py`, `tests/test_hub_mouse_concat.py` (add round-trip). Steps: round-trip save/load test for `MouseData` in `tmp_path` (sf/tf lists, dt_date); default-arg test; replace `Expr.apply` with polars list expressions; move the `None -> "last_saved"` default above validation; fix `isSaved` message attrs; `getattr(self, "saved_dir", None)` in `save`. Non-goals: file names/format, sort keys, session numbering. Invariants: CSV/parquet formats readable from existing files. Acceptance: tests pass; `Mouse(...).gather_data()` with no args runs. Risk: low-medium. Deps: none.

**H5 (Plan 5) Transition translation.** Goal: unmapped transitions raise `StateMachineError` listing keys; vectorized. Files: `piepy/core/run.py` (`translate_state_changes`), `tests/test_stimpy_primitives.py` or new `tests/test_translate_states.py`. Steps: test mapped frame -> expected `transition` list; test unmapped -> `StateMachineError` containing `'1->7'`; implement with `replace_strict(default=None)`; delete dead `except WrongSessionTypeError`. Non-goals: fallback dict, `cycle` rename. Invariants: `transition` String values/order; golden identical. Acceptance: tests + golden. Risk: low. Deps: none.

**H6 (Plan 6) set_trial guards + early-check order.** Goal: None/foreign channels skipped; incomplete first trial returns None; clear errors. Files: `piepy/core/trial.py` (`set_trial`), `piepy/psychophysics/tasks/wheel_detection/wheelDetectionTrial.py` (`get_trial`, `check_early`, `set_state_events`), `tests/test_stimpy_primitives.py`. Steps: tests (None channel; `photo` channel with `presentTime`; incomplete trial 1); implement `continue` guards, reorder `check_early`, `StateMachineError` with where/fix, `display` instead of `print`. Non-goals: vstim window logic, thresholds. Invariants: `handler.data["state"]`, vstim slice, golden identical. Acceptance: tests + golden. Risk: low. Deps: none.

**H7 (Plan 7) Model-rebuild caching.** Goal: ~50% faster trial extraction, identical output. Files: `piepy/core/trial.py` (`set_model`, `_update_model`). Steps: ensure golden snapshots current; cache `_model_dtypes`/`_model_columns` refreshed in `set_model` and after `with_fields`; only retry Null columns whose current value is not None; single `with_fields` per trial; run golden; re-profile one session (`cProfile` on `run.analyze_run()`) and record before/after. Non-goals: `list_field_fixer` rules, `get_trials` chain, `set_trial`. Invariants: final model fields/dtypes, parquet bytes. Acceptance: golden identical; `_update_model` cumulative time < 1 s on the 499-trial session (was 5.85 s). Risk: low-medium. Deps: none.

**H8 (Plan 8, after Q2) Pseudo camlog removal.** Goal: no writes to raw camera dirs; in-memory frame period. Files: `piepy/core/log_repair_functions.py` (`convert_riglog_to_camloglike` pure), `piepy/core/run.py` (`read_run_data` mesorig branch reuses `rawdata["onepcam"]`), `piepy/imaging/widefield.py` (`widefield_from_run` uses `run.rawdata["onepcam_log"]` first), `tests/test_imaging_widefield.py`. Non-goals: `frame_period_ms` math, `load_stack`. Invariants: `rawdata["onepcam_log"]` columns `frame_id`, `timestamp`. Acceptance: tests pass; one mesorig session re-parses twice without creating files (user check). Risk: medium. Deps: Q2; coordinate with H11 if both land.

**H9 (Plan 9) Dead-code and removed-API sweep.** Goal: every module imports; zero ruff F821/F811/E722. Delete: `piepy/core/{dbinterface,experimenter,session_launcher,logger}.py`, `piepy/psychophysics/psychophysicalRunData.py`, `piepy/psychophysics/tasks/multisensory/`, `piepy/imaging/scripts.py`, `piepy/imaging/onep/{eyecam,facecam}/`, `piepy/models/glmhmmModel.py`; `Session.save_session/load_session`; `transforms.set_outcome` + its call; duplicate `ScreenPulseError`; `parsers.py` Py2 import and dead `faulty` block. Fix: `parsers.py` `map_dict`->`replace_strict`, `pd.io.common.EmptyDataError`->`pandas.errors.EmptyDataError`; `stacks.py` remove undefined-class branches (raise `NotImplementedError`), `tf.imsave`->`tf.imwrite`; `filters.py` import matplotlib locally; `utils.timeit` add `functools.wraps`. Update `tests/test_smoke.py` module list (remove deleted, add `piepy.viz`, `piepy.temporal`, `piepy.imaging.widefield`, `piepy.cli`). Non-goals: `models/{lapseModel,glmModel,transformer}.py`, `onepAnalysis.py`, `cameraAnalysis.py` until Q3/Q4 are answered; no renames. Invariants: golden identical. Acceptance: suite green; `ruff check piepy` clean of F-class errors. Risk: low. Deps: H3 (both touch `_filter_session_list`; whichever lands second skips it).

**H10 (Plan 10) analyze signature + visual builtin.** Goal: one `analyze` signature; `get_paradigm("visual")` works. Files: `piepy/core/session.py`, the three `*Session.py` (delete `analyze` overrides), `piepy/core/registry.py` (`_BUILTIN_MODULES`), `tests/conftest.py` (use `get_session_class`), `tests/test_hub.py`/`test_registry_synthesize.py` (add visual resolution + signature test). Non-goals: `concatenate_runs`, `register_paradigm`. Invariants: `paradigm` column values; golden identical. Acceptance: tests; `piepy session <visual sess> -p visual` resolves. Risk: low. Deps: none.

**H11 (Plan 11, after Q6) Lazy rawdata.** Goal: load path skips raw parsing. Files: `piepy/core/run.py` (`rawdata` property over `_rawdata`, `get_rawdata` unchanged body), `piepy/core/session.py` (`init_session_runs` without `get_rawdata`), new `tests/test_lazy_rawdata.py`. Non-goals: save/load formats, Hub/Mouse. Invariants: attribute name `run.rawdata`; parse path output identical (golden). Acceptance: tests; `Session(...)` construction on a saved session does not open `.stimlog` (assert via monkeypatched `parse_stimpy_log`). Risk: medium (error timing moves). Deps: Q6; H8 if adopted.

Suggested execution order: H1, H9, H2, H3, H4, H5, H6, H7 (all independent), then H10, then H8/H11 once Q2/Q6 are answered.
