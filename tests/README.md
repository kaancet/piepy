# piepy tests

Two layers:

| Layer | Marker | Needs data? | Runs in CI? |
|-------|--------|-------------|-------------|
| Smoke (`test_smoke.py`) | — | no | yes |
| Golden-master parse (`test_golden_parsing.py`) | `needs_data` | yes (local) | no |

## Running

```bash
# data-free tests (what CI runs)
pytest -m "not needs_data"

# everything, including golden-master parse on local sessions
pytest
```

## Golden-master workflow

The golden tests parse the real sessions listed in `golden_sessions.toml`
(resolved through `~/.piepy/config.json`) and compare the parsed trial table to a
stored snapshot. Sessions that aren't present locally are **skipped**, never failed.

1. Capture current behavior once (do this on a known-good commit):
   ```bash
   pytest --update-golden -m needs_data
   ```
2. Then on every change, plain `pytest` flags any drift in parsed output.

Snapshots are written to `tests/_snapshots/` (gitignored). Point them elsewhere
(e.g. shared scratch on the cluster) with:

```bash
PIEPY_GOLDEN_DIR=/path/to/snapshots pytest -m needs_data
```

Re-parsing never touches your real `analysis/` dir: the `redirect_analysis`
fixture sends saves to a temp directory for the duration of each test.

## Adding sessions / paradigms

- Add a session: append a `[[detection]]` / `[[discrimination]]` table to
  `golden_sessions.toml`, then re-run with `--update-golden`.
- Add a paradigm: register it in `PARADIGMS` in `conftest.py`.
