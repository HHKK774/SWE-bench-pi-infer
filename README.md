# SWE-bench π Infer

A SWE-bench inference harness built around the [pi](https://github.com/badlogic/pi-mono) coding agent. It fetches SWE-bench task instances, lets a pi agent solve them inside the official SWE-bench Docker image, records full trajectories, and grades the produced patches with the official evaluation harness.

Two inference arms are supported per instance:

- **plain** — a single isolated pi session with four Docker-wrapped tools only.
- **routing** — the user's installed model-router package is enabled; the main session may delegate subtasks to configured child models via `darknavstar-spawn-model-pool`, and all context/token usage across the main session and its children is accounted for.

## Layout

```
src/
  fetch_instance.py   # instance fetching + offline parquet cache
  infer.py            # single-bundle inference run (pi in Docker)
  evaluate.py         # official SWE-bench grading of a recorded run
  orchestrate.py      # one-shot runner: fetch → infer (plain+routing) → evaluate
  run_tracking.py     # runs.db / runs.jsonl persistence helpers
tests/                # pytest suite (offline; no Docker needed)
pi-extension/         # docker-tools.ts: Docker-wrapped pi tools + routing notes
config.json           # inference arm config (model + routing flag per arm)
runs.db               # SQLite run tracking database
runs.jsonl            # append-only JSONL run log
trajectories/         # per-run trajectories, patches, prompts, metrics
logs/                 # orchestration logs and evaluation reports
PROGRESS.md           # stable design, rules, and living plans
RUNS.md               # older batch table history (local, git-ignored)
HISTORY.md            # batch records and verification evidence (local, git-ignored)
```

## Requirements

- Python 3.11+, Docker (daemon running; SWE-bench images are pulled per instance)
- The `pi` coding agent installed and on `PATH`
- For routing runs: the installed user model-router package (`~/.pi/model-router.json` pool config)
- Network access for first-time instance fetch (parquet shards) and image pulls; afterwards an offline cache is used

## Usage

### One-shot (recommended)

`orchestrate.py` is the entry point for a full run. It fetches the instance once, builds private per-arm bundles under `/tmp/swe-orchestrate/<UTC>__<instance>__<unique>/`, runs the arms from `config.json` in parallel, evaluates each arm with the official harness, removes the Docker image, and prints a JSON summary per arm plus a final summary.

```bash
python3 src/orchestrate.py <instance_id>            # e.g. astropy__astropy-12907
python3 src/orchestrate.py <instance_id> --dry-run  # print commands, create nothing
python3 src/orchestrate.py <instance_id> --timeout 3600 --keep-image
```

Arms come from `config.json`:

```json
{
  "inferences": [
    {"label": "qwen-plain", "model": "provider/model", "routing": false},
    {"label": "qwen-routing", "model": "provider/model", "routing": true}
  ]
}
```

Each arm's result is written to `runs.db` and `runs.jsonl`, with its trajectory under `trajectories/` and its grading report under `logs/evaluation/<run_id>/...`.

### Individual steps

Fetch an instance (metadata, environment spec, gold patch, image pull; caches rows in an offline SQLite backed by parquet shards):

```bash
python3 src/fetch_instance.py <instance_id> [output_dir] [--no-pull] [--force] [--refresh]
```

Run inference on an existing bundle directory (a pi session inside the instance's Docker container, working in `/testbed`):

```bash
python3 src/infer.py <bundle_dir> --model provider/model \
    [--routing] [--thinking-level LEVEL] [--workdir /testbed] \
    [--predictions-out preds.jsonl] [--keep-container] [--dry-run]
```

`--routing` enables the model-router extension and appends the router tools to the tool allowlist; see `pi-extension/ROUTING.md` for the routing mechanism and its guarantees.

Grade a recorded run with the official SWE-bench evaluation:

```bash
python3 src/evaluate.py <run_id> [--bundle-dir DIR] [--timeout 1200]
python3 src/evaluate.py --migrate        # convert legacy verdicts in runs.db
```

`run_id` is resolved against `runs.db` + `trajectories/`. The report is written to `logs/evaluation/<run_id>/<model>/<instance>/report.json` and the verdict stored in `runs.db` (`evaluation` column).

### Tests

```bash
python3 -m pytest tests/ -q
```

The suite is fully offline (Docker and the network are mocked or optional); a few integration tests are skipped by default.

## Run tracking

Every inference run appends a row to `runs.db` (`runs` table) and `runs.jsonl` with:

- outcome (`ok` / `fail` / `error`), routing flag, duration, model
- token accounting: main-session and total context tokens, input/output/cache tokens, `router_child_count`
- evaluation verdict once graded (`success` / `fail`)

Task difficulty bookkeeping lives in `tasks_short` / `tasks_medium`.

**Outcome rules** (see `PROGRESS.md` for the full text):

- `outcome=ok` means pi exited successfully, a patch was captured, and no router-spawn fallback occurred — it does *not* mean the SWE-bench instance was resolved.
- A `darknavstar-spawn-model-pool` result with `fallback_to_main_loop=true` forces `outcome=fail`, zeros the counted children, and (currently) ends the run early via a `fallback.json` receipt in the trajectory directory.
- A routing run that simply never spawns is allowed and stays `ok` as long as there is no fallback.

## Trajectories

Each run keeps a private directory `trajectories/<UTC>__<instance>__<plain|routing>__<run_id>/` containing `prompt.txt`, `patch.diff`, `run.json` (metrics), `meta.json`, `trajectory.jsonl` (full pi session), and `children/` (router child session records).

## History

See `PROGRESS.md` for stable design decisions, rules, and living plans. Local, git-ignored history lives in `RUNS.md` (older batch tables) and `HISTORY.md` (migrated batch records, verification evidence, and future dated entries).
