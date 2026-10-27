# SWE-bench Local Evaluation Framework — Design & Rules

This file keeps the plan, implementation status, repo structure, and design decisions.
Dated batch records and verification evidence live in [HISTORY.md](HISTORY.md) (local, intentionally git-ignored); [RUNS.md](RUNS.md) holds the older batch table history.

## Goal

Given a SWE-bench Verified `instance_id`, run the full single-instance evaluation pipeline locally:

```
fetch (pull resources) → inference (agent produces model patch) → evaluation (run eval) → check (grading/report)
```

Fixed decisions:
- **Agent is pi** (the pi coding agent, `@earendil-works/pi-coding-agent`), running on the host.
- Repo under evaluation lives in the SWE-bench Docker image (`/testbed`). pi tools that operate on that filesystem/processes must be redirected into the container **transparently** (same tool interface); wrapping method decided: same-name tool replacement via extension `registerTool` (see 'Pi tool wrapping decision' section).
- Grading rules must align with the official SWE-bench harness (reference commit `02e7a74`).
- No image archive export; test bundles are generated per run (ephemeral in phase 1; batch-phase caching/persistence to be decided by resource availability).
- All code and docs in English.

## Roadmap

- **Phase 1 — E2E complete**: fetch → real pi agent → official CLI evaluation → verified graded report. `orchestrate.py` provides a one-shot single-instance runner with isolated parallel inferences, an evaluation barrier, and image teardown. Dummy/gold E2E assertion and standalone check automation remain unimplemented. Verification evidence: see HISTORY.md (local, not committed).
- **Phase 2 — pending**: batch runner and resource-aware scheduling; scope and concurrency sized to available resources (disk/network/time).

## Per-run records (required, every run)

- patch duration
- token consumption
- main-process / total context length
- inference outcome (`ok` / `fail` / `error` / `cancelled`), separate from the evaluation verdict
- `runs.evaluation` stores only the objective harness verdict: `success` / `fail` / NULL (not evaluated); it is never used for generation invalidation. Provenance and test details remain in `logs/evaluation/<run_id>/` (`report.json`, `test_output.txt`).
- Except for early-killed router fallback runs (no patch required; see Implemented behavior), every inference run persists `trajectories/<UTC>__<instance_id>__<routing-or-plain>__<short-uuid>/` (configurable with `--trajectories-dir`, relative to the repo root): `trajectory.jsonl`, `run.json`, `prompt.txt`, `meta.json`, `patch.diff`, and `children/` (router records or a no-children README); `runs.jsonl` includes `trajectory_dir`.

### Task-table access convention

- `runs.db.tasks_short` (`<15 min fix`) and `tasks_medium` (`15 min - 1 hour`, i.e. 15–60 min fixes) are the **first access point for task references, selection, and already-run checks**. Each has `instance_id TEXT PRIMARY KEY`, `difficulty TEXT NOT NULL`, and `runs TEXT NOT NULL DEFAULT '[]'` (a JSON array of run ids).
- Tables are seeded once from the read-only upstream `~/.cache/swebench/instances.db` by `run_tracking.initialize_runs_db(..., cache_path)` (the current implementation has no separate `seed_task_tables` function). The instance cache is only the seed source and bundle-metadata lookup source, not the task-selection interface.
- A task counts as run if its task-row `runs` list is nonempty **or** the `runs` table contains a run id for that instance (excluding `instance_id='unknown'`). For “first” tasks, use explicit `ORDER BY rowid` in the task table. Inference tracking appends unique run ids to the task list; orchestration invokes that tracking through infer.

## Components & file responsibilities

| File | Responsibility | Status |
|---|---|---|
| `fetch_instance.py` | Fetch one instance row from local SQLite (built from HF `/parquet`, `--refresh` to rebuild); write the 5-file test bundle: `instance.json` (single-row JSON array, usable as official `--dataset_name`), `eval.sh` (eval_script, executable), `gold.patch`, `test_patch.diff`, `image.txt`; `docker pull --platform linux/amd64` by default (`--no-pull` to skip) | ✅ done (verification: HISTORY.md, local) |
| `infer.py` (inference runner) | Container lifecycle; transparent docker redirection layer for pi tools; run pi agent on the instance; collect `git diff` as model patch (use `git add -N` for new files); write `predictions.jsonl` (`{instance_id, model_name_or_path, model_patch}`) and per-run `patch.diff` | ✅ runner + opt-in inherited routing; sampled memory, stable per-run full trajectories and metadata, context/token accounting, and append-only `runs.jsonl`; live-run evidence in RUNS.md |
| `run_tracking.py` / `runs.db` | SQLite tracking maintained by `infer.py` (`--runs-db` overrides the repo-root default): `tasks_short`, `tasks_medium`, and `runs`; task rows seeded once from the read-only instance cache, then only their JSON run-id lists maintained; one flattened metrics row per run | ✅ implemented; `evaluation` is always NULL on inference insert, populated externally with only `success` / `fail` by `evaluate.py` |
| pi tool classification | Decide which pi built-in tools are docker-wrapped (transparently) vs kept host-side; enumerate pi extension mechanisms; same-name `registerTool` replacement chosen | ✅ done (findings below) |
| `evaluate.py` (evaluation runner) | Wrap the official `swebench.harness.run_evaluation` CLI for one inference run; prefer persisted `patch.diff`, accept raw `--patch FILE`, and populate `runs.evaluation` with only `success` / `fail` from the report | ✅ implemented; provenance under `logs/evaluation/<run_id>/`, informative JSON stdout summary, transactional overwrite on re-evaluation |
| `orchestrate.py` | Fetch once, private five-file bundles and predictions per config entry, parallel inference, barrier then parallel evaluation, final image removal and JSON summaries | ✅ done; offline tests and live verification in HISTORY.md (local) |
| `config.json` | Validated inference requirements; checked-in Qwen plain/routing pair, optional per-entry `thinking_level` | ✅ done |
| `check.py` (report/validation) | Planned parser for per-instance `report.json` (`resolved`, `tests_status`, patch/infra status) and aggregate `results.json` (resolved/unresolved/error/infra buckets), emitting verdict + diagnostics | 📋 file absent; existing official reports verified manually, no local structured-output checker |
| run log / metrics | Per-run records (duration, tokens, context length, outcome), appended to `runs.jsonl` | ✅ inference metrics implemented; outcome is execution status, not evaluation verdict |
| CLI entry | One-shot `orchestrate.py <instance_id>` plus separate fetch / infer / evaluate entry points | ✅ one-shot pipeline implemented; standalone check remains pending |
| `pi-extension/docker-tools.ts` | Session-local Docker tool backends inherited by router children | ✅ implemented for `bash/read/edit/write`; blocks unwrapped `powershell/grep/find/ls` and interactive host bash; container-side output artifacts and process-group cancellation |
| `test_fetch_instance.py`, `test_infer.py`, `test_evaluate.py`, `test_orchestrate.py` | Offline regression suite plus opt-in Docker/router integration smoke tests | ✅ offline regression tests plus opt-in smoke tests; command below, historical verification in HISTORY.md (local) |

## Test bundle files (per-instance)

`fetch_instance.py` produces all five files in the output directory via `write_bundle`. All are UTF-8; consumers below distinguish the implemented inference runner, the external official evaluation CLI, and the still-planned dummy/check automation.

### `instance.json`

- **Content:** the complete fetched dataset row in a single-row JSON array: `json.dumps([row], ensure_ascii=False, indent=2) + "\n"`. `FAIL_TO_PASS` and `PASS_TO_PASS` are decoded from the cache into lists before export. This is a JSON serialization of the row, not a byte-for-byte copy of the source parquet.
- **Consumer and role:** evaluation passes this file as the official `--dataset_name`. It preserves the test specifications (`FAIL_TO_PASS`, `PASS_TO_PASS`, `eval_script`, `eval_type`, `log_parser`) and repository/environment metadata (including `repo`, `base_commit`, `version`, `environment_setup_commit`) used for evaluation setup and grading.
- **Inference boundary:** read host-side by infer for `instance_id` and, by default, `problem_statement` to build the agent prompt; the full JSON is not agent input. The complete row also contains `patch` and `test_patch`, so exposing this JSON would leak the reference solution and evaluation tests even if the separate patch files were hidden.

### `eval.sh`

- **Content:** exactly `row["eval_script"].encode("utf-8")`, byte-identical to the dataset field's UTF-8 encoding, with no added newline. Fetch sets its mode to `0o755` (executable).
- **Consumer and role:** evaluation runs the instance's evaluation script inside the container through the official harness. The standalone file is the exported copy of the script also preserved in `instance.json`; the official CLI receives the dataset row rather than requiring infer to run this file. As noted in the grading rules below, the script itself applies `test_patch` and performs the reset.
- **Inference boundary:** NOT used by infer; keep it hidden from the agent as evaluation material.

### `gold.patch`

- **Content:** exactly `row["patch"].encode("utf-8")`, byte-identical to the reference-solution dataset field's UTF-8 encoding, with no added newline.
- **Consumer and role:** only the dummy-agent E2E sanity check and debugging use this reference solution. The planned dummy mode would output it as `model_patch`, then evaluate and assert `resolved=true`; no dummy mode or gold E2E assertion is currently implemented in `infer.py` or the tests. It is not the real agent's prediction or a patch to apply during real inference.
- **Inference boundary:** MUST NOT be visible to the real agent (solution leakage).

### `test_patch.diff`

- **Content:** exactly `row["test_patch"].encode("utf-8")`, byte-identical to the dataset field's UTF-8 encoding, with no added newline.
- **Consumer and role:** evaluation material, exported separately for inspection/debugging and also retained in `instance.json`. The instance's `eval.sh` / `eval_script` applies the test patch during evaluation; infer must not apply it.
- **Inference boundary:** MUST NOT be visible to the real agent (evaluation-test leakage).

### `image.txt`

- **Content:** `(row["image"] + "\n").encode("utf-8")`: the Docker image reference followed by one added newline. Unlike the patch/script files, the file is not byte-identical to the field alone.
- **Consumers and role:** fetch uses the same `row["image"]` value directly (it does not reread this file) for `docker pull --platform linux/amd64` by default; `--no-pull` skips this, and missing Docker or pull failure is nonfatal. Infer reads the reference from `image.txt`, removing the trailing newline, to start the per-run container with the repository workspace at `/testbed`.

### Relevance to infer

- Consume `image.txt` for the container; read `instance.json` host-side for `instance_id` and the default problem-statement prompt. Do not run `eval.sh` or apply `test_patch.diff` during inference.
- **Concrete isolation rule:** keep the entire bundle on the host outside the pi resource-loading workspace; never mount or copy it into the inference container or include it in agent prompts/tool results. Host-side runner code may read metadata, but the agent works only inside container `/testbed` through redirected tools. Any host-side tools, nested calls, or artifact routing must also deny agent access to the bundle. This hides `gold.patch`, `test_patch.diff`, and their copies in `instance.json` (as well as the evaluation script).
- Capture the agent's `git diff` as the model patch (use `git add -N` to include new files), then write `predictions.jsonl` with `{instance_id, model_name_or_path, model_patch}`. On run-level success, failure, or cancellation, extract the diff **before** killing and discarding the container, per the teardown decision below.

## Implemented behavior

Batch records and verification evidence: see HISTORY.md (local, not committed).

Regression command: `python3 -m pytest tests/ -q`. Optional Docker/router smoke tests use `SWE_INFER_IT=1 SWE_INFER_ROUTING_IT=1`; explicit test paths avoid collecting saved evaluation logs.

- [x] Router-spawn early-kill: routing runs poll the live session file about every 200ms, retaining byte offsets and incomplete LF-delimited lines (including split UTF-8), tolerating delayed creation and resetting on truncation/inode replacement. The first fallback receipt writes `<trajectory_dir>/fallback.json` with run id, UTC detection time, verbatim receipt, best-effort spawn category/role/task and reason, then logs immediately and terminates the pi process group (SIGTERM, five-second grace, SIGKILL). This rule supersedes patch-before-teardown for this path: no diff capture, `patch.diff`, or predictions append; partial byproducts are intentionally not preserved. Existing predictions remain untouched. Container removal still respects `--keep-container`. Normal bookkeeping order remains run.json, runs.jsonl/stdout, database row and task-list append; outcome is `fail` (exit 1), child/missing counts are zero, accounting complete, context and best-effort usage main-only. End-of-scan fallback detection remains a safety net with its existing semantics. Plain runs retain their original wait and patch path without polling. `orchestrate.py` needs no changes: fail/error rows already skip evaluation.

- [x] Simplified `runs.evaluation` to plain `success` / `fail`; inference still inserts NULL. Repeatable offline migration: `python3 evaluate.py --migrate [--runs-db PATH]`. It atomically converts legacy JSON using `resolved is true`, skips NULL/already-migrated rows, and leaves unparseable or missing-`resolved` values untouched with run-id warnings on stderr. No Docker is invoked. `orchestrate.py` reads the plain verdict and locates reports under `logs/evaluation/<run_id>/`.

- [x] Single-instance orchestration: `python3 orchestrate.py <instance_id> [--config PATH] [--keep-image] [--runs-db PATH] [--trajectories-dir PATH] [--timeout SECONDS] [--dry-run]`. Stdlib only. Config rejects unknown keys, missing model/label, unsafe or case-insensitively duplicate labels, invalid routing/thinking types, and empty lists. Defaults use repo-root config/database/trajectories; relative database paths follow infer's current-working-directory semantics, relative trajectories resolve against the repo root.
  - Retained work directories: `/tmp/swe-orchestrate/<UTC-compact>__<sanitized-instance_id>__<random>/`. One fetch/pull writes `staging/`; each `run-<label>/bundle/` contains five independent copies (not links). Each inference gets its own container, predictions, log and uniquely generated trajectory. Logs are replayed to stderr after each child exits; stdout is per-run JSON followed by aggregate JSON. `--dry-run` only prints planned commands and creates no artifacts.
  - Correlation uses the final inference record in that subprocess's private `infer.log` (identical to the record infer appends to `runs.jsonl`), obtains the 8-hex run id from its trajectory basename, and reads that id's database row for trajectory and verdict. This avoids shared-log offset races and ambiguity with concurrent same-model runs. Missing patches produce `evaluated: null` and a diagnostic. Inference/evaluation failures do not stop siblings; all inference futures are drained before evaluations launch, and all evaluations finish before `docker rmi`. `--keep-image` retains the image; an already absent image is noted as successful cleanup.
  - `--timeout` bounds inference waiting (SIGTERM invokes infer's patch/container cleanup, which is awaited), and forwards a rounded-up test timeout to the evaluator/official harness. Fetch, evaluation setup, and teardown are deliberately not hard-killed, preventing orphaned harness children or premature image deletion. Without this option infer's 1500-second agent bound and evaluate's 1200-second test timeout apply.
  - CLI audit found a positional bundle argument, `--routing`, and no thinking-level flag in infer. Minimal additive change: `infer.py --thinking-level STRING` forwards pi's `--thinking STRING` before its literal-prompt separator. Default behavior is unchanged; fetch/evaluate behavior is unchanged. The checked-in Qwen pair omits thinking levels.

- [x] Local evaluation and durable patches: `infer.py` now writes UTF-8 `patch.diff` immediately after diff capture, before teardown, on success/failure/cancellation (including empty patches). Predictions and inference record shapes are unchanged. `evaluate.py <8-hex-run_id>` resolves DB records or trajectory directories, prefers the persisted patch, supports raw `--patch FILE` and unambiguous legacy predictions, fetches missing default bundles without refreshing the instance cache, wraps the official CLI, and transactionally overwrites `runs.evaluation` with only `success` / `fail`; provenance remains in `logs/evaluation/<run_id>/`. Optional flags: `--runs-db`, `--trajectories-dir` (relative to repo root), `--bundle-dir`, `--harness-python`, `--timeout` (1200 default), `--keep-predictions`. Missing/ambiguous legacy predictions require an explicit patch.

- [x] `infer.py` implements real pi execution, opt-in routing, diff capture before teardown, predictions, resource/token/context metrics and persistent trajectories. Inference records are recorded in `runs.jsonl` and their trajectory `run.json` files. These are inference outcomes; grading verdicts are stored separately as only `success` / `fail` in `runs.evaluation`, with provenance in `logs/evaluation/<run_id>/`.
- [x] SQLite run tracking in `run_tracking.py`, maintained by `infer.py`: `runs.db` is separate from the refreshable instance cache. `tasks_short` (`<15 min fix`) and `tasks_medium` (`15 min - 1 hour`) are seeded once; subsequent runs only append unique run ids to their JSON lists. `runs` records each completed inference regardless of execution outcome, with flattened token usage. Initialization imports only the last three lines of `runs.jsonl`, once; the historical backfill excludes the probe runs. The `evaluation` column is populated with only `success` / `fail` by `evaluate.py` (provenance in `logs/evaluation/<run_id>/`); inference inserts NULL and does not overwrite existing evaluations. Missing cache is an error at seed time only; later runs still record metrics without task-list updates when the cache is absent. Dry runs do not create or update the database.

- [x] Research: dataset API (`/filter` endpoint, schema, image naming), official grading rules, official runner structure (grading summary below).
- [x] `fetch_instance.py`:
  - replaced flaky HF `/filter` with a SQLite cache at `~/.cache/swebench/instances.db`; first use or `--refresh` downloads default/test shards via `/parquet` (3 download attempts, 1s/2s backoff);
  - atomic cache replacement, 500 rows × 17 TEXT columns, UNIQUE `instances_instance_id` index; duplicate IDs fail without damaging an existing cache;
  - `FAIL_TO_PASS` and `PASS_TO_PASS` lists are JSON-encoded as TEXT and decoded on lookup; all other strings (including long patches/scripts) are preserved verbatim. Existing local row validation and Docker default/`--no-pull` behavior remain;
  - cache builds require PyArrow (`python3 -m pip install pyarrow`, preferably in a venv); cached reads need only stdlib `sqlite3`;
  - output-dir protection: refuse existing dir unless `--force` (overwrites generated files only).
- [x] Architecture decisions: 5-file bundle; no README; no image archive export.
- [x] Facts confirmed with user: agent = pi; transparent docker tool wrapping; E2E-first then batch; per-run metrics; all English.

## Planned (suggested order)

1. **Done:** pi tool classification completed; wrapping mechanism decided: same-name tool replacement via extension `registerTool`.
2. **Done:** `infer.py` runner and Docker tool wrapping, predictions, per-run metrics and trajectories; real pi runs in plain and routing modes officially graded successfully. Phase-1 E2E no longer awaits a live LLM run.
3. **Evaluation and one-shot orchestration done:** `evaluate.py` wraps the official CLI and populates `runs.evaluation` with only `success` / `fail`; `orchestrate.py` runs the single-instance configurable parallel pipeline. Standalone `check.py` remains pending; inference itself does not grade runs.
4. **Pending sanity regression:** implement dummy/gold predictions and an E2E assertion that gold → `resolved=true`. Real-agent E2E completed without this originally proposed prerequisite.
5. **Pending Phase 2:** batch runner + resource-aware scheduling.
6. (Optional, deferred) replace official CLI with self-hosted execution reusing `get_eval_report`.

## Open (needs a decision)

1. **DECIDED — pi tool wrapping mechanism:** same-name tool replacement via pi extension `registerTool`; registering a tool named e.g. `bash` or `read` replaces the built-in (see pi `docs/extensions.md` and `examples/extensions/ssh.ts`). User-confirmed lifecycle: per-session; the replacement takes effect only for the pi session that loads the extension. A freshly started session without the extension is unaffected.
2. **DECIDED — official CLI for Phase 1:** installed and successfully invoked directly via `/Users/cc/workplace/SWE-bench/.venv/bin/python -m swebench.harness.run_evaluation`; `evaluate.py` now wraps this. See `RUNS.md` official grading commands and `logs/evaluation/` reports.
3. **DECIDED — harness pin:** official checkout `/Users/cc/workplace/SWE-bench` is at `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` (audit: `git -C /Users/cc/workplace/SWE-bench rev-parse HEAD`), matching the recorded install/checkout and grading commands in `RUNS.md`. Pinning is operational/documented, not yet enforced by a local evaluation wrapper.
4. Phase 2: concurrency / retry policy and bundle caching strategy.
5. Offline support (some `eval_script`s run `pip install` at runtime) — default: no.

## Pi tool classification (findings, pi 1.2.0 at `/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent`)

**Core built-in tools (8):** `read`, `bash`, `edit`, `write`, `powershell`, `grep`, `find`, `ls`. An uncustomized session enables only `read`, `bash`, `edit`, `write` (`DEFAULT_TOOL_NAMES` in `dist/core/settings-manager.js`; registry in `dist/core/tools/index.js`). All 8 are target-sensitive (they touch the repo filesystem or process space) → **WRAP if enabled**. Non-core tools (`codemode`, `tool_search`, `mcp__*`, plus any session-injected utilities such as calculators/web search) can stay host-side, subject to nested-tool coverage and temp-artifact path handling. (Session-injected tools like `todo`/`calculator`/web search are extensions, not pi core built-ins.)

**Wrapping mechanisms (enumerated; decision: option 1, same-name replacement):**
1. **Same-name replacement via extensions** — registering a tool named `bash`/`read`/... replaces the built-in (`_refreshToolRegistry()` in `dist/core/agent-session.js`; example `examples/extensions/tool-override.ts`). This is the key mechanism for full transparency. Official `examples/extensions/ssh.ts` does exactly this: replaces `read/write/edit/bash` with a remote backend while keeping schemas.
2. **Built-in tool factories with injected operations** — factories expose operations interfaces for remote file/process backends; note the `grep` seam is incomplete (still spawns local `rg`), so grep may need a custom tool anyway.
3. **SDK `customTools`** — same replacement registry without an extension file (`docs/sdk.md`).
4. **MCP servers** — tools surface as `mcp__<server>__<tool>`, NOT bare `bash`/`read` → no same-name transparency without an adapter layer.
5. **Hooks** (`tool_call`/`tool_result`, shell `spawnHook`, `user_bash`) — can mutate args or results (e.g. rewrite commands with `docker exec` prefix) but are not a general "execute remotely" seam.

**Transparency risks to handle in the wrapper:**
- Path resolution: tools resolve paths against host `ctx.cwd` and `read` does host-side existence checks → need explicit path mapping (relative / `/testbed` / absolute / home).
- Large `bash`/codemode output spills to host temp files whose paths a container-only `read` can't see → artifact transfer or host-path routing needed.
- Result contract must be preserved fully: `content`, `details`, `structuredContent`, images, truncation notices, streaming updates, nonzero exit codes.
- Cancellation/timeout: two-level semantics decided — see 'Pi tool wrapping decision'.
- Wrapping is not full isolation: extension code runs with host permissions; resource loading uses the host workspace.

## Pi tool wrapping decision

- **Chosen mechanism:** same-name tool replacement via pi extension `registerTool`, as documented in pi `docs/extensions.md` and demonstrated in `examples/extensions/ssh.ts`.
- **Lifecycle (user-confirmed):** per-session; the wrapper applies only to the pi session started with the extension loaded. A freshly started session without the extension is unaffected.
- **Required wrapper handling:** all items in **Transparency risks to handle in the wrapper** above remain requirements. Resolve and map host paths to container `/testbed` paths explicitly, including home, relative, and absolute paths. Large `bash`/codemode output can spill to host temp files that a container-only `read` cannot access; artifact transfer/file synchronization or host-path routing is required. Preserve the full result contract (`content`, `details`, `structuredContent`, images, truncation notices, streaming updates, nonzero exit codes).

### Cancellation/timeout semantics (decided)

- **Run-level end (success, failure, or user cancellation):** extract `git diff` FIRST as the model patch, then kill and discard the per-run container (`docker stop` / `docker rm`). No per-process propagation is needed at this level. This ordering fixes the patch snapshot before teardown; late background writes cannot change the captured patch.
- **Tool-level timeout/cancellation (one wrapped tool call while the session continues):** match native pi behavior by propagating cancellation/timeout into the container and killing the container-side process tree, not just the host client. Return a timeout/aborted tool result so the agent session continues. Orphaned processes could consume resources, hold file locks, or mutate the repo after the agent has moved on, potentially corrupting the final `git diff`.

## Resource limits & per-run measurements

### Resource-limit facts

Dated investigation evidence: see HISTORY.md (local, not committed).

- The runner imposes no per-container memory or CPU cap. Containers share the Docker VM's memory and CPU; concurrent runs do not multiply available resources and can exhaust them.
- Infer launches no local model server. Configured context ceilings apply per request, not cumulatively across turns or agents; server enforcement and automatic compaction may differ.

- Host-side pi is a Node process with in-process extensions, plus router child pi processes and transient Docker CLI subprocesses. No runner-imposed host RSS cap exists. RSS sampling sums the pi descendants and its isolated process group, including router children; shared pages can be double-counted. It excludes the Python runner, the sampling CLI itself, Docker VM/daemon overhead, and unrelated host processes. Short-lived processes/peaks between samples can be missed.

### Measurement definitions and evidence

Historical measurement and regression evidence: see HISTORY.md (local, not committed).

- A background stdlib thread samples `ps -axo pid=,ppid=,pgid=,rss=` and `docker stats --no-stream --format '{{.MemUsage}}'` approximately every two seconds. Each CLI call has a four-second timeout. No pip dependencies. Docker's CLI memory is cache-adjusted working-set-style usage on Linux, **not** raw cgroup allocation or VM RSS. `_mb` fields are **MiB (2^20 bytes)**. Peak is the largest observed sample; final is the last successful sample immediately before patch capture/container teardown. Errors are counted, unavailable measurements are null, not fabricated zeros.
- Main pi previously used `--no-session`. Live runs now replace that with a unique explicit `--session <trajectory_dir>/trajectory.jsonl`, permanently stored outside the temporary workspace and identified in each run record. Inherited caller session/router identity variables are removed. Reference: installed Pi `docs/session-format.md`, assistant `message.usage` fields.
- `main_context_tokens` = **last assistant request's input + cacheRead + cacheWrite + output** with nonzero recorded usage. This includes completion and cached prompt tokens, not cumulative billing. It measures the last observed request, not the maximum context before any compaction or unsent tool output. `token_usage` separately sums all recorded usage across requests, including explicit session usage/compaction entries. Missing/zero usage leaves context unknown.
- Router reference: `/Users/cc/workplace/darknavstar-model-router/src/runtime/records.ts` (`NodeRecord`, `RunDetail`, `RecordStore.appendEvent`) stores each node's own `detail.events`; `src/harness/extension.ts:ensure` binds `session_id` to the main persisted session. We match that session, traverse only its direct-child `entries` recursively, and count each descendant once. Child `message_end.message.usage` remains available with `--no-session`. Root trajectory events are not counted again. `total_context_tokens` sums the main and each child's last recorded context, **not a single shared context window**. Missing records yield an observed subtotal with `context_accounting_complete=false` and `missing_context_records`; they are not silently treated as zero. `router_child_count` includes descendants. Exception: any main-session `darknavstar-spawn-model-pool` result with `fallback_to_main_loop=true` forces `outcome=fail`, `router_child_count=0`, `missing_context_records=0`, and `context_accounting_complete=true`. Context is the main-only subtotal, even if other spawns succeeded; token usage and raw child evidence remain unchanged. Detection parses result text as JSON, with a substring-pattern fallback for non-JSON text; `false` does not trigger failure.
- Runs append identical final JSON to stdout and repository `runs.jsonl`, preserving the original fields. Duration includes startup, inference, final sampling, diff capture and cleanup. `outcome=ok` means pi exited successfully and a patch was captured without a router-spawn fallback, **not** SWE-bench resolved. A fallback forces the distinct `fail` outcome; the orchestrator skips evaluation for both `fail` and `error` runs, even with partial patches (evaluation remains NULL), while tracking still appends their run ids to task lists. `ok` runs continue to be evaluated as before. This is a forward-only orchestration rule; historical rows are left as-is. Agent wait is bounded to 1,500 seconds; timeout/error/cancellation still captures partial metrics/patch and tears down the container. Docker lifecycle commands retain their existing behavior.

### Post-hoc generation bookkeeping

Dated bookkeeping corrections: see HISTORY.md (local, not committed).

- `evaluation` records only the objective harness verdict (`success` / `fail` / NULL = not evaluated), never generation invalidation.
- Generation invalidation is marked with `outcome='fail'`: both runs of a comparison-losing new pair (`routing.main_context_tokens >= plain.main_context_tokens`), and both runs of the superseded old generation when a re-run is performed under the user's re-run rule. Reuse the existing outcome value; no schema change. Preserve all measurements and objective evaluation verdicts.
- Comparison-invalidated evaluated runs retain a real evaluation verdict. Spawn-fallback runs have `outcome='fail'` and evaluation NULL under the current infer/orchestrate rule; historical evaluated fallback records remain historical.
- These are post-hoc bookkeeping markings, mirrored into existing outcome fields in `runs.jsonl` and trajectory `run.json`. The orchestrator's forward-only skip rule for outcome in (`fail`, `error`) is unchanged; no pipeline code changes are needed.

## Official grading rules (alignment reference, commit `02e7a74`, `harness/grading.py` + `log_parsers/python.py`)

- `resolved = (f2p == 1.0) and (p2p == 1.0)`; f2p/p2p are success ratios; no partial credit.
- F2P success = `PASSED` or `XFAIL`; P2P success = `PASSED`/`XFAIL`/`SKIPPED`; `FAILED`/`ERROR`/missing from log = failure.
- Test-name matching is exact string; exception: truncated parameterized IDs (more `[` than `]`) use startswith prefix matching, and all candidates must share the same pass/fail boolean.
- Log-validity gate first: log containing `Patch Apply Failed` / `Reset Failed` / `Tests Errored` / `Tests Timed Out`, missing Start/End Test Output markers, empty parsed map, or "non-zero exit code but no FAILED/ERROR in map" → invalid → `resolved=False`.
- Non-zero test exit code does NOT by itself mean unresolved; newly failing tests outside F2P/P2P do not affect resolved.
- `test_patch` application and reset are done by `eval_script` itself (`set -uxo pipefail`, no `-e`; no automatic short-circuit).
- Model patch apply: 4-level fallback (`git apply` → `--3way` → `--reject` → `patch --fuzz=5`); total failure → instance lands in `error_ids` (infra failure tracked separately from "ran but unresolved").
- Parser status enum: `PASSED/FAILED/SKIPPED/ERROR/XFAIL` (no XPASS); duplicate IDs: last occurrence wins.
- Reports: `logs/evaluation/<run_id>/<model>/<instance_id>/report.json` (`resolved`, `patch_successfully_applied`, ...) + aggregated `results.json` (resolved/unresolved/error/infra_failure buckets).
