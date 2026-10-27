# Inference routing

Run `python3 infer.py <bundle> --model <model> --routing` to enable the installed user model-router package. Without `--routing`, the original isolated-extension CLI is unchanged. Both modes explicitly load `docker-tools.ts`; metrics include `routing`.

## Cross-process loading

Investigation references (installed versions at implementation time):

- `/Users/cc/workplace/darknavstar-model-router/src/runtime/dispatch.ts:181-210`: children use `--mode json -p --no-session --provider ... --model ... --thinking ... -e <router> --tools ... -- <task>`. They do **not** forward the parent's extension, resource-disable, or tool flags. Their allowed tools come from the pool, role, depth, and parent ceiling. The environment spreads `process.env`, including the Docker variables.
- Router `src/runtime/runner.ts:19-27,88-93`: launches the current Pi executable/script with fresh arguments, passing the dispatch environment and project cwd. A PATH shim is therefore insufficient. No child extra-extension environment/config option exists in the inspected router; the invocation callback is an internal dispatch API, not user configuration.
- Installed Pi `docs/environment-variables.md:81` documents `PI_CODING_AGENT_DIR`; it does not document an extra-extension environment variable. `docs/configuration.md:7-13` describes the agent-directory settings. `docs/extensions.md:44-50` and `docs/settings.md:149-162` document extension discovery and settings resource paths.

`infer.py:routing_settings` creates a per-run agent-directory overlay. It symlinks existing user resources (including credentials/models/package installations), copies settings, rebases relative resource/package paths, explicitly includes the Docker extension, selects the four default tools, and disables built-in MCP. It does not edit global settings or the router package. The overlay is removed with the neutral temporary workspace after Pi exits. Dry-run uses `$PI_RUN_AGENT_DIR` as the placeholder for this runtime-allocated path.

The parent drops `--no-extensions`, preserving the installed user router package.

**Tool allowlist pitfall (fixed 2026-10-16):** Pi `docs/cli.md#tools` says plain-name `--tools` replaces the selection for extension tools too. Loading/registering the router is not sufficient: routing mode now appends `darknavstar-get-model-pool,darknavstar-spawn-model-pool,darknavstar-get-agent-run-records` to the four-tool allowlist; plain mode is unchanged. The overlay's defaults are overridden by this parent CLI and by each child's explicit selection. The router only injects delegation instructions and its request-local reminder when spawn is active. Verify system `toolsAdded` (the prose `<tools>` block may omit extensions); reminders require observing request context, not grepping normal persisted history.

Every normal router child inherits the overlay and the same `SWE_INFER_CONTAINER`/`SWE_INFER_WORKDIR`, so it discovers the wrapper independently of its CLI flags. This also covers recursive descendants. Pool/category configuration remains unchanged at `~/.pi/model-router.json`. Router role restrictions may narrow the child's tool set; they must not be overridden to force write access on read-only children.

`docker-tools.ts` additionally blocks `powershell`, `grep`, `find`, and `ls` through `tool_call`, including nested calls: a pool must not re-enable unwrapped built-ins from Pi's full registry. Use wrapped bash for these operations. Missing/invalid Docker configuration remains fatal when the wrapper loads.

This is tool redirection, **not a hostile-extension sandbox**. Routing deliberately enables trusted user extensions, which execute on the host. Do not install extensions that expose other host filesystem/process tools or replace the wrapper. Child flags do not inherit the parent's skill/context/template switches; user resources should be trusted and must not expose evaluation bundles. No bundle is mounted, copied, or used as a Pi workspace.

## Verification

- `python3 -m pytest -q`: 22 passed, 2 opt-in integration tests skipped (2026-10-16).
- Live fixed main-session/control verification: both remote Qwen sessions returned OK with no tool calls; routing declared all three router tools and plain declared only four Docker tools. An observational `context_with_system` hook persisted the request-local reminder only for routing. Evidence: `/tmp/router-verify-kv7xzrbw/{routing,plain}.jsonl`; scripts: `/tmp/router-verify.py`, `/tmp/router-verify-probe.ts`; container removed.
- `python3 infer.py ~/.cache/swebench/astropy__astropy-12907 --model test --predictions-out /tmp/swe-infer-plain.jsonl --dry-run`: exit 0; original `--no-extensions` command.
- `python3 infer.py ~/.cache/swebench/astropy__astropy-12907 --model test --predictions-out /tmp/swe-infer-routing.jsonl --routing --dry-run`: exit 0; inherited config overlay, no `--no-extensions`, explicit Docker extension.
- `SWE_INFER_ROUTING_IT=1 python3 -m pytest -q -s test_infer.py::test_routing_child_cli`: passed. Replicates the router's **exact child CLI flags and child configuration/environment**, with no explicit Docker extension flag. A deterministic localhost OpenAI-compatible model issues write/edit/read/bash calls. Real Pi child startup validates the router identity; results show `/testbed`, `Linux`, and the edited container-only file. The host workspace has no such file. Container cleanup runs in `finally`.
- `SWE_INFER_IT=1 python3 -m pytest -q -s test_infer.py::test_docker_tools`: passed, including timeout/abort process-tree cleanup.

The integration test also caught a development regression: `+path` alone enables but does not discover a new extension. The final overlay includes both the plain path (discovery) and `+path` (enablement); the final rerun passed.

Verification level: real child CLI and Docker tools with a deterministic local model, not a live LLM selecting/invoking spawn-model-pool. The child invocation is replicated from dispatch rather than initiated by its spawn tool. Integration tests use local images only; Docker reports the expected amd64-image/arm64-host warning.
