/** Session-local Docker backends. Pi loads TypeScript directly using jiti.
 * Requires bash, setsid, base64 and standard coreutils in the image.
 */
import { spawn, execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { posix } from "node:path";
import {
  type ExtensionAPI, createBashToolDefinition, createReadToolDefinition,
  createWriteToolDefinition, createEditToolDefinition,
} from "@earendil-works/pi-coding-agent";

const quote = (s: string) => "'" + s.replace(/'/g, "'\\''") + "'";

export default function (pi: ExtensionAPI) {
  const cid = process.env.SWE_INFER_CONTAINER;
  // Pi normally catches extension-load errors and keeps native tools. Never allow
  // that fallback for this isolation boundary: invalid configuration is fatal.
  const fatal = (message: string): never => { console.error(message); process.exit(1); };
  if (!cid) fatal("SWE_INFER_CONTAINER must identify a running Docker container");
  try {
    if (execFileSync("docker", ["inspect", "-f", "{{.State.Running}}", cid], { encoding: "utf8" }).trim() !== "true")
      fatal("SWE_INFER_CONTAINER is not running");
  } catch { fatal("Cannot inspect SWE_INFER_CONTAINER: Docker/container unavailable"); }
  const cwd = process.env.SWE_INFER_WORKDIR || "/testbed";
  const home = execFileSync("docker", ["exec", cid, "bash", "-lc", 'printf %s "$HOME"'], { encoding: "utf8" });
  const path = (p: string) => posix.resolve(cwd, p === "~" ? home : p.startsWith("~/") ? home + p.slice(1) : p);

  function raw(command: string, input = ""): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const child = spawn("docker", ["exec", "-i", cid!, "bash", "-lc", command]);
      const out: Buffer[] = [], err: Buffer[] = [];
      child.stdout.on("data", b => out.push(b));
      child.stderr.on("data", b => err.push(b));
      child.stdin.on("error", () => {});
      child.stdin.end(input);
      child.on("error", reject);
      child.on("close", code => code === 0 ? resolve(Buffer.concat(out)) :
        reject(new Error(`docker exec exited with code ${code}: ${Buffer.concat(err)}`)));
    });
  }

  async function execute(command: string, signal?: AbortSignal, timeout?: number, input = ""): Promise<{output: string; code: number; artifact: string; truncated: boolean}> {
    if (signal?.aborted) throw new Error("Command aborted");
    if (timeout !== undefined && (!Number.isFinite(timeout) || timeout <= 0 || timeout * 1000 > 2147483647))
      throw new Error("Invalid timeout");
    const dir = `/tmp/swe-pi-${randomUUID()}`;
    await raw(`mkdir -m 700 ${quote(dir)}`);
    // A dedicated session owns the whole shell process group. The cancellation marker
    // closes the race where abort arrives before the session has published its PGID.
    // Kill inside Docker BEFORE waiting for the client; killing only docker is insufficient.
    let reason = "", killing: Promise<unknown> | undefined;
    const cancel = (why: string) => {
      if (reason) return;
      reason = why;
      killing = raw(`touch ${quote(dir + "/cancel")}; if test -s ${quote(dir + "/pid")}; then kill -KILL -- -$(cat ${quote(dir + "/pid")}) 2>/dev/null || true; fi`);
    };
    const abort = () => cancel("Command aborted");
    signal?.addEventListener("abort", abort, { once: true });
    const timer = timeout === undefined ? undefined : setTimeout(() => cancel(`Command timed out after ${timeout} seconds`), timeout * 1000);
    try {
      if (signal?.aborted) abort();
      const inner = `echo $$ > ${quote(dir + "/pid")}; test ! -e ${quote(dir + "/cancel")} || exit 137; cd ${quote(cwd)} && ${command}`;
      const result = await raw(`setsid bash -c ${quote(inner)} > ${quote(dir + "/output")} 2>&1; printf '%s' $?`, input);
      await killing;
      if (reason) throw new Error(reason);
      const size = Number((await raw(`wc -c < ${quote(dir + "/output")}`)).toString());
      return { output: (await raw(`tail -c 51200 ${quote(dir + "/output")}`)).toString(), code: Number(result.toString()), artifact: dir + "/output", truncated: size > 51200 };
    } finally {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      await killing;
      // Output stays container-side, readable through the wrapped read tool.
    }
  }

  // Factory definitions preserve schemas, image handling, exact edits and mutation queues.
  // Normalize paths before factories so ~ never expands to the HOST home directory.
  // TODO: optimize base64 transfers with docker cp/artifact staging for large files.
  for (const kind of ["read", "write", "edit"] as const) {
    const make = (signal?: AbortSignal, requestedPath?: string) => {
      const readFile = async (p: string) => {
        // Ignore the factory's macOS host-path spelling probes: only the exact
        // container path may select data; no host file contents are ever read.
        p = requestedPath ?? p;
        const r = await execute(`base64 -w0 -- ${quote(p)}`, signal);
        if (r.code) throw new Error(`Read exited with code ${r.code}: ${r.output}`);
        // Read the full encoded artifact, not the bounded bash display tail.
        return Buffer.from((await raw(`cat ${quote(r.artifact)}`)).toString(), "base64");
      };
      const writeFile = async (p: string, content: string) => {
        // stdin avoids ARG_MAX; decoding/writing share the cancellable process group.
        const r = await execute(`base64 -d > ${quote(p)}`, signal, undefined, Buffer.from(content).toString("base64"));
        if (r.code) throw new Error(`Write exited with code ${r.code}: ${r.output}`);
      };
      const access = async () => {}; // readFile/writeFile report container-side failures.
      if (kind === "read") return createReadToolDefinition(cwd, { operations: { readFile, access,
        detectImageMimeType: async p => {
          const ext = posix.extname(p).toLowerCase();
          return ({".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".gif":"image/gif", ".webp":"image/webp", ".bmp":"image/bmp"} as Record<string,string>)[ext] || null;
        } } });
      if (kind === "edit") return createEditToolDefinition(cwd, { operations: { readFile, writeFile, access } });
      return createWriteToolDefinition(cwd, { operations: { writeFile, mkdir: async p => {
        const r = await execute(`mkdir -p -- ${quote(p)}`, signal);
        if (r.code) throw new Error(`mkdir exited with code ${r.code}: ${r.output}`);
      } } });
    };
    const definition = make();
    pi.registerTool({ ...definition, async execute(id, params: any, signal, update, ctx) {
      return make(signal, path(params.path)).execute(id, { ...params, path: path(params.path) }, signal, update, { ...ctx, cwd } as any);
    } } as any);
  }
  const bash = createBashToolDefinition(cwd);
  pi.registerTool({ ...bash, async execute(_id, { command, timeout }, signal) {
    const start = Date.now();
    const r = await execute(command, signal, timeout);
    const output = r.output + `\n[Full output: ${r.artifact}]` + (r.code ? `\nCommand exited with code ${r.code}` : "");
    return { content: [{ type: "text", text: output }], details: { fullOutputPath: r.artifact },
      isError: r.code !== 0, structuredContent: { output: r.output, truncated: r.truncated,
        full_output_path: r.artifact, exit_code: r.code, wall_time_seconds: (Date.now() - start) / 1000 } };
  } });
  pi.on("before_agent_start", event => ({ systemPrompt: event.systemPrompt.replace(process.cwd(), cwd) }));
  // Router pools can select inactive built-ins from Pi's full registry. Never
  // let a child re-enable an unwrapped host filesystem/process tool (including
  // nested codemode calls, which also pass through tool_call).
  pi.on("tool_call", event => {
    if (["powershell", "grep", "find", "ls"].includes(event.toolName))
      return { block: true, reason: "Use the Docker-backed bash/read/edit/write tools" };
  });
  // No local fallback, including interactive ! commands.
  pi.on("user_bash", () => { throw new Error("Use the Docker-backed bash tool"); });
}
