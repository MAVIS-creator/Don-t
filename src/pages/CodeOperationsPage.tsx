import { useCallback, useEffect, useState } from "react";

type Job = { name: string; region: string; state: string };
type Step = {
  id: string;
  label: string;
  state: string;
  kind?: string;
  job?: string;
  detail?: string;
};
type Log = { timestamp?: string; severity?: string; message?: string };
type Status = {
  state?: string;
  action?: string;
  reason?: string;
  stage?: string;
  submission_enabled?: boolean;
  global_gpu_quota?: number;
  max_active_gpu_jobs?: number;
  required_global_gpu_quota?: number;
  scheduler_interval_minutes?: number;
  last_controller_execution?: string;
  next_scheduled_execution?: string;
  jobs?: Job[];
  checkpoint?: string | null;
  pipeline?: {
    campaign?: string;
    state?: string;
    approval_gate?: string;
    steps?: Step[];
  };
  error?: string;
};
type View = "overview" | "commands" | "logs" | "diagnostics" | "uploads";
async function api<T = Status>(
  action: string,
  method: "GET" | "POST" = "GET",
  body?: object,
  query = "",
) {
  const response = await fetch(`/api/code?action=${action}${query}`, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let payload: T & { error?: string };
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(
      `Gateway returned an invalid response (${response.status})`,
    );
  }
  if (!response.ok)
    throw Object.assign(
      new Error(payload.error || `Request failed (${response.status})`),
      { status: response.status },
    );
  return payload;
}
const save = (name: string, content: string, type = "text/plain") => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
};
const downloadName = (prefix: string, extension: string) =>
  `${prefix}-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;

export function CodeOperationsPage() {
  const [authorized, setAuthorized] = useState<boolean | null>(null),
    [password, setPassword] = useState(""),
    [status, setStatus] = useState<Status | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [view, setView] = useState<View>("overview"),
    [source, setSource] = useState<"build" | "batch" | "run">("build"),
    [logs, setLogs] = useState<Log[]>([]),
    [diagnostic, setDiagnostic] = useState<Record<string, unknown> | null>(
      null,
    ),
    [uploadResult, setUploadResult] = useState("");
  const refresh = useCallback(async () => {
    try {
      setStatus(await api("status"));
      setAuthorized(true);
      setError("");
    } catch (cause) {
      const failure = cause as Error & { status?: number };
      if (failure.status === 401) setAuthorized(false);
      else setError(failure.message);
    }
  }, []);
  const loadLogs = useCallback(
    async (next = source) => {
      setBusy(true);
      try {
        const data = await api<{ entries?: Log[] }>(
          "logs",
          "GET",
          undefined,
          `&source=${next}`,
        );
        setLogs(data.entries || []);
        setError("");
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Log request failed");
      } finally {
        setBusy(false);
      }
    },
    [source],
  );
  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 30000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);
  useEffect(() => {
    if (authorized && view === "logs") {
      const initial = window.setTimeout(() => void loadLogs(), 0);
      const timer = window.setInterval(() => void loadLogs(), 15000);
      return () => {
        window.clearTimeout(initial);
        window.clearInterval(timer);
      };
    }
  }, [authorized, view, loadLogs]);
  const inventory = async () => {
    setBusy(true);
    try {
      const data = await api<Record<string, unknown>>("inventory");
      setDiagnostic(data);
      setView("diagnostics");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Inventory failed");
    } finally {
      setBusy(false);
    }
  };
  const github = async (kind: "inventory" | "dispatch" | "issue") => {
    setBusy(true);
    try {
      if (kind === "inventory") {
        setDiagnostic(await api<Record<string, unknown>>("github_inventory"));
      } else if (kind === "dispatch") {
        const workflow_id = window.prompt("Workflow filename or numeric workflow ID");
        const ref = window.prompt("Approved branch or commit SHA");
        if (!workflow_id || !ref) return;
        const confirmation = window.prompt(`Type exactly to confirm:\ngithub_dispatch:${workflow_id}:${ref}`) || "";
        setDiagnostic(await api<Record<string, unknown>>("github_dispatch", "POST", { workflow_id, ref, confirmation }));
      } else {
        const title = window.prompt("Issue title");
        if (!title) return;
        const body = window.prompt("Issue description") || "";
        const confirmation = window.prompt(`Type exactly to confirm:\ngithub_issue:${title}`) || "";
        setDiagnostic(await api<Record<string, unknown>>("github_issue", "POST", { title, body, confirmation }));
      }
      setView("diagnostics");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "GitHub operation failed");
    } finally {
      setBusy(false);
    }
  };
  const operate = async (action: string) => {
    const target = window.prompt(`Target for ${action} (provider ID, service, job, or object path)`);
    if (!target) return;
    const payload: Record<string, unknown> = { action, target };
    if (action.startsWith("run.") || action.startsWith("batch.")) {
      const region = window.prompt("Allow-listed GCP region");
      if (!region) return;
      payload.region = region;
    }
    if (action === "build.trigger") payload.branch = window.prompt("Git branch", "main") || "main";
    if (action === "run.rollback") payload.revision = window.prompt("Cloud Run revision name") || "";
    if (action === "run.deploy_image") payload.image = window.prompt("Artifact Registry image URI") || "";
    if (action === "batch.submit") payload.spec_object = window.prompt("Approved Batch JSON object path in the orchestration bucket") || "";
    if (action === "storage.delete") payload.generation = Number(window.prompt("Exact GCS object generation") || "0");
    if (action !== "storage.download") {
      const expected = `${action}:${target}`;
      payload.confirmation = window.prompt(`Type exactly to confirm:\n${expected}`) || "";
    }
    setBusy(true);
    try {
      const data = await api<Record<string, unknown>>("command", "POST", payload);
      if (action === "storage.download") {
        const result = data.provider_result as { filename?: string; content_type?: string; content_base64?: string } | undefined;
        if (result?.content_base64) {
          const bytes = Uint8Array.from(atob(result.content_base64), (value) => value.charCodeAt(0));
          const url = URL.createObjectURL(new Blob([bytes], { type: result.content_type || "application/octet-stream" }));
          const link = document.createElement("a");
          link.href = url;
          link.download = result.filename || "artifact";
          link.click();
          URL.revokeObjectURL(url);
        }
      }
      setDiagnostic(data);
      setView("diagnostics");
      setError("");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Cloud operation failed");
    } finally {
      setBusy(false);
    }
  };
  const run = async () => {
    setBusy(true);
    try {
      setStatus(await api("run", "POST"));
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Preflight failed");
    } finally {
      setBusy(false);
    }
  };
  const report = async () => {
    setBusy(true);
    try {
      const data = await api<Record<string, unknown>>("report");
      save(
        downloadName("jointinspect-gcp-report", "json"),
        JSON.stringify(data, null, 2),
        "application/json",
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Report failed");
    } finally {
      setBusy(false);
    }
  };
  const diagnostics = async () => {
    setBusy(true);
    try {
      setDiagnostic(await api<Record<string, unknown>>("diagnostics"));
      setView("diagnostics");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Diagnostics failed");
    } finally {
      setBusy(false);
    }
  };
  const upload = async (file?: File) => {
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) {
      setError("Uploads are limited to 2 MiB");
      return;
    }
    setBusy(true);
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      const data = await api<{ gcs_uri?: string }>("upload", "POST", {
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        content_base64: btoa(binary),
      });
      setUploadResult(data.gcs_uri || "Uploaded");
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };
  if (authorized !== true)
    return (
      <div className="code-auth-shell">
        <form
          className="code-auth-card"
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            try {
              await api("login", "POST", { password });
              setPassword("");
              setAuthorized(true);
              await refresh();
            } catch (cause) {
              setError(
                cause instanceof Error
                  ? cause.message
                  : "Authentication failed",
              );
            } finally {
              setBusy(false);
            }
          }}
        >
          <span className="code-eyebrow">Restricted operations</span>
          <h1>JointInspect Cloud Console</h1>
          <p>
            Authenticate to monitor and operate the protected GCP training
            campaign.
          </p>
          <label className="field">
            <span>Operator access password</span>
            <input
              autoComplete="current-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {error && (
            <div className="code-alert code-alert-error" role="alert">
              {error}
            </div>
          )}
          <button className="button button-primary" disabled={busy}>
            {busy ? "Authenticating…" : "Open cloud console"}
          </button>
        </form>
      </div>
    );
  const jobs = status?.jobs || [],
    active = jobs.filter(
      (j) =>
        !["SUCCEEDED", "FAILED", "CANCELLED", "DELETION_IN_PROGRESS"].includes(
          j.state,
        ),
    ),
    steps = status?.pipeline?.steps || [],
    logText = logs
      .map(
        (x) =>
          `${x.timestamp || ""} [${x.severity || "DEFAULT"}] ${x.message || ""}`,
      )
      .join("\n");
  const overview = (
    <>
      <section className="code-metrics">
        <article>
          <span>Stage</span>
          <strong>{status?.stage || "Unknown"}</strong>
          <small>{status?.action || "No action reported"}</small>
        </article>
        <article>
          <span>Active GPU jobs</span>
          <strong>{active.length}</strong>
          <small>GCP limit: {status?.max_active_gpu_jobs ?? "—"}</small>
        </article>
        <article>
          <span>Global GPU quota</span>
          <strong>{status?.global_gpu_quota ?? "—"}</strong>
          <small>Required: {status?.required_global_gpu_quota ?? "—"}</small>
        </article>
        <article>
          <span>Scheduler</span>
          <strong>
            {status?.submission_enabled ? "Enabled" : "Safe mode"}
          </strong>
          <small>Every {status?.scheduler_interval_minutes ?? "—"} min</small>
        </article>
      </section>
      {steps.length > 0 && (
        <section className="code-panel code-pipeline">
          <div className="code-panel-heading">
            <div>
              <span className="code-eyebrow">GCP campaign</span>
              <h2>{status?.pipeline?.campaign}</h2>
            </div>
            <span
              className={`code-state code-state-${(status?.pipeline?.state || "pending").toLowerCase()}`}
            >
              {status?.pipeline?.state}
            </span>
          </div>
          <div className="code-step-grid">
            {steps.map((step) => (
              <article key={step.id}>
                <div className="code-step-marker" />
                <div>
                  <strong>{step.label}</strong>
                  <span>
                    {step.detail || step.kind || step.job || "Reported by GCP"}
                  </span>
                </div>
                <span
                  className={`code-state code-state-${step.state.toLowerCase()}`}
                >
                  {step.state}
                </span>
              </article>
            ))}
          </div>
          {status?.pipeline?.approval_gate && (
            <div className="code-approval-gate">
              <span>Final gate</span>
              <strong>{status.pipeline.approval_gate}</strong>
            </div>
          )}
        </section>
      )}
      <section className="code-panel">
        <div className="code-panel-heading">
          <div>
            <span className="code-eyebrow">Live workload</span>
            <h2>Batch jobs</h2>
          </div>
          <span className="code-live">Updates every 30s</span>
        </div>
        <div className="code-job-list">
          {jobs.map((job) => (
            <article key={job.name}>
              <div>
                <strong>{job.name.split("/").at(-1)}</strong>
                <span>{job.region}</span>
              </div>
              <span
                className={`code-state code-state-${job.state.toLowerCase()}`}
              >
                {job.state}
              </span>
            </article>
          ))}
        </div>
      </section>
      <section className="code-panel code-command-bar">
        <div>
          <span className="code-eyebrow">Safe command</span>
          <h2>Re-run guarded preflight</h2>
          <p>
            Refreshes quota, jobs and financial gates. It cannot approve
            spending, bypass limits or create parallel GPU work.
          </p>
        </div>
        <button
          className="button button-primary"
          onClick={() => void run()}
          disabled={busy}
        >
          {busy ? "Running safeguards…" : "Run guarded preflight"}
        </button>
      </section>
    </>
  );
  const command = (id: string) => {
    if (id === "refresh") void refresh();
    else if (id === "inventory") void inventory();
    else if (id === "preflight") void run();
    else if (id === "github.inventory") void github("inventory");
    else if (id === "github.dispatch") void github("dispatch");
    else if (id === "github.issue") void github("issue");
    else if (id.includes(".")) void operate(id);
    else if (id.endsWith("-logs")) {
      const next = id.split("-")[0] as typeof source;
      setSource(next);
      setView("logs");
      void loadLogs(next);
    } else if (id === "diagnostics") void diagnostics();
    else if (id === "report") void report();
    else if (id === "upload") setView("uploads");
  };
  const commandItems = [
    [
      "refresh",
      "Refresh GCP inventory",
      "Reload quota, campaign and Batch state",
    ],
    ["inventory", "Open live GCP inventory", "Cloud Build, Cloud Run, Batch and Storage resources"],
    [
      "preflight",
      "Run guarded preflight",
      "Re-check quota, budget and duplicate-job gates",
    ],
    ["build.trigger", "Trigger approved Cloud Build", "Run an existing trigger against an explicit branch"],
    ["build.retry", "Retry Cloud Build", "Retry an existing build by exact build ID"],
    ["build.cancel", "Cancel Cloud Build", "Cancel a running build by exact build ID"],
    ["run.deploy_image", "Deploy Cloud Run image", "Update an allow-listed service to an Artifact Registry image"],
    ["run.rollback", "Rollback Cloud Run", "Send 100% traffic to an explicit prior revision"],
    ["batch.submit", "Submit approved Batch job", "Load a reviewed job specification from GCS"],
    ["batch.cancel", "Cancel Batch job", "Cancel an exact job in an allow-listed region"],
    ["storage.download", "Download GCS artifact", "Download an exact object up to 2 MiB"],
    ["storage.delete", "Delete GCS object", "Generation-matched deletion with exact confirmation"],
    ["github.inventory", "Review GitHub deployment inputs", "List open PRs and available Actions workflows"],
    ["github.dispatch", "Deploy approved Git ref", "Dispatch a selected workflow for an explicit branch or commit"],
    ["github.issue", "Create GitHub issue", "Record a reviewed operational problem in the repository"],
    [
      "build-logs",
      "Open Cloud Build logs",
      "Trainer, smoke-test and build output",
    ],
    ["batch-logs", "Open Batch logs", "L4 training and CPU inference output"],
    ["run-logs", "Open Cloud Run logs", "Scheduler and controller activity"],
    [
      "diagnostics",
      "Generate diagnostics",
      "Create a downloadable status snapshot",
    ],
    [
      "report",
      "Download campaign report",
      "Export current GCP controller data",
    ],
    ["upload", "Upload inert artifact", "Store a file without execution"],
  ];
  const commands = (
    <section className="code-panel code-console-panel">
      <span className="code-eyebrow">Reviewed operations</span>
      <h2>Command center</h2>
      <p>
        Fixed authenticated operations only. Arbitrary shell commands cannot be
        entered or executed.
      </p>
      <div className="code-command-grid">
        {commandItems.map(([id, label, detail]) => (
          <article key={id}>
            <div>
              <strong>{label}</strong>
              <span>{detail}</span>
            </div>
            <button
              className="button button-secondary"
              onClick={() => command(id)}
              disabled={busy}
            >
              Run
            </button>
          </article>
        ))}
      </div>
      <div className="code-approval-gate">
        <span>Final model decision</span>
        <strong>
          {status?.pipeline?.approval_gate || "Not reported by GCP"}
        </strong>
      </div>
    </section>
  );
  const content =
    view === "overview" ? (
      overview
    ) : view === "commands" ? (
      commands
    ) : view === "logs" ? (
      <section className="code-panel code-console-panel">
        <div className="code-panel-heading">
          <div>
            <span className="code-eyebrow">Live GCP logs</span>
            <h2>
              {source === "build"
                ? "Cloud Build"
                : source === "batch"
                  ? "Batch jobs"
                  : "Cloud Run controller"}
            </h2>
          </div>
          <div className="code-actions">
            <select
              value={source}
              onChange={(e) => {
                const next = e.target.value as typeof source;
                setSource(next);
                void loadLogs(next);
              }}
            >
              <option value="build">Cloud Build</option>
              <option value="batch">Batch</option>
              <option value="run">Cloud Run</option>
            </select>
            <button
              className="button button-secondary"
              onClick={() => void loadLogs()}
              disabled={busy}
            >
              Refresh logs
            </button>
            <button
              className="button button-secondary"
              onClick={() =>
                save(downloadName(`jointinspect-${source}-logs`, "txt"), logText)
              }
              disabled={!logs.length}
            >
              Download TXT
            </button>
          </div>
        </div>
        <pre className="code-log-stream" aria-live="polite">
          {logText || "No log entries returned by GCP."}
        </pre>
      </section>
    ) : view === "diagnostics" ? (
      <section className="code-panel code-console-panel">
        <div className="code-panel-heading">
          <div>
            <span className="code-eyebrow">Read-only inspection</span>
            <h2>GCP diagnostics</h2>
          </div>
          <button
            className="button button-secondary"
            onClick={() =>
              diagnostic &&
              save(
                downloadName("jointinspect-diagnostics", "txt"),
                JSON.stringify(diagnostic, null, 2),
              )
            }
            disabled={!diagnostic}
          >
            Download TXT
          </button>
        </div>
        <pre className="code-log-stream">
          {diagnostic
            ? JSON.stringify(diagnostic, null, 2)
            : "Select Run diagnostics."}
        </pre>
      </section>
    ) : (
      <section className="code-panel code-console-panel">
        <span className="code-eyebrow">Quarantined input</span>
        <h2>Upload file to GCP</h2>
        <p>
          Stored under the protected operator-upload prefix, never executed
          automatically, maximum 2 MiB.
        </p>
        <label className="code-dropzone">
          <strong>Select code, config or report</strong>
          <span>Accepted as inert storage only</span>
          <input
            type="file"
            onChange={(e) => void upload(e.target.files?.[0])}
            disabled={busy}
          />
        </label>
        {uploadResult && (
          <div className="code-alert code-alert-info">
            <strong>Uploaded</strong>
            <span>{uploadResult}</span>
          </div>
        )}
      </section>
    );
  return (
    <div className="code-console">
      <aside className="code-console-nav">
        <div className="code-console-brand">
          <span>JI</span>
          <div>
            <strong>JointInspect</strong>
            <small>Cloud operations</small>
          </div>
        </div>
        <nav>
          {(
            ["overview", "commands", "logs", "diagnostics", "uploads"] as View[]
          ).map((item) => (
            <button
              key={item}
              className={view === item ? "is-active" : ""}
              onClick={() => {
                setView(item);
                if (item === "diagnostics") void diagnostics();
              }}
            >
              {item === "overview"
                ? "Campaign"
                : item === "commands"
                  ? "Commands"
                  : item === "logs"
                    ? "Live logs"
                    : item === "diagnostics"
                      ? "Diagnostics"
                      : "Uploads"}
            </button>
          ))}
        </nav>
        <div className="code-console-nav-bottom">
          <span className="code-live">GCP connected</span>
          <button
            onClick={async () => {
              await api("logout", "POST");
              setAuthorized(false);
              setStatus(null);
            }}
          >
            Sign out and lock
          </button>
          <a href="/">Return to JointInspect</a>
        </div>
      </aside>
      <main className="code-page">
        <header className="code-hero">
          <div>
            <span className="code-eyebrow">Cloud operations · protected</span>
            <h1>Training operations</h1>
            <p>Live GCP inventory, logs, diagnostics and confirmation-gated controls.</p>
          </div>
          <div className="code-actions">
            <button
              className="button button-secondary"
              onClick={() => void refresh()}
              disabled={busy}
            >
              Refresh GCP status
            </button>
            <button
              className="button button-secondary"
              onClick={() => void report()}
              disabled={busy}
            >
              Download JSON report
            </button>
          </div>
        </header>
        {error && (
          <div className="code-alert code-alert-error" role="alert">
            <strong>Operation failed</strong>
            <span>{error}</span>
          </div>
        )}
        {status?.reason && (
          <div className="code-alert code-alert-warning">
            <strong>{status.state}</strong>
            <span>{status.reason}</span>
          </div>
        )}
        {content}
      </main>
    </div>
  );
}
