from __future__ import annotations

import json
import os
import socket
import base64
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from flask import Flask, jsonify, request
from google.cloud import batch_v1, compute_v1, storage, logging_v2
from google.api_core.exceptions import PreconditionFailed

try:
    from .engine import FinancialPolicy, Snapshot, decide
    from .cloud_operations import CloudOperations, OperationError
except ImportError:  # Cloud Run source root
    from engine import FinancialPolicy, Snapshot, decide
    from cloud_operations import CloudOperations, OperationError


PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "joint-inspection-510310")
STATE_BUCKET = os.getenv("STATE_BUCKET", "joint-inspection-510310-orchestration")
STATE_PREFIX = os.getenv("STATE_PREFIX", "orchestration/gpu-scheduler/v1")
SUBMISSION_ENABLED = os.getenv("SUBMISSION_ENABLED", "false").lower() == "true"
LOCK_TTL_SECONDS = int(os.getenv("LOCK_TTL_SECONDS", "840"))
UPLOAD_PREFIX = os.getenv("UPLOAD_PREFIX", "operator-uploads")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
app = Flask(__name__)

def cloud() -> CloudOperations:
    return CloudOperations(PROJECT, STATE_BUCKET)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_blob(name: str):
    return storage.Client(project=PROJECT).bucket(STATE_BUCKET).blob(f"{STATE_PREFIX}/{name}")


def acquire_lock(now: datetime) -> tuple[Any, dict[str, Any]]:
    blob = _json_blob("controller.lock.json")
    lease = {
        "owner": f"{socket.gethostname()}:{os.getpid()}",
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=LOCK_TTL_SECONDS)).isoformat(),
    }
    try:
        blob.upload_from_string(json.dumps(lease), content_type="application/json", if_generation_match=0)
        return blob, lease
    except PreconditionFailed:
        blob.reload()
        current = json.loads(blob.download_as_text())
        if datetime.fromisoformat(current["expires_at"]) > now:
            raise RuntimeError("controller lock is already held")
        generation = blob.generation
        blob.upload_from_string(json.dumps(lease), content_type="application/json", if_generation_match=generation)
        return blob, lease


def release_lock(blob: Any) -> None:
    try:
        blob.delete(if_generation_match=blob.generation)
    except Exception:
        app.logger.exception("failed to release controller lock")


def global_gpu_quota() -> float:
    project = compute_v1.ProjectsClient().get(project=PROJECT)
    for quota in project.quotas:
        if quota.metric == "GPUS_ALL_REGIONS":
            return float(quota.limit)
    return 0.0


def configured_financial_policy() -> FinancialPolicy:
    credit = os.getenv("VERIFIED_PROMO_CREDIT_USD")
    verified = os.getenv("CREDIT_VERIFIED_AT")
    expires = os.getenv("PROMO_CREDIT_EXPIRES_AT")
    pricing_verified = os.getenv("PRICING_VERIFIED_AT")
    return FinancialPolicy(
        credit_usd=float(credit) if credit else None,
        credit_verified_at=datetime.fromisoformat(verified) if verified else None,
        credit_expires_at=datetime.fromisoformat(expires) if expires else None,
        pricing_verified_at=datetime.fromisoformat(pricing_verified) if pricing_verified else None,
        estimated_spend_usd=float(os.getenv("ESTIMATED_CAMPAIGN_SPEND_USD", "0")),
        active_liability_usd=float(os.getenv("ACTIVE_LIABILITY_USD", "0")),
        candidate_worst_case_usd=float(os.getenv("CANDIDATE_WORST_CASE_USD", "0")),
        reserve_usd=float(os.getenv("CREDIT_RESERVE_USD", "50")),
        campaign_cap_usd=min(float(os.getenv("CAMPAIGN_CAP_USD", "100")), 240.0),
    )


def discover_campaign_jobs(now: datetime) -> tuple[list[dict[str, Any]], list[str], int]:
    client = batch_v1.BatchServiceClient()
    regions = os.getenv(
        "CANDIDATE_REGIONS",
        "europe-west4,europe-west3,europe-west2,europe-west6,us-central1,us-east1,us-west1,asia-southeast1,asia-east1,asia-south1",
    ).split(",")
    jobs: list[dict[str, Any]] = []
    events: list[str] = []
    oldest_minutes = 0
    for region in regions:
        parent = f"projects/{PROJECT}/locations/{region.strip()}"
        try:
            for job in client.list_jobs(parent=parent):
                job_id = job.name.rsplit("/", 1)[-1]
                if not job_id.startswith("sewer-ml-"):
                    continue
                state = batch_v1.JobStatus.State(job.status.state).name
                jobs.append({"name": job.name, "region": region.strip(), "state": state})
                if state not in {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}:
                    created = job.create_time.replace(tzinfo=timezone.utc) if job.create_time.tzinfo is None else job.create_time
                    oldest_minutes = max(oldest_minutes, int((now - created).total_seconds() // 60))
                    events.extend(event.description for event in job.status.status_events)
        except Exception as exc:
            events.append(f"inventory failure in {region.strip()}: {type(exc).__name__}")
    return jobs, events, oldest_minutes


def load_state() -> dict[str, Any]:
    blob = _json_blob("state.json")
    if not blob.exists():
        return {"stage": "efficientnet_b0_20k", "attempted_regions": [], "active_jobs": []}
    return json.loads(blob.download_as_text())


def save_status(status: dict[str, Any]) -> None:
    _json_blob("status.json").upload_from_string(json.dumps(status, indent=2), content_type="application/json")


def read_status() -> dict[str, Any]:
    blob = _json_blob("status.json")
    status = json.loads(blob.download_as_text()) if blob.exists() else {
        "state": "DISCOVERING", "reason": "Controller inventory has not run yet.",
        "submission_enabled": SUBMISSION_ENABLED, "jobs": [],
    }
    provider = cloud()
    inventory = provider.inventory()
    pipeline_blob = _json_blob("pipeline-status.json")
    stored_pipeline = json.loads(pipeline_blob.download_as_text()) if pipeline_blob.exists() else {}
    status["pipeline"] = provider.reconcile_pipeline(stored_pipeline, inventory)
    status["inventory"] = inventory
    status["jobs"] = inventory.get("batch_jobs", status.get("jobs", []))
    status["provider_verified_at"] = inventory.get("verified_at")
    status["provider_inventory_complete"] = inventory.get("complete", False)
    status["max_active_gpu_jobs"] = int(os.getenv("MAX_ACTIVE_GPU_JOBS", "1"))
    status["required_global_gpu_quota"] = float(os.getenv("REQUIRED_GLOBAL_GPU_QUOTA", "1"))
    status["scheduler_interval_minutes"] = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", "15"))
    if inventory.get("errors"):
        status["reason"] = "Live GCP inventory is incomplete; no missing provider is shown as healthy."
    return status

def recent_logs(source: str, limit: int) -> list[dict[str, Any]]:
    filters = {
        "build": 'resource.type="build"',
        "batch": '(resource.type="batch_task" OR resource.type="batch_job")',
        "run": 'resource.type="cloud_run_revision" AND resource.labels.service_name="jointinspect-gpu-controller"',
    }
    if source not in filters:
        raise ValueError("Unsupported log source")
    entries = logging_v2.Client(project=PROJECT).list_entries(
        filter_=filters[source], order_by=logging_v2.DESCENDING, max_results=min(max(limit, 1), 300)
    )
    result = []
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, dict):
            message = json.dumps(payload, sort_keys=True, default=str)
        else:
            message = str(payload)
        result.append({"timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                       "severity": str(entry.severity or "DEFAULT"), "message": message[:12000]})
    return result


def diagnostics() -> dict[str, Any]:
    status = read_status()
    return {
        "generated_at": utcnow().isoformat(), "project": PROJECT,
        "controller_revision": os.getenv("K_REVISION"), "service": os.getenv("K_SERVICE"),
        "region": os.getenv("GOOGLE_CLOUD_REGION", "europe-west2"),
        "global_gpu_quota": global_gpu_quota(), "status": status,
        "storage": {"bucket": STATE_BUCKET, "state_prefix": STATE_PREFIX, "upload_prefix": UPLOAD_PREFIX},
        "commands": [
            {"id":"refresh","label":"Refresh all GCP status","mutates":False},
            {"id":"preflight","label":"Run guarded quota and budget preflight","mutates":True},
            {"id":"build-logs","label":"Read Cloud Build logs","mutates":False},
            {"id":"batch-logs","label":"Read Batch logs","mutates":False},
            {"id":"run-logs","label":"Read controller logs","mutates":False},
        ],
    }


def run_controller() -> dict[str, Any]:
    now = utcnow()
    lock, _ = acquire_lock(now)
    try:
        state = load_state()
        discovered_jobs, discovered_events, wait_minutes = discover_campaign_jobs(now)
        snapshot = Snapshot(
            now=now,
            submission_enabled=SUBMISSION_ENABLED,
            global_gpu_quota=global_gpu_quota(),
            active_jobs=discovered_jobs,
            completed_stage=state.get("completed_stage", False),
            wait_minutes=wait_minutes,
            events=discovered_events,
            checkpoint=state.get("checkpoint"),
            financial=configured_financial_policy(),
        )
        decision = decide(snapshot)
        status = {
            "state": decision.state.value,
            "action": decision.action,
            "reason": decision.reason,
            "submission_enabled": SUBMISSION_ENABLED,
            "global_gpu_quota": snapshot.global_gpu_quota,
            "last_controller_execution": now.isoformat(),
            "next_scheduled_execution": (now + timedelta(minutes=15)).isoformat(),
            "stage": state.get("stage", "efficientnet_b0_20k"),
            "attempted_regions": state.get("attempted_regions", []),
            "jobs": discovered_jobs,
            "checkpoint": snapshot.checkpoint,
        }
        # Preflight records a decision only. Mutations are separate,
        # confirmation-gated operations exposed through /command.
        save_status(status)
        return status
    finally:
        release_lock(lock)


@app.post("/run")
def run_endpoint():
    try:
        return jsonify(run_controller())
    except RuntimeError as exc:
        return jsonify({"state": "MANUAL_REVIEW_REQUIRED", "reason": str(exc)}), 409


@app.get("/status")
def status_endpoint():
    return jsonify(read_status())


@app.get("/report")
def report_endpoint():
    return jsonify({
        "schema_version": 1,
        "generated_at": utcnow().isoformat(),
        "project": PROJECT,
        "controller": "jointinspect-gpu-controller",
        "status": read_status(),
        "security": {"submission_enabled": SUBMISSION_ENABLED, "credentials_included": False, "arbitrary_commands_allowed": False},
    })


@app.get("/diagnostics")
def diagnostics_endpoint():
    return jsonify(diagnostics())


@app.get("/logs")
def logs_endpoint():
    try:
        source = request.args.get("source", "build")
        limit = int(request.args.get("limit", "100"))
        return jsonify({"source": source, "generated_at": utcnow().isoformat(), "entries": recent_logs(source, limit)})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/upload")
def upload_endpoint():
    payload = request.get_json(silent=True) or {}
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", str(payload.get("filename", "")))[:160]
    if not filename or filename in {".", ".."}:
        return jsonify({"error": "A safe filename is required"}), 400
    try:
        content = base64.b64decode(str(payload.get("content_base64", "")), validate=True)
    except Exception:
        return jsonify({"error": "Invalid base64 upload"}), 400
    if not content or len(content) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"Upload must be between 1 and {MAX_UPLOAD_BYTES} bytes"}), 413
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    object_name = f"{STATE_PREFIX}/{UPLOAD_PREFIX}/{stamp}-{filename}"
    blob = storage.Client(project=PROJECT).bucket(STATE_BUCKET).blob(object_name)
    blob.metadata = {"execution_allowed": "false", "uploaded_via": "jointinspect-code-console"}
    blob.upload_from_string(content, content_type=str(payload.get("content_type") or "application/octet-stream"), if_generation_match=0)
    return jsonify({"uploaded": True, "gcs_uri": f"gs://{STATE_BUCKET}/{object_name}", "bytes": len(content), "execution_allowed": False})


@app.get("/inventory")
def inventory_endpoint():
    provider = cloud()
    prefix = request.args.get("prefix", "")
    service = request.args.get("service")
    region = request.args.get("region")
    result = provider.inventory()
    result["objects"] = provider.objects(prefix)
    if service and region:
        result["revisions"] = provider.run_revisions(service, region)
    return jsonify(result)


@app.post("/command")
def command_endpoint():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", ""))
    try:
        return jsonify(cloud().command(action, payload))
    except OperationError as exc:
        return jsonify({"error": str(exc), "action": action}), exc.status
    except Exception:
        app.logger.exception("cloud command failed")
        return jsonify({"error": "Cloud command failed; inspect controller logs", "action": action}), 502


@app.get("/healthz")
def health():
    return {"ok": True, "submission_enabled": SUBMISSION_ENABLED}
