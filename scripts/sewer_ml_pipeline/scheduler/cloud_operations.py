"""Allow-listed GCP operations used by the protected /code console.

Every mutation requires an exact confirmation phrase and provider identifiers.
There is deliberately no shell, arbitrary URL, or free-form API proxy.
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage


TERMINAL_BUILD = {"SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED"}
TERMINAL_BATCH = {"SUCCEEDED", "FAILED", "CANCELLED", "DELETION_IN_PROGRESS"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class OperationError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CloudOperations:
    def __init__(self, project: str, bucket: str):
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.http = AuthorizedSession(credentials)
        self.project = project
        self.bucket_name = bucket
        self.run_regions = [x.strip() for x in os.getenv("RUN_REGIONS", "europe-west2,us-central1").split(",") if x.strip()]
        self.batch_regions = [x.strip() for x in os.getenv("CANDIDATE_REGIONS", "europe-west4,europe-west2,us-central1").split(",") if x.strip()]

    def _request(self, method: str, url: str, *, payload: Any = None, ok: tuple[int, ...] = (200,)) -> Any:
        response = self.http.request(method, url, json=payload, timeout=30)
        if response.status_code not in ok:
            try:
                detail = response.json().get("error", {}).get("message")
            except Exception:
                detail = response.text[:500]
            raise OperationError(detail or f"GCP request failed ({response.status_code})", response.status_code)
        if response.status_code == 204 or not response.content:
            return {"ok": True}
        return response.json()

    def builds(self, limit: int = 30) -> list[dict[str, Any]]:
        data = self._request("GET", f"https://cloudbuild.googleapis.com/v1/projects/{self.project}/builds?pageSize={min(limit, 100)}")
        return [{
            "id": item.get("id"), "name": item.get("name"), "status": item.get("status", "UNKNOWN"),
            "create_time": item.get("createTime"), "finish_time": item.get("finishTime"),
            "log_url": item.get("logUrl"), "substitutions": item.get("substitutions", {}),
        } for item in data.get("builds", [])]

    def run_services(self) -> list[dict[str, Any]]:
        services: list[dict[str, Any]] = []
        for region in self.run_regions:
            data = self._request("GET", f"https://run.googleapis.com/v2/projects/{self.project}/locations/{region}/services?pageSize=100")
            for item in data.get("services", []):
                services.append({
                    "name": item.get("name"), "region": region, "uri": item.get("uri"),
                    "latest_ready_revision": item.get("latestReadyRevision"),
                    "latest_created_revision": item.get("latestCreatedRevision"),
                    "traffic": item.get("trafficStatuses", []), "conditions": item.get("conditions", []),
                })
        return services

    def run_revisions(self, service: str, region: str) -> list[dict[str, Any]]:
        self._safe(service); self._region(region)
        parent = f"projects/{self.project}/locations/{region}/services/{service}"
        data = self._request("GET", f"https://run.googleapis.com/v2/{parent}/revisions?pageSize=100")
        return [{"name": x.get("name"), "create_time": x.get("createTime"), "conditions": x.get("conditions", [])} for x in data.get("revisions", [])]

    def batch_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for region in self.batch_regions:
            url = f"https://batch.googleapis.com/v1/projects/{self.project}/locations/{region}/jobs?pageSize={min(limit, 100)}"
            data = self._request("GET", url)
            for item in data.get("jobs", []):
                status = item.get("status", {})
                result.append({"name": item.get("name"), "region": region, "state": status.get("state", "UNKNOWN"),
                               "create_time": item.get("createTime"), "update_time": item.get("updateTime"),
                               "status_events": status.get("statusEvents", [])[-10:]})
        return result

    def objects(self, prefix: str = "", limit: int = 200) -> list[dict[str, Any]]:
        if prefix and (prefix.startswith("/") or ".." in prefix.split("/")):
            raise OperationError("Unsafe object prefix")
        blobs = storage.Client(project=self.project).list_blobs(self.bucket_name, prefix=prefix, max_results=min(limit, 500))
        return [{"name": b.name, "size": b.size, "content_type": b.content_type, "updated": b.updated.isoformat() if b.updated else None,
                 "generation": int(b.generation) if b.generation else None, "metadata": b.metadata or {}} for b in blobs]

    def inventory(self) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        output: dict[str, Any] = {"verified_at": _now(), "source": "LIVE_GCP"}
        for key, fn in (("builds", self.builds), ("run_services", self.run_services), ("batch_jobs", self.batch_jobs)):
            try:
                output[key] = fn()
            except Exception as exc:
                output[key] = []
                errors.append({"provider": key, "error": f"{type(exc).__name__}: {exc}"})
        output["errors"] = errors
        output["complete"] = not errors
        return output

    def reconcile_pipeline(self, stored: Any, inventory: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(stored, dict):
            return {"state": "UNVERIFIED", "verification": "INVALID_SNAPSHOT", "steps": []}
        safe = json.loads(json.dumps(stored))
        builds = {x.get("id"): x for x in inventory.get("builds", [])}
        jobs = {str(x.get("name", "")).rsplit("/", 1)[-1]: x for x in inventory.get("batch_jobs", [])}
        verified_states: list[str] = []
        for step in safe.get("steps", []):
            reported = str(step.get("state", "UNKNOWN")).upper()
            resource_id = str(step.get("resource_id") or step.get("build_id") or step.get("job") or "")
            resource = builds.get(resource_id) or jobs.get(resource_id)
            step["reported_state"] = reported
            if resource:
                state = str(resource.get("status") or resource.get("state") or "UNKNOWN").upper()
                step.update({"state": state, "status_source": "LIVE_GCP", "verified_live_at": inventory.get("verified_at"),
                             "resource_id": resource_id, "detail": f"Verified from live GCP resource {resource_id}"})
                verified_states.append(state)
            else:
                step.update({"state": "PLANNED" if reported in {"QUEUED", "PENDING"} else "UNVERIFIED",
                             "status_source": "PERSISTED_GCS", "verified_live_at": None,
                             "detail": "No matching live GCP resource ID; stored status is not treated as live"})
        if any(x in {"WORKING", "QUEUED", "PENDING", "RUNNING", "SCHEDULED"} for x in verified_states):
            state = "RUNNING"
        elif verified_states and all(x in TERMINAL_BUILD | TERMINAL_BATCH for x in verified_states):
            state = "FAILED" if any(x in {"FAILURE", "FAILED", "CANCELLED", "TIMEOUT", "INTERNAL_ERROR"} for x in verified_states) else "COMPLETED"
        else:
            state = "UNVERIFIED"
        safe.update({"reported_state": safe.get("state", "UNKNOWN"), "state": state,
                     "verification": "LIVE_GCP" if verified_states else "STORED_SNAPSHOT_UNVERIFIED",
                     "verified_live_at": inventory.get("verified_at") if verified_states else None})
        return safe

    def command(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "build.trigger": self._trigger_build, "build.retry": self._retry_build, "build.cancel": self._cancel_build,
            "run.rollback": self._rollback_run, "run.deploy_image": self._deploy_image,
            "batch.submit": self._submit_batch, "batch.cancel": self._cancel_batch,
            "storage.delete": self._delete_object, "storage.download": self._download_object,
        }
        handler = handlers.get(action)
        if not handler:
            raise OperationError("Unsupported operation")
        target = str(payload.get("target", ""))
        if action != "storage.download":
            expected = f"{action}:{target}"
            if payload.get("confirmation") != expected:
                raise OperationError(f"Confirmation must exactly equal {expected}", 409)
        result = handler(payload)
        return {"ok": True, "action": action, "target": target, "completed_at": _now(), "provider_result": result}

    def _trigger_build(self, p: dict[str, Any]) -> Any:
        trigger = self._safe(str(p.get("target", "")))
        branch = self._safe(str(p.get("branch", "main")))
        return self._request("POST", f"https://cloudbuild.googleapis.com/v1/projects/{self.project}/triggers/{quote(trigger, safe='')}:run",
                             payload={"branchName": branch})

    def _retry_build(self, p: dict[str, Any]) -> Any:
        build = self._safe(str(p.get("target", "")))
        return self._request("POST", f"https://cloudbuild.googleapis.com/v1/projects/{self.project}/builds/{build}:retry", payload={})

    def _cancel_build(self, p: dict[str, Any]) -> Any:
        build = self._safe(str(p.get("target", "")))
        return self._request("POST", f"https://cloudbuild.googleapis.com/v1/projects/{self.project}/builds/{build}:cancel", payload={})

    def _rollback_run(self, p: dict[str, Any]) -> Any:
        service, region = self._service_target(p)
        revision = self._safe(str(p.get("revision", "")))
        name = f"projects/{self.project}/locations/{region}/services/{service}"
        body = {"name": name, "traffic": [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": revision, "percent": 100}]}
        return self._request("PATCH", f"https://run.googleapis.com/v2/{name}?updateMask=traffic", payload=body)

    def _deploy_image(self, p: dict[str, Any]) -> Any:
        service, region = self._service_target(p)
        image = str(p.get("image", ""))
        if not image or not image.startswith(("gcr.io/", "us-docker.pkg.dev/", "europe-docker.pkg.dev/", "europe-west2-docker.pkg.dev/")):
            raise OperationError("An Artifact Registry or GCR image is required")
        name = f"projects/{self.project}/locations/{region}/services/{service}"
        current = self._request("GET", f"https://run.googleapis.com/v2/{name}")
        containers = current.get("template", {}).get("containers", [])
        if not containers:
            raise OperationError("Service has no deployable container")
        containers[0]["image"] = image
        body = {"name": name, "template": {**current.get("template", {}), "containers": containers}}
        return self._request("PATCH", f"https://run.googleapis.com/v2/{name}?updateMask=template.containers", payload=body)

    def _submit_batch(self, p: dict[str, Any]) -> Any:
        job_id = self._safe(str(p.get("target", "")))
        region = self._region(str(p.get("region", "")))
        spec_object = self._safe(str(p.get("spec_object", "")))
        blob = storage.Client(project=self.project).bucket(self.bucket_name).blob(spec_object)
        if not blob.exists():
            raise OperationError("Approved Batch specification was not found")
        spec = json.loads(blob.download_as_text())
        if not isinstance(spec, dict) or "taskGroups" not in spec or "allocationPolicy" not in spec:
            raise OperationError("Invalid Batch job specification")
        return self._request("POST", f"https://batch.googleapis.com/v1/projects/{self.project}/locations/{region}/jobs?job_id={quote(job_id)}", payload=spec)

    def _cancel_batch(self, p: dict[str, Any]) -> Any:
        job_id = self._safe(str(p.get("target", "")))
        region = self._region(str(p.get("region", "")))
        name = f"projects/{self.project}/locations/{region}/jobs/{job_id}"
        return self._request("POST", f"https://batch.googleapis.com/v1/{name}:cancel", payload={"reason": "Cancelled from authenticated JointInspect control room"})

    def _delete_object(self, p: dict[str, Any]) -> Any:
        name = self._safe(str(p.get("target", "")))
        generation = int(p.get("generation", 0))
        if not generation:
            raise OperationError("Object generation is required")
        storage.Client(project=self.project).bucket(self.bucket_name).blob(name).delete(if_generation_match=generation)
        return {"deleted": f"gs://{self.bucket_name}/{name}", "generation": generation}

    def _download_object(self, p: dict[str, Any]) -> Any:
        name = self._safe(str(p.get("target", "")))
        blob = storage.Client(project=self.project).bucket(self.bucket_name).blob(name)
        blob.reload()
        if not blob.size or blob.size > 2 * 1024 * 1024:
            raise OperationError("Console downloads are limited to 2 MiB; use a signed artifact workflow")
        content = blob.download_as_bytes(if_generation_match=blob.generation)
        return {"filename": name.rsplit("/", 1)[-1], "content_type": blob.content_type,
                "content_base64": base64.b64encode(content).decode(), "generation": int(blob.generation)}

    def _service_target(self, p: dict[str, Any]) -> tuple[str, str]:
        return self._safe(str(p.get("target", ""))), self._region(str(p.get("region", "")))

    def _safe(self, value: str) -> str:
        if not value or not SAFE_NAME.fullmatch(value) or ".." in value.split("/"):
            raise OperationError("Invalid provider identifier")
        return value

    def _region(self, value: str) -> str:
        value = self._safe(value)
        if value not in set(self.run_regions) | set(self.batch_regions):
            raise OperationError("Region is not allow-listed")
        return value
