"""Stream one approved remote ZIP directly into GCS from a GCP worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from urllib.parse import urlsplit

import requests
from google.cloud import storage


def stage(url: str, project: str, bucket: str, object_name: str, max_bytes: int,
          reuse_existing: bool = False) -> dict:
    parsed = urlsplit(url)
    allowed_hosts = {"sciencedata.dk", "www.sciencedata.dk"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("Archive URL must use HTTPS on an approved Sewer-ML host")
    if not object_name.endswith(".zip") or object_name.startswith("/") or ".." in object_name.split("/"):
        raise ValueError("A safe .zip GCS object path is required")

    digest = hashlib.sha256(url.encode()).hexdigest()
    blob = storage.Client(project=project).bucket(bucket).blob(object_name)
    if reuse_existing and blob.exists():
        blob.reload()
        if (blob.metadata or {}).get("source_url_sha256") != digest or not blob.size:
            raise RuntimeError(f"Existing archive provenance mismatch: gs://{bucket}/{object_name}")
        return {"gcs_uri": f"gs://{bucket}/{object_name}", "object": object_name,
                "generation": int(blob.generation), "bytes": int(blob.size),
                "source_url_sha256": digest, "reused": True}

    with requests.get(url, stream=True, allow_redirects=False, timeout=(30, 900),
                      headers={"Accept-Encoding": "identity"}) as response:
        if response.is_redirect:
            raise RuntimeError("Archive redirects are not accepted; approve the final HTTPS URL explicitly")
        response.raise_for_status()
        length = int(response.headers.get("Content-Length", "0"))
        if length <= 0 or length > max_bytes:
            raise RuntimeError(f"Archive size must be known and no larger than {max_bytes} bytes")
        blob.metadata = {"source_url_sha256": digest, "execution_allowed": "false"}
        blob.upload_from_file(response.raw, size=length, content_type="application/zip",
                              rewind=False, if_generation_match=0, timeout=900)
        blob.reload()
    return {"gcs_uri": f"gs://{bucket}/{object_name}", "object": object_name,
            "generation": int(blob.generation), "bytes": int(blob.size),
            "source_url_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output", default="/workspace/staged-archive.json")
    args = parser.parse_args()
    result = stage(args.url, args.project, args.bucket, args.object, args.max_bytes,
                   reuse_existing=args.reuse_existing)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
