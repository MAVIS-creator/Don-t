"""Cloud wrapper for metadata-only 250K Sewer-ML selection."""
from __future__ import annotations

import argparse
import hashlib
import json

from google.cloud import storage

from curate_staged import gcp_attestation
from select_250k_manifest import manifest_csv, parse_csv, select_rows, shard_plan


def object_bytes(bucket, name: str) -> bytes:
    blob = bucket.blob(name)
    if not blob.exists():
        raise RuntimeError(f"Required GCS object missing: gs://{bucket.name}/{name}")
    return blob.download_as_bytes()


def reserved_names(bucket, objects: list[str]) -> set[str]:
    names = set()
    for name in objects:
        for row in parse_csv(object_bytes(bucket, name)):
            filename = str(row.get("Filename", "")).strip()
            if filename:
                names.add(filename.casefold())
    return names


def upload_create_only(bucket, name: str, payload: bytes, content_type: str) -> dict:
    blob = bucket.blob(name)
    blob.metadata = {"sha256": hashlib.sha256(payload).hexdigest(), "production_approved": "false"}
    blob.upload_from_string(payload, content_type=content_type, if_generation_match=0)
    return {"object": f"gs://{bucket.name}/{name}", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--reserve-object", action="append", default=[])
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--frame-block", type=int, default=100)
    parser.add_argument("--group-cap", type=int, default=30)
    parser.add_argument("--candidate-reserve", type=int, default=0)
    args = parser.parse_args()
    gcp_attestation(args.project)
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)
    source = object_bytes(bucket, args.train_annotations)
    exclusions = reserved_names(bucket, args.reserve_object)
    selected, report = select_rows(parse_csv(source), args.target, excluded_names=exclusions,
                                   seed=args.seed, frame_block=args.frame_block, group_cap=args.group_cap,
                                   candidate_reserve=args.candidate_reserve)
    report["source_annotation_uri"] = f"gs://{args.bucket}/{args.train_annotations}"
    report["source_annotation_sha256"] = hashlib.sha256(source).hexdigest()
    report["reserved_manifests"] = args.reserve_object
    report["reserved_unique_filenames"] = len(exclusions)
    prefix = args.output_prefix.rstrip("/")
    outputs = [
        upload_create_only(bucket, f"{prefix}/selection.csv", manifest_csv(selected).encode(), "text/csv"),
        upload_create_only(bucket, f"{prefix}/shard_plan.json", json.dumps(shard_plan(selected), indent=2).encode(), "application/json"),
    ]
    report["outputs"] = outputs
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode()
    report_output = upload_create_only(bucket, f"{prefix}/selection_report.json", report_bytes, "application/json")
    print(json.dumps({"selected_candidates": len(selected), "final_target": args.target,
                      "status": report["status"], "outputs": outputs + [report_output]}, indent=2))


if __name__ == "__main__":
    main()
