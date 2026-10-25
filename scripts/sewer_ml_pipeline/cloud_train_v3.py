"""Cloud-only GCS staging and artifact publication for Sewer-ML v3.

The worker downloads only objects named by immutable split manifests, verifies every
image digest, invokes the native trainer, validates its required outputs, and uploads
the trained model with create-only GCS preconditions. It never reads locked-test
data and never permits a workstation execution path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from curate_staged import TARGETS, gcp_attestation, group_id, verify_disjoint

MAX_FILES = 30_000
BOOT_DISK_SIZE_BYTES = 150 * 1024**3
MAX_TOTAL_BYTES = 80 * 1024**3  # Reconciled for 150 GiB boot disk with OS, container layers, and headroom
FILESYSTEM_SAFETY_MARGIN_BYTES = 15 * 1024**3
MIN_INITIAL_FREE_BYTES = 25 * 1024**3
REQUIRED_ARTIFACTS = ("best.pt", "model.onnx", "metrics.json", "history.json", "model_card.json")
REQUIRED_MANIFEST_COLUMNS = ("Filename", "sha256", "GCS_URI")
REQUIRED_CHECKPOINT_KEYS = {"epoch", "model", "optimizer", "scheduler", "best_f1", "wait", "config_id"}
SPLITS = ("train", "early_stop", "threshold_tune", "dev_report")


def validate_checkpoint_state(ckpt: dict, expected_config_id: str) -> None:
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint resume error: checkpoint state is not a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS - set(ckpt.keys())
    if missing:
        raise RuntimeError(f"Checkpoint resume error: missing required keys {sorted(missing)}")
    if ckpt.get("config_id") != expected_config_id:
        raise RuntimeError(f"Checkpoint resume error: config_id mismatch (checkpoint: '{ckpt.get('config_id')}', expected: '{expected_config_id}')")
    if not isinstance(ckpt.get("epoch"), int) or ckpt["epoch"] < 0:
        raise RuntimeError(f"Checkpoint resume error: invalid epoch in checkpoint: {ckpt.get('epoch')}")


def check_disk_space(target_path: Path, required_bytes: int, safety_margin: int = FILESYSTEM_SAFETY_MARGIN_BYTES) -> dict[str, int]:
    target = target_path if target_path.exists() else target_path.parent
    total, used, free = shutil.disk_usage(target)
    usable = free - safety_margin
    if usable < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space in {target}: free={free / (1024**3):.2f} GiB, "
            f"required={required_bytes / (1024**3):.2f} GiB with "
            f"{safety_margin / (1024**3):.2f} GiB safety margin (usable: {usable / (1024**3):.2f} GiB)"
        )
    return {"total": total, "used": used, "free": free, "usable": usable}


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI: {uri}")
    bucket, sep, name = uri[5:].partition("/")
    if not sep or not bucket or not name or ".." in PurePosixPath(name).parts:
        raise ValueError(f"Unsafe GCS URI: {uri}")
    return bucket, name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(split_rows: dict[str, list[dict]], allowed_bucket: str | None = None) -> dict[str, int]:
    if set(split_rows) != set(SPLITS) or any(not split_rows[name] for name in SPLITS):
        raise ValueError("All four non-empty training/development splits are required")
    verify_disjoint(split_rows)
    total = 0
    for split, rows in split_rows.items():
        if not rows:
            raise ValueError(f"{split}: split manifest has no records")
        for row in rows:
            filename = str(row.get("Filename", ""))
            if not filename or Path(filename).name != filename or not group_id(row):
                raise ValueError(f"{split}: unsafe filename or missing group provenance")
            if len(str(row.get("sha256", ""))) != 64:
                raise ValueError(f"{split}: missing or invalid SHA-256 for {filename}")
            uri = str(row.get("GCS_URI", ""))
            bucket_name, _ = parse_gs_uri(uri)
            if allowed_bucket and bucket_name != allowed_bucket:
                raise ValueError(f"{split}: image {uri} is outside the approved GCS bucket {allowed_bucket}")
            if not any(row.get(code) in ("1", "1.0", 1, 1.0) for code in TARGETS):
                raise ValueError(f"{split}: {filename} has no recognized target label")
            total += 1
    if total > MAX_FILES:
        raise ValueError(f"Manifest exceeds {MAX_FILES} image safety cap (got {total})")
    return {name: len(rows) for name, rows in split_rows.items()}


def read_csv_bytes(payload: bytes) -> list[dict]:
    try:
        text = payload.decode("utf-8-sig").splitlines()
    except Exception as exc:
        raise ValueError(f"Corrupt manifest: cannot decode UTF-8 payload: {exc}") from exc
    non_empty_lines = [line for line in text if line.strip()]
    if not non_empty_lines:
        raise ValueError("Corrupt manifest: payload is empty")
    reader = csv.DictReader(non_empty_lines)
    if not reader.fieldnames or all(f is None or not f.strip() for f in reader.fieldnames):
        raise ValueError("Corrupt manifest: missing CSV header")
    missing = set(REQUIRED_MANIFEST_COLUMNS) - set(reader.fieldnames)
    if missing:
        raise ValueError(f"Corrupt manifest: missing required columns: {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError("Corrupt manifest: CSV has header but no data rows")
    return rows


def download_verified(
    blob, destination: Path, expected_sha256: str, remaining_bytes: int,
    safety_margin: int = FILESYSTEM_SAFETY_MARGIN_BYTES,
) -> int:
    blob.reload()
    size = int(blob.size or 0)
    if size < 1:
        raise RuntimeError(f"Missing or empty image blob: {getattr(blob, 'name', destination.name)}")
    if size > remaining_bytes:
        raise RuntimeError(f"GCS staging byte limit exceeded by {getattr(blob, 'name', destination.name)}: {size} > {remaining_bytes}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_disk_space(destination.parent, size, safety_margin)
    blob.download_to_filename(str(destination))
    actual_size = destination.stat().st_size
    if actual_size != size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Interrupted staging for {getattr(blob, 'name', destination.name)}: expected {size} bytes, got {actual_size}")
    actual_sha = sha256_file(destination)
    if actual_sha != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {getattr(blob, 'name', destination.name)}: expected {expected_sha256}, got {actual_sha}")
    return size


def artifact_inventory(output: Path) -> list[dict]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing training artifacts: {missing}")
    card = json.loads((output / "model_card.json").read_text())
    if card.get("production_approved") is not False or card.get("locked_test_evaluated") is not False:
        raise RuntimeError("Model safety flags are invalid")
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "resume.pt"
    ]


def publish_artifacts(output_dir: Path, client, out_bucket: str, out_prefix: str, config_id: str, inventory: list[dict]) -> None:
    if not inventory:
        raise RuntimeError("Artifact publication failure: artifact inventory is empty")
    for item in inventory:
        path = output_dir / item["name"]
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Artifact publication failure: missing or empty artifact {item['name']}")
        target = client.bucket(out_bucket).blob(f"{out_prefix.rstrip('/')}/{item['name']}")
        target.metadata = {"sha256": item["sha256"], "config_id": config_id, "production_approved": "false"}
        target.upload_from_filename(str(path), if_generation_match=0)


def run(args) -> None:
    gcp_attestation(args.project)
    from google.cloud import storage

    client = storage.Client(project=args.project)
    work = Path(tempfile.mkdtemp(prefix="sewerml-v3-", dir=args.scratch_root))
    try:
        manifests = work / "manifests"
        images = work / "images"
        output = work / "output"
        manifests.mkdir(); images.mkdir(); output.mkdir()
        # Fail-closed disk space preflight on scratch directory
        check_disk_space(work, MIN_INITIAL_FREE_BYTES)
        split_rows = {}
        manifest_hashes = {}
        for split in SPLITS:
            bucket_name, object_name = parse_gs_uri(getattr(args, split))
            if bucket_name != args.allowed_bucket:
                raise ValueError("Manifest is outside the approved GCS bucket")
            payload = client.bucket(bucket_name).blob(object_name).download_as_bytes()
            manifest_hashes[split] = hashlib.sha256(payload).hexdigest()
            (manifests / f"{split}.csv").write_bytes(payload)
            split_rows[split] = read_csv_bytes(payload)
        split_counts = validate_rows(split_rows, args.allowed_bucket)
        consumed = 0
        seen = set()
        for rows in split_rows.values():
            for row in rows:
                filename = row["Filename"]
                if filename in seen:
                    continue
                seen.add(filename)
                bucket_name, object_name = parse_gs_uri(row["GCS_URI"])
                consumed += download_verified(
                    client.bucket(bucket_name).blob(object_name), images / filename,
                    row["sha256"], MAX_TOTAL_BYTES - consumed,
                )
        out_bucket, out_prefix = parse_gs_uri(args.output_prefix)
        if out_bucket != args.allowed_bucket:
            raise ValueError("Output is outside the approved GCS bucket")
        checkpoint_bucket = client.bucket(out_bucket)
        for name in ("resume.pt", "best.pt"):
            checkpoint = checkpoint_bucket.blob(f"{out_prefix.rstrip('/')}/checkpoints/{name}")
            if checkpoint.exists():
                checkpoint.download_to_filename(str(output / name))
        command = [
            sys.executable, str(Path(__file__).with_name("train_v3.py")),
            "--project", args.project,
            "--train", str(manifests / "train.csv"),
            "--early-stop", str(manifests / "early_stop.csv"),
            "--threshold-tune", str(manifests / "threshold_tune.csv"),
            "--dev-report", str(manifests / "dev_report.csv"),
            "--images-dir", str(images), "--output", str(output),
            "--config-id", args.config_id, "--architecture", args.architecture,
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--resume",
            "--checkpoint-prefix", f"gs://{out_bucket}/{out_prefix.rstrip('/')}/checkpoints",
        ]
        subprocess.run(command, check=True)
        inventory = artifact_inventory(output)
        provenance = {
            "config_id": args.config_id, "manifest_sha256": manifest_hashes,
            "split_counts": split_counts, "staged_image_bytes": consumed,
            "artifacts": inventory, "production_approved": False,
            "locked_test_evaluated": False,
        }
        manifest_path = output / "training_manifest.json"
        manifest_path.write_text(json.dumps(provenance, indent=2))
        inventory.append({
            "name": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        })
        publish_artifacts(output, client, out_bucket, out_prefix, args.config_id, inventory)
        print(json.dumps({"uploaded": True, "output_prefix": args.output_prefix,
                          "artifacts": len(inventory), "production_approved": False}))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="joint-inspection-510310")
    parser.add_argument("--allowed-bucket", default="joint-inspection-510310-data")
    for name in SPLITS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--architecture", choices=("efficientnet_b0", "convnext_tiny"), default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scratch-root", default="/workspace/tmp")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
