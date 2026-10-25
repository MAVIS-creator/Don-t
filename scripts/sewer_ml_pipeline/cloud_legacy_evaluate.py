"""Same-image development evaluation for V1, V2, and original WRc.

Thresholds for the native multilabel models are fitted only on threshold_tune.
The original WRc single-label head is reported only for its directly compatible
displaced-joint class. The locked test is never read by this worker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from google.cloud import storage
from PIL import Image

from curate_staged import TARGETS, gcp_attestation, verify_disjoint
from evaluate_same_images import preprocess
from metrics_v3 import calibration, complete_report, metrics, tune_thresholds


def split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"GCS URI required: {uri}")
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name or ".." in Path(name).parts:
        raise ValueError(f"Unsafe GCS URI: {uri}")
    return bucket, name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(client: storage.Client, uri: str, path: Path, expected_sha: str | None = None) -> None:
    bucket, name = split_gs(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket).blob(name).download_to_filename(path)
    if expected_sha and sha256(path) != expected_sha.lower():
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Artifact digest mismatch: {uri}")


def read_manifest(client: storage.Client, uri: str) -> list[dict[str, str]]:
    bucket, name = split_gs(uri)
    rows = list(csv.DictReader(client.bucket(bucket).blob(name).download_as_text().splitlines()))
    if not rows or not {"Filename", "GCS_URI", "sha256", *TARGETS}.issubset(rows[0]):
        raise ValueError(f"Invalid evaluation manifest: {uri}")
    return rows


def stage_images(client: storage.Client, rows: list[dict[str, str]], directory: Path) -> None:
    seen: set[str] = set()
    for row in rows:
        filename = row["Filename"]
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe image filename: {filename}")
        if filename in seen:
            continue
        seen.add(filename)
        download(client, row["GCS_URI"], directory / filename, row["sha256"])


def infer(session: ort.InferenceSession, rows: list[dict[str, str]], images: Path, spec: dict) -> tuple[np.ndarray, dict]:
    output_name = session.get_outputs()[0].name
    values, latencies = [], []
    for row in rows:
        with Image.open(images / row["Filename"]) as image:
            input_value = preprocess(image, spec)
        started = time.perf_counter()
        result = np.asarray(session.run([output_name], {session.get_inputs()[0].name: input_value})[0])[0]
        latencies.append((time.perf_counter() - started) * 1000)
        if result.shape != (len(spec["class_codes"]),) or not np.isfinite(result).all():
            raise RuntimeError(f"Invalid output from {spec['name']}: {result.shape}")
        if spec["kind"] == "sewerml_multilabel_sigmoid":
            if (result < 0).any() or (result > 1).any():
                raise RuntimeError(f"{spec['name']} did not emit probabilities")
        elif (result < 0).any() or abs(float(result.sum()) - 1.0) > 0.01:
            raise RuntimeError("Original WRc output is not a normalized softmax")
        values.append(result)
    return np.asarray(values), {
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }


def truth(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row.get(code) in {"1", "1.0"}) for code in TARGETS] for row in rows])


def write_predictions(path: Path, rows: list[dict[str, str]], values: np.ndarray, classes: list[str]) -> None:
    with path.open("w") as destination:
        for row, prediction in zip(rows, values, strict=True):
            destination.write(json.dumps({
                "Filename": row["Filename"], "source_sha256": row["sha256"],
                "class_codes": classes, "probabilities": prediction.tolist(),
            }, sort_keys=True) + "\n")


def upload_create_only(client: storage.Client, path: Path, uri: str) -> None:
    bucket, name = split_gs(uri)
    client.bucket(bucket).blob(name).upload_from_filename(str(path), if_generation_match=0)


def run(args) -> None:
    gcp_attestation(args.project)
    client = storage.Client(project=args.project)
    splits = {
        "train": read_manifest(client, args.train),
        "early_stop": read_manifest(client, args.early_stop),
        "threshold_tune": read_manifest(client, args.threshold_tune),
        "dev_report": read_manifest(client, args.dev_report),
    }
    verify_disjoint(splits)
    with tempfile.TemporaryDirectory(prefix="legacy-eval-") as temporary:
        root = Path(temporary)
        images = root / "images"
        stage_images(client, splits["threshold_tune"] + splits["dev_report"], images)
        model_root = root / "models"
        specs = [
            {
                "name": "sewer-condition-v1", "kind": "sewerml_multilabel_sigmoid",
                "preprocess": "imagenet_rgb_nchw", "image_size": 128,
                "class_codes": list(TARGETS), "model_uri": args.v1_model,
                "model_sha": args.v1_sha, "data_uri": args.v1_data,
                "data_sha": args.v1_data_sha,
            },
            {
                "name": "sewer-condition-v2", "kind": "sewerml_multilabel_sigmoid",
                "preprocess": "imagenet_rgb_nchw", "image_size": 224,
                "class_codes": list(TARGETS), "model_uri": args.v2_model,
                "model_sha": args.v2_sha, "data_uri": args.v2_data,
                "data_sha": args.v2_data_sha,
            },
            {
                "name": "wrc-inceptionresnetv2-original", "kind": "wrc_single_softmax",
                "preprocess": "wrc_raw_rgb_nhwc", "image_size": 299,
                "class_codes": ["B", "CN", "CK", "CND", "D", "DE", "JD", "F", "H", "J", "LL", "R", "W"],
                "model_uri": args.wrc_model, "model_sha": args.wrc_sha,
            },
        ]
        reports = {}
        dev_truth = truth(splits["dev_report"])
        for spec in specs:
            directory = model_root / spec["name"]
            model_path = directory / "model.onnx"
            download(client, spec["model_uri"], model_path, spec["model_sha"])
            if spec.get("data_uri"):
                download(client, spec["data_uri"], directory / "model.onnx.data", spec["data_sha"])
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            if spec["kind"] == "sewerml_multilabel_sigmoid":
                tune_values, _ = infer(session, splits["threshold_tune"], images, spec)
                thresholds = tune_thresholds(truth(splits["threshold_tune"]), tune_values)
                dev_values, timing = infer(session, splits["dev_report"], images, spec)
                report = complete_report(dev_truth, dev_values, TARGETS, thresholds)
                report["thresholds"] = {code: float(value) for code, value in zip(TARGETS, thresholds)}
            else:
                dev_values, timing = infer(session, splits["dev_report"], images, spec)
                top1 = dev_values.argmax(axis=1)
                fs_true = dev_truth[:, 0].astype(bool)
                fs_pred = top1 == 6
                report = {
                    "common_taxonomy": {"FS_displaced_joint_DIRECT_only": metrics(fs_true, fs_pred)},
                    "fs_probability_calibration": calibration(fs_true[:, None], dev_values[:, 6:7]),
                    "native_13_class_top1_counts": {code: int((top1 == index).sum()) for index, code in enumerate(spec["class_codes"])},
                    "uncomparable_classes": list(TARGETS[1:]),
                    "single_label_head_limitation": True,
                }
            report.update(timing)
            report.update({
                "model_id": spec["name"], "model_sha256": spec["model_sha"],
                "model_size_bytes": model_path.stat().st_size,
                "development_report_count": len(splits["dev_report"]),
                "locked_test_evaluated": False, "production_approved": False,
            })
            reports[spec["name"]] = report
            predictions = root / f"{spec['name']}_predictions.jsonl"
            write_predictions(predictions, splits["dev_report"], dev_values, spec["class_codes"])
            upload_create_only(client, predictions, f"{args.output_prefix.rstrip('/')}/{predictions.name}")
        combined = root / "same_image_report.json"
        combined.write_text(json.dumps({
            "models": reports, "identical_ordered_images": True,
            "sample_count": len(splits["dev_report"]),
            "threshold_tuning_count": len(splits["threshold_tune"]),
            "locked_test_used_for_tuning": False, "production_approved": False,
        }, indent=2))
        upload_create_only(client, combined, f"{args.output_prefix.rstrip('/')}/{combined.name}")
        print(json.dumps({"evaluated_models": list(reports), "images": len(splits["dev_report"]),
                          "output_prefix": args.output_prefix, "locked_test_evaluated": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    for name in ("train", "early-stop", "threshold-tune", "dev-report", "output-prefix"):
        parser.add_argument("--" + name, required=True)
    for name in ("v1-model", "v1-sha", "v1-data", "v1-data-sha",
                 "v2-model", "v2-sha", "v2-data", "v2-data-sha", "wrc-model", "wrc-sha"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
