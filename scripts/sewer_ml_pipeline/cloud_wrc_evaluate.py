"""Evaluate an adapted WRc candidate without retraining or touching locked test data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from google.cloud import storage

from curate_staged import TARGETS, gcp_attestation, verify_disjoint
from metrics_v3 import complete_report, tune_thresholds


def split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"GCS URI required: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"GCS object URI required: {uri}")
    return parts[0], parts[1]


def download_text(client: storage.Client, uri: str) -> str:
    bucket, name = split_gs(uri)
    return client.bucket(bucket).blob(name).download_as_text()


def download(client: storage.Client, uri: str, path: Path) -> None:
    bucket, name = split_gs(uri)
    client.bucket(bucket).blob(name).download_to_filename(path)


def upload(client: storage.Client, path: Path, uri: str, content_type: str) -> None:
    bucket, name = split_gs(uri)
    client.bucket(bucket).blob(name).upload_from_filename(path, content_type=content_type)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rows(text: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("Evaluation manifest is empty")
    required = {"GCS_URI", *TARGETS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Evaluation manifest is missing columns: {sorted(missing)}")
    return rows


def truth(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(code) in {"1", "1.0"}) for code in TARGETS] for row in rows],
        dtype=np.float32,
    )


def make_dataset(tf, rows: list[dict[str, str]], batch_size: int):
    uris = [row["GCS_URI"] for row in rows]
    labels = truth(rows)
    dataset = tf.data.Dataset.from_tensor_slices((uris, labels))

    def load_image(uri, label):
        image = tf.io.decode_image(tf.io.read_file(uri), channels=3, expand_animations=False)
        image = tf.image.resize_with_pad(tf.cast(image, tf.float32), 299, 299)
        return image, tf.cast(label, tf.float32)

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predictions(model, dataset) -> np.ndarray:
    values = np.asarray(model.predict(dataset, verbose=1), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(TARGETS):
        raise RuntimeError(f"Expected [N,{len(TARGETS)}] model output, got {values.shape}")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RuntimeError("Model emitted invalid probabilities")
    return values


def write_predictions(path: Path, rows: list[dict[str, str]], values: np.ndarray) -> None:
    with path.open("w") as destination:
        for row, probability in zip(rows, values, strict=True):
            destination.write(
                json.dumps(
                    {
                        "Filename": row.get("Filename"),
                        "GCS_URI": row["GCS_URI"],
                        "class_order": list(TARGETS),
                        "probabilities": [float(value) for value in probability],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--early-stop", required=True)
    parser.add_argument("--threshold-tune", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    gcp_attestation(args.project)
    client = storage.Client(project=args.project)
    split_uris = {
        "train": args.train,
        "early_stop": args.early_stop,
        "threshold_tune": args.threshold_tune,
        "dev_report": args.dev_report,
    }
    split_rows = {name: parse_rows(download_text(client, uri)) for name, uri in split_uris.items()}
    verify_disjoint(split_rows)

    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        model_path = output / "model.keras"
        download(client, args.model, model_path)

        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        tune_probabilities = predictions(
            model, make_dataset(tf, split_rows["threshold_tune"], args.batch_size)
        )
        thresholds = tune_thresholds(truth(split_rows["threshold_tune"]), tune_probabilities)
        dev_probabilities = predictions(
            model, make_dataset(tf, split_rows["dev_report"], args.batch_size)
        )
        report = complete_report(
            truth(split_rows["dev_report"]), dev_probabilities, TARGETS, thresholds
        )
        report.update(
            {
                "schema_version": 1,
                "model_id": "wrc-inceptionresnetv2-v3-candidate",
                "config_id": args.config_id,
                "model_uri": args.model,
                "model_sha256": sha256(model_path),
                "threshold_tuning_count": len(split_rows["threshold_tune"]),
                "development_report_count": len(split_rows["dev_report"]),
                "split_counts": {name: len(rows) for name, rows in split_rows.items()},
                "split_uris": split_uris,
                "locked_test_evaluated": False,
                "production_approved": False,
                "approval_gate": "AWAITING_MODEL_COMPARISON_AND_HUMAN_REVIEW",
            }
        )

        metrics_path = output / "metrics.json"
        tune_path = output / "threshold_tune_predictions.jsonl"
        dev_path = output / "dev_report_predictions.jsonl"
        metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        write_predictions(tune_path, split_rows["threshold_tune"], tune_probabilities)
        write_predictions(dev_path, split_rows["dev_report"], dev_probabilities)

        prefix = args.output_prefix.rstrip("/")
        upload(client, metrics_path, f"{prefix}/metrics.json", "application/json")
        upload(client, tune_path, f"{prefix}/threshold_tune_predictions.jsonl", "application/x-ndjson")
        upload(client, dev_path, f"{prefix}/dev_report_predictions.jsonl", "application/x-ndjson")
        print(
            json.dumps(
                {
                    "evaluated": True,
                    "macro_f1": report["macro_f1"],
                    "macro_f2": report["macro_f2"],
                    "micro_f1": report["micro"]["f1"],
                    "output_prefix": prefix,
                    "production_approved": False,
                    "locked_test_evaluated": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
