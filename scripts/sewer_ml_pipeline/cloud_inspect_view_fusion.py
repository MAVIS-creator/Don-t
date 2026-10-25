"""Build and development-test a class-aware Inspect View fusion prototype.

All weights and thresholds are selected on threshold_tune. The selected fusion is
then evaluated once on dev_report. This script never reads the locked test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from google.cloud import storage

from cloud_legacy_evaluate import (
    download, infer, read_manifest, split_gs, stage_images, truth, upload_create_only,
    write_predictions,
)
from curate_staged import TARGETS, gcp_attestation, group_id, verify_disjoint
from metrics_v3 import complete_report, metrics
from select_250k_manifest import proxy_group


MULTILABEL_MODELS = (
    "efficientnet-b0", "convnext-tiny", "sewer-condition-v2",
    "sewer-condition-v1", "wrc-adapted",
)
ORIGINAL_WRC = "wrc-original"
GRID_UNITS = 4  # convex weights in 0.25 increments
MIN_BLEND_GAIN = 0.02
CV_FOLDS = 5
MIN_POSITIVES_FOR_BLEND = 20


def compositions(total: int, length: int, prefix: tuple[int, ...] = ()):
    if length == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from compositions(total - value, length - 1, prefix + (value,))


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    best = None
    for threshold in np.arange(0.05, 0.951, 0.05):
        result = metrics(y_true.astype(bool), probability >= threshold)
        candidate = (result["f1"], result["f2"], -abs(float(threshold) - 0.5), float(threshold), result)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], best[4]


def cross_validated_score(y_true: np.ndarray, probability: np.ndarray, folds: np.ndarray) -> dict:
    decisions = np.zeros(len(y_true), dtype=bool)
    fold_thresholds = []
    for fold in range(CV_FOLDS):
        held_out = folds == fold
        training = ~held_out
        threshold, _ = best_threshold(y_true[training], probability[training])
        decisions[held_out] = probability[held_out] >= threshold
        fold_thresholds.append(threshold)
    score = metrics(y_true.astype(bool), decisions)
    score["fold_thresholds"] = fold_thresholds
    return score


def select_class_fusion(y_true: np.ndarray, probabilities: dict[str, np.ndarray], folds: np.ndarray) -> dict:
    names = tuple(probabilities)
    best_single = None
    for name in names:
        cv_score = cross_validated_score(y_true, probabilities[name], folds)
        threshold, tuning_score = best_threshold(y_true, probabilities[name])
        candidate = (cv_score["f1"], cv_score["f2"], name, threshold, tuning_score, cv_score)
        if best_single is None or candidate[:2] > best_single[:2]:
            best_single = candidate

    best_blend = None
    if int(y_true.sum()) >= MIN_POSITIVES_FOR_BLEND:
        for units in compositions(GRID_UNITS, len(names)):
            nonzero = sum(value > 0 for value in units)
            if nonzero != 2:
                continue
            weights = np.asarray(units, dtype=np.float32) / GRID_UNITS
            blended = sum(weights[index] * probabilities[name] for index, name in enumerate(names))
            cv_score = cross_validated_score(y_true, blended, folds)
            threshold, tuning_score = best_threshold(y_true, blended)
            candidate = (cv_score["f1"], cv_score["f2"], tuple(units), threshold, tuning_score, cv_score)
            if best_blend is None or candidate[:2] > best_blend[:2]:
                best_blend = candidate

    if best_blend and best_blend[0] >= best_single[0] + MIN_BLEND_GAIN:
        units = best_blend[2]
        return {
            "weights": {name: units[index] / GRID_UNITS for index, name in enumerate(names) if units[index]},
            "threshold": best_blend[3], "tuning_metrics": best_blend[4],
            "cross_validated_metrics": best_blend[5],
            "selection": "cross_validated_blend", "minimum_required_gain": MIN_BLEND_GAIN,
        }
    return {
        "weights": {best_single[2]: 1.0}, "threshold": best_single[3],
        "tuning_metrics": best_single[4], "cross_validated_metrics": best_single[5],
        "selection": "cross_validated_single_model",
        "minimum_required_gain": MIN_BLEND_GAIN,
    }


def load_jsonl(client: storage.Client, uri: str, rows: list[dict]) -> np.ndarray:
    bucket, name = split_gs(uri)
    records = [json.loads(line) for line in client.bucket(bucket).blob(name).download_as_text().splitlines() if line]
    by_name = {record["Filename"]: record for record in records}
    if set(by_name) != {row["Filename"] for row in rows}:
        raise RuntimeError(f"Prediction/image identity mismatch: {uri}")
    ordered = []
    for row in rows:
        record = by_name[row["Filename"]]
        if record.get("class_order", record.get("class_codes")) != list(TARGETS):
            raise RuntimeError(f"Prediction class order mismatch: {uri}")
        ordered.append(record["probabilities"])
    values = np.asarray(ordered, dtype=np.float32)
    if values.shape != (len(rows), len(TARGETS)) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid prediction matrix: {uri}")
    return values


def onnx_probabilities(session, rows, images, spec):
    tune, _ = infer(session, rows["threshold_tune"], images, spec)
    dev, timing = infer(session, rows["dev_report"], images, spec)
    return tune, dev, timing


def fused_matrix(predictions: dict[str, np.ndarray], selections: dict[str, dict]) -> np.ndarray:
    output = np.zeros((next(iter(predictions.values())).shape[0], len(TARGETS)), dtype=np.float32)
    for class_index, class_code in enumerate(TARGETS):
        for name, weight in selections[class_code]["weights"].items():
            source_index = 6 if name == ORIGINAL_WRC else class_index
            output[:, class_index] += float(weight) * predictions[name][:, source_index]
    return output


def run(args) -> None:
    gcp_attestation(args.project)
    client = storage.Client(project=args.project)
    rows = {
        "train": read_manifest(client, args.train),
        "early_stop": read_manifest(client, args.early_stop),
        "threshold_tune": read_manifest(client, args.threshold_tune),
        "dev_report": read_manifest(client, args.dev_report),
    }
    verify_disjoint(rows)
    with tempfile.TemporaryDirectory(prefix="inspect-view-") as temporary:
        root = Path(temporary)
        images = root / "images"
        stage_images(client, rows["threshold_tune"] + rows["dev_report"], images)
        specs = [
            ("efficientnet-b0", args.efficientnet_model, args.efficientnet_sha, 224, "logits", None, None),
            ("convnext-tiny", args.convnext_model, args.convnext_sha, 224, "logits", None, None),
            ("sewer-condition-v2", args.v2_model, args.v2_sha, 224, "probability", args.v2_data, args.v2_data_sha),
            ("sewer-condition-v1", args.v1_model, args.v1_sha, 128, "probability", args.v1_data, args.v1_data_sha),
            (ORIGINAL_WRC, args.wrc_model, args.wrc_sha, 299, "probability", None, None),
        ]
        tune_predictions, dev_predictions, timings, model_hashes = {}, {}, {}, {}
        for name, uri, expected_sha, image_size, activation, data_uri, data_sha in specs:
            directory = root / "models" / name
            model_path = directory / "model.onnx"
            download(client, uri, model_path, expected_sha)
            if data_uri:
                download(client, data_uri, directory / "model.onnx.data", data_sha)
            kind = "wrc_single_softmax" if name == ORIGINAL_WRC else "sewerml_multilabel_sigmoid"
            spec = {
                "name": name, "kind": kind,
                "preprocess": "wrc_raw_rgb_nhwc" if name == ORIGINAL_WRC else "imagenet_rgb_nchw",
                "image_size": image_size,
                "class_codes": (["B", "CN", "CK", "CND", "D", "DE", "JD", "F", "H", "J", "LL", "R", "W"]
                                if name == ORIGINAL_WRC else list(TARGETS)),
                "output_activation": activation,
            }
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            tune_predictions[name], dev_predictions[name], timings[name] = onnx_probabilities(session, rows, images, spec)
            model_hashes[name] = expected_sha

        tune_predictions["wrc-adapted"] = load_jsonl(client, args.wrc_adapted_tune_predictions, rows["threshold_tune"])
        dev_predictions["wrc-adapted"] = load_jsonl(client, args.wrc_adapted_dev_predictions, rows["dev_report"])
        model_hashes["wrc-adapted"] = args.wrc_adapted_sha

        tune_truth = truth(rows["threshold_tune"])
        dev_truth = truth(rows["dev_report"])
        tuning_groups = [group_id(row) for row in rows["threshold_tune"]]
        if any(not group for group in tuning_groups):
            raise RuntimeError("Cross-validation requires provenance for every tuning image")
        if len(set(tuning_groups)) >= CV_FOLDS:
            cv_groups = tuning_groups
            cv_grouping_strategy = "verified_source_group"
        else:
            # The historical v4 validation manifests expose one conservative
            # `official_val_unsegmented` group. Keep nearby numeric frames together
            # rather than pretending individual images are independent groups.
            cv_groups = [proxy_group(row["Filename"]) for row in rows["threshold_tune"]]
            cv_grouping_strategy = "numeric_filename_block_proxy_100"
            if len(set(cv_groups)) < CV_FOLDS:
                raise RuntimeError("Cross-validation proxy grouping did not produce enough groups")
        ordered_groups = sorted(set(cv_groups), key=lambda group: hashlib.sha256(group.encode()).digest())
        group_folds = {group: index % CV_FOLDS for index, group in enumerate(ordered_groups)}
        folds = np.asarray([group_folds[group] for group in cv_groups])
        if set(folds.tolist()) != set(range(CV_FOLDS)):
            raise RuntimeError("Cross-validation fold assignment did not populate every fold")
        selections = {}
        for class_index, class_code in enumerate(TARGETS):
            candidates = {name: tune_predictions[name][:, class_index] for name in MULTILABEL_MODELS}
            if class_code == "FS":
                candidates[ORIGINAL_WRC] = tune_predictions[ORIGINAL_WRC][:, 6]
            selections[class_code] = select_class_fusion(tune_truth[:, class_index], candidates, folds)

        tune_fused = fused_matrix(tune_predictions, selections)
        dev_fused = fused_matrix(dev_predictions, selections)
        thresholds = [selections[code]["threshold"] for code in TARGETS]
        development_report = complete_report(dev_truth, dev_fused, TARGETS, thresholds)

        standalone_reports = {}
        for name in MULTILABEL_MODELS:
            standalone_thresholds = []
            for class_index in range(len(TARGETS)):
                threshold, _ = best_threshold(tune_truth[:, class_index], tune_predictions[name][:, class_index])
                standalone_thresholds.append(threshold)
            standalone_reports[name] = complete_report(
                dev_truth, dev_predictions[name], TARGETS, standalone_thresholds,
            )
        best_standalone_name, best_standalone = max(
            standalone_reports.items(), key=lambda item: (item[1]["macro_f1"], item[1]["macro_f2"]),
        )
        acceptance = {
            "accepted": (
                development_report["macro_f1"] > best_standalone["macro_f1"]
                and development_report["macro_f2"] > best_standalone["macro_f2"]
            ),
            "rule": "fusion must strictly exceed strongest standalone macro_f1 and macro_f2 on dev_report",
            "best_standalone_model": best_standalone_name,
            "best_standalone_macro_f1": best_standalone["macro_f1"],
            "best_standalone_macro_f2": best_standalone["macro_f2"],
        }

        selected_names = sorted({name for selection in selections.values() for name in selection["weights"]})
        ablations = {}
        for removed in selected_names:
            altered = json.loads(json.dumps(selections))
            valid = True
            for code in TARGETS:
                weights = altered[code]["weights"]
                if removed not in weights:
                    continue
                del weights[removed]
                total = sum(weights.values())
                if total == 0:
                    valid = False
                    break
                altered[code]["weights"] = {name: weight / total for name, weight in weights.items()}
            if valid:
                ablated = fused_matrix(dev_predictions, altered)
                report = complete_report(dev_truth, ablated, TARGETS, thresholds)
                ablations[removed] = {
                    "macro_f1": report["macro_f1"],
                    "delta_macro_f1_vs_full": report["macro_f1"] - development_report["macro_f1"],
                }
            else:
                ablations[removed] = {"not_removable": True, "reason": "sole provider for at least one class"}

        for split_name, matrix in (("threshold_tune", tune_fused), ("dev_report", dev_fused)):
            path = root / f"inspect_view_{split_name}_predictions.jsonl"
            write_predictions(path, rows[split_name], matrix, list(TARGETS))
            upload_create_only(client, path, f"{args.output_prefix.rstrip('/')}/{path.name}")
        report_path = root / "inspect_view_fusion_report.json"
        report_path.write_text(json.dumps({
            "schema_version": 1, "model_id": "inspect-view-development-prototype",
            "selection_split": "threshold_tune", "evaluation_split": "dev_report",
            "selection": selections, "development_report": development_report,
            "standalone_reports": standalone_reports, "acceptance_gate": acceptance,
            "ablations": ablations, "selected_models": selected_names,
            "model_sha256": model_hashes, "onnx_cpu_timings": timings,
            "cross_validation_grouping": cv_grouping_strategy,
            "threshold_tuning_count": len(rows["threshold_tune"]),
            "development_report_count": len(rows["dev_report"]),
            "locked_test_evaluated": False, "production_approved": False,
        }, indent=2))
        upload_create_only(client, report_path, f"{args.output_prefix.rstrip('/')}/{report_path.name}")
        print(json.dumps({
            "built": True, "selected_models": selected_names,
            "macro_f1": development_report["macro_f1"], "macro_f2": development_report["macro_f2"],
            "accepted": acceptance["accepted"],
            "output_prefix": args.output_prefix, "locked_test_evaluated": False,
        }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    for name in ("train", "early-stop", "threshold-tune", "dev-report", "output-prefix"):
        parser.add_argument("--" + name, required=True)
    for name in (
        "efficientnet-model", "efficientnet-sha", "convnext-model", "convnext-sha",
        "v1-model", "v1-sha", "v1-data", "v1-data-sha",
        "v2-model", "v2-sha", "v2-data", "v2-data-sha",
        "wrc-model", "wrc-sha", "wrc-adapted-tune-predictions",
        "wrc-adapted-dev-predictions", "wrc-adapted-sha",
    ):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
