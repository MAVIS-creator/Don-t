"""Deterministic metadata-only selection for a 250K Sewer-ML research corpus.

This stage never downloads images. It produces a preliminary selection manifest
and shard plan; image hashes and near-duplicate checks remain mandatory before the
manifest can be promoted to training-ready status.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

TARGETS = ("FS", "IS", "RB", "DE", "RO", "AF", "BE", "FO", "ND")
SOURCE_DEFECTS = ("VA", "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO",
                  "GR", "PH", "PB", "OS", "OP", "OK")

TARGET_SIZE = 250_000
DEFAULT_SEED = 20261025
DEFAULT_FRAME_BLOCK = 100
DEFAULT_GROUP_CAP = 30
DEFAULT_SHARD_ROWS = 512


def labels(row: dict) -> list[str]:
    return [code for code in TARGETS if row.get(code) in ("1", "1.0", 1, 1.0)]


def normalize_target_row(row: dict) -> dict | None:
    normalized = dict(row)
    active_source = [code for code in SOURCE_DEFECTS if row.get(code) in ("1", "1.0", 1, 1.0)]
    active_targets = [code for code in TARGETS[:-1] if row.get(code) in ("1", "1.0", 1, 1.0)]
    explicit_nd = row.get("ND") in ("1", "1.0", 1, 1.0)
    if explicit_nd and active_source:
        return None
    if active_targets:
        normalized["ND"] = "0"
    elif active_source:
        return None
    elif explicit_nd or row.get("Defect") in ("0", "0.0", 0, 0.0):
        normalized["ND"] = "1"
    else:
        return None
    for code in TARGETS:
        normalized[code] = "1" if normalized.get(code) in ("1", "1.0", 1, 1.0) else "0"
    return normalized


def stable_unit(seed: int, name: str) -> float:
    number = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
    return (number + 1) / (2**64 + 1)


def numeric_frame(filename: str) -> int | None:
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else None


def proxy_group(filename: str, frame_block: int = DEFAULT_FRAME_BLOCK) -> str:
    frame = numeric_frame(filename)
    if frame is None:
        digest = hashlib.sha256(filename.casefold().encode()).hexdigest()[:12]
        return f"filename-hash-block:{digest}"
    return f"numeric-frame-block-{frame_block}:{frame // frame_block:08d}"


def parse_csv(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def select_rows(
    rows: list[dict[str, str]], target: int = TARGET_SIZE, *,
    excluded_names: set[str] | None = None, seed: int = DEFAULT_SEED,
    frame_block: int = DEFAULT_FRAME_BLOCK, group_cap: int = DEFAULT_GROUP_CAP,
    candidate_reserve: int = 0,
) -> tuple[list[dict], dict]:
    if target < 1:
        raise ValueError("Target must be positive")
    if candidate_reserve < 0:
        raise ValueError("Candidate reserve cannot be negative")
    selection_target = target + candidate_reserve
    excluded = {name.casefold() for name in (excluded_names or set())}
    unique: dict[str, dict] = {}
    rejected = Counter()
    for source in rows:
        filename = str(source.get("Filename", "")).strip()
        if not filename or Path(filename).name != filename:
            rejected["unsafe_filename"] += 1
            continue
        key = filename.casefold()
        if key in excluded:
            rejected["reserved_or_prior_split"] += 1
            continue
        if key in unique:
            rejected["duplicate_filename"] += 1
            continue
        row = normalize_target_row(source)
        if row is None:
            rejected["outside_target_contract"] += 1
            continue
        row = dict(row)
        row["source_group"] = proxy_group(filename, frame_block)
        unique[key] = row
    candidates = list(unique.values())
    if len(candidates) < selection_target:
        raise ValueError(f"Only {len(candidates)} eligible unique rows for target {selection_target}")

    counts = Counter(code for row in candidates for code in labels(row))
    rarity = {code: (len(candidates) / max(1, counts[code])) ** 0.55 for code in TARGETS}
    ranked = []
    for row in candidates:
        active = labels(row)
        weight = max((rarity[code] for code in active), default=1.0)
        difficulty_bonus = 1.0 + 0.12 * max(0, len(active) - 1)
        score = -math.log(stable_unit(seed, row["Filename"])) / (weight * difficulty_bonus)
        ranked.append((score, row["Filename"], row))
    ranked.sort(key=lambda value: (value[0], value[1]))

    selected = []
    groups = Counter()
    for score, _, row in ranked:
        group = row["source_group"]
        if groups[group] >= group_cap:
            continue
        selected.append({**row, "selection_score": f"{score:.12f}"})
        groups[group] += 1
        if len(selected) == selection_target:
            break
    if len(selected) != selection_target:
        raise ValueError(f"Proxy-group cap selected only {len(selected)} of {selection_target}")

    selected_counts = Counter(code for row in selected for code in labels(row))
    report = {
        "schema_version": 1,
        "status": "METADATA_SELECTED_IMAGE_INTEGRITY_PENDING",
        "target": target,
        "candidate_reserve": candidate_reserve,
        "selected_candidates": len(selected),
        "source_rows": len(rows),
        "eligible_unique_rows": len(candidates),
        "excluded_or_rejected": dict(rejected),
        "selection_seed": seed,
        "group_strategy": "conservative numeric filename block proxy; official annotations expose no verified sequence ID",
        "frame_block": frame_block,
        "group_cap": group_cap,
        "selected_proxy_groups": len(groups),
        "source_class_counts": dict(counts),
        "selected_class_counts": dict(selected_counts),
        "dataset_access_status": "USER_REPORTED_GRANTED",
        "dataset_access_basis": "project_owner_attestation_of_prior_dataset-contact_approval",
        "authorization_evidence_status": "NOT_YET_ATTACHED",
        "acceptable_authorization_evidence": [
            "verifiable email or correspondence",
            "verifiable dataset access-process record",
            "applicable licence or permission document",
        ],
        "commercial_training_rights": "UNVERIFIED",
        "production_use_rights": "UNVERIFIED",
        "training_ready": False,
        "remaining_gates": [
            "attach verifiable authorization evidence and determine its granted uses",
            "document commercial-training rights if commercial training is intended",
            "document production-use rights before production approval or deployment",
            "resolve every filename to an authorized source archive member",
            "verify decompressed image SHA-256",
            "reject exact and perceptual near-duplicates",
            "verify no image/group overlap with validation or locked test",
            "publish immutable image shards and shard SHA-256 inventory",
        ],
        "production_approved": False,
    }
    return selected, report


def manifest_csv(rows: list[dict]) -> str:
    fields = ["Filename", "source_group", *TARGETS, "selection_score", "shard_id"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow({**row, "shard_id": f"shard-{index // DEFAULT_SHARD_ROWS:05d}"})
    return output.getvalue()


def shard_plan(rows: list[dict], shard_rows: int = DEFAULT_SHARD_ROWS) -> dict:
    shards = []
    for start in range(0, len(rows), shard_rows):
        chunk = rows[start:start + shard_rows]
        shards.append({
            "shard_id": f"shard-{start // shard_rows:05d}",
            "row_start": start,
            "row_end_exclusive": start + len(chunk),
            "rows": len(chunk),
            "first_filename": chunk[0]["Filename"],
            "last_filename": chunk[-1]["Filename"],
        })
    return {"schema_version": 1, "format": "tar+index (planned, not materialized)",
            "target_rows_per_shard": shard_rows, "shard_count": len(shards), "shards": shards}
