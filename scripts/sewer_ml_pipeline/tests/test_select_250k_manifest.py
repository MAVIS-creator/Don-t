from select_250k_manifest import manifest_csv, proxy_group, select_rows, shard_plan


def row(index, **labels):
    value = {"Filename": f"{index:08d}.png", "Defect": "0", "ND": "1"}
    value.update({name: "1" if enabled else "0" for name, enabled in labels.items()})
    if any(labels.values()):
        value["Defect"] = "1"
        value["ND"] = "0"
    return value


def test_selection_is_deterministic_excludes_reserved_and_caps_proxy_groups():
    rows = [row(i, FS=i % 7 == 0, FO=i % 31 == 0) for i in range(1, 1001)]
    selected_a, report_a = select_rows(rows, 120, excluded_names={"00000007.png"},
                                       frame_block=20, group_cap=4)
    selected_b, report_b = select_rows(rows, 120, excluded_names={"00000007.png"},
                                       frame_block=20, group_cap=4)
    assert [item["Filename"] for item in selected_a] == [item["Filename"] for item in selected_b]
    assert "00000007.png" not in {item["Filename"] for item in selected_a}
    counts = {}
    for item in selected_a:
        counts[item["source_group"]] = counts.get(item["source_group"], 0) + 1
    assert max(counts.values()) <= 4
    assert report_a == report_b
    assert report_a["training_ready"] is False


def test_manifest_and_shard_plan_cover_every_selected_row():
    selected, _ = select_rows([row(i) for i in range(1, 81)], 40, frame_block=10, group_cap=6)
    csv_text = manifest_csv(selected)
    plan = shard_plan(selected, shard_rows=16)
    assert csv_text.count("\n") == 41
    assert plan["shard_count"] == 3
    assert sum(item["rows"] for item in plan["shards"]) == 40


def test_proxy_group_is_stable_and_explicitly_a_proxy():
    assert proxy_group("00000199.png", 100) == "numeric-frame-block-100:00000001"
    assert proxy_group("odd-name.png", 100).startswith("filename-hash-block:")


def test_candidate_reserve_provides_dedup_replacements():
    selected, report = select_rows([row(i) for i in range(1, 301)], 100,
                                   candidate_reserve=25, frame_block=10, group_cap=6)
    assert len(selected) == 125
    assert report["target"] == 100
    assert report["candidate_reserve"] == 25
    assert report["selected_candidates"] == 125
