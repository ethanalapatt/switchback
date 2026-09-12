"""Workload selection must be reproducible and independent of row content."""

from __future__ import annotations

import hashlib

import pytest
from bench.prepare import DATASETS, select_rows, workload_document
from bench.validate import expected_keys_for
from data.synthetic_generator import TARGET_LENGTHS, generate


def rows(count: int) -> list[dict[str, object]]:
    return [{"task_id": 100 + index, "text": f"problem {index}"} for index in range(count)]


def test_selection_is_deterministic() -> None:
    first = select_rows(rows(50), "abc", "train", "task_id", 8)
    second = select_rows(rows(50), "abc", "train", "task_id", 8)
    assert [row_id for row_id, _ in first] == [row_id for row_id, _ in second]


def test_selection_follows_the_documented_hash_order() -> None:
    """Recomputable from the manifest alone, by hand if necessary."""
    population = rows(20)
    chosen = [row_id for row_id, _ in select_rows(population, "rev", "test", "task_id", 5)]
    expected = sorted(
        (str(row["task_id"]) for row in population),
        key=lambda row_id: (
            hashlib.sha256(f"rev{'test'}{row_id}".encode()).hexdigest(),
            row_id,
        ),
    )[:5]
    assert chosen == expected


def test_a_different_revision_selects_a_different_subset() -> None:
    left = {row_id for row_id, _ in select_rows(rows(200), "rev-a", "train", "task_id", 16)}
    right = {row_id for row_id, _ in select_rows(rows(200), "rev-b", "train", "task_id", 16)}
    assert left != right


def test_splits_select_independently() -> None:
    train = {row_id for row_id, _ in select_rows(rows(200), "rev", "train", "task_id", 16)}
    test = {row_id for row_id, _ in select_rows(rows(200), "rev", "test", "task_id", 16)}
    assert train != test


def test_selection_ignores_row_content() -> None:
    """Choosing prompts by their text would be tuning on the workload."""
    plain = select_rows(rows(40), "rev", "train", "task_id", 6)
    shouted = select_rows(
        [{"task_id": row["task_id"], "text": str(row["text"]).upper() * 20} for row in rows(40)],
        "rev",
        "train",
        "task_id",
        6,
    )
    assert [row_id for row_id, _ in plain] == [row_id for row_id, _ in shouted]


def test_source_order_is_the_identity_when_there_is_no_row_id() -> None:
    chosen = select_rows(rows(30), "rev", "train", None, 5)
    assert all(row_id.isdigit() and int(row_id) < 30 for row_id, _ in chosen)


def test_asking_for_more_rows_than_exist_fails_loudly() -> None:
    with pytest.raises(SystemExit, match="only 10"):
        select_rows(rows(10), "rev", "train", "task_id", 11)


def test_dataset_revisions_are_pinned_commits() -> None:
    for name, entry in DATASETS.items():
        assert len(entry["revision"]) == 40, name
        assert entry["calibration_split"] != entry["heldout_split"], name


# --- workload document -----------------------------------------------------


def test_the_workload_hash_covers_the_prompts(tmp_path) -> None:
    from bench.prepare import WorkloadPrompt

    def prompt(index: int) -> WorkloadPrompt:
        return WorkloadPrompt(
            prompt_id=f"p{index}",
            dataset="mbpp",
            split="test",
            role="heldout",
            source_row_id=str(index),
            prompt_tokens=40,
            token_sha256="a" * 64,
        )

    left = workload_document("c", "b" * 64, [prompt(0), prompt(1)], {})
    right = workload_document("c", "b" * 64, [prompt(0), prompt(2)], {})
    assert left["workload_sha256"] != right["workload_sha256"]
    again = workload_document("c", "b" * 64, [prompt(0), prompt(1)], {})
    assert again["workload_sha256"] == left["workload_sha256"]


def test_the_workload_stores_hashes_not_prompt_text() -> None:
    from bench.prepare import WorkloadPrompt

    document = workload_document(
        "c",
        "b" * 64,
        [WorkloadPrompt("p0", "mbpp", "test", "heldout", "1", 40, "a" * 64)],
        {},
    )
    entry = document["prompts"][0]
    assert set(entry) == {
        "prompt_id",
        "dataset",
        "split",
        "role",
        "source_row_id",
        "prompt_tokens",
        "token_sha256",
        "target_tokens",
    }
    assert "text" not in entry


# --- expected coverage -----------------------------------------------------


def config_for(engines: list[str], repeats: int, seeds: list[int]) -> dict:
    return {
        "cohort": "smoke",
        "condition": {
            "name": "fixed_length_32",
            "mode": "greedy",
            "seeds": seeds,
            "repeats": repeats,
        },
        "run": {"engines": engines},
    }


def workload_for(count: int) -> dict:
    return {
        "prompts": [
            {"prompt_id": f"p{index}", "dataset": "mbpp", "role": "heldout"}
            for index in range(count)
        ]
        + [{"prompt_id": "cal0", "dataset": "mbpp", "role": "calibration"}]
    }


def test_expected_coverage_is_the_full_cross_product() -> None:
    keys = expected_keys_for(config_for(["hf_ar", "adaptive"], 3, [42]), workload_for(4))
    assert len(keys) == 2 * 3 * 4
    assert len({tuple(key) for key in keys}) == len(keys)


def test_calibration_prompts_are_never_evaluated() -> None:
    """The controller's calibration split must not appear in the cohort."""
    keys = expected_keys_for(config_for(["hf_ar"], 1, [42]), workload_for(2))
    assert all("cal0" not in key for key in keys)
    assert len(keys) == 2


# --- controlled prompts ----------------------------------------------------


def test_controlled_prompts_are_deterministic() -> None:
    assert [p.text for p in generate(17, 8)] == [p.text for p in generate(17, 8)]
    assert [p.text for p in generate(17, 8)] != [p.text for p in generate(29, 8)]


def test_controlled_prompts_span_the_length_buckets() -> None:
    prompts = generate(17, len(TARGET_LENGTHS) * 2)
    assert {p.target_tokens for p in prompts} == set(TARGET_LENGTHS)
    # Longer buckets really do produce longer text.
    by_bucket = {p.target_tokens: len(p.text) for p in prompts}
    assert by_bucket[TARGET_LENGTHS[-1]] > by_bucket[TARGET_LENGTHS[0]] * 4


def test_a_target_length_is_a_label_not_a_measurement() -> None:
    """SPEC.md section 9.1: do not claim requested lengths equal measured ones."""
    prompt = generate(17, 1)[0]
    assert prompt.target_tokens in TARGET_LENGTHS
    assert "target_tokens" in prompt.__dataclass_fields__


def test_no_controlled_prompts_when_none_are_asked_for() -> None:
    assert generate(17, 0) == []
