"""Flatten PatchEval's verified CVE metadata JSON into an analysis-friendly CSV.

The export preserves list- and object-valued metadata as JSON strings. It does
not fabricate the patch-level BPD features or reviewer labels required to train
FuzzyGen-BPD; those require candidate generation and benchmark evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/PatchEval/patcheval/datasets/patcheval_verified.json"
DEFAULT_OUTPUT = ROOT / "data/patcheval_verified.csv"

FIELDS = [
    "cve_id",
    "cve_description",
    "cwe_ids",
    "cwe_names",
    "cwe_descriptions",
    "repo",
    "patch_urls",
    "programming_language",
    "vul_func_ids",
    "vul_func_files",
    "vul_func_snippets",
    "fix_func_ids",
    "fix_func_files",
    "fix_func_snippets",
    "image_url",
]


def json_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def flatten_case(case: dict) -> dict[str, str]:
    cwe = case.get("cwe_info") or {}
    vul = case.get("vul_func") or []
    fix = case.get("fix_func") or []
    return {
        "cve_id": json_cell(case.get("cve_id")),
        "cve_description": json_cell(case.get("cve_description")),
        "cwe_ids": json_cell(list(cwe.keys())),
        "cwe_names": json_cell([item.get("name", "") for item in cwe.values()]),
        "cwe_descriptions": json_cell([item.get("description", "") for item in cwe.values()]),
        "repo": json_cell(case.get("repo")),
        "patch_urls": json_cell(case.get("patch_url")),
        "programming_language": json_cell(case.get("programming_language")),
        "vul_func_ids": json_cell([item.get("id", "") for item in vul]),
        "vul_func_files": json_cell([item.get("file_path", "") for item in vul]),
        "vul_func_snippets": json_cell([item.get("snippet", "") for item in vul]),
        "fix_func_ids": json_cell([item.get("id", "") for item in fix]),
        "fix_func_files": json_cell([item.get("file_path", "") for item in fix]),
        "fix_func_snippets": json_cell([item.get("snippet", "") for item in fix]),
        "image_url": json_cell(case.get("image_url")),
    }


def convert(source: Path, destination: Path) -> int:
    with source.open(encoding="utf-8") as stream:
        cases = json.load(stream)
    if not isinstance(cases, list):
        raise ValueError("Expected PatchEval dataset JSON to contain a list of cases.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flatten_case(case) for case in cases)
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(f"Wrote {count} CVE records to {args.output}")


if __name__ == "__main__":
    main()
