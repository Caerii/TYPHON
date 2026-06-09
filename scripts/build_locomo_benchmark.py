"""Build a TYPHON LoCoMo benchmark pack from snap-research ``locomo10.json``.

Usage:
    python scripts/build_locomo_benchmark.py \
        --input /path/to/locomo10.json \
        --id locomo_real --conversations 1 --max-per-conversation 20

Writes ``data/benchmarks/<id>/samples.jsonl``. The raw dataset is fetched
separately (https://github.com/snap-research/locomo, ``data/locomo10.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.benchmarks.locomo_importer import build_locomo_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a LoCoMo benchmark pack.")
    parser.add_argument("--input", required=True, help="Path to locomo10.json")
    parser.add_argument("--id", default="locomo_real", help="Benchmark id (output dir name)")
    parser.add_argument("--conversations", type=int, default=None, help="Cap number of conversations")
    parser.add_argument("--max-per-conversation", type=int, default=None, help="Cap QA per conversation")
    parser.add_argument("--no-adversarial", action="store_true", help="Drop category-5 adversarial QA")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = build_locomo_samples(
        raw,
        conversations=args.conversations,
        max_per_conversation=args.max_per_conversation,
        include_adversarial=not args.no_adversarial,
    )

    out_dir = Path(args.repo_root) / "data" / "benchmarks" / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    categories: dict[str, int] = {}
    for sample in samples:
        key = str(sample["metadata"].get("category"))
        categories[key] = categories.get(key, 0) + 1
    print(f"wrote {len(samples)} samples -> {out_path}")
    print(f"by category: {categories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
