# Benchmark Packs

TYPHON uses manifest-backed benchmark packs so local benchmark data stays traceable and does not degrade into one-off sample files.

## Preferred Layout

- `data/benchmarks/<benchmark_id>/pack.json`
- `data/benchmarks/<benchmark_id>/packs/<pack_id>/samples.jsonl`

Legacy layouts are still read for compatibility:

- `data/benchmarks/<benchmark_id>/samples.jsonl`
- `data/benchmarks/<benchmark_id>/samples.json`

## Manifest Shape

```json
{
  "benchmark_id": "memorybench",
  "format_version": 1,
  "preferred_source": "local_pack",
  "packs": [
    {
      "pack_id": "seed_v1",
      "path": "packs/seed_v1/samples.jsonl",
      "format": "jsonl",
      "sample_count": 2,
      "default_split": "local",
      "description": "Seed local MemoryBench pack",
      "source_label": "data\\imports\\memorybench_seed_v1.jsonl",
      "metadata": {
        "importer": "typhon.import_benchmark_pack"
      }
    }
  ]
}
```

## Normalized Sample Fields

Each normalized row should contain:

- `sample_id`
- `split`
- `task_type`
- `question`
- `context`
- `expected_answer_type`
- `reference_answer` or `reference_answers`
- `metadata`

## Import Commands

Generic importer example:

```powershell
uv run typhon import-benchmark-pack `
  --benchmark memorybench `
  --input data/imports/memorybench_seed_v1.jsonl `
  --pack-id seed_v1 `
  --description "Seed local MemoryBench pack" `
  --sample-id-field id `
  --task-type-field kind `
  --question-field prompt `
  --context-field history `
  --reference-answer-field gold `
  --metadata-field details
```

LongBench adapter example:

```powershell
uv run --extra adapter_hf typhon import-longbench `
  --config configs/benchmarks/adapters/longbench_english_smoke.json `
  --replace
```

## Validation and Inspection

Validate one benchmark:

```powershell
uv run typhon validate-benchmark-pack --benchmark memorybench
```

Inspect the selected source path:

```powershell
uv run typhon inspect-benchmark-data --benchmark longbench
```

Run a smoke slice:

```powershell
uv run typhon smoke-test --benchmark longbench --sample-source local --sample-limit 10
```

## Governance

- Every committed local pack should have clear provenance in `pack.json`.
- Upstream adapters should normalize into the same pack structure instead of inventing benchmark-specific local layouts.
- Do not hand-append unrelated ad hoc rows to an existing pack. Create a new `pack_id` when provenance changes.
- Use `data/imports/` for raw import sources and `data/benchmarks/` for normalized local assets.
