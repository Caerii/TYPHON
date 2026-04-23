# Benchmark Packs

TYPHON now prefers manifest-backed local benchmark packs over ad hoc root-level sample files.

## Layout

Preferred layout:

- `data/benchmarks/<benchmark_id>/pack.json`
- `data/benchmarks/<benchmark_id>/packs/<pack_id>/samples.jsonl`

Legacy layout still supported:

- `data/benchmarks/<benchmark_id>/samples.jsonl`
- `data/benchmarks/<benchmark_id>/samples.json`

## Manifest

`pack.json` records which local pack files belong to one benchmark and which split each pack defaults to.

Current manifest shape:

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

## Sample Schema

Each normalized sample row should contain:

- `sample_id`
- `split`
- `task_type`
- `question`
- `context`
- `expected_answer_type`
- `reference_answer`
- `metadata`

`reference_answer` is optional, but you want it when running aggregate scoring or live memory compares.

## Importing

Example import for `memorybench`:

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

Example import for `evo_memory`:

```powershell
uv run typhon import-benchmark-pack `
  --benchmark evo_memory `
  --input data/imports/evo_memory_seed_v1.jsonl `
  --pack-id seed_v1 `
  --description "Seed local Evo-Memory pack" `
  --sample-id-field id `
  --task-type-field kind `
  --question-field prompt `
  --context-field history `
  --reference-answer-field gold `
  --metadata-field details
```

Example import for the official LongBench English smoke pack:

```powershell
uv run --extra adapter_hf typhon import-longbench `
  --config configs/benchmarks/adapters/longbench_english_smoke.json `
  --replace
```

That adapter currently targets the Hugging Face `THUDM/LongBench` dataset and writes a manifest-backed pack under `data/benchmarks/longbench/`.

## Validation

Validate one benchmark:

```powershell
uv run typhon validate-benchmark-pack --benchmark memorybench
```

Validate a family:

```powershell
uv run typhon validate-benchmark-pack --family continual_learning
```

Inspect source selection:

```powershell
uv run typhon inspect-benchmark-data --benchmark memorybench
```

## Notes

- Pack manifests are the preferred local path because they let one benchmark accumulate multiple curated slices without turning a single `samples.jsonl` file into a dumping ground.
- The current importer is schema-mapping based. It is meant for controlled local ingestion, not raw benchmark replication from arbitrary upstream formats.
- The LongBench adapter is the first dedicated upstream importer in the repo. It normalizes official Hugging Face rows into the same local pack format used by the curated benchmarks.
- The LongBench adapter currently depends on `datasets<4`, because the upstream dataset still uses a dataset script. Use the `adapter_hf` extra when running it through `uv`.
