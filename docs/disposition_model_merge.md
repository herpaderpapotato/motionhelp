# Disposition Model Merge Tool

The compare and merge tool lives at `scripts/disposition_model_merge.py`.

## GUI

Launch the interactive UI:

```powershell
python scripts/disposition_model_merge.py gui \
  --model-a data/models/checkpoints_disposition/best_disposition.pt \
  --model-b data/models/checkpoints_disposition/best_disposition_ddl_1e5.pt
```

The GUI keeps both checkpoints loaded, shows component-level weight statistics and deltas, and lets you select the whole model or individual compatible modules before merging.

## CLI Compare

Write comparison artifacts without producing a merged checkpoint:

```powershell
python scripts/disposition_model_merge.py compare \
  --model-a data/models/checkpoints_disposition/best_disposition.pt \
  --model-b data/models/checkpoints_disposition/best_disposition_ddl_1e5.pt \
  --output-root tmp/disposition_model_merges \
  --run-name compare_example
```

This creates CSV and JSON summaries plus a top-component difference plot under the run folder.

## CLI Merge

Merge the full compatible model and automatically benchmark model A, model B, and model C on the same 10 validation sequences:

```powershell
python scripts/disposition_model_merge.py merge \
  --model-a data/models/checkpoints_disposition/best_disposition.pt \
  --model-b data/models/checkpoints_disposition/best_disposition_ddl_1e5.pt \
  --weight-a 0.5 \
  --weight-b 0.5
```

Merge only selected compatible components:

```powershell
python scripts/disposition_model_merge.py merge \
  --model-a data/models/checkpoints_disposition/best_disposition.pt \
  --model-b data/models/checkpoints_disposition/best_disposition_1e5.pt \
  --include spatial_encoder proj output_head person_attn \
  --weight-a 0.6 \
  --weight-b 0.4
```

## Output Layout

Each merge run is written under `tmp/disposition_model_merges` by default:

```text
tmp/disposition_model_merges/<run>/
  comparison/
  benchmark_suite/
  model_a/
    checkpoint/
    results/
  model_b/
    checkpoint/
    results/
  model_c/
    checkpoint/
    results/
  manifest.json
```

If the root models are incompatible, full-model merge is rejected with a specific compatibility message. In that case, pick compatible subcomponents instead.