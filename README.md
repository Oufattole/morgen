# MoRGen Reproduction Guide

This repository is the primary code release for **MoRGen**. It contains the model
training code, MEDS preprocessing entry points, trajectory generation code, compute
benchmark launchers, and the scripts used to materialize saved evaluation
artifacts for downstream scoring.

The Python package and command-line entry points are named `morgen`. The commands
below use the released CLI names, such as `morgen_process_data` and
`morgen_pretrain`.

## Installation

Use Python 3.12.11.

```bash
git clone https://github.com/Oufattole/morgen.git
cd morgen

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Histogram-aware MEDS Torch data utilities.
python -m pip install git+https://github.com/Oufattole/meds-histogram-dataset.git

# Install this repository.
python -m pip install -e .
```

For development or experiment logging, install the optional extras:

```bash
python -m pip install -e ".[wandb,dev]"
```

Some training runs can use FlashAttention when it is available:

```bash
python -m pip install flash-attn --no-build-isolation
```

Exact paper evaluation also needs the companion evaluation repository:

```bash
git clone https://github.com/Oufattole/morgen-eval.git ../morgen-eval
python -m pip install -e ../morgen-eval
```

## Repository Boundary

The public release is split across three repositories.

`morgen` is this repository. It owns MEDS preprocessing, baseline autoregressive
training, histogram/coarse model training, trajectory generation, compute
benchmark launchers, and evaluation-materialization helpers.

`morgen-eval` is the evaluation companion. It owns conversion from generated
trajectories to time-to-event caches, AUROC, Brier/BCE-style score artifacts,
log-loss scoring, jackknife uncertainty, MoRGen convex-fusion fitting, and
paper table and figure generation.

`ethos-eic` is an optional ETHOS comparator repository. It should own conversion
from MEDS-EIC data to ETHOS-compatible inputs, ETHOS model training, and ETHOS
trajectory generation. Core MoRGen reproduction does not require retraining
ETHOS; ETHOS reproduction can begin from stored ETHOS trajectory artifacts.

The intended paper flow is:

1. Use `morgen` to preprocess data, train/generate baseline and histogram
   trajectories, and run compute benchmarks.
2. Use `morgen-eval` to cache TTEs and compute score parquet files.
3. Use `morgen-eval` to turn saved score artifacts into tables and figures.
4. Use `ethos-eic` only for the extended ETHOS comparator path.

## Data Layout

The commands below assume each dataset has an experiment root:

```bash
BASE_DIR=/path/to/dataset/eic
RAW_MEDS_DIR=/path/to/raw/MEDS
INTERMEDIATE_DIR=${BASE_DIR}/preprocessing
FINAL_DATA_DIR=${BASE_DIR}/final
TASK_ROOT_DIR=${BASE_DIR}/tasks
```

Expected subdirectories after preprocessing and evaluation:

```text
${BASE_DIR}/preprocessing/data/       # MEDS-transforms output
${BASE_DIR}/final/                    # tensorized data for meds-torch-data
${BASE_DIR}/tasks/...                 # task labels / prediction times
${BASE_DIR}/generated_trajectories/   # minute-level generated trajectories
${BASE_DIR}/em_generated_trajectories # histogram/coarse generated trajectories
${BASE_DIR}/ttes/{tuning,held_out}/   # TTE caches written by morgen-eval
${BASE_DIR}/ground_truth/             # held-out/tuning label horizons
${BASE_DIR}/scores/{task}/{horizon}.parquet
```

The default paper tasks are:

```bash
MIMIC_TASKS=leukocyte,death,hematocrit,readmission,timeline_end,platelet,creatinine,hemoglobin
EHRSHOT_TASKS=leukocyte,death,hematocrit,admission,timeline_end,platelet,creatinine,hemoglobin
HORIZONS=30,60,90,182,365,730
```

## Core Reproduction

### 1. Preprocess MEDS Data

`morgen_process_data` runs the MEDS-transforms pipeline and then tensorizes
the result for `meds-torch-data`.

```bash
morgen_process_data \
  input_dir="${RAW_MEDS_DIR}" \
  intermediate_dir="${INTERMEDIATE_DIR}" \
  output_dir="${FINAL_DATA_DIR}" \
  pipeline=mimic_filtered
```

For EHRSHOT-style filtered data:

```bash
morgen_process_data \
  input_dir="${RAW_MEDS_DIR}" \
  intermediate_dir="${INTERMEDIATE_DIR}" \
  output_dir="${FINAL_DATA_DIR}" \
  pipeline=ehrshot_filtered
```

Task labels are generated outside this package with the relevant MEDS task
configuration. The trajectory generation commands expect a task-label directory
such as:

```text
${TASK_ROOT_DIR}/mimiciv/hospital_discharge/timeline_end/
```

or, for EHRSHOT:

```text
/path/to/ehrshot/meds/labels/ehrshot/timeline_end/
```

### 2. Train the Minute-Level Baseline

```bash
PRETRAINED_MODEL_DIR=${BASE_DIR}/results/micro/pretrained_lr_5e-4

morgen_pretrain \
  trainer/logger=csv \
  lightning_module=micro \
  datamodule.config.tensorized_cohort_dir="${FINAL_DATA_DIR}" \
  output_dir="${PRETRAINED_MODEL_DIR}" \
  datamodule.batch_size=128 \
  trainer.max_epochs=100 \
  lightning_module.optimizer.lr=5e-4
```

The paper launch wrapper for baseline training is:

```bash
PYENV_VERSION=morgen \
bash scripts/launch_baseline_train_only.sh --gpu 0 --datasets mimic,ehrshot
```

### 3. Generate Minute-Level Trajectories

```bash
GENERATED_TRAJECTORIES_DIR=${BASE_DIR}/generated_trajectories/micro/hospital_discharge
TASK_LABELS_DIR=${TASK_ROOT_DIR}/mimiciv/hospital_discharge/timeline_end

morgen_generate_trajectories \
  inference=test \
  inference.generate_for_splits=[tuning,held_out] \
  inference.N_trajectories_per_task_sample=50 \
  datamodule.config.tensorized_cohort_dir="${FINAL_DATA_DIR}" \
  datamodule.config.task_labels_dir="${TASK_LABELS_DIR}" \
  model_initialization_dir="${PRETRAINED_MODEL_DIR}" \
  output_dir="${GENERATED_TRAJECTORIES_DIR}" \
  datamodule.batch_size=512 \
  datamodule.config.max_seq_len=512 \
  seq_lens.frac_seq_len_as_context=0.25
```

`seq_lens.frac_seq_len_as_context=0.25` uses 25% of the pretrained context for
conditioning and the remaining context for generation. Alternatively, set
`seq_lens.frac_seq_len_as_context=null` and specify either
`seq_lens.generation_context_size` or `seq_lens.max_generated_trajectory_len`.

### 4. Train Histogram / Coarse Models

Histogram models first learn a vocabulary over fixed-width patient-history
windows, then train an autoregressive model over those window tokens.

Fit an analytic EM vocabulary:

```bash
morgen_fit_em \
  datamodule.config.tensorized_cohort_dir="${FINAL_DATA_DIR}" \
  output_dir="${BASE_DIR}/em_params/K128_month_level" \
  window_size_days=30 \
  quantizer_config.n_embeddings=128 \
  gap_config.max_gap_length=256
```

Train the histogram autoregressive model:

```bash
AR_OUTPUT_DIR=${BASE_DIR}/results/K128_month_level
EM_PARAMS_DIR=${BASE_DIR}/em_params/K128_month_level

morgen_train_autoregressive_histogram \
  output_dir="${AR_OUTPUT_DIR}" \
  em_params_dir="${EM_PARAMS_DIR}" \
  datamodule.config.tensorized_cohort_dir="${FINAL_DATA_DIR}" \
  quantizer_config.n_embeddings=128 \
  window_size_days=30 \
  model_config.max_position_embeddings=512 \
  trainer.max_epochs=100 \
  datamodule.batch_size=256
```

Generate histogram trajectories:

```bash
HISTOGRAM_TRAJECTORIES_DIR=${BASE_DIR}/em_generated_trajectories/K128_month_level

morgen_generate_histogram_trajectories \
  inference=test \
  inference.generate_for_splits=[tuning,held_out] \
  inference.N_trajectories_per_task_sample=50 \
  datamodule.config.tensorized_cohort_dir="${FINAL_DATA_DIR}" \
  datamodule.config.task_labels_dir="${TASK_LABELS_DIR}" \
  model_initialization_dir="${AR_OUTPUT_DIR}" \
  em_params_dir="${EM_PARAMS_DIR}" \
  output_dir="${HISTOGRAM_TRAJECTORIES_DIR}" \
  window_size_days=30 \
  max_new_tokens=26 \
  datamodule.batch_size=512
```

The main launchers for histogram/coarse model reproduction are:

```bash
PYENV_VERSION=morgen bash scripts/mimic_fit_em.sh
PYENV_VERSION=morgen bash scripts/ehrshot_fit_em.sh
PYENV_VERSION=morgen bash scripts/mimic_gpu_0.sh
PYENV_VERSION=morgen bash scripts/mimic_gpu_1.sh
```

### 5. Convert Trajectories to TTE Caches

Run this from `morgen-eval`.

```bash
cd ../morgen-eval

for SPLIT in tuning held_out; do
  python scripts/cache_ttes.py \
    dataset=mimic \
    split="${SPLIT}" \
    input_dir="${BASE_DIR}/generated_trajectories/micro/hospital_discharge/" \
    output_dir="${BASE_DIR}/ttes/" \
    index_dir="${TASK_LABELS_DIR}" \
    output_file_name=minute_level

  python scripts/cache_ttes.py --multirun \
    hydra/launcher=joblib \
    dataset=mimic \
    split="${SPLIT}" \
    input_dir="$(echo ${BASE_DIR}/em_generated_trajectories/K*/ | tr ' ' ',')" \
    output_dir="${BASE_DIR}/ttes/" \
    index_dir="${TASK_LABELS_DIR}"
done
```

TTE cache files are written to:

```text
${BASE_DIR}/ttes/tuning/{model_name}.parquet
${BASE_DIR}/ttes/held_out/{model_name}.parquet
```

Each model parquet stores per-patient event-time lists for each task.

### 6. Compute AUROC Artifacts and MoRGen Fusion

Run from `morgen-eval`.

```bash
python scripts/compute_scores.py --multirun \
  hydra/launcher=joblib \
  dataset=mimic \
  dataset.base_dir="${BASE_DIR}" \
  horizon="${HORIZONS}" \
  task="${MIMIC_TASKS}"
```

For EHRSHOT:

```bash
python scripts/compute_scores.py --multirun \
  hydra/launcher=joblib \
  dataset=ehrshot \
  dataset.base_dir="${BASE_DIR}" \
  horizon="${HORIZONS}" \
  task="${EHRSHOT_TASKS}"
```

`compute_scores.py` reads:

```text
${BASE_DIR}/ground_truth/tuning.parquet
${BASE_DIR}/ground_truth/held_out.parquet
${BASE_DIR}/ttes/tuning/*.parquet
${BASE_DIR}/ttes/held_out/*.parquet
```

and writes one parquet per task and horizon:

```text
${BASE_DIR}/scores/{task}/{horizon}.parquet
```

Each score parquet contains both tuning and held-out rows. The important columns
are:

```text
split      # train or test; test is the held-out split
task       # task name
time       # horizon in days
strategy   # minute_level, K*_level, p_convex_fusion, etc.
metric     # auroc, auprc, brier_score, log_loss, pi, pi_hat, ...
score      # metric value
std        # jackknife standard deviation where available
```

The MoRGen result is the `strategy == "p_convex_fusion"` row. Held-out AUROC is:

```bash
export BASE_DIR=/path/to/dataset/eic
python - <<'PY'
import os
import polars as pl
from pathlib import Path

score_fp = Path(os.environ["BASE_DIR"]) / "scores" / "creatinine" / "730.parquet"
df = pl.read_parquet(score_fp)
print(
    df.filter(
        (pl.col("split") == "test")
        & (pl.col("strategy") == "p_convex_fusion")
        & (pl.col("metric") == "auroc")
    )
)
PY
```

## Extended ETHOS Reproduction

ETHOS is an optional comparator path. For exact paper reproduction, the primary
path starts from stored ETHOS trajectories generated with:

- ETHOS model context length: `512`
- inference history conditioning: `128` tokens
- generated trajectory root: `${BASE_DIR}/ethos_results_512CL/`

`ethos-eic` should document how to train ETHOS and generate those trajectories
from scratch. Once trajectories exist, `morgen-eval` consumes them exactly like
other generated trajectories.

Materialize ETHOS TTEs:

```bash
cd ../morgen-eval

ETHOS_EVAL_DIR=${BASE_DIR}/ethos_auc_eval_512CL
mkdir -p "${ETHOS_EVAL_DIR}"
ln -sfn "${BASE_DIR}/ground_truth" "${ETHOS_EVAL_DIR}/ground_truth"

for SPLIT in tuning held_out; do
  python scripts/cache_ttes.py \
    dataset=mimic \
    split="${SPLIT}" \
    input_dir="${BASE_DIR}/ethos_results_512CL/" \
    output_dir="${ETHOS_EVAL_DIR}/ttes/" \
    index_dir="${TASK_LABELS_DIR}" \
    output_file_name=ethos
done
```

Score ETHOS alone:

```bash
python scripts/compute_scores.py --multirun \
  hydra/launcher=joblib \
  dataset=mimic \
  dataset.base_dir="${ETHOS_EVAL_DIR}" \
  horizon="${HORIZONS}" \
  task="${MIMIC_TASKS}"
```

The resulting AUROC artifacts live under:

```text
${BASE_DIR}/ethos_auc_eval_512CL/scores/{task}/{horizon}.parquet
```

To evaluate ETHOS together with the MoRGen experts, copy or link the ETHOS TTEs
into an evaluation root that also contains the baseline and histogram TTEs, then
rerun `compute_scores.py`:

```bash
ETHOS_MORGEN_DIR=${BASE_DIR}/ethos_morgen
rsync -a "${ETHOS_EVAL_DIR}/" "${ETHOS_MORGEN_DIR}/"
rsync -a "${BASE_DIR}/ttes/" "${ETHOS_MORGEN_DIR}/ttes/"

python scripts/compute_scores.py --multirun \
  hydra/launcher=joblib \
  dataset=mimic \
  dataset.base_dir="${ETHOS_MORGEN_DIR}" \
  horizon=30,730 \
  task="${MIMIC_TASKS}"
```

The ETHOS-augmented fusion artifacts are then:

```text
${BASE_DIR}/ethos_morgen/scores/{task}/{horizon}.parquet
```

Again, use `strategy == "p_convex_fusion"` and `split == "test"` for the held-out
MoRGen fusion score. Use `strategy == "ethos"` for the standalone ETHOS score.

## Additional Paper Experiments

### Long-Horizon Scores

```bash
cd ../morgen-eval

python scripts/compute_scores.py --multirun \
  hydra/launcher=joblib \
  dataset=mimic \
  dataset.base_dir="${BASE_DIR}" \
  horizon=1825,3650,5300 \
  task="${MIMIC_TASKS}"
```

The notes identify 1825, 3650, and 5300 days as the saved long-horizon artifact
set.

### Joint / Conjunctive Tasks

```bash
cd ../morgen-eval

python scripts/compute_conjunctive_task_scores.py \
  dataset=mimic \
  task_a=death \
  horizon_a=30 \
  task_b=creatinine \
  horizon_b=730
```

Outputs are written under:

```text
${BASE_DIR}/conjunctive_task_scores_v2/
```

### Shared-Weight and Uniform Fusion

```bash
cd ../morgen-eval

python scripts/compute_single_weight.py \
  dataset=mimic \
  tasks='[creatinine,death,hematocrit,hemoglobin,leukocyte,platelet,readmission]' \
  eval_times='[30,730]' \
  fit_times='[30,730]'
```

Default output:

```text
${BASE_DIR}/single_weight_scores/30_730.parquet
```

### VQ Count Ablation

Train and generate:

```bash
cd ../morgen

PYENV_VERSION=morgen \
bash scripts/launch_vq_count_grid.sh ehrshot --gpu 0 --output-root-name vq_count

PYENV_VERSION=morgen \
bash scripts/launch_vq_count_grid.sh ehrshot --gpu 0 --max-count 2 --output-root-name vq_count_mc2
```

Materialize TTEs and scores:

```bash
PYENV_VERSION=morgen \
bash scripts/materialize_vq_count_eval.sh ehrshot --output-root-name vq_count

PYENV_VERSION=morgen \
bash scripts/materialize_vq_count_eval.sh ehrshot --output-root-name vq_count_mc2
```

Score artifacts:

```text
${BASE_DIR}/vq_count/scores/
${BASE_DIR}/vq_count_mc2/scores/
```

### Compute Benchmarks

Histogram benchmark:

```bash
PYENV_VERSION=morgen \
bash scripts/launch_compute_grid.sh --gpu 0 --datasets mimic,ehrshot
```

Baseline benchmark:

```bash
PYENV_VERSION=morgen \
bash scripts/launch_baseline_compute.sh --gpu 0 --datasets mimic,ehrshot
```

Benchmark summaries are written below each dataset's:

```text
${BASE_DIR}/compute_benchmark/
```

## Release Scope

This public `morgen` release focuses on installation, preprocessing, model
training, trajectory generation, compute benchmarks, and the handoff to
`morgen-eval` for scoring, tables, and figures.
