
export CUDA_VISIBLE_DEVICES=0
K=64

export POLARS_MAX_THREADS=1
BASE_DIR="/storage/shared/ehr-shot/full/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/K${K}_day_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/day_level/"

morgen_fit_em \
    cache_dir="$CACHE_DIR" \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=2048 \
    window_size_days=1 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=0 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="discharge/IP" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=false \
    trainer.val_check_interval=null

export POLARS_MAX_THREADS=1
BASE_DIR="/storage/shared/ehr-shot/full/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/K${K}_week_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/week_level/"

morgen_fit_em \
    cache_dir=${CACHE_DIR} \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=512 \
    window_size_days=7 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=0 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="discharge/IP" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=false \
    trainer.val_check_interval=null

export POLARS_MAX_THREADS=1
BASE_DIR="/storage/shared/ehr-shot/full/"
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/K${K}_month_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/month_level/"

morgen_fit_em \
    cache_dir=${CACHE_DIR} \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=256 \
    window_size_days=30 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=0 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="discharge/IP" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=false \
    trainer.val_check_interval=null

export POLARS_MAX_THREADS=1
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/K${K}_quarter_annual_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/quarter_annual_level/"

morgen_fit_em \
    cache_dir=${CACHE_DIR} \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=400 \
    window_size_days=90 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=0 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="discharge/IP" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=false \
    trainer.val_check_interval=null


BASE_DIR="/storage/shared/ehr-shot/full/"
export POLARS_MAX_THREADS=1
QUANTIZER_OUTPUT_DIR="${BASE_DIR}/eic/results/em_quantizer/K${K}_semi_annual_level/"
DATA_DIR="${BASE_DIR}/eic/final/"
CACHE_DIR="${BASE_DIR}/eic/results/fit_em_cache/semi_annual_level/"

morgen_fit_em \
    cache_dir=${CACHE_DIR} \
    quantizer_config.n_embeddings=${K} \
    seed=1 \
    output_dir="$QUANTIZER_OUTPUT_DIR" \
    datamodule.config.tensorized_cohort_dir="$DATA_DIR" \
    gap_config.max_gap_length=200 \
    window_size_days=182 \
    trainer.max_epochs=50 \
    lightning_module.learning_rate=1e-3 \
    do_overwrite=true \
    datamodule.batch_size=512 \
    datamodule.num_workers=0 \
    datamodule.anchoring_strategy=SPECIFIC_EVENT \
    datamodule.specific_event_regex="discharge/IP" \
    datamodule.include_anchor_token=false \
    datamodule.empty_window_mode=IGNORE \
    datamodule.persistent_workers=false \
    trainer.val_check_interval=null
