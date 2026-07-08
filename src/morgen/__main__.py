import logging
import os
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import hydra
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from meds import held_out_split, train_split, tuning_split
from meds_torchdata import MEDSTorchDataConfig
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema
from MEDS_transforms.runner import load_yaml_file
from omegaconf import DictConfig, OmegaConf
from torchmetrics.functional import auroc, average_precision
from tqdm.auto import tqdm

from .generation import format_trajectories, get_timeline_end_token_idx
from .generation.histogram_trajectories import format_histogram_trajectories
from .training import find_checkpoint_path, MorgenModule, validate_resume_directory

# Import OmegaConf Resolvers
from .utils import (
    gpus_available,
    hash_based_seed,
    int_prod,
    is_mlflow_logger,
    num_cores,
    num_gpus,
    oc_min,
    resolve_generation_context_size,
    save_resolved_config,
    sub,
)

logger = logging.getLogger(__name__)

CONFIGS = files("morgen") / "configs"

MEDSTorchDataConfig.add_to_config_store("datamodule/config")


def build_histogram_metadata(
    processor,
    window_size_days: int,
    device: torch.device,
) -> "pl.DataFrame":
    """Construct metadata describing every histogram/gap/special token.

    Args:
        processor: HistogramSequenceProcessor configured with the quantizer + gap layout.
        window_size_days: Number of days represented by each histogram window.
        device: Device to use when decoding the quantizer codebook.

    Returns:
        A Polars DataFrame indexed by vocab id with decoded histograms and gap metadata.
    """

    import polars as pl

    histogram_metadata_schema = {
        "histogram": pl.List(pl.Float32),
        "histogram/vocab_index": pl.Int64,
        "gap_count": pl.Int64,
        "token_type": pl.Categorical,
    }

    decoded_histograms = processor.decode_histograms(device=device)

    code_indices = list(range(1, processor.quantizer_token_start))
    code_metadata_df = pl.DataFrame(
        {
            "histogram": [[]] * len(code_indices),
            "histogram/vocab_index": code_indices,
            "gap_count": [0] * len(code_indices),
            "token_type": ["CODE"] * len(code_indices),
        },
        schema=histogram_metadata_schema,
    )

    if processor.do_fusion:
        assert code_metadata_df.height > 0
    histogram_indices = list(range(processor.quantizer_token_start, processor.quantizer_token_end + 1))
    histogram_metadata_df = pl.DataFrame(
        {
            "histogram": decoded_histograms.detach().cpu(),
            "histogram/vocab_index": histogram_indices,
            "gap_count": [1] * len(decoded_histograms),
            "token_type": ["HISTOGRAM"] * len(decoded_histograms),
        },
        schema=histogram_metadata_schema,
    )

    if processor.special_tokens["PAD"] != 0:
        raise ValueError("Padding token must be 0.")

    padding_metadata_df = pl.DataFrame(
        {"histogram": [[]], "histogram/vocab_index": [0], "gap_count": [0], "token_type": ["PAD"]},
        schema=histogram_metadata_schema,
    )

    gap_indices = list(range(processor.gap_token_start, processor.gap_token_end + 1))
    gap_metadata_df = pl.DataFrame(
        {
            "histogram": [[]] * len(gap_indices),
            "histogram/vocab_index": gap_indices,
            "gap_count": range(1, len(gap_indices) + 1),
            "token_type": ["GAP"] * len(gap_indices),
        },
        schema=histogram_metadata_schema,
    )

    special_metadata_df = pl.DataFrame(
        {
            "histogram": [[], [], []],
            "histogram/vocab_index": [
                processor.special_tokens["ANCHOR"],
                processor.special_tokens["BOS"],
                processor.special_tokens["EOS"],
            ],
            "gap_count": [0, 0, 1],
            "token_type": ["ANCHOR", "BOS", "EOS"],
        },
        schema=histogram_metadata_schema,
    )

    histogram_metadata_df = pl.concat(
        [padding_metadata_df, code_metadata_df, histogram_metadata_df, gap_metadata_df, special_metadata_df]
    )

    if (
        not histogram_metadata_df.with_row_index()
        .select(pl.col("histogram/vocab_index").eq(pl.col("index")).all())
        .item()
    ):
        raise ValueError("Histogram metadata vocab indices do not match row indices.")

    histogram_metadata_df = histogram_metadata_df.with_columns(
        pl.duration(days=pl.col("gap_count") * window_size_days).alias("number_of_days")
    )

    return histogram_metadata_df



@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_train_quantizer")
def train_quantizer(cfg: DictConfig):
    """Train histogram quantizer (Phase 1 of phenotype compression)."""
    import multiprocessing

    multiprocessing.set_start_method("forkserver", force=True)
    st = datetime.now(tz=UTC)

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    output_dir = Path(cfg.output_dir)

    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")

    cfg_path = output_dir / "config.yaml"

    ckpt_path = None

    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing output directory {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            validate_resume_directory(output_dir, cfg)
            ckpt_path = find_checkpoint_path(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    # Create simplified quantizer for training
    from lightning import LightningModule

    from .model.simplified_quantizer import SimplifiedAutoencoderVQ

    # Instantiate datamodule (real MEDS data)
    D = instantiate(cfg.datamodule)

    # Create quantizer directly
    quantizer = SimplifiedAutoencoderVQ(**cfg.quantizer_config)

    # Load code metadata once so it is available inside the lightning module
    import polars as pl

    tensorized_dir = Path(D.config.tensorized_cohort_dir)
    metadata_dir = tensorized_dir / "metadata"
    code_metadata_path = metadata_dir / "codes.parquet"
    if not code_metadata_path.is_file():
        raise FileNotFoundError(f"Code metadata not found at {code_metadata_path}")
    code_metadata_df = pl.read_parquet(code_metadata_path, columns=["code/vocab_index", "code"])

    # Create histogram-based lightning module
    class HistogramQuantizerModule(LightningModule):
        def __init__(self, quantizer, vocab_size, window_size_days=30):
            super().__init__()
            self.quantizer = quantizer
            self.vocab_size = vocab_size
            self.window_size_days = window_size_days
            # Track mortality index consistently across hooks
            self.mortality_idx = 283
            self.special_codes: dict[str, torch.Tensor] = {}
            for name, code_regex in cfg.special_codes.items():
                code_vocab_indices = code_metadata_df.filter(pl.col("code").str.contains(code_regex))[
                    "code/vocab_index"
                ].to_list()
                filtered_indices = sorted({idx for idx in code_vocab_indices if 0 <= idx < self.vocab_size})
                if not filtered_indices:
                    logger.warning(
                        "No codes matched regex %s for special code %s; metrics for this group will be skipped.",
                        code_regex,
                        name,
                    )
                    indices_tensor = torch.empty(0, dtype=torch.long)
                else:
                    indices_tensor = torch.tensor(filtered_indices, dtype=torch.long)
                self.special_codes[name] = indices_tensor

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(),
                lr=cfg.lightning_module.learning_rate,
                weight_decay=cfg.lightning_module.weight_decay,
            )

        def training_step(self, batch, batch_idx):
            # Convert MEDS batch to histogram windows
            histograms = batch.histograms[batch.histograms.sum(-1) != 0]

            # Skip if no histograms generated
            if histograms.sum() == 0:
                return None

            result = self.quantizer(histograms)
            loss = result["total_loss"]

            self.log("train_loss", loss, prog_bar=True)
            self.log("vq_loss", result["vq_loss"])
            self.log("reconstruction_loss", result["reconstruction_loss"])
            if "usage_loss" in result:
                self.log("usage_loss", result["usage_loss"], on_step=True, on_epoch=True)

            # Codebook usage metrics (how many codes are active this step)
            if "indices" in result and result["indices"] is not None and result["indices"].numel() > 0:
                indices = result["indices"]
                unique_codes = torch.unique(indices).numel()
                n_embeddings = int(self.quantizer.n_embeddings)
                usage_frac = unique_codes / max(n_embeddings, 1)
                self.log("train_codebook_unique", unique_codes, on_step=True, on_epoch=True)
                self.log("train_codebook_usage_frac", usage_frac, on_step=True, on_epoch=True)

            # Mean mortality logit for quick sanity (should move away from ~0 if learning)
            if self.vocab_size > self.mortality_idx and "reconstruction" in result:
                mort_logit_mean = result["reconstruction"][:, self.mortality_idx].mean()
                self.log("train_mortality_logit_mean", mort_logit_mean, on_step=True, on_epoch=True)

            self._log_special_code_metrics(
                result,
                prefix="train_special",
                on_step=True,
                on_epoch=False,
            )

            return loss

        def _log_special_code_metrics(
            self,
            result: dict,
            prefix: str,
            *,
            on_step: bool,
            on_epoch: bool,
        ) -> None:
            if not self.special_codes:
                return
            if "reconstruction" not in result or "binary_input" not in result:
                return

            logits = result["reconstruction"]
            if logits.ndim != 2 or logits.shape[0] == 0:
                return

            batch_size = logits.shape[0]
            probs = torch.sigmoid(logits).detach()
            targets = (result["binary_input"] > 0).detach()

            for name, indices in self.special_codes.items():
                if indices.numel() == 0:
                    continue

                idx = indices.to(logits.device)
                group_probs = probs[:, idx]
                group_targets = targets[:, idx]

                prob_any = (1 - torch.prod(1 - group_probs, dim=1)).clamp(min=0.0, max=1.0)
                prob_any = prob_any.detach()
                target_any = group_targets.any(dim=1)

                if target_any.numel() == 0:
                    continue

                preds = prob_any >= 0.5
                tp = (preds & target_any).float().sum()
                fp = (preds & ~target_any).float().sum()
                fn = (~preds & target_any).float().sum()
                denom = 2 * tp + fp + fn
                f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))
                prevalence = target_any.float().mean()

                positive_count = int(target_any.sum().item())
                total_count = target_any.shape[0]

                auprc = (
                    average_precision(prob_any, target_any.int(), task="binary")
                    if positive_count > 0
                    else torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
                )

                auroc_value = (
                    auroc(prob_any, target_any.int(), task="binary")
                    if 0 < positive_count < total_count
                    else torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
                )

                metric_prefix = f"{prefix}_{name.lower()}"
                self.log(
                    f"{metric_prefix}_f1",
                    f1,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )
                self.log(
                    f"{metric_prefix}_auprc",
                    auprc,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )
                self.log(
                    f"{metric_prefix}_auroc",
                    auroc_value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )
                self.log(
                    f"{metric_prefix}_prevalence",
                    prevalence,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )

        def validation_step(self, batch, batch_idx):
            # Run same logic as training but log as validation metrics
            with torch.no_grad():
                histograms = batch.histograms[batch.histograms.sum(-1) != 0]
                result = self.quantizer(histograms)
                loss = result["total_loss"]

                # Track how often reconstructions produce any active codes
                if histograms.shape[0] > 0:
                    recon_logits = result["reconstruction"]
                    recon_active = torch.sigmoid(recon_logits) > 0.5
                    non_empty_percent = recon_active.any(dim=1).float().mean() * 100.0
                    self.log(
                        "val_non_empty_reconstruction_percent",
                        non_empty_percent,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

                    binary_hist = result["binary_input"]
                    non_zero_counts = binary_hist.sum(dim=1)
                    empty_percent = (non_zero_counts == 0).float().mean() * 100.0

                    self.log(
                        "val_ground_truth_empty_percent",
                        empty_percent,
                        on_step=False,
                        on_epoch=True,
                    )

                    mortality_idx = self.mortality_idx
                    if self.vocab_size > mortality_idx:
                        mortality_targets = result["binary_input"][:, mortality_idx] > 0
                        mortality_logits = recon_logits[:, mortality_idx]
                        mortality_probs = torch.sigmoid(mortality_logits)
                        mortality_preds = mortality_probs >= 0.5

                        positive_count = mortality_targets.float().sum()
                        true_positive = (mortality_preds & mortality_targets).float().sum()

                        mortality_recall = torch.where(
                            positive_count > 0,
                            true_positive / positive_count,
                            torch.tensor(0.0, device=histograms.device),
                        )
                        mortality_recall_t10 = torch.where(
                            positive_count > 0,
                            ((mortality_probs >= 0.1) & mortality_targets).float().sum() / positive_count,
                            torch.tensor(0.0, device=histograms.device),
                        )
                        mortality_recall_t02 = torch.where(
                            positive_count > 0,
                            ((mortality_probs >= 0.02) & mortality_targets).float().sum() / positive_count,
                            torch.tensor(0.0, device=histograms.device),
                        )

                        mortality_prevalence = mortality_targets.float().mean() * 100.0

                        self.log(
                            "val_mortality_recall",
                            mortality_recall,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        )
                        self.log(
                            "val_mortality_recall_T10",
                            mortality_recall_t10,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        )
                        self.log(
                            "val_mortality_recall_T02",
                            mortality_recall_t02,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                        )
                        self.log(
                            "val_mortality_prevalence_percent",
                            mortality_prevalence,
                            on_step=False,
                            on_epoch=True,
                        )

                        auprc = average_precision(
                            mortality_probs,
                            mortality_targets.int(),
                            task="binary",
                        )
                        self.log(
                            "val_mortality_auprc",
                            auprc,
                            on_step=False,
                            on_epoch=True,
                        )

                        if 0 < positive_count < histograms.shape[0]:
                            auroc_value = auroc(
                                mortality_probs,
                                mortality_targets.int(),
                                task="binary",
                            )
                            self.log(
                                "val_mortality_auroc",
                                auroc_value,
                                on_step=False,
                                on_epoch=True,
                            )

                        fp = (mortality_preds & ~mortality_targets).float().sum()
                        fn = (~mortality_preds & mortality_targets).float().sum()
                        denom = 2 * true_positive + fp + fn
                        mortality_f1 = torch.where(
                            denom > 0,
                            (2 * true_positive) / denom,
                            torch.tensor(0.0, device=histograms.device),
                        )
                        self.log(
                            "val_mortality_f1",
                            mortality_f1,
                            on_step=False,
                            on_epoch=True,
                        )

                        # Also log mean mortality logit on validation
                        self.log(
                            "val_mortality_logit_mean",
                            mortality_logits.mean(),
                            on_step=False,
                            on_epoch=True,
                        )

                    self._log_special_code_metrics(
                        result,
                        prefix="val_special",
                        on_step=False,
                        on_epoch=True,
                    )

                    if batch_idx == 0:
                        codebook_indices = torch.arange(
                            0, self.quantizer.n_embeddings, device=histograms.device
                        )
                        learned_histograms = self.quantizer.decode_indices(codebook_indices)
                        learned_active = learned_histograms >= 0
                        learned_non_empty_percent = learned_active.any(dim=1).float().mean() * 100.0
                        self.log(
                            "val_learned_hist_non_empty_percent",
                            learned_non_empty_percent,
                            on_step=False,
                            on_epoch=True,
                        )

                # Log validation metrics
                self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("val_vq_loss", result["vq_loss"], on_step=False, on_epoch=True)
                self.log(
                    "val_reconstruction_loss", result["reconstruction_loss"], on_step=False, on_epoch=True
                )
                if "usage_loss" in result:
                    self.log(
                        "val_usage_loss",
                        result["usage_loss"],
                        on_step=False,
                        on_epoch=True,
                    )

                # Codebook usage on validation batch
                if "indices" in result and result["indices"] is not None and result["indices"].numel() > 0:
                    indices = result["indices"]
                    unique_codes = torch.unique(indices).numel()
                    n_embeddings = int(self.quantizer.n_embeddings)
                    usage_frac = unique_codes / max(n_embeddings, 1)
                    self.log("val_codebook_unique", unique_codes, on_step=False, on_epoch=True)
                    self.log("val_codebook_usage_frac", usage_frac, on_step=False, on_epoch=True)

                # Add tuning/loss for early stopping callback
                self.log("tuning/loss", loss, on_step=False, on_epoch=True)

                return loss

    # Instantiate the lightning module with real MEDS data processing
    M = HistogramQuantizerModule(quantizer=quantizer, vocab_size=cfg.quantizer_config.vocab_size)

    if cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    trainer = instantiate(cfg.trainer)

    M = trainer.precision_plugin.convert_module(M)
    trainer.strategy.connect(M)
    trainer.strategy.setup_environment()
    device = trainer.strategy.root_device
    M = M.to(device)

    if any(is_mlflow_logger(logger) for logger in trainer.loggers):
        import mlflow

        if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" not in os.environ:
            mlflow.enable_system_metrics_logging()

    # Use the real MEDS datamodule - no dummy data!
    trainer_kwargs = {"model": M, "datamodule": D}

    trainer.fit(**trainer_kwargs)

    # Handle checkpoint saving - always ensure we have a checkpoint file
    output_fp = Path(cfg.output_dir) / "best_quantizer.ckpt"

    # Try to copy from checkpoint callback if available and file exists
    checkpoint_copied = False
    if (
        hasattr(trainer, "checkpoint_callback")
        and trainer.checkpoint_callback is not None
        and hasattr(trainer.checkpoint_callback, "best_model_path")
        and trainer.checkpoint_callback.best_model_path is not None
    ):
        best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
        if best_ckpt_path.is_file():
            try:
                shutil.copyfile(best_ckpt_path, output_fp)
                best_score = trainer.checkpoint_callback.best_model_score
                logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")
                checkpoint_copied = True
            except Exception as e:
                logger.warning(f"Failed to copy checkpoint from {best_ckpt_path}: {e}")

    # If no checkpoint was copied, save the current model state
    if not checkpoint_copied:
        trainer.save_checkpoint(output_fp)
        logger.info(f"Saved current model state to {output_fp!s}.")
    logger.info(f"Quantizer training complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_train_autoregressive_histogram")
def train_autoregressive_histogram(cfg: DictConfig):
    """Train autoregressive model on quantized histogram sequences (Phase 2)."""
    import multiprocessing

    multiprocessing.set_start_method("forkserver", force=True)
    st = datetime.now(tz=UTC)

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    output_dir = Path(cfg.output_dir)

    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")

    cfg_path = output_dir / "config.yaml"

    ckpt_path = None

    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing output directory {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            validate_resume_directory(output_dir, cfg)
            ckpt_path = find_checkpoint_path(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    import torch
    torch.set_float32_matmul_precision("medium")

    # Create histogram dataset (EmptyWindowMode.SINGLE_GAP for AR training)
    from meds_torchdata.histogram_dataset import EmptyWindowMode, HistogramConfig, HistogramPytorchDataset

    from .data.histogram_sequence_processor import HistogramSequenceProcessor
    from .model.histogram_model import GapTokenConfig, HistogramModel, ModelMode

    # Instantiate histogram dataset with SINGLE_GAP mode
    D = instantiate(cfg.datamodule)
    
    # Load pretrained quantizer    
    if cfg.em_params_dir is not None:
        logger.info("#################################################Using EM quantizer#################################################")
        from .model.em_quantizer import EMQuantizer
        import numpy as np

        logger.info(f"Loading Bernoulli EM quantizer from {cfg.em_params_dir}")
        em_path = Path(cfg.em_params_dir)
        theta = torch.from_numpy(np.load(em_path / "cluster_theta.npy")).float()
        pi = torch.from_numpy(np.load(em_path / "cluster_pi.npy")).float()
        
        # Instantiate EM quantizer directly
        quantizer = EMQuantizer(theta=theta, pi=pi)
        
        # Pass the instantiated object to the processor
        processor = HistogramSequenceProcessor(
            quantizer=quantizer,
            vocab_size=cfg.quantizer_config.vocab_size,
            max_gap_length=cfg.gap_config.max_gap_length,
            do_fusion=cfg.do_fusion,
        )
    else:
        logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Using VQ-VAE quantizer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Default VQ-VAE workflow
        quantizer_checkpoint_path = cfg.quantizer_checkpoint_path
        if not Path(quantizer_checkpoint_path).exists():
            raise FileNotFoundError(f"Quantizer checkpoint not found: {quantizer_checkpoint_path}")

        processor = HistogramSequenceProcessor(
            quantizer_checkpoint_path=quantizer_checkpoint_path,
            vocab_size=cfg.quantizer_config.vocab_size,
            max_gap_length=cfg.gap_config.max_gap_length,
            do_fusion=cfg.do_fusion,
        )

    gap_config = cfg.gap_config

    # Update model config with correct vocab size from processor
    model_config = cfg.model_config.copy()
    model_config["vocab_size"] = processor.total_vocab_size

    model = HistogramModel(
        model_config=model_config,
        quantizer_config=cfg.quantizer_config,
        gap_token_config=gap_config,
        mode=ModelMode.AUTOREGRESSIVE,
        quantizer=processor.quantizer,
        quantizer_token_start=processor.quantizer_token_start,
        total_vocab_size=processor.total_vocab_size,
    )

    # Ensure EOS token id is set on the underlying HF config (don’t mutate Hydra struct)
    model.ar_model.HF_model_config.eos_token_id = processor.special_tokens["EOS"]
    model.ar_model.HF_model.config.eos_token_id = processor.special_tokens["EOS"]

    # Create Lightning module for autoregressive training
    from .training.histogram_module import HistogramLightningModule

    M = HistogramLightningModule(model=model, processor=processor, **cfg.lightning_module)

    if M.model.mode == ModelMode.AUTOREGRESSIVE or cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    trainer = instantiate(cfg.trainer)
    if any(is_mlflow_logger(logger) for logger in trainer.loggers):
        import mlflow

        if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" not in os.environ:
            mlflow.enable_system_metrics_logging()

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path

    trainer.fit(**trainer_kwargs)

    # Get best checkpoint path directly - no fallbacks
    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError(f"Best checkpoint not found at {best_ckpt_path}")

    output_fp = Path(cfg.output_dir) / "best_model.ckpt"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_ckpt_path, output_fp)
    logger.info(f"Copied best checkpoint to {output_fp}")

    # Get best score directly - no fallbacks
    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(
        f"Best autoregressive model checkpoint (with score {best_score if best_score is not None else 'N/A'}) copied to {output_fp!s}."
    )
    logger.info(f"Autoregressive training complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_generate_histogram_trajectories")
def generate_histogram_trajectories(cfg: DictConfig):
    """Generate synthetic trajectories using histogram model (Phase 3)."""
    import multiprocessing

    import polars as pl

    multiprocessing.set_start_method("forkserver", force=True)
    st = datetime.now(tz=UTC)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)

    # Load histogram model from checkpoint
    from .training.histogram_module import HistogramLightningModule

    M = HistogramLightningModule.load_from_checkpoint(Path(cfg.ckpt_path))
    M.max_new_tokens = cfg.max_new_tokens
    M.eval()
    from .data.histogram_sequence_processor import HistogramSequenceProcessor

    # ------------------ Store Decoded Histograms ------------------
    processor = HistogramSequenceProcessor(
        quantizer=M.model.quantizer,
        vocab_size=cfg.quantizer_config.vocab_size,
        max_gap_length=cfg.gap_config.max_gap_length,
        do_fusion=cfg.do_fusion,
    )
    M.processor = processor
    histogram_metadata_df = build_histogram_metadata(
        processor,
        window_size_days=cfg.window_size_days,
        device=M.device,
    )
    Path(cfg.output_metadata_dir).mkdir(parents=True, exist_ok=True)
    histogram_metadata_df.write_parquet(Path(cfg.output_metadata_dir) / "histogram_metadata.parquet")
    pl.read_parquet(
        Path(cfg.datamodule.config.tensorized_cohort_dir) / "metadata" / "codes.parquet"
    ).write_parquet(Path(cfg.output_metadata_dir) / "code_metadata.parquet")

    # ------------------ Generate Histogram Trajectories ------------------

    trainer = instantiate(cfg.trainer)

    M = trainer.precision_plugin.convert_module(M)
    trainer.strategy.connect(M)
    trainer.strategy.setup_environment()
    device = trainer.strategy.root_device
    M = M.to(device)

    inference = cfg.inference

    if cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    written_files = []
    for split in inference.generate_for_splits:
        if split == train_split:
            dataloader = D.train_dataloader(shuffle=False)
        elif split == tuning_split:
            dataloader = D.val_dataloader()
        elif split == held_out_split:
            dataloader = D.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split}.")

        for sample in range(inference.N_trajectories_per_task_sample):
            histogram_out_fp = Path(cfg.output_dir) / "histogram" / split / f"{sample}.parquet"
            code_out_fp = Path(cfg.output_dir) / "code" / split / f"{sample}.parquet"
            histogram_out_fp.parent.mkdir(parents=True, exist_ok=True)

            if histogram_out_fp.is_file() and not cfg.do_overwrite:
                logger.info(f"Skipping {histogram_out_fp} as it already exists.")
                continue
            else:
                histogram_out_fp.parent.mkdir(parents=True, exist_ok=True)
                code_out_fp.parent.mkdir(parents=True, exist_ok=True)

            seed = hash_based_seed(cfg.get("seed", None), split, sample)

            logger.info(
                f"Generating histogram trajectories for {split} sample {sample} to {histogram_out_fp} with seed {seed}."
            )

            seed_everything(seed, workers=True)

            sampler = getattr(dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(sample)

            predictions: list[torch.Tensor] = []
            with torch.no_grad():
                precision_ctx = trainer.precision_plugin.forward_context()
                with precision_ctx:
                    for batch in dataloader:
                        batch_on_device = trainer.strategy.batch_to_device(batch, device=device)
                        batch_predictions = M.predict_step(batch_on_device)
                        predictions.append(batch_predictions.detach().cpu())

            # In fusion mode, extract only histogram tokens for histogram trajectory formatting
            # if cfg.do_fusion:
            #     histogram_only_predictions = []
            #     for batch_pred in predictions:
            #         # Filter to keep only histogram tokens (not codes, gaps, anchors, BOS, EOS, PAD)
            #         is_hist = processor.is_histogram_token(batch_pred)
            #         # Extract histogram tokens for each sequence in batch
            #         batch_hist = []
            #         for seq_idx in range(batch_pred.size(0)):
            #             hist_tokens = batch_pred[seq_idx][is_hist[seq_idx]]
            #             batch_hist.append(hist_tokens)
            #         histogram_only_predictions.append(torch.stack(batch_hist))

            #     predictions_for_hist = histogram_only_predictions
            # else:
            #     predictions_for_hist = predictions
            predictions_for_hist = predictions

            predictions_df = format_histogram_trajectories(
                dataloader.dataset, predictions_for_hist, histogram_metadata_df, cfg.window_size_days
            )
            pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
            pq.write_table(pa_table, histogram_out_fp)

            if cfg.do_fusion:
                predictions_df = format_trajectories(dataloader.dataset, predictions)
                pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
                pq.write_table(pa_table, code_out_fp)

    logger.info(f"Generation of histogram trajectories complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    output_dir = Path(cfg.output_dir)

    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")

    cfg_path = output_dir / "config.yaml"

    ckpt_path = None

    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing output directory {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            validate_resume_directory(output_dir, cfg)
            ckpt_path = find_checkpoint_path(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)

    gpt_kwargs = {"vocab_size": D.config.vocab_size, "eos_token_id": get_timeline_end_token_idx(D.config)}

    M = instantiate(
        cfg.lightning_module,
        model={"gpt_kwargs": gpt_kwargs},
        metrics={"vocab_size": D.config.vocab_size},
    )

    if M.model.do_demo or cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    trainer = instantiate(cfg.trainer)
    if any(is_mlflow_logger(logger) for logger in trainer.loggers):
        # We do the import only here to avoid importing mlflow if it isn't installed.
        import mlflow

        if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" not in os.environ:
            # The user can set this environment variable to enable or disable system metrics logging on their
            # own, but if they don't, it will by default be enabled.
            mlflow.enable_system_metrics_logging()

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path

    trainer.fit(**trainer_kwargs)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")

    output_fp = Path(cfg.output_dir) / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")
    logger.info(f"Training complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_generate_trajectories")
def generate_trajectories(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)

    M = MorgenModule.load_from_checkpoint(Path(cfg.ckpt_path))
    M.eval()

    trainer = instantiate(cfg.trainer)

    M = trainer.precision_plugin.convert_module(M)
    trainer.strategy.connect(M)
    trainer.strategy.setup_environment()
    device = trainer.strategy.root_device
    M = M.to(device)

    inference = cfg.inference

    if cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    for split in inference.generate_for_splits:
        if split == train_split:
            dataloader = D.train_dataloader(shuffle=False)
        elif split == tuning_split:
            dataloader = D.val_dataloader()
        elif split == held_out_split:
            dataloader = D.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split}.")

        for sample in (range(inference.N_trajectories_per_task_sample)):
            out_fp = Path(cfg.output_dir) / split / f"{sample}.parquet"
            out_fp.parent.mkdir(parents=True, exist_ok=True)

            if out_fp.is_file() and not cfg.do_overwrite:
                logger.info(f"Skipping {out_fp} as it already exists.")
                continue
            else:
                out_fp.parent.mkdir(parents=True, exist_ok=True)

            seed = hash_based_seed(cfg.get("seed", None), split, sample)

            logger.info(f"Generating trajectories for {split} sample {sample} to {out_fp} with seed {seed}.")

            seed_everything(seed, workers=True)

            sampler = getattr(dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(sample)

            predictions: list[torch.Tensor] = []
            with torch.no_grad():
                precision_ctx = trainer.precision_plugin.forward_context()
                with precision_ctx:
                    logger.info("Generating")
                    for i, batch in enumerate(tqdm(dataloader)):
                        if i == 0:
                            logger.info("First batch")
                        batch_on_device = trainer.strategy.batch_to_device(batch, device=device)
                        batch_predictions = M.predict_step(batch_on_device)
                        predictions.append(batch_predictions.detach().cpu())
            logger.info("Formatting and Storing")
            predictions_df = format_trajectories(dataloader.dataset, predictions)

            pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
            pq.write_table(pa_table, out_fp)

    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_process_generated_histograms")
def process_generated_histograms(cfg: DictConfig):
    import numpy as np
    import polars as pl

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    probability_threshold = float(cfg.threshold)
    threshold = np.log(probability_threshold / (1 - probability_threshold))

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    metadata_dir = input_dir / "metadata"
    histogram_metadata_path = metadata_dir / "histogram_metadata.parquet"
    code_metadata_path = metadata_dir / "code_metadata.parquet"

    if not histogram_metadata_path.is_file():
        raise FileNotFoundError(f"Histogram metadata not found at {histogram_metadata_path}")
    if not code_metadata_path.is_file():
        raise FileNotFoundError(f"Code metadata not found at {code_metadata_path}")

    histogram_metadata_df = pl.read_parquet(histogram_metadata_path).with_columns(
        pl.col("histogram")
        .list.eval(pl.int_range(0, pl.len()).filter(pl.element() > threshold))
        .alias("code/vocab_index")
    )
    code_metadata_df = pl.read_parquet(code_metadata_path)

    metadata_df = (
        histogram_metadata_df.select("histogram", "histogram/vocab_index", "code/vocab_index", "token_type")
        .explode("code/vocab_index")
        .join(code_metadata_df["code/vocab_index", "code"], on="code/vocab_index", how="left")
        .with_columns(pl.col("code").fill_null(pl.col("token_type")))
        .group_by(["histogram", "histogram/vocab_index", "token_type"], maintain_order=True)
        .agg(pl.all())
    )

    for name, code_regex in cfg.special_codes.items():
        code_vocab_indices = code_metadata_df.filter(pl.col("code").str.contains(code_regex))[
            "code/vocab_index"
        ].to_list()
        assert len(code_vocab_indices) > 0, f"No code found for {name}"
        special_token_col = f"SPECIAL_TOKEN//{name}"
        special_token_df = metadata_df.with_columns(
            pl.col("histogram")
            .list.eval(pl.element().filter(pl.int_range(0, pl.len()).is_in(code_vocab_indices)))
            .alias(special_token_col)
        )
        # Confirm we loaded a non-zero number of codes and the number of code_vocab indices we filtered is corrected
        is_correct = (
            special_token_df.filter(pl.col("token_type").eq("HISTOGRAM"))
            .select(pl.col(special_token_col).list.eval(pl.len()))
            .explode(special_token_col)
            .select(pl.col(special_token_col).eq(len(code_vocab_indices)))[special_token_col]
        )
        assert is_correct.all(), f"{name} special token not found in histogram metadata"
        assert len(is_correct) == metadata_df.filter(pl.col("token_type").eq("HISTOGRAM")).height, (
            "Number of histograms is not equal to the number of special tokens"
        )

        special_token_df = special_token_df.with_columns(
            (
                1
                - (
                    pl.col(special_token_col)
                    .list.eval(-pl.element().exp().add(1).log())  # -softplus(logit)
                    .list.sum()
                    .exp()
                )
            ).alias(special_token_col)
        ).with_columns(
            pl.concat_list(
                [
                    pl.col("code"),
                    pl.when(pl.col(special_token_col) > probability_threshold)
                    .then([special_token_col])
                    .otherwise([]),
                ]
            ).alias("code")
        )
        logger.info(
            f"Special token {name} has prevalence {special_token_df[special_token_col].mean() * 100:.4f}% with p threshold {probability_threshold}"
        )
        metadata_df = special_token_df.drop(special_token_col)

        logger.info(f"Processed {name} special token")

    metadata_df.drop("histogram")

    def is_in_metadata(path: Path) -> bool:
        if path == metadata_dir or metadata_dir in path.parents:
            return True
        else:
            return False

    histogram_files = [
        path for path in input_dir.rglob("*.parquet") if path.is_file() and not is_in_metadata(path)
    ]

    if not histogram_files:
        logger.warning(f"No histogram parquet files found in {input_dir}.")
        return

    for histogram_path in sorted(histogram_files):
        histogram_df = pl.read_parquet(histogram_path)
        code_df = (
            histogram_df.with_columns(pl.col("code").cast(pl.Int64))
            .rename({"code": "histogram_index"})
            .join(
                metadata_df["histogram/vocab_index", "code"],
                left_on="histogram_index",
                right_on="histogram/vocab_index",
            )
            .explode("code")
        )

        relative_path = histogram_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pa_table = GeneratedTrajectorySchema.align(code_df.to_arrow())
        pq.write_table(pa_table, output_path)
        logger.info(f"Wrote converted codes to {output_path}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_process_generated_histograms")
def process_generated_histograms_soft_decoding(cfg: DictConfig):
    import numpy as np
    import polars as pl

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    metadata_dir = input_dir / "metadata"
    if not metadata_dir.is_dir():
        metadata_dir = input_dir.parent / "metadata"
        if not metadata_dir.is_dir():
            raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")
    histogram_metadata_path = metadata_dir / "histogram_metadata.parquet"
    code_metadata_path = metadata_dir / "code_metadata.parquet"

    if not histogram_metadata_path.is_file():
        raise FileNotFoundError(f"Histogram metadata not found at {histogram_metadata_path}")
    if not code_metadata_path.is_file():
        raise FileNotFoundError(f"Code metadata not found at {code_metadata_path}")

    histogram_metadata_df = pl.read_parquet(histogram_metadata_path).with_columns(
        pl.col("histogram").list.eval(1 / (1 + (-pl.element()).exp()))
    )
    code_metadata_df = pl.read_parquet(code_metadata_path)
    assert code_metadata_df["code/vocab_index"].is_sorted()
    code_names = ["PAD//OBSERVATION"] + code_metadata_df["code"].to_list()
    histogram_metadata_df = histogram_metadata_df.with_columns(
        pl.when(pl.col("token_type").eq("HISTOGRAM"))
        .then(pl.lit(code_names))
        .otherwise(pl.lit([]))
        .alias("code")
    )
    # check length
    assert histogram_metadata_df.select(
        pl.col("code").list.eval(pl.len()).eq(pl.col("histogram").list.eval(pl.len())).alias("check")
    )["check"].all()

    metadata_df = histogram_metadata_df

    for name, code_regex in cfg.special_codes.items():
        code_vocab_indices = code_metadata_df.filter(pl.col("code").str.contains(code_regex))[
            "code/vocab_index"
        ].to_list()
        assert len(code_vocab_indices) > 0, f"No code found for {name}"
        special_token_col = f"SPECIAL_TOKEN//{name}"
        special_token_df = metadata_df.with_columns(
            pl.col("histogram")
            .list.eval(pl.element().filter(pl.int_range(0, pl.len()).is_in(code_vocab_indices)))
            .alias(special_token_col)
        )
        # Confirm we loaded a non-zero number of codes and the number of code_vocab indices we filtered is corrected
        is_correct = (
            special_token_df.filter(pl.col("token_type").eq("HISTOGRAM"))
            .select(pl.col(special_token_col).list.eval(pl.len()))
            .explode(special_token_col)
            .select(pl.col(special_token_col).eq(len(code_vocab_indices)))[special_token_col]
        )
        assert is_correct.all(), f"{name} special token not found in histogram metadata"
        assert len(is_correct) == metadata_df.filter(pl.col("token_type").eq("HISTOGRAM")).height, (
            "Number of histograms is not equal to the number of special tokens"
        )

        special_token_df = special_token_df.with_columns(
            (
                1
                - (
                    pl.col(special_token_col)
                    .list.eval(1 - pl.element())  # per-element (1 - p)
                    .list.eval(pl.element().product())  # ∏(1 - p)
                    .list.get(0)
                    .clip(lower_bound=0.0, upper_bound=1.0)
                )
            ).alias(special_token_col)
        )

        special_token_df = special_token_df.with_columns(
            pl.when(pl.col("token_type").eq("HISTOGRAM"))
            .then(pl.concat_list([pl.col("histogram"), pl.col(special_token_col)]))
            .otherwise(pl.col("histogram"))
            .alias("histogram"),
            pl.when(pl.col("token_type").eq("HISTOGRAM"))
            .then(pl.concat_list([pl.col("code"), pl.lit(special_token_col)]))
            .otherwise(pl.col("code"))
            .alias("code"),
        )
        logger.info(
            f"Special token {name} has prevalence {special_token_df[special_token_col].mean() * 100:.4f}%"
        )
        metadata_df = special_token_df.drop(special_token_col)

        logger.info(f"Processed {name} special token")

    def is_in_metadata(path: Path) -> bool:
        if path == metadata_dir or metadata_dir in path.parents:
            return True
        else:
            return False

    histogram_files = [input_dir / cfg.split / f"{i}.parquet" for i in range(cfg.num_trajectories)]

    if not histogram_files:
        logger.warning(f"No histogram parquet files found in {input_dir}.")
        return

    for histogram_path in histogram_files:
        assert histogram_path.is_file(), f"Histogram file {histogram_path} does not exist"
        histogram_df = pl.read_parquet(histogram_path)
        code_df = (
            histogram_df.with_columns(pl.col("code").cast(pl.Int64))
            .rename({"code": "histogram_index"})
            .join(
                metadata_df["histogram/vocab_index", "code", "histogram"],
                left_on="histogram_index",
                right_on="histogram/vocab_index",
            )
            .explode("code", "histogram")
            .with_columns(
                pl.col("code").fill_null(pl.col("token_type")),
                pl.col("histogram").fill_null(pl.lit(1)),
            )
        )
        for i in range(cfg.num_histogram_samples):
            relative_path = histogram_path.relative_to(input_dir)
            relative_path = relative_path.with_stem(
                str(int(relative_path.stem) * cfg.num_histogram_samples + i)
            )
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # if not output_path.exists():
            sampled_code_df = code_df.filter(
                (pl.col("histogram") > pl.lit(np.random.rand(code_df.height))).alias("sampled")
            )
            pa_table = GeneratedTrajectorySchema.align(sampled_code_df.to_arrow())
            pq.write_table(pa_table, output_path)
            logger.info(f"Wrote converted codes to {output_path}")
            # else:
            #     logger.info(f"Already exists, skipping: {output_path}")
