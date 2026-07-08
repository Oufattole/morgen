"""Lightning module for histogram model training.

This module provides PyTorch Lightning wrappers for training the histogram-based phenotype compression models.
It supports both quantizer training (Phase 1) and autoregressive training (Phase 2).
"""

import dataclasses
import logging

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig

from ..model.histogram_model import GapTokenConfig, HistogramModel, ModelMode



class HistogramLightningModule(L.LightningModule):
    """Lightning module for histogram model training.

    This module handles the training logic for both quantizer and autoregressive
    modes of the histogram model.

    Examples:
        >>> from morgen.model.histogram_model import HistogramModel, ModelMode, GapTokenConfig
        >>>
        >>> # Create model
        >>> model_config = {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 4, "vocab_size": 10}
        >>> quantizer_config = {
        ...     "vocab_size": 5, "embedding_dim": 4, "n_embeddings": 8,
        ...     "beta": 1.0, "encoder_hidden_dims": [8], "decoder_hidden_dims": [8], "dropout": 0.0
        ... }
        >>> gap_config = GapTokenConfig()
        >>> model = HistogramModel(model_config, quantizer_config, gap_config, ModelMode.QUANTIZER)
        >>>
        >>> # Create lightning module
        >>> module = HistogramLightningModule(model=model, learning_rate=1e-3)
        >>> module.model.mode
        <ModelMode.QUANTIZER: 'quantizer'>
    """

    def __init__(
        self,
        model: HistogramModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        processor=None,
        max_new_tokens=None,
        inference_mode: str = "generate",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        # Optional sequence processor used in AR mode to turn histograms into code sequences
        self.processor = processor
        self.inference_mode = inference_mode

        self.training_step_outputs = []
        self.max_new_tokens = max_new_tokens

        # Save hyperparameters needed for checkpoint loading
        self.save_hyperparameters(
            {
                "model": {
                    "model_config": model.model_config if hasattr(model, "model_config") else {},
                    "quantizer_config": model.quantizer_config if hasattr(model, "quantizer_config") else {},
                    "gap_token_config": model.gap_token_config if hasattr(model, "gap_token_config") else {},
                    "mode": model.mode.value if hasattr(model, "mode") else "generation",
                    "quantizer_token_start": getattr(model, "quantizer_token_start", 1),
                    "total_vocab_size": getattr(model, "total_vocab_size", None),
                },
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
            }
        )

    def _prepare_code_from_histogram_batch(self, batch):
        """Convert histogram batch into token sequences ready for model input.

        Delegates to processor.process_batch() which handles both fusion and histogram-only modes.

        Examples:
            >>> import torch
            >>> class DummyProcessor:
            ...     special_tokens = {"PAD": 0, "BOS": 11, "EOS": 12}
            ...     do_fusion = False
            ...     def process_batch(self, batch):
            ...         return torch.tensor([[11, 4, 5, 12, 0]])
            >>> from types import SimpleNamespace
            >>> dummy_model = SimpleNamespace(
            ...     model_config={},
            ...     quantizer_config={},
            ...     gap_token_config={},
            ...     mode=SimpleNamespace(value="autoregressive"),
            ... )
            >>> module = HistogramLightningModule(model=dummy_model, processor=DummyProcessor())
            >>> dummy_batch = SimpleNamespace(
            ...     histograms=torch.zeros(1, 3, 2),
            ...     gap_counts=torch.zeros(1, 3, dtype=torch.long),
            ...     anchor_token_indices=None,
            ... )
            >>> module._prepare_code_from_histogram_batch(dummy_batch)
            tensor([[11,  4,  5, 12,  0]])

        Expects `batch.histograms` shaped [B, T, V]. In fusion mode, also expects
        `batch.code` with histogram placeholders and `batch.first_code_indices`.
        """
        # Unified interface handles both fusion and histogram-only modes
        code = self.processor.process_batch(batch)
        # print(f"batch before process_batch:")
        # print(batch.__repr__)
        # # Debug: Check for zeros
        # if (code == 0).any():
        #     zero_positions = (code == 0).nonzero(as_tuple=False)
        #     print(f"\n🔴 FOUND ZEROS after process_batch!")
        #     print(f"   Batch shape: {code.shape}")
        #     print(f"   Zero count: {(code == 0).sum()}")
        #     print(f"   Zero positions (first 10): {zero_positions[:10]}")

        #     # Check batch.code BEFORE quantization
        #     if hasattr(batch, 'code'):
        #         print(f"\n   Checking batch.code BEFORE quantization:")
        #         print(f"   Has zeros: {(batch.code == 0).any()}")
        #         if (batch.code == 0).any():
        #             before_zeros = (batch.code == 0).nonzero(as_tuple=False)
        #             print(f"   Zero positions in batch.code: {before_zeros[:10]}")
        # breakpoint()
        return code

    def _decode_histogram_tokens(self, tokens):
        """Decodes sequence of histogram tokens back to a sequence of codes and timedeltas—in days.

        Expects `histograms` shaped [B, T].
        """
        codes, timedelta_days = self.processor.decode_histogram_tokens(tokens)
        return codes, timedelta_days

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        **kwargs,
    ):
        """Load HistogramLightningModule from checkpoint.

        This method reconstructs the HistogramModel from saved hyperparameters before loading the checkpoint,
        similar to how MorgenModule does it.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        # Handle both old and new checkpoint formats
        assert "model" in hparams
        # New format: model hyperparameters are nested under "model" key
        model_hparams = hparams["model"]
        model_config = model_hparams.get("model_config", {})
        quantizer_config = model_hparams.get("quantizer_config", {})
        gap_config_dict = model_hparams.get("gap_token_config", {})

        # Coerce DictConfig to plain dicts
        if not isinstance(model_config, dict):
            model_config = OmegaConf.to_container(model_config, resolve=True)
        if not isinstance(quantizer_config, dict):
            quantizer_config = OmegaConf.to_container(quantizer_config, resolve=True)
        if isinstance(gap_config_dict, DictConfig):
            gap_config_dict = OmegaConf.to_container(gap_config_dict, resolve=True)
        elif isinstance(gap_config_dict, GapTokenConfig):
            gap_config_dict = dataclasses.asdict(gap_config_dict)

        # Extract only supported fields to avoid Hydra-specific keys
        if isinstance(gap_config_dict, dict):
            gap_config_dict = {k: int(v) for k, v in gap_config_dict.items() if k == "max_gap_length"}
        mode_str = model_hparams.get("mode", "generation")
        quantizer_token_start = model_hparams.get("quantizer_token_start", 1)
        total_vocab_size = model_hparams.get("total_vocab_size")

        # Create GapTokenConfig object
        gap_config = GapTokenConfig(**gap_config_dict) if gap_config_dict else GapTokenConfig()

        # Create ModelMode enum
        mode = ModelMode(mode_str)

        # Reconstruct the HistogramModel
        if quantizer_config['em_params_dir'] is not None:
            logger.warning("############################################################ USING EM QUANTIZER! ############################################################")
            import numpy as np
            from morgen.model.em_quantizer import EMQuantizer

            em_path = Path(quantizer_config['em_params_dir'])
            theta = torch.from_numpy(np.load(em_path / "cluster_theta.npy")).float()
            pi = torch.from_numpy(np.load(em_path / "cluster_pi.npy")).float()
            
            # Instantiate EM quantizer directly
            quantizer = EMQuantizer(theta=theta, pi=pi)
        else:
            quantizer_config.pop('em_params_dir', None)
            logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! USING VQ VAE QUANTIZER! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ
            quantizer = SimplifiedAutoencoderVQ(**quantizer_config)
        model = HistogramModel(
            model_config=model_config,
            quantizer_config=quantizer_config,
            quantizer=quantizer,
            gap_token_config=gap_config,
            mode=mode,
            quantizer_token_start=quantizer_token_start,
            total_vocab_size=total_vocab_size,
        )

        # Load the module with reconstructed model
        return super().load_from_checkpoint(
            ckpt_path,
            model=model,
            learning_rate=hparams.get("learning_rate", 1e-3),
            weight_decay=hparams.get("weight_decay", 0.01),
            warmup_steps=hparams.get("warmup_steps", 1000),
            **kwargs,
        )

    def _predict_generate(self, batch):
        if not self.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be set to a positive integer, but is {self.max_new_tokens}"
            )

        with torch.no_grad():
            code = self._prepare_code_from_histogram_batch(batch)

        # Drop EOS tokens - find the last non-PAD token per sequence
        # In fusion mode, sequences may be right-padded: [BOS, ..., EOS, PAD, PAD]
        # In non-fusion (histogram-only) mode, sequences end with EOS (no padding)
        pad_id = batch.PAD_INDEX
        eos_id = self.processor.special_tokens["EOS"]

        # Find last non-PAD position for each sequence
        non_pad_mask = code != pad_id
        last_non_pad_idx = non_pad_mask.sum(dim=1) - 1  # [batch_size]

        # Check that last non-PAD token is EOS for all sequences
        batch_indices = torch.arange(code.size(0), device=code.device)
        last_non_pad_tokens = code[batch_indices, last_non_pad_idx]

        # In fusion mode, EOS might not be present (dataset handles this differently)
        # Only enforce EOS check in histogram-only mode
        if not self.processor.do_fusion:
            if not (code[:, -1] == eos_id).all().item():
                raise ValueError(
                    "Input code sequences must end with EOS token before padding. "
                    "Found non-EOS tokens at last non-PAD positions."
                )
            # Remove EOS from each sequence so we can generate forward
            code = code[:, :-1]
        else:
            # In fusion mode, check if EOS is present and remove it if so
            raise NotImplementedError(
                "For Early Fusion you should remove the trailing EOS, this is an easy fix her but I just didn't implement it for now"
            )
            eos_mask = last_non_pad_tokens == eos_id
            if eos_mask.any():
                # Remove EOS only from sequences that have it
                code[batch_indices[eos_mask], last_non_pad_idx[eos_mask]] = pad_id

        for_hf = {
            "input_ids": code,
            "attention_mask": (code != batch.PAD_INDEX),
        }

        generation_config = self._build_generation_config(
            pad_token_id=batch.PAD_INDEX,
            eos_token_id=self.processor.special_tokens["EOS"],
            max_new_tokens=self.max_new_tokens,
        )

        output_ids = self.model.ar_model.HF_model.generate(
            for_hf.pop("input_ids"),
            generation_config=generation_config,
            **for_hf,
        )

        input_seq_len = code.shape[1]
        output_ids = output_ids[:, input_seq_len:]

        return output_ids

    @staticmethod
    def _build_generation_config(
        *,
        pad_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
    ) -> GenerationConfig:
        """Return a generation config tuned for histogram autoregressive decoding.

        Examples:
            >>> cfg = HistogramLightningModule._build_generation_config(
            ...     pad_token_id=0,
            ...     eos_token_id=99,
            ...     max_new_tokens=16,
            ...     do_sample=False,
            ...     num_beams=2,
            ...     temperature=0.7,
            ... )
            >>> (cfg.pad_token_id, cfg.eos_token_id, cfg.max_new_tokens)
            (0, 99, 16)
            >>> (cfg.do_sample, cfg.num_beams, cfg.temperature)
            (False, 2, 0.7)
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            pad_token_id=pad_token_id,
            bos_token_id=None,
            eos_token_id=eos_token_id,
        )

    def predict_step(self, batch):
        """Generate histogram token trajectories for a batch."""
        if self.model.ar_model is None:
            raise RuntimeError("Autoregressive model is not available. Ensure model is in GENERATION mode.")
        return self._predict_generate(batch)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.warmup_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """Training step for both quantizer and autoregressive modes."""

        if self.model.mode == ModelMode.QUANTIZER:
            return self._training_step_quantizer(batch, batch_idx)
        elif self.model.mode == ModelMode.AUTOREGRESSIVE:
            return self._training_step_autoregressive(batch, batch_idx)
        else:
            raise NotImplementedError(f"Training not implemented for mode {self.model.mode}")

    def _training_step_quantizer(self, batch, batch_idx):
        """Training step for quantizer mode (Phase 1)."""

        # Extract histograms from batch
        histograms = batch.histograms

        # Forward pass through quantizer
        result = self.model(histograms)

        loss = result["loss"]

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_vq_loss", result["vq_loss"], on_step=True, on_epoch=True)
        self.log("train_reconstruction_loss", result["reconstruction_loss"], on_step=True, on_epoch=True)
        self.log("train_non_empty_ratio", result["num_non_empty"] / result["total_samples"], on_step=True)

        # Store for epoch end
        self.training_step_outputs.append(
            {
                "loss": loss.detach(),
                "vq_loss": result["vq_loss"].detach(),
                "reconstruction_loss": result["reconstruction_loss"].detach(),
                "num_non_empty": result["num_non_empty"],
                "total_samples": result["total_samples"],
            }
        )

        return loss

    def _training_step_autoregressive(self, batch, batch_idx):
        """Training step for autoregressive mode (Phase 2) using MEDSTorchHistogramBatch.

        Assumes `batch.histograms` exists and converts windowed histograms to code sequences
        via the provided processor (quantize + gap tokens), then adds BOS/EOS and pads to
        model max length with PAD=0.
        """

        if not hasattr(batch, "histograms"):
            raise ValueError("Autoregressive training expects MEDSTorchHistogramBatch with 'histograms'.")

        # Turn histogram windows into AR code sequences and align device
        code = self._prepare_code_from_histogram_batch(batch)
        if hasattr(batch.histograms, "device"):
            code = code.to(batch.histograms.device)

        batch.code = code
        batch.PAD_INDEX = 0  # enforce PAD=0
        # Safety: ensure indices are within vocabulary to avoid CUDA device asserts
        vocab_size = self.model.ar_model.vocab_size
        max_id = int(code.max().item()) if code.numel() > 0 else -1
        min_id = int(code.min().item()) if code.numel() > 0 else 0
        if not (min_id >= 0 and max_id < vocab_size):
            raise ValueError(f"Code ids out of range: min={min_id}, max={max_id}, vocab_size={vocab_size}")
        loss, _ = self.model.ar_model._forward(batch)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True)

        self.training_step_outputs.append({"loss": loss.detach(), "perplexity": perplexity.detach()})

        return loss

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        if not self.training_step_outputs:
            return

        if self.model.mode == ModelMode.QUANTIZER:
            # Aggregate quantizer metrics
            avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
            avg_vq_loss = torch.stack([x["vq_loss"] for x in self.training_step_outputs]).mean()
            avg_recon_loss = torch.stack(
                [x["reconstruction_loss"] for x in self.training_step_outputs]
            ).mean()

            total_non_empty = sum(x["num_non_empty"] for x in self.training_step_outputs)
            total_samples = sum(x["total_samples"] for x in self.training_step_outputs)
            non_empty_ratio = total_non_empty / total_samples if total_samples > 0 else 0

            self.log("epoch_train_loss", avg_loss)
            self.log("epoch_train_vq_loss", avg_vq_loss)
            self.log("epoch_train_reconstruction_loss", avg_recon_loss)
            self.log("epoch_non_empty_ratio", non_empty_ratio)

            logger.info(
                f"Epoch {self.current_epoch}: avg_loss={avg_loss:.4f}, "
                f"vq_loss={avg_vq_loss:.4f}, recon_loss={avg_recon_loss:.4f}, "
                f"non_empty_ratio={non_empty_ratio:.3f}"
            )

        elif self.model.mode == ModelMode.AUTOREGRESSIVE:
            # Aggregate AR metrics
            avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
            avg_perplexity = torch.stack([x["perplexity"] for x in self.training_step_outputs]).mean()

            self.log("epoch_train_loss", avg_loss)
            self.log("epoch_train_perplexity", avg_perplexity)

            logger.info(
                f"Epoch {self.current_epoch}: avg_loss={avg_loss:.4f}, perplexity={avg_perplexity:.2f}"
            )

        # Clear outputs for next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if self.model.mode == ModelMode.QUANTIZER:
            return self._validation_step_quantizer(batch, batch_idx)
        elif self.model.mode == ModelMode.AUTOREGRESSIVE:
            return self._validation_step_autoregressive(batch, batch_idx)
        else:
            raise NotImplementedError(f"Validation not implemented for mode {self.model.mode}")

    def _validation_step_quantizer(self, batch, batch_idx):
        """Validation step for quantizer mode."""
        # Similar logic to training step but without gradient updates
        histograms = batch.histograms

        with torch.no_grad():
            result = self.model(histograms)

        loss = result["loss"]

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_vq_loss", result["vq_loss"], on_step=False, on_epoch=True)
        self.log("val_reconstruction_loss", result["reconstruction_loss"], on_step=False, on_epoch=True)

        # Add tuning/loss for early stopping callback
        self.log("tuning/loss", loss, on_step=False, on_epoch=True)

        return loss

    def _validation_step_autoregressive(self, batch, batch_idx):
        """Validation step for autoregressive mode using MEDSTorchHistogramBatch only."""
        if not hasattr(batch, "histograms"):
            raise ValueError("Autoregressive validation expects MEDSTorchHistogramBatch with 'histograms'.")

        with torch.no_grad():
            code = self._prepare_code_from_histogram_batch(batch)
            if hasattr(batch.histograms, "device"):
                code = code.to(batch.histograms.device)
            batch.code = code
            batch.PAD_INDEX = 0
            vocab_size = self.model.ar_model.vocab_size
            max_id = int(code.max().item()) if code.numel() > 0 else -1
            min_id = int(code.min().item()) if code.numel() > 0 else 0
            if not (min_id >= 0 and max_id < vocab_size):
                raise ValueError(
                    f"Code ids out of range (val): min={min_id}, max={max_id}, vocab_size={vocab_size}"
                )
            loss, _ = self.model.ar_model._forward(batch)
        perplexity = torch.exp(loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True)

        # Add tuning/loss for early stopping callback
        self.log("tuning/loss", loss, on_step=False, on_epoch=True)

        return loss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
