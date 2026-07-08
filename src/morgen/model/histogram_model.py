"""Integrated histogram model for phenotype-token compression.

This module provides a unified interface for training and using the histogram-based
phenotype compression system. It integrates:
1. HistogramPytorchDataset for windowed histogram generation
2. SimplifiedAutoencoderVQ for quantization
3. Existing Model (GPT-NeoX) for autoregressive modeling
4. Gap token handling for empty windows

The model operates in three modes:
- "quantizer": Train VQ-VAE on binary histograms (Phase 1)
- "autoregressive": Train AR model on quantized sequences (Phase 2)
- "generation": Generate synthetic trajectories (Phase 3)
"""

import logging
from dataclasses import dataclass, fields
from enum import StrEnum

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .model import Model
from .simplified_quantizer import SimplifiedAutoencoderVQ

logger = logging.getLogger(__name__)


class ModelMode(StrEnum):
    """Operating mode for the histogram model."""

    QUANTIZER = "quantizer"  # Train VQ-VAE on histograms
    AUTOREGRESSIVE = "autoregressive"  # Train AR model on quantized sequences
    GENERATION = "generation"  # Generate synthetic trajectories


@dataclass
class GapTokenConfig:
    """Configuration for gap token vocabulary.

    Uses multiple gap tokens (GAP_1, GAP_2, ..., GAP_max_gap_length) to preserve
    temporal information about gap durations, which is crucial for medical timelines.

    Examples:
        >>> # Gap tokens for different durations
        >>> config = GapTokenConfig(max_gap_length=10)
        >>> config.max_gap_length
        10

        >>> # Support very long gaps (~100 years)
        >>> config2 = GapTokenConfig(max_gap_length=1200)
        >>> config2.max_gap_length
        1200
    """

    max_gap_length: int = 1200  # Maximum consecutive gap windows to support (~100 years)
    gap_token_start: int | None = None  # Starting index for gap tokens (set automatically)
    gap_token_end: int | None = None  # Ending index for gap tokens (set automatically)


class HistogramModel(nn.Module):
    """Integrated model for histogram-based phenotype compression.

    This model orchestrates the entire pipeline from raw medical events to
    quantized phenotype tokens and back to synthetic trajectories.

    Examples:
        >>> # Histogram model supports 3 modes: quantizer, autoregressive, generation
        >>> # Vocabulary is automatically calculated: n_embeddings + max_gap_length + 3
        >>> # PAD token is always at index 0 for autoregressive compatibility
    """

    def __init__(
        self,
        model_config: dict,
        quantizer_config: dict,
        gap_token_config: GapTokenConfig,
        mode: ModelMode = ModelMode.QUANTIZER,
        precision: str = "32-true",
        quantizer: SimplifiedAutoencoderVQ | None = None,
        quantizer_token_start: int = 1,
        total_vocab_size: int | None = None,
    ):
        super().__init__()
        self.mode = mode

        if not isinstance(gap_token_config, GapTokenConfig):
            gap_token_config_dict = (
                OmegaConf.to_container(gap_token_config, resolve=True)
                if isinstance(gap_token_config, DictConfig)
                else gap_token_config
            )
            if isinstance(gap_token_config_dict, dict):
                allowed_keys = {f.name for f in fields(GapTokenConfig)}
                filtered_config = {k: v for k, v in gap_token_config_dict.items() if k in allowed_keys}
                gap_token_config = GapTokenConfig(**filtered_config)
            else:
                raise TypeError("gap_token_config must be a GapTokenConfig or dict-like object")

        self.gap_token_config = gap_token_config

        # Persist configs for potential reuse
        self.model_config = model_config
        self.quantizer_config = (
            OmegaConf.to_container(quantizer_config, resolve=True)
            if not isinstance(quantizer_config, dict)
            else quantizer_config
        )

        # Initialize quantizer (always needed). Prefer provided quantizer (e.g., from checkpoint)
        self.quantizer = quantizer if quantizer is not None else SimplifiedAutoencoderVQ(**quantizer_config)
        # Ensure quantizer_config reflects the actual codebook size for checkpoint reproducibility
        try:
            actual_ne = int(self.quantizer.n_embeddings)
            self.quantizer_config["n_embeddings"] = actual_ne
        except Exception:
            pass

        # Calculate vocabulary sizes
        self._setup_vocabulary(
            model_config,
            quantizer_config,
            quantizer_token_start=quantizer_token_start,
            total_vocab_size=total_vocab_size,
        )

        # Initialize autoregressive model for AR and generation modes
        if mode in [ModelMode.AUTOREGRESSIVE, ModelMode.GENERATION]:
            # Update model config with correct vocab size
            gpt_cfg = model_config.copy()
            gpt_cfg["vocab_size"] = self.total_vocab_size

            self.ar_model = Model(gpt_kwargs=gpt_cfg, precision=precision)
        else:
            self.ar_model = None

    @property
    def hidden_size(self) -> int:
        """Hidden size of the autoregressive histogram model."""

        if self.ar_model is None:
            raise RuntimeError("Autoregressive model is not initialised in the current mode.")
        return self.ar_model.HF_model_config.hidden_size

    def embed_tokens(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Return input embeddings for histogram token ids.

        Args:
            token_ids: Tensor of token indices in the histogram vocabulary.

        Returns:
            Embedded representations with shape ``token_ids.shape + (hidden_size,)``.
        """

        if self.ar_model is None:
            raise RuntimeError("Autoregressive model is required for token embeddings.")

        embedding_layer = self.ar_model.HF_model.get_input_embeddings()
        return embedding_layer(token_ids)

    def _setup_vocabulary(
        self,
        model_config: dict,
        quantizer_config: dict,
        *,
        quantizer_token_start: int = 1,
        total_vocab_size: int | None = None,
    ):
        """Setup vocabulary mapping for quantized tokens and gap tokens.

        Vocabulary layout (PAD at 0 for autoregressive compatibility):
        - Token 0: PAD token (for cross_entropy ignore_index compatibility)
        - Tokens 1 to n_embeddings: Quantized histogram tokens
        - Tokens n_embeddings+1 to n_embeddings+max_gap_length: Gap tokens (GAP_1, GAP_2, ...)
        - Token n_embeddings+max_gap_length+1: Anchor token marking discharge-aligned window
        - Token n_embeddings+max_gap_length+2: BOS token
        - Token n_embeddings+max_gap_length+3: EOS token
        """
        # Use the actual instantiated quantizer's codebook size to avoid config/checkpoint mismatches
        self.quantizer_vocab_size = self.quantizer.n_embeddings

        # Always use multiple gap tokens to preserve temporal information
        self.gap_vocab_size = self.gap_token_config.max_gap_length

        # Token ranges (shifted by quantizer_token_start to make room for PAD at 0)
        self.quantizer_token_start = quantizer_token_start
        self.quantizer_token_end = self.quantizer_token_start + self.quantizer_vocab_size - 1
        self.gap_token_start = self.quantizer_token_end + 1
        self.gap_token_end = self.gap_token_start + self.gap_vocab_size - 1

        # Populate the config fields
        self.gap_token_config.gap_token_start = self.gap_token_start
        self.gap_token_config.gap_token_end = self.gap_token_end

        # Special tokens - PAD must be 0 for autoregressive model compatibility
        anchor_token_id = self.gap_token_end + 1
        self.special_tokens = {
            "PAD": 0,  # Always 0 for cross_entropy ignore_index compatibility
            "ANCHOR": anchor_token_id,
            "BOS": anchor_token_id + 1,
            "EOS": anchor_token_id + 2,
        }

        eos_plus_one = self.special_tokens["EOS"] + 1
        if total_vocab_size is not None:
            if total_vocab_size < eos_plus_one:
                raise ValueError(
                    "total_vocab_size must accommodate the EOS token: "
                    f"got {total_vocab_size}, required at least {eos_plus_one}"
                )
            self.total_vocab_size = total_vocab_size
        else:
            self.total_vocab_size = eos_plus_one

        logger.info(
            f"Vocabulary setup: quantizer={self.quantizer_vocab_size}, "
            f"gap={self.gap_vocab_size}, special={len(self.special_tokens)}, "
            f"total={self.total_vocab_size}"
        )

    def decode_gap_token(self, token_idx: int) -> int | None:
        """Decode gap token index to gap length.

        Args:
            token_idx: Token index

        Returns:
            Gap length if token is a gap token, None otherwise

        Examples:
            >>> # Decodes gap tokens back to gap lengths
            >>> # Gap tokens: GAP_1, GAP_2, ..., GAP_max_gap_length
            >>> # Returns None for non-gap tokens
        """
        gap_start = self.gap_token_start
        gap_end = gap_start + self.gap_vocab_size

        if gap_start <= token_idx < gap_end:
            return (token_idx - gap_start) + 1  # GAP_1, GAP_2, etc.
        return None

    def is_gap_token(self, token_idx: int) -> bool:
        """Check if token index corresponds to a gap token.

        Examples:
            >>> # Gap tokens are in range [gap_start, gap_start + gap_vocab_size)
            >>> # Quantizer tokens are [0, n_embeddings)
            >>> # Special tokens (PAD, BOS, EOS) are after gap tokens
        """
        gap_start = self.gap_token_start
        gap_end = gap_start + self.gap_vocab_size
        return gap_start <= token_idx < gap_end

    def forward_quantizer_mode(self, batch_histograms: torch.Tensor) -> dict:
        """Forward pass in quantizer training mode.

        Args:
            batch_histograms: Binary histograms [batch_size, seq_len, vocab_size]

        Returns:
            Quantizer training results
        """
        # Reshape to [batch_size * seq_len, vocab_size] for quantizer
        batch_size, seq_len, vocab_size = batch_histograms.shape
        flat_histograms = batch_histograms.view(-1, vocab_size)

        # Filter out empty histograms (all zeros) for training
        non_empty_mask = flat_histograms.sum(dim=1) > 0
        if non_empty_mask.sum() == 0:
            # All histograms are empty, return dummy loss
            return {
                "loss": torch.tensor(0.0, device=batch_histograms.device, requires_grad=True),
                "vq_loss": torch.tensor(0.0, device=batch_histograms.device),
                "reconstruction_loss": torch.tensor(0.0, device=batch_histograms.device),
                "num_non_empty": 0,
                "total_samples": flat_histograms.shape[0],
            }

        non_empty_histograms = flat_histograms[non_empty_mask]

        # Train quantizer only on non-empty histograms
        result = self.quantizer(non_empty_histograms)

        return {
            "loss": result["total_loss"],
            "vq_loss": result["vq_loss"],
            "reconstruction_loss": result["reconstruction_loss"],
            "num_non_empty": non_empty_mask.sum().item(),
            "total_samples": flat_histograms.shape[0],
            "quantizer_result": result,
        }

    def forward_autoregressive_mode(self, batch) -> dict[str, torch.Tensor]:
        """Forward pass for autoregressive training mode.

        Uses real MEDSTorchHistogramBatch directly from dataloader - no manual batch creation!

        Args:
            batch: Real MEDSTorchHistogramBatch from histogram dataloader

        Returns:
            Dictionary with loss and other training outputs
        """
        if self.ar_model is None:
            raise RuntimeError("Autoregressive model not initialized for this mode")

        # Use real MEDSTorchHistogramBatch directly - exactly like pretrain does!
        # No manual MEDSTorchBatch creation needed at all

        # MEDSTorchHistogramBatch inherits from MEDSTorchBatch, so we can use it directly
        # Set the PAD_INDEX as a class attribute if needed
        batch.PAD_INDEX = self.special_tokens["PAD"]

        # Forward through autoregressive model using the real batch directly
        loss, outputs = self.ar_model._forward(batch)

        return {"loss": loss, "logits": outputs.logits, "ar_outputs": outputs}

    def forward(self, batch_data: torch.Tensor | dict) -> dict:
        """Main forward pass - delegates to mode-specific methods.

        Args:
            batch_data: Input data (format depends on mode)

        Returns:
            Mode-specific results
        """
        if self.mode == ModelMode.QUANTIZER:
            if isinstance(batch_data, dict):
                histograms = batch_data["histograms"]
            else:
                histograms = batch_data
            return self.forward_quantizer_mode(histograms)

        elif self.mode == ModelMode.AUTOREGRESSIVE:
            if isinstance(batch_data, dict):
                sequences = batch_data["token_sequences"]
            else:
                sequences = batch_data
            return self.forward_autoregressive_mode(sequences)

        else:  # GENERATION mode
            # For generation, we use the autoregressive model to predict next tokens
            # Input should be token sequences (not histograms)
            if isinstance(batch_data, torch.Tensor):
                sequences = batch_data
            else:
                raise ValueError("Generation mode expects token sequences as input, not histogram batches")

            return self.forward_autoregressive_mode(sequences)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
