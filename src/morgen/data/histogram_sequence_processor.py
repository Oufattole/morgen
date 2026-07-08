"""Histogram sequence processor for autoregressive training.

This module handles the conversion of histogram sequences into quantized token sequences
with gap tokens for autoregressive model training (Phase 2).

Key functionality:
- Load histogram sequences from HistogramPytorchDataset (EmptyWindowMode.SINGLE_GAP)
- Quantize non-empty histograms using pretrained VQ-VAE
- Insert gap tokens for empty windows
- Create proper token sequences for autoregressive training
"""

import logging
from pathlib import Path

import torch

from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ

logger = logging.getLogger(__name__)


class HistogramSequenceProcessor:
    """Processes histogram sequences into quantized token sequences.

    This class handles the conversion pipeline from histogram windows to
    quantized tokens with gap handling for autoregressive training.

    Examples:
        >>> from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ
        >>> # Create processor with quantizer object
        >>> processor = HistogramSequenceProcessor(
        ...     quantizer=SimplifiedAutoencoderVQ(
        ...         vocab_size=10, embedding_dim=8, n_embeddings=16, beta=1.0,
        ...         encoder_hidden_dims=[16], decoder_hidden_dims=[16], dropout=0.0
        ...     ),
        ...     vocab_size=10,
        ...     max_gap_length=5
        ... )
        >>> processor.vocab_size
        10

        >>> # Process sample histogram sequences
        >>> batch_size, seq_len, vocab_size = 2, 4, 10
        >>> histograms = torch.randint(0, 3, (batch_size, seq_len, vocab_size)).float()
        >>> gap_mask = torch.tensor([[False, True, False, False], [True, False, True, False]])
        >>> gap_counts = torch.tensor([[0, 2, 0, 0], [3, 0, 1, 0]])
        >>>
        >>> token_sequences = processor.process_histogram_batch(histograms, gap_counts)
        >>> token_sequences.shape[0] == batch_size
        True
        >>> token_sequences.shape[1] == seq_len
        True
    """

    def __init__(
        self,
        quantizer_checkpoint_path: str | None = None,
        quantizer: SimplifiedAutoencoderVQ | None = None,
        vocab_size: int = 100,
        max_gap_length: int = 10,
        gap_token_strategy: str = "multiple",
        do_fusion: bool = False,
    ):
        """Initialize histogram sequence processor.

        Args:
            quantizer_checkpoint_path: Path to pretrained quantizer checkpoint
            quantizer: Pretrained quantizer (alternative to checkpoint path)
            vocab_size: Histogram vocabulary size
            max_gap_length: Maximum gap length to support
            gap_token_strategy: Gap token strategy (currently only "multiple" supported)
            do_fusion: Whether the processor will emit fused histogram+code sequences
        """
        self.vocab_size = vocab_size
        self.max_gap_length = max_gap_length
        self.gap_token_strategy = gap_token_strategy
        if do_fusion:
            self.quantizer_token_start = 309
        else:
            self.quantizer_token_start = 1
        self.do_fusion = do_fusion
        self._fusion_eos_token_id: int | None = None

        # Load or use provided quantizer
        if quantizer is not None:
            self.quantizer = quantizer
        elif quantizer_checkpoint_path is not None:
            self.quantizer = self._load_quantizer(quantizer_checkpoint_path)
        else:
            raise ValueError("Either quantizer_checkpoint_path or quantizer must be provided")

        # Set up vocabulary indices
        self._setup_vocabulary()

        logger.info(
            f"Initialized HistogramSequenceProcessor with vocab_size={vocab_size}, "
            f"gap_strategy={gap_token_strategy}, max_gap_length={max_gap_length}"
        )

    def _load_quantizer(self, checkpoint_path: str) -> SimplifiedAutoencoderVQ:
        """Load pretrained quantizer from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Quantizer checkpoint not found: {checkpoint_path}")

        # Load checkpoint (weights_only=False for trusted internal checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract model state dict (assuming it's saved in Lightning format)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

            # Find quantizer keys with flexible prefix matching
            quantizer_keys = [k for k in state_dict.keys() if "quantizer" in k]
            if not quantizer_keys:
                raise ValueError(
                    f"No quantizer keys found in checkpoint. Available keys: {list(state_dict.keys())[:10]}"
                )

            # Determine the actual prefix used (keys start with 'quantizer.')
            first_quantizer_key = quantizer_keys[0]
            if first_quantizer_key.startswith("quantizer."):
                prefix = "quantizer."
                logger.info(f"Removing quantizer prefix: '{prefix}'")

                # Extract quantizer state dict by removing the quantizer prefix
                state_dict = {
                    k.replace(prefix, "", 1): v  # Only replace first occurrence
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
            else:
                # Fallback: use all keys (if quantizer is at root level)
                logger.info("No quantizer prefix found, using all keys")

        else:
            state_dict = checkpoint

        # Extract model configuration from checkpoint or use provided config
        # First, try to extract configuration from the checkpoint structure
        embedding_dim = self._extract_embedding_dim_from_state_dict(state_dict)
        n_embeddings = self._extract_n_embeddings_from_state_dict(state_dict)
        encoder_hidden_dims, decoder_hidden_dims = self._extract_hidden_dims_from_state_dict(state_dict)

        # Store hidden dimensions as attributes for later access
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims

        logger.info(f"Creating quantizer with embedding_dim={embedding_dim}, n_embeddings={n_embeddings}")
        logger.info(f"Hidden dims - encoder: {encoder_hidden_dims}, decoder: {decoder_hidden_dims}")

        max_count = SimplifiedAutoencoderVQ.infer_max_count_from_state_dict(state_dict, self.vocab_size)

        quantizer = SimplifiedAutoencoderVQ(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            n_embeddings=n_embeddings,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            beta=1.0,
            dropout=0.0,
            max_count=max_count,
        )

        # Load state dict
        quantizer.load_state_dict(state_dict)
        quantizer.eval()

        logger.info(f"Loaded pretrained quantizer from {checkpoint_path}")
        return quantizer

    def _extract_embedding_dim_from_state_dict(self, state_dict: dict) -> int:
        """Extract embedding dimension from quantizer state dict.

        The embedding dimension can be inferred from the quantizer.embedding.weight shape.

        Args:
            state_dict: Model state dictionary

        Returns:
            int: Embedding dimension

        Examples:
            >>> processor = HistogramSequenceProcessor.__new__(HistogramSequenceProcessor)
            >>> # Test with typical checkpoint structure
            >>> import torch
            >>> state_dict = {
            ...     'quantizer.embedding.weight': torch.randn(128, 64),  # [n_embeddings, embedding_dim]
            ...     'encoder.encoder.network.0.weight': torch.randn(64, 8)
            ... }
            >>> processor._extract_embedding_dim_from_state_dict(state_dict)
            64

            >>> # Test with different dimensions
            >>> state_dict = {
            ...     'quantizer.embedding.weight': torch.randn(512, 16),  # [n_embeddings, embedding_dim]
            ... }
            >>> processor._extract_embedding_dim_from_state_dict(state_dict)
            16
        """
        if "quantizer.embedding.weight" in state_dict:
            embedding_weight = state_dict["quantizer.embedding.weight"]
            return embedding_weight.shape[1]  # [n_embeddings, embedding_dim]
        else:
            # Fallback: try to infer from encoder layers
            for key in state_dict:
                if "encoder.encoder.network.0.weight" in key:
                    return state_dict[key].shape[1]  # Input dimension to first encoder layer

            # Last resort: use default
            logger.warning("Could not extract embedding_dim from checkpoint, using default 64")
            return 64

    def _extract_n_embeddings_from_state_dict(self, state_dict: dict) -> int:
        """Extract number of embeddings from quantizer state dict.

        The number of embeddings can be inferred from the quantizer.embedding.weight shape.

        Args:
            state_dict: Model state dictionary

        Returns:
            int: Number of embeddings

        Examples:
            >>> processor = HistogramSequenceProcessor.__new__(HistogramSequenceProcessor)
            >>> # Test with typical checkpoint structure
            >>> import torch
            >>> state_dict = {
            ...     'quantizer.embedding.weight': torch.randn(128, 64),  # [n_embeddings, embedding_dim]
            ...     'encoder.encoder.network.0.weight': torch.randn(64, 8)
            ... }
            >>> processor._extract_n_embeddings_from_state_dict(state_dict)
            128

            >>> # Test with different dimensions
            >>> state_dict = {
            ...     'quantizer.embedding.weight': torch.randn(512, 16),  # [n_embeddings, embedding_dim]
            ... }
            >>> processor._extract_n_embeddings_from_state_dict(state_dict)
            512
        """
        if "quantizer.embedding.weight" in state_dict:
            embedding_weight = state_dict["quantizer.embedding.weight"]
            return embedding_weight.shape[0]  # [n_embeddings, embedding_dim]
        else:
            # Last resort: use default
            logger.warning("Could not extract n_embeddings from checkpoint, using default 128")
            return 128

    def _extract_hidden_dims_from_state_dict(self, state_dict: dict) -> tuple[list[int], list[int]]:
        """Extract encoder and decoder hidden dimensions from state dict.

        Infers the network architecture from the layer weight shapes.

        Args:
            state_dict: Model state dictionary

        Returns:
            tuple: (encoder_hidden_dims, decoder_hidden_dims) as lists of integers

        Examples:
            >>> processor = HistogramSequenceProcessor.__new__(HistogramSequenceProcessor)
            >>> import torch
            >>> # Test with actual checkpoint structure (vocab_size=8, embedding_dim=64)
            >>> state_dict = {
            ...     'encoder.encoder.network.0.weight': torch.randn(64, 8),     # 8 -> 64 (first layer)
            ...     'encoder.encoder.network.4.weight': torch.randn(32, 64),    # 64 -> 32 (second layer)
            ...     'encoder.encoder.network.8.weight': torch.randn(16, 32),    # 32 -> 16 (third layer)
            ...     'encoder.encoder.network.12.weight': torch.randn(64, 16),   # 16 -> 64 (to embedding)
            ...     'decoder.decoder.network.0.weight': torch.randn(16, 64),    # 64 -> 16 (first layer)
            ...     'decoder.decoder.network.4.weight': torch.randn(32, 16),    # 16 -> 32 (second layer)
            ...     'decoder.decoder.network.8.weight': torch.randn(64, 32),    # 32 -> 64 (third layer)
            ...     'decoder.decoder.network.12.weight': torch.randn(8, 64),    # 64 -> 8 (to vocab)
            ...     'quantizer.embedding.weight': torch.randn(128, 64)
            ... }
            >>> encoder_dims, decoder_dims = processor._extract_hidden_dims_from_state_dict(state_dict)
            >>> encoder_dims
            [64, 32, 16]
            >>> decoder_dims
            [16, 32, 64]
        """
        # Extract encoder dimensions
        encoder_dims = []
        decoder_dims = []

        # Find encoder linear layers (only even indices: 0, 4, 8, 12) - odd indices are batch norm
        encoder_keys = [
            k
            for k in state_dict
            if k.startswith("encoder.encoder.network.")
            and k.endswith(".weight")
            and int(k.split(".")[-2]) % 4 == 0
        ]  # Only even indices that are divisible by 4
        encoder_keys = sorted(encoder_keys, key=lambda x: int(x.split(".")[-2]))  # Sort by layer index
        for key in encoder_keys:
            weight = state_dict[key]
            out_dim = weight.shape[0]  # Output dimension
            encoder_dims.append(out_dim)

        # Find decoder linear layers (only even indices: 0, 4, 8, 12) - odd indices are batch norm
        decoder_keys = [
            k
            for k in state_dict
            if k.startswith("decoder.decoder.network.")
            and k.endswith(".weight")
            and int(k.split(".")[-2]) % 4 == 0
        ]  # Only even indices that are divisible by 4
        decoder_keys = sorted(decoder_keys, key=lambda x: int(x.split(".")[-2]))  # Sort by layer index
        for key in decoder_keys:
            weight = state_dict[key]
            out_dim = weight.shape[0]  # Output dimension
            decoder_dims.append(out_dim)

        # Remove the last layer (output layer) from each to get hidden dims
        if encoder_dims:
            encoder_hidden_dims = encoder_dims[:-1]  # Remove last layer (to embedding)
        else:
            encoder_hidden_dims = [64, 32, 16]  # Default

        if decoder_dims:
            decoder_hidden_dims = decoder_dims[:-1]  # Remove last layer (to vocab)
        else:
            decoder_hidden_dims = [16, 32, 64]  # Default

        # Handle empty case
        if not encoder_hidden_dims:
            encoder_hidden_dims = [64, 32, 16]
        if not decoder_hidden_dims:
            decoder_hidden_dims = [16, 32, 64]

        return encoder_hidden_dims, decoder_hidden_dims

    def _setup_vocabulary(self):
        """Setup vocabulary indices for tokens.

        Vocabulary layout (PAD at 0 for autoregressive compatibility):
        - Token 0: PAD token
        - Tokens 1 to n_embeddings: Quantized histogram tokens
        - Tokens n_embeddings+1 to n_embeddings+max_gap_length: Gap tokens
        - Token n_embeddings+max_gap_length+1: Anchor token
        - Token n_embeddings+max_gap_length+2: BOS token
        - Token n_embeddings+max_gap_length+3: EOS token
        """
        # Quantized histogram tokens occupy [quantizer_token_start, quantizer_token_end]
        self.quantizer_vocab_size = self.quantizer.n_embeddings
        self.quantizer_token_end = self.quantizer_token_start + self.quantizer_vocab_size - 1

        # Gap tokens start immediately after histogram tokens
        self.gap_vocab_size = self.max_gap_length
        self.gap_token_start = self.quantizer_token_end + 1
        self.gap_token_end = self.gap_token_start + self.gap_vocab_size - 1

        # Special tokens - PAD at 0 for autoregressive model compatibility
        anchor_token_id = self.gap_token_end + 1
        self.special_tokens = {
            "PAD": 0,  # Always 0 for cross_entropy ignore_index compatibility
            "ANCHOR": anchor_token_id,
            "BOS": anchor_token_id + 1,
            "EOS": anchor_token_id + 2,
        }

        self.total_vocab_size = self.special_tokens["EOS"] + 1

        logger.info(
            f"Vocabulary setup: quantizer={self.quantizer_vocab_size}, "
            f"gap={self.gap_vocab_size}, special={len(self.special_tokens)}, "
            f"total={self.total_vocab_size}"
        )

    def process_batch(self, batch):
        """Unified batch processing that handles both fusion and histogram-only modes.

        In histogram-only mode:
            - Quantizes histograms
            - Adds gap/anchor tokens
            - Wraps with BOS/EOS

        In fusion mode:
            - Takes pre-fused code sequence from batch (already has BOS/EOS from dataset)
            - Quantizes histograms and fills them into -1 placeholder positions
            - No BOS/EOS wrapping needed (already in the code sequence)

        Args:
            batch: MEDSTorchHistogramBatch containing histograms, gap_counts, and optionally
                   code sequences with placeholders (fusion mode)

        Returns:
            torch.Tensor: Token sequences ready for model input
                - Histogram-only mode: [batch, seq_len] with BOS/EOS
                - Fusion mode: [batch, seq_len] with codes and fused histograms

        Examples:
            >>> from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ
            >>> quantizer = SimplifiedAutoencoderVQ(
            ...     vocab_size=3, embedding_dim=4, n_embeddings=3, beta=0.25,
            ...     encoder_hidden_dims=[4], decoder_hidden_dims=[4], dropout=0.0
            ... )
            >>> processor = HistogramSequenceProcessor(quantizer=quantizer, vocab_size=3)
            >>> # Histogram-only mode
            >>> from types import SimpleNamespace
            >>> batch = SimpleNamespace(
            ...     histograms=torch.tensor([[[0., 1., 0.], [1., 0., 0.]]]),
            ...     gap_counts=torch.tensor([[0, 0]]),
            ...     anchor_token_indices=None,
            ...     PAD_INDEX=0
            ... )
            >>> result = processor.process_batch(batch)
            >>> result.shape
            torch.Size([1, 4])
            >>> result[0, 0].item() == processor.special_tokens["BOS"]
            True
            >>> result[0, -1].item() == processor.special_tokens["EOS"]
            True
        """
        if self.do_fusion:
            # Fusion mode: batch.code already has BOS/EOS, just fill histogram placeholders
            tokens = self.quantize_and_fuse_histograms(
                histogram_sequence=batch.histograms,
                code_sequence=batch.code,
                first_code_indices=batch.first_code_indices,
                gap_counts=batch.gap_counts,
                anchor_token_indices=batch.anchor_token_indices,
            )
            return tokens
        else:
            # Histogram-only mode: quantize histograms and wrap with BOS/EOS
            tokens = self.process_histogram_batch(
                batch.histograms,
                batch.gap_counts,
                batch.anchor_token_indices,
            )
            return self.create_code_with_bos_eos(tokens)

    def is_gap_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """Check if token index corresponds to a gap token.

        Examples:
        >>> processor = HistogramSequenceProcessor(
        ...     quantizer=SimplifiedAutoencoderVQ(
        ...         vocab_size=10, embedding_dim=4, n_embeddings=4, beta=1.0,
        ...         encoder_hidden_dims=[8], decoder_hidden_dims=[8], dropout=0.0
        ...     ),
        ...     vocab_size=10, max_gap_length=3, gap_token_strategy="multiple"
        ... )
            >>> processor.is_gap_token(torch.tensor([5])).item()   # GAP_1 (gap tokens start at 5)
            True
            >>> processor.is_gap_token(torch.tensor([7])).item()   # GAP_3
            True
            >>> processor.is_gap_token(torch.tensor([8])).item()   # BOS token
            False
            >>> processor.is_gap_token(torch.tensor([2])).item()   # Quantizer token
            False
            >>> processor.is_gap_token(torch.tensor([0])).item()   # PAD token
            False
        """
        gap_start = self.gap_token_start
        gap_end = gap_start + self.gap_vocab_size
        return (gap_start <= tokens) & (tokens < gap_end)

    def is_histogram_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """Check if token index corresponds to a gap token.

        Examples:
        >>> processor = HistogramSequenceProcessor(
        ...     quantizer=SimplifiedAutoencoderVQ(
        ...         vocab_size=10, embedding_dim=4, n_embeddings=4, beta=1.0,
        ...         encoder_hidden_dims=[8], decoder_hidden_dims=[8], dropout=0.0
        ...     ),
        ...     vocab_size=10, max_gap_length=3, gap_token_strategy="multiple"
        ... )
            >>> processor.is_histogram_token(torch.tensor([5])).item()   # GAP_1 (gap tokens start at 5)
            False
            >>> processor.is_histogram_token(torch.tensor([7])).item()   # GAP_3
            False
            >>> processor.is_histogram_token(torch.tensor([8])).item()   # BOS token
            False
            >>> processor.is_histogram_token(torch.tensor([2])).item()   # Quantizer token
            True
            >>> processor.is_histogram_token(torch.tensor([0])).item()   # PAD token
            False
        """
        return (tokens >= 1) & (tokens <= self.quantizer_vocab_size)

    def decode_histograms(self, device: torch.device) -> list[torch.Tensor]:
        return self.quantizer.decode_indices(torch.arange(0, self.quantizer_vocab_size, device=device))

    def quantize_histogram(self, histogram: torch.Tensor) -> torch.Tensor:
        """Quantize a single histogram to get discrete token index.

        Args:
            histogram: Histogram tensor [vocab_size]

        Returns:
            Quantized token index

        Examples:
            >>> processor = HistogramSequenceProcessor(
            ...     quantizer=SimplifiedAutoencoderVQ(
            ...         vocab_size=5, embedding_dim=4, n_embeddings=8, beta=1.0,
            ...         encoder_hidden_dims=[8], decoder_hidden_dims=[8], dropout=0.0
            ...     ),
            ...     vocab_size=5
            ... )
            >>> hist = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)
            >>> tokens = processor.quantize_histogram(hist)
            >>> isinstance(tokens, torch.Tensor)
            True
            >>> 1 <= tokens[0].item() <= 8  # Should be within quantizer token range (1-8)
            True
        """
        # Ensure histogram is 2D for quantizer
        if len(histogram.shape) == 1:
            histogram = histogram.unsqueeze(0)  # [1, vocab_size]

        with torch.no_grad():
            # Encode to continuous latent
            z_e = self.quantizer.encode(histogram)

            # Quantize to get discrete code
            z_q, vq_loss, indices = self.quantizer.quantize(z_e)

            # Return the quantized index shifted by quantizer_token_start.
            # Raw quantizer returns 0-based indices, we adjust to the configured vocab start.
            return indices + self.quantizer_token_start

    def process_histogram_batch(
        self,
        histogram_sequence: torch.Tensor,
        gap_counts: torch.Tensor | None = None,
        anchor_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a single histogram sequence into quantized tokens.

        Args:
            histogram_sequence: Sequence of histograms [seq_len, vocab_size]
            gap_mask: Boolean mask indicating gap windows [seq_len]
            gap_counts: Optional tensor with run-length encoded gap counts [seq_len]
            anchor_token_indices: Optional tensor containing indices of anchor tokens per sequence,
                padded with -1

        Returns:
            Quantized token sequence [output_seq_len]

        Examples:
            >>> processor = HistogramSequenceProcessor(
            ...     quantizer=SimplifiedAutoencoderVQ(
            ...         vocab_size=5, embedding_dim=4, n_embeddings=8, beta=1.0,
            ...         encoder_hidden_dims=[8], decoder_hidden_dims=[8], dropout=0.0
            ...     ),
            ...     vocab_size=5, max_gap_length=3
            ... )
            >>> seq_len, vocab_size = 4, 5
            >>> histograms = torch.tensor([
            ...     [
            ...     [0., 1., 0., 1., 0.],
            ...     [0., 0., 0., 0., 0.],
            ...     [1., 0., 1., 0., 0.],
            ...     [0., 1., 0., 0., 1.],
            ...     [0., 0., 0., 0., 0.],
            ... ],
            ... [
            ...     [0., 1., 0., 1., 0.],
            ...     [0., 0., 0., 0., 0.],
            ...     [1., 0., 1., 0., 0.],
            ...     [0., 1., 0., 0., 1.],
            ...     [0., 0., 0., 0., 0.],
            ... ]
            ... ], dtype=torch.float32)
            >>> gap_counts = torch.tensor([[0, 0, 2, 0, 0], [0, 0, 2, 0, 0]])
            >>>
            >>> tokens = processor.process_histogram_batch(histograms, gap_counts)
            >>> tokens.shape
            torch.Size([2, 5])
            >>> (tokens[0] == tokens[1]).all().item()
            True
            >>> tokens = tokens[0]
            >>> ((tokens[[0,3]] > 0) & (tokens[[0,3]] <= 8)).all().item()
            True
            >>> tokens[2].item()
            10
            >>> (tokens[[1,4]] == 0).all().item()
            True

            >>> anchor_indices = torch.tensor([[2], [1]])
            >>> tokens_with_anchor = processor.process_histogram_batch(histograms, gap_counts, anchor_indices)
            >>> anchor_id = processor.special_tokens["ANCHOR"]
            >>> tokens_with_anchor[0].eq(anchor_id)
            tensor([False, False,  True, False, False])
            >>> tokens_with_anchor[1].eq(anchor_id)
            tensor([False,  True, False, False, False])
        """
        B, T, V = histogram_sequence.shape
        tokens = self.quantize_histogram(histogram_sequence.reshape(B * T, V)).reshape(B, T)
        gap_codes = (gap_counts - 1 + self.gap_token_start).clip(max=self.gap_token_end)
        gap_mask = gap_counts > 0
        tokens = torch.where(gap_mask, gap_codes, tokens)
        pad_tokens = torch.full_like(tokens, self.special_tokens["PAD"])
        is_pad = (histogram_sequence.sum(dim=-1) == 0) & ~gap_mask
        tokens = torch.where(is_pad, pad_tokens, tokens)

        if anchor_token_indices is not None and anchor_token_indices.numel() > 0:
            anchor_token_indices = anchor_token_indices.to(tokens.device)
            anchor_token_id = self.special_tokens["ANCHOR"]
            valid_mask = anchor_token_indices >= 0
            if valid_mask.any():
                batch_indices, position_indices = valid_mask.nonzero(as_tuple=True)
                anchor_positions = anchor_token_indices[batch_indices, position_indices]
                tokens[batch_indices, anchor_positions] = anchor_token_id
        return tokens

    def create_code_with_bos_eos(self, token_sequences: torch.Tensor) -> torch.Tensor:
        """Build a single code sequence that includes BOS at the start and EOS at the end.

        This utility prepares model-ready input where the model learns to predict EOS. It pads
        with PAD=0 to `max_seq_len` if provided; otherwise, pads to the longest sequence.

        Args:
            token_sequences: Tensor of shape [batch_size, seq_len] containing content tokens
                followed by PAD (0). No BOS/EOS present.

        Returns:
            code: Tensor of shape [batch_size, final_len] that starts with BOS, contains the content
                tokens, ends with EOS, and is right-padded with PAD (0).

        Examples:
            >>> processor = HistogramSequenceProcessor(
            ...     quantizer=SimplifiedAutoencoderVQ(
            ...         vocab_size=5, embedding_dim=4, n_embeddings=8, beta=1.0,
            ...         encoder_hidden_dims=[8], decoder_hidden_dims=[8], dropout=0.0
            ...     ),
            ...     vocab_size=5
            ... )
            >>> toks = torch.tensor([[4, 5, 6], [7, 8, 0]])  # right padding in second row
            >>> code = processor.create_code_with_bos_eos(toks)
            >>> code.shape
            torch.Size([2, 5])
            >>> # First row layout: BOS, 4, 5, 6, EOS
            >>> # Second row layout: BOS, 7, 8, EOS, PAD
            >>> code
            tensor([[20,  4,  5,  6, 21],
                    [20,  7,  8, 21,  0]])
            >>> left_padded = torch.tensor([[3, 4, 5], [0, 0, 7]])
            >>> left_code = processor.create_code_with_bos_eos(left_padded)
            >>> # First row layout: PAD, BOS, 4, 5, EOS
            >>> # Second row layout: PAD, PAD, BOS, 7, EOS
            >>> left_code
            tensor([[20,  3,  4,  5, 21],
                    [ 0,  0, 20,  7, 21]])
            >>> # Multiple Trailing Pads:
            >>> toks = torch.tensor([[7, 8, 0, 0], [4, 5, 6, 0]])  # multiple trailing PADs in row 1
            >>> code = processor.create_code_with_bos_eos(toks)
            >>> code  # expected EOS right after content
            tensor([[20,  7,  8, 21,  0,  0],
                    [20,  4,  5,  6, 21,  0]])
            >>> left_padded_toks = torch.tensor([[0, 0, 7, 8], [0, 4, 5, 6]])  # multiple trailing PADs in row 1
            >>> left_code = processor.create_code_with_bos_eos(left_padded_toks)
            >>> left_code  # expected EOS right after content
            tensor([[ 0,  0, 20,  7,  8, 21],
                    [ 0, 20,  4,  5,  6, 21]])
        """
        bos = self.special_tokens["BOS"]
        eos = self.special_tokens["EOS"]
        pad = self.special_tokens["PAD"]

        batch_size, seq_len = token_sequences.shape
        device = token_sequences.device

        target_len = seq_len + 2

        is_left_padded = bool(token_sequences[:, 0].eq(pad).any().item())

        # Prepend and Postpend an Empty vector for BOS/EOS
        code = torch.full((batch_size, target_len), pad, dtype=torch.long, device=device)
        code[:, 1 : 1 + seq_len] = token_sequences
        pad_mask = token_sequences == pad

        start_idx = 1
        if is_left_padded:
            code[:, -1] = eos
            # returns shape batch_size, with 0 (first index) for rows with no pads. Adds a +1 offset as we prepend and postpend 0 vector
            last_pad_idx = (
                torch.where(pad_mask, torch.arange(seq_len, device=pad_mask.device) + 1, 0).max(dim=-1).values
            )
            row_indices = torch.arange(batch_size, device=device)
            code[row_indices, last_pad_idx] = bos
        else:
            code[:, 0] = bos
            # returns shape batch_size, with -1 (last index) for rows with no pads. Adds a +1 offset as we prepend and postpend 0 vector
            first_pad_idx = (
                torch.where(pad_mask, torch.arange(seq_len, device=pad_mask.device) + 1, seq_len + 1)
                .min(dim=-1)
                .values
            )
            row_indices = torch.arange(batch_size, device=device)
            code[row_indices, first_pad_idx] = eos

        return code

    def quantize_and_fuse_histograms(
        self,
        histogram_sequence: torch.Tensor,
        code_sequence: torch.Tensor,
        first_code_indices: torch.Tensor,
        gap_counts: torch.Tensor | None = None,
        anchor_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quantize histograms and place them into ``code_sequence`` placeholders.

        ``code_sequence`` must contain ``-1`` at the locations where the quantized
        histogram tokens should land. ``first_code_indices`` provides the column index
        for each window; entries with ``-1`` are treated as padding and skipped. Anchor
        (``-2``) and gap (``-3``) placeholders are replaced with their corresponding
        special tokens once fusion completes.

        Examples:
            >>> import torch
            >>> from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ
            >>> quantizer = SimplifiedAutoencoderVQ(
            ...     vocab_size=3,
            ...     embedding_dim=2,
            ...     n_embeddings=4,
            ...     beta=0.25,
            ...     encoder_hidden_dims=[4],
            ...     decoder_hidden_dims=[4],
            ...     dropout=0.0,
            ... )
            >>> processor = HistogramSequenceProcessor(quantizer=quantizer, vocab_size=3, do_fusion=True)
            >>> hist = torch.tensor([[[0., 1., 0.], [1., 0., 0.]]])
            >>> codes = torch.tensor([[-1, -2, 101, -1, 0]])
            >>> first_idx = torch.tensor([[0, 3]])
            >>> fused = processor.quantize_and_fuse_histograms(
            ...     hist,
            ...     codes,
            ...     first_idx,
            ...     gap_counts=torch.zeros(1, 2, dtype=torch.long),
            ...     anchor_token_indices=torch.tensor([[0]])
            ... )
            >>> fused.shape
            torch.Size([1, 5])
            >>> fused[0, 0].item() >= processor.quantizer_token_start
            True
            >>> fused[0, 1].item() == processor.special_tokens["ANCHOR"]
            True
            >>> (fused < 0).any().item()
            False

            >>> # BUG REPRODUCTION: Gap tokens not being inserted correctly
            >>> # When a histogram window has gap_count > 0, we should insert a gap token, not quantize
            >>> codes_with_gap = torch.tensor([[-1, 101, -1, 102, -1, 103, 0]])  # 3 histogram placeholders
            >>> hist_with_gap = torch.tensor([
            ...     [[0., 1., 0.],   # Window 0: real histogram
            ...      [0., 0., 0.],   # Window 1: gap (gap_count=1)
            ...      [1., 0., 0.]]   # Window 2: real histogram
            ... ])
            >>> gap_counts_test = torch.tensor([[0, 1, 0]])  # Window 1 is a gap
            >>> first_idx_gap = torch.tensor([[0, 2, 4]])  # Positions of the 3 placeholders
            >>> fused_gap = processor.quantize_and_fuse_histograms(
            ...     hist_with_gap,
            ...     codes_with_gap,
            ...     first_idx_gap,
            ...     gap_counts=gap_counts_test,
            ...     anchor_token_indices=None
            ... )
            >>> # Position 2 should have a gap token (gap_token_start), not 0 or a quantized histogram
            >>> gap_token_id = processor.gap_token_start  # For gap_count=1
            >>> print(f"Position 2 token: {fused_gap[0, 2].item()}, Expected gap token: {gap_token_id}")
            Position 2 token: 313, Expected gap token: 313
            >>> fused_gap[0, 2].item() == gap_token_id
            True
            >>> # Positions 0 and 4 should have quantized histograms
            >>> fused_gap[0, 0].item() >= processor.quantizer_token_start
            True
            >>> fused_gap[0, 4].item() >= processor.quantizer_token_start
            True

            >>> # Test anchor window placeholders (-2) get converted correctly
            >>> # When code sequence has -2 (anchor window placeholder), it should NOT get a 0!
            >>> codes_with_anchor = torch.tensor([[-1, 101, -2, 102, -1, 103, 0]])  # Placeholder at pos 2 is -2
            >>> hist_with_anchor = torch.tensor([
            ...     [[0., 1., 0.],   # Window 0: normal histogram
            ...      [1., 0., 0.],   # Window 1: anchor window histogram
            ...      [0., 1., 0.]]   # Window 2: normal histogram
            ... ])
            >>> gap_counts_anchor = torch.tensor([[0, 0, 0]])  # No gaps
            >>> anchor_indices_anchor = torch.tensor([[1]])  # Window 1 is anchor
            >>> first_idx_anchor = torch.tensor([[0, 2, 4]])  # Positions of the 3 placeholders
            >>> fused_anchor = processor.quantize_and_fuse_histograms(
            ...     hist_with_anchor,
            ...     codes_with_anchor,
            ...     first_idx_anchor,
            ...     gap_counts=gap_counts_anchor,
            ...     anchor_token_indices=anchor_indices_anchor
            ... )
            >>> # Position 2 has -2 placeholder (anchor window), should get ANCHOR token, NOT 0!
            >>> anchor_token_id = processor.special_tokens["ANCHOR"]
            >>> print(f"Position 2 token: {fused_anchor[0, 2].item()}, Expected anchor token: {anchor_token_id}")  # doctest: +ELLIPSIS
            Position 2 token: ..., Expected anchor token: ...
            >>> fused_anchor[0, 2].item() == anchor_token_id
            True
            >>> # Position 0 and 4 should have quantized histograms (normal windows)
            >>> fused_anchor[0, 0].item() >= processor.quantizer_token_start
            True
            >>> fused_anchor[0, 4].item() >= processor.quantizer_token_start
            True

            >>> # BUG REPRODUCTION: When batch has different numbers of histogram placeholders
            >>> # Sample 0 (short): 2 histogram placeholders
            >>> # Sample 1 (long): 4 histogram placeholders
            >>> # After padding first_code_indices to match longest, sample 0 gets extra -1 padding
            >>> # But the histogram_tokens also gets padded with PAD (0), creating the bug!
            >>> batch_codes_bug = torch.tensor([
            ...     [-1, 101, -1, 102, 0, 0, 0, 0],  # Sample 0: 2 placeholders + padding
            ...     [-1, 201, -1, 202, -1, 203, -1, 204],  # Sample 1: 4 placeholders
            ... ])
            >>> # first_code_indices gets padded to 4 columns
            >>> indices_bug = torch.tensor([
            ...     [0, 2, -1, -1],  # Sample 0: only 2 valid indices, rest are -1 (padding)
            ...     [0, 2, 4, 6],    # Sample 1: all 4 valid
            ... ])
            >>> # Histogram tokens: sample 0 has 2 real histograms, sample 1 has 4
            >>> # After quantization and batching, they're padded to same width
            >>> # Let's simulate the quantized histogram tokens (already offset by quantizer_token_start)
            >>> # In fusion mode, quantizer_token_start = 309, so histogram tokens are 309+
            >>> hist_tokens_bug = torch.tensor([
            ...     [310, 320, 0, 0],  # Sample 0: 2 quantized histograms + 2 PAD
            ...     [315, 325, 335, 345],  # Sample 1: 4 quantized histograms
            ... ])
            >>> # When fuse_histograms_and_codes processes this:
            >>> # - valid_mask = (indices_bug >= 0) = [[True, True, False, False], [True, True, True, True]]
            >>> # - It should skip indices with -1
            >>> # - But if histogram_tokens has PAD (0) at invalid positions, check at line 854 validates this
            >>> # The bug is: what if histogram_tokens[~valid_mask] has non-zero PAD due to other reasons?
            >>> result_bug = processor.fuse_histograms_and_codes(
            ...     batch_codes_bug,
            ...     hist_tokens_bug,
            ...     indices_bug,
            ...     pad_token_id=0
            ... )
            >>> # Verify: positions 0,2 in sample 0 should have histogram tokens
            >>> result_bug[0, 0].item() == 310
            True
            >>> result_bug[0, 2].item() == 320
            True
            >>> # Positions 4,5 should remain as padding (0)
            >>> result_bug[0, 4:6].tolist()
            [0, 0]
        """

        if not self.do_fusion:
            raise ValueError("quantize_and_fuse_histograms requires do_fusion=True on the processor")

        if gap_counts is None:
            gap_counts = torch.zeros(
                histogram_sequence.shape[:2],
                dtype=torch.long,
                device=histogram_sequence.device,
            )

        if gap_counts.shape != histogram_sequence.shape[:2]:
            raise ValueError(
                "gap_counts must have shape [batch_size, histogram_seq_len] matching histogram_sequence"
            )

        # Compute histogram tokens without overwriting anchor positions so that
        # anchor placeholders (-2) can be resolved explicitly after fusion.
        histogram_tokens = self.process_histogram_batch(
            histogram_sequence,
            gap_counts,
            anchor_token_indices=None,
        )

        fused = HistogramSequenceProcessor.fuse_histograms_and_codes(
            code_sequence,
            histogram_tokens,
            first_code_indices,
            pad_token_id=self.special_tokens["PAD"],
        )

        fused = self._apply_special_placeholder_tokens(
            fused,
            gap_counts,
        )

        if self._fusion_eos_token_id is not None:
            fused = torch.where(
                fused == self.special_tokens["EOS"],
                torch.full_like(fused, self._fusion_eos_token_id),
                fused,
            )

        if (fused < 0).any():
            raise ValueError("Fusion produced unresolved placeholder tokens")

        return fused

    @staticmethod
    def fuse_histograms_and_codes(
        code_sequence: torch.Tensor,
        histogram_tokens: torch.Tensor,
        first_code_indices: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Replace ``-1`` placeholders with histogram tokens.

        Examples:
            >>> import torch
            >>> codes = torch.tensor([[0, -1, 101, 102, -1, 0]])
            >>> hist_tokens = torch.tensor([[7, 8, 0]])
            >>> first_idx = torch.tensor([[1, 4, -1]])
            >>> fused = HistogramSequenceProcessor.fuse_histograms_and_codes(
            ...     codes, hist_tokens, first_idx, pad_token_id=0
            ... )
            >>> fused
            tensor([[  0,   7, 101, 102,   8,   0]])
            >>> fused.eq(-1).any().item()
            False
        """

        if histogram_tokens.shape != first_code_indices.shape:
            raise ValueError("histogram_tokens and first_code_indices must share the same shape")

        if code_sequence.ndim != 2:
            raise ValueError("code_sequence must be 2D [batch, seq_len]")

        if histogram_tokens.ndim != 2:
            raise ValueError("histogram_tokens must be 2D [batch, hist_seq_len]")

        if code_sequence.dtype != torch.long:
            raise ValueError("code_sequence must use dtype torch.long")

        if histogram_tokens.dtype != torch.long:
            raise ValueError("histogram_tokens must use dtype torch.long")

        if first_code_indices.dtype != torch.long:
            raise ValueError("first_code_indices must use dtype torch.long")

        if histogram_tokens.device != code_sequence.device:
            raise ValueError("histogram_tokens and code_sequence must be on the same device")

        if first_code_indices.device != code_sequence.device:
            raise ValueError("first_code_indices and code_sequence must be on the same device")

        batch_hist, hist_len = histogram_tokens.shape
        batch_codes, seq_len = code_sequence.shape
        if batch_hist != batch_codes:
            raise ValueError("Batch dimension mismatch between code_sequence and histogram_tokens")

        device = code_sequence.device
        fused = code_sequence.clone()

        valid_mask = first_code_indices >= 0

        if (first_code_indices[valid_mask] >= seq_len).any():
            raise ValueError("first_code_indices contains positions beyond code_sequence length")

        if (histogram_tokens[~valid_mask] != pad_token_id).any():
            raise ValueError("Histogram padding tokens must align with invalid first_code_indices entries")

        if valid_mask.any():
            batch_indices = torch.arange(batch_hist, device=device).unsqueeze(1).expand_as(first_code_indices)
            original_values = fused[batch_indices[valid_mask], first_code_indices[valid_mask]]
            if (original_values >= 0).any():
                raise ValueError("Expected negative placeholders (-1, -2, or -3) at histogram insertion positions")

            # Only replace -1 (histogram) placeholders with histogram tokens
            # Leave -2 (anchor) and -3 (gap) placeholders for _apply_special_placeholder_tokens to handle
            histogram_mask = original_values == -1
            if histogram_mask.any():
                # Get flat indices where we have -1 placeholders
                histogram_positions = histogram_mask.nonzero(as_tuple=True)[0]
                batch_hist_indices = batch_indices[valid_mask][histogram_positions]
                seq_hist_indices = first_code_indices[valid_mask][histogram_positions]
                tokens_to_insert = histogram_tokens[valid_mask][histogram_positions]
                fused[batch_hist_indices, seq_hist_indices] = tokens_to_insert

        return fused

    def _apply_special_placeholder_tokens(
        self,
        fused_sequence: torch.Tensor,
        gap_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve anchor (-2) and gap (-3) placeholders after fusion.

        Examples:
            >>> import torch
            >>> from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ
            >>> quantizer = SimplifiedAutoencoderVQ(
            ...     vocab_size=3,
            ...     embedding_dim=2,
            ...     n_embeddings=4,
            ...     beta=0.25,
            ...     encoder_hidden_dims=[4],
            ...     decoder_hidden_dims=[4],
            ...     dropout=0.0,
            ... )
            >>> processor = HistogramSequenceProcessor(quantizer=quantizer, vocab_size=3, do_fusion=True)
            >>> fused = torch.tensor([[-2, -3, 5]])
            >>> gap_counts = torch.tensor([[0, 2, 0]])
            >>> resolved = processor._apply_special_placeholder_tokens(fused, gap_counts)
            >>> resolved[0, 0].item() == processor.special_tokens["ANCHOR"]
            True
            >>> resolved[0, 1].item() == processor.gap_token_start + 1
            True
            >>> (resolved < 0).any().item()
            False
        """

        if fused_sequence.ndim != 2:
            raise ValueError("fused_sequence must be 2D [batch, seq_len]")

        anchor_placeholder = -2
        gap_placeholder = -3

        anchor_token_id = self.special_tokens["ANCHOR"]

        anchor_mask = fused_sequence == anchor_placeholder
        if anchor_mask.any():
            fused_sequence = fused_sequence.clone()
            fused_sequence[anchor_mask] = anchor_token_id

        gap_mask = fused_sequence == gap_placeholder
        if not gap_mask.any():
            return fused_sequence

        if gap_counts is None:
            raise ValueError("gap_counts must be provided when gap placeholders are present")

        if gap_counts.ndim != 2 or gap_counts.shape[0] != fused_sequence.shape[0]:
            raise ValueError(
                "gap_counts must be a 2D tensor matching the batch dimension of the fused sequence"
            )
        if gap_counts.shape[1] == 0:
            raise ValueError("gap_counts must have non-zero window dimension when gap placeholders exist")

        gap_counts = gap_counts.to(fused_sequence.device)
        gap_token_start = self.gap_token_start
        gap_token_end = self.gap_token_end

        # Vectorized gap token insertion across all batches
        # Count gaps per batch and validate against positive gap counts
        gaps_per_batch = gap_mask.sum(dim=1)  # [batch_size]
        positive_gaps_per_batch = (gap_counts > 0).sum(dim=1)  # [batch_size]

        if not torch.all(gaps_per_batch == positive_gaps_per_batch):
            mismatches = (gaps_per_batch != positive_gaps_per_batch).nonzero(as_tuple=False).squeeze(-1)
            for batch_idx in mismatches.tolist():
                raise ValueError(
                    "Mismatch between gap placeholders and positive gap counts for batch index "
                    f"{batch_idx}: placeholders={gaps_per_batch[batch_idx].item()}, "
                    f"gaps={positive_gaps_per_batch[batch_idx].item()}"
                )

        # Get all gap positions as (batch_idx, seq_idx) pairs
        batch_indices, seq_indices = gap_mask.nonzero(as_tuple=True)

        if batch_indices.numel() > 0:
            # Vectorized gap token generation across all batches
            # Get all positive gap counts (flattens in row-major order, matching gap_mask.nonzero order)
            positive_mask = gap_counts > 0
            all_gap_tokens = (
                gap_counts[positive_mask]
                .sub(1)
                .add(gap_token_start)
                .clamp(max=gap_token_end)
                .to(device=fused_sequence.device, dtype=fused_sequence.dtype)
            )
            # Single scatter operation
            fused_sequence[batch_indices, seq_indices] = all_gap_tokens

        return fused_sequence

    @property
    def fusion_eos_token_id(self) -> int | None:
        return self._fusion_eos_token_id

    def set_fusion_eos_token_id(self, token_id: int | None) -> None:
        if token_id is not None and token_id < 0:
            raise ValueError("fusion_eos_token_id must be non-negative or None")
        self._fusion_eos_token_id = token_id

    def get_ar_eos_token(self) -> int:
        if self.do_fusion and self._fusion_eos_token_id is not None:
            return self._fusion_eos_token_id
        return self.special_tokens["EOS"]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
