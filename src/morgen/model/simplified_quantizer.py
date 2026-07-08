"""Simplified VQ-VAE for count histograms with binary decode outputs.

The quantizer now trains on clipped count histograms using per-code cross-entropy
over count bins ``[0, ..., max_count]``. The public decode path still returns
binary logits, so downstream inference code can keep thresholding decoded
histograms exactly as before.
"""

import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CountHistogramNormalizer(nn.Module):
    """Clamp count histograms and provide normalized encoder inputs."""

    def __init__(self, vocab_size: int, max_count: int):
        super().__init__()
        if max_count < 1:
            raise ValueError(f"max_count must be >= 1, got {max_count}")
        self.vocab_size = vocab_size
        self.max_count = max_count

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0, max=self.max_count).float()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.clamp(x) / float(self.max_count)

    def to_binary(self, x: torch.Tensor) -> torch.Tensor:
        return (self.clamp(x) > 0).float()


class VectorQuantizer(nn.Module):
    """Vector quantization layer operating on latent vectors."""

    def __init__(self, n_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(z.shape) == 2, f"Expected 2D input, got shape {z.shape}"
        assert z.shape[1] == self.embedding_dim, (
            f"Expected embedding_dim={self.embedding_dim}, got {z.shape[1]}"
        )

        z_flattened = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn -> bn", z_flattened, self.embedding.weight.t())
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)

        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = commitment_loss + self.beta * codebook_loss
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class SimpleMLP(nn.Module):
    """Simple MLP implementation without torchvision dependency."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Encoder(nn.Module):
    """MLP encoder for normalized count histograms."""

    def __init__(
        self, input_dim: int, latent_dim: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.0
    ):
        super().__init__()
        self.encoder = SimpleMLP(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=latent_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """MLP decoder producing per-code count-bin logits."""

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [32, 64],
        dropout: float = 0.0,
        max_count: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_count = max_count
        self.decoder = SimpleMLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim * (max_count + 1),
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return logits.view(z.shape[0], self.output_dim, self.max_count + 1)


class SimplifiedAutoencoderVQ(nn.Module):
    """Simplified VQ-VAE for count histograms.

    Histograms are clipped to ``max_count`` during training. Reconstruction loss is
    per-code cross-entropy over count bins, while ``decode`` and ``decode_indices``
    still expose binary logits for downstream thresholding.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_embeddings: int,
        beta: float,
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
        dropout: float,
        max_count: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.max_count = max_count

        self.normalizer = CountHistogramNormalizer(vocab_size, max_count)
        self.encoder = Encoder(
            input_dim=vocab_size, latent_dim=embedding_dim, hidden_dims=encoder_hidden_dims, dropout=dropout
        )
        self.quantizer = VectorQuantizer(n_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=beta)
        self.decoder = Decoder(
            output_dim=vocab_size,
            latent_dim=embedding_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            max_count=max_count,
        )

    @staticmethod
    def infer_max_count_from_state_dict(state_dict: dict[str, torch.Tensor], vocab_size: int) -> int:
        """Infer max_count from decoder output shape, defaulting to binary."""
        pattern = re.compile(r"decoder\.decoder\.network\.(\d+)\.weight$")
        candidates: list[tuple[int, int]] = []
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match:
                candidates.append((int(match.group(1)), int(value.shape[0])))

        if not candidates:
            return 1

        _, output_dim = max(candidates, key=lambda item: item[0])
        if output_dim % vocab_size != 0:
            logger.warning(
                "Could not infer max_count cleanly from decoder output_dim=%s and vocab_size=%s; defaulting to 1",
                output_dim,
                vocab_size,
            )
            return 1

        return max(1, output_dim // vocab_size - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalizer.transform(x))

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z_e)

    def decode_count_logits(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def _count_logits_to_binary_logits(self, count_logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(count_logits, dim=-1)
        log_prob_zero = log_probs[..., 0]
        log_prob_nonzero = torch.logsumexp(log_probs[..., 1:], dim=-1)
        return log_prob_nonzero - log_prob_zero

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self._count_logits_to_binary_logits(self.decode_count_logits(z_q))

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.get_codebook_entry(indices)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> dict:
        count_input = self.normalizer.clamp(x)
        count_targets = count_input.long()
        binary_input = self.normalizer.to_binary(x)

        z_e = self.encode(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        count_logits = self.decode_count_logits(z_q)
        reconstruction = self._count_logits_to_binary_logits(count_logits)
        reconstruction_loss = F.cross_entropy(count_logits.transpose(1, 2), count_targets, reduction="mean")
        total_loss = vq_loss + reconstruction_loss

        return {
            "reconstruction": reconstruction,
            "count_reconstruction": count_logits,
            "vq_loss": vq_loss,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": total_loss,
            "indices": indices,
            "binary_input": binary_input,
            "count_input": count_input,
            "z_e": z_e,
            "z_q": z_q,
        }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
