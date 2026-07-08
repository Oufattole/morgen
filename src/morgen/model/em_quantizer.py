import torch
import torch.nn as nn

class EMQuantizer(nn.Module):
    """
    An EM-based Bernoulli Mixture Model implementation that follows the 
    SimplifiedAutoencoderVQ API for hot-swapping.
    """
    def __init__(
        self,
        theta: torch.Tensor,
        pi: torch.Tensor,
        alpha: float = 1e-6
    ):
        """
        Args:
            theta: Tensor of shape [n_embeddings, vocab_size] (cluster probabilities)
            pi: Tensor of shape [n_embeddings] (cluster priors)
            alpha: Smoothing factor for log calculations
        """
        super().__init__()
        # Register as buffers so they move to the correct device with the module
        self.register_buffer("theta", theta)
        self.register_buffer("pi", pi)
        
        self.n_embeddings = theta.shape[0]
        self.vocab_size = theta.shape[1]
        self.alpha = alpha

        # These attributes are needed for HistogramSequenceProcessor metadata extraction
        # but don't perform operations in an EM model.
        self.embedding_dim = self.n_embeddings 

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps binary histograms to log-likelihood scores (logits) for each cluster.
        This serves as the 'continuous latent' in the EM framework.
        """
        # Ensure input is binary [batch, vocab_size]
        x_bool = (x > 0).float()
        
        # log P(x|k) = x*log(theta) + (1-x)*log(1-theta) + log(pi)
        log_theta = torch.log(self.theta + self.alpha)
        log_one_minus_theta = torch.log(1 - self.theta + self.alpha)
        
        # Weights: log(theta / (1-theta))
        weights = log_theta - log_one_minus_theta
        # Bias: sum(log(1-theta)) + log(pi)
        bias = log_one_minus_theta.sum(dim=1) + torch.log(self.pi + self.alpha)
        
        # [batch, n_embeddings]
        logits = x_bool @ weights.T + bias
        return logits

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assigns the input to the most likely cluster.
        
        Returns:
            z_q: The logit vector of the assigned cluster
            vq_loss: Negative Log-Likelihood (NLL) of the mixture
            indices: The discrete cluster IDs
        """
        # Assign to the cluster with highest log-likelihood
        indices = torch.argmax(z_e, dim=1)
        
        # Retrieval of 'latent' for assigned cluster
        z_q = z_e[torch.arange(z_e.size(0)), indices].unsqueeze(1)

        # Compute VQ loss as the Negative Mean Log-Likelihood
        # log P(x) = logsumexp(logits)
        log_probs = torch.logsumexp(z_e, dim=1)
        vq_loss = -torch.mean(log_probs)

        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        In the EM/BMM framework, the reconstruction of a cluster is its 
        exemplar probabilities (theta) converted back to logits.
        """
        # Note: z_q here is the score. In this API, we map back to the 
        # probability-based logits for the most likely code.
        # Since we use argmax in quantize, we retrieve the theta for the winner.
        pass # Not typically used directly in HistogramSequenceProcessor

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Maps cluster indices directly to reconstructed binary logits.
        """
        # Retrieve the theta for selected indices
        probs = self.theta[indices]
        # Convert probabilities to logits for reconstruction
        logits = torch.log(probs + self.alpha) - torch.log(1 - probs + self.alpha)
        return logits

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass for compatibility."""
        z_e = self.encode(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        x_logits = self.decode_indices(indices)
        
        # Binary input for loss calculation
        binary_x = (x > 0).float()
        
        # This matches the dictionary structure expected by your evaluation scripts
        return {
            "reconstruction": x_logits,
            "vq_loss": vq_loss,
            "reconstruction_loss": torch.tensor(0.0, device=x.device), # Integrated into NLL
            "total_loss": vq_loss,
            "indices": indices,
            "binary_input": binary_x
        }