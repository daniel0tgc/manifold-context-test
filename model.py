"""
model.py

A small fully-connected Variational Autoencoder (VAE) suitable for
low-dimensional synthetic data or flattened embeddings.

Requires: torch
Python: 3.10+
"""

from __future__ import annotations

from typing import Sequence, Tuple, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    A simple MLP-based VAE.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input vectors.
    hidden_dims : Sequence[int]
        Sizes of hidden layers (encoder). Decoder mirrors encoder.
    latent_dim : int
        Dimensionality of the latent z.
    activation : nn.Module
        Activation applied between hidden layers (default: ReLU).
    recon_activation : Optional[nn.Module]
        Final activation for decoder output (e.g., Sigmoid) or None for identity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        latent_dim: int = 16,
        activation: Optional[nn.Module] = None,
        recon_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation or nn.ReLU(inplace=True)
        self.recon_activation = recon_activation

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(self.activation)
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(self.activation)
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        if self.recon_activation is not None:
            dec_layers.append(self.recon_activation)
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent mu and logvar.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample z ~ N(mu, sigma^2)
        """
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruct input.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run full pass: input -> (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        recon_loss_type: str = "mse",
        kld_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss + KLD.

        recon_loss_type: "mse" or "bce"
        kld_weight: multiplicative weight on KLD term
        """
        if recon_loss_type == "mse":
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        elif recon_loss_type == "bce":
            # BCE expects inputs in [0,1]
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="mean")
        else:
            raise ValueError("recon_loss_type must be 'mse' or 'bce'")

        # KLD per batch (mean)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kld_weight * kld
        return {"loss": loss, "recon_loss": recon_loss, "kld": kld}

    def sample(self, num_samples: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from standard normal in latent space and decode.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)