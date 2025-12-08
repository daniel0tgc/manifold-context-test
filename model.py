"""
model.py

Small fully-connected autoencoders for low-dimensional synthetic data or
flattened embeddings.

Includes:
- ``VAE``: a minimal variational autoencoder
- ``ManifoldAE``: a compact, symmetric MLP autoencoder with 3-layer encoder/decoder
- ``random_event_embedding``: utility to generate a normalized latent vector

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


class ManifoldAE(nn.Module):
    """
    A compact MLP autoencoder with a 3-layer encoder and symmetric decoder.

    The encoder applies three Linear+ReLU layers and then projects to a latent
    vector ``z`` of size ``latent_dim``. The decoder mirrors the encoder to
    reconstruct the input.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    latent_dim : int, default 64
        Dimensionality of the latent vector ``z``.
    hidden_dims : Sequence[int], default (256, 128, 64)
        Sizes of the three hidden layers in the encoder. The decoder uses the
        reversed sequence symmetrically. Must contain exactly three elements
        for the default architecture. Custom sequences are supported.
    final_activation : Optional[nn.Module]
        Optional activation applied to the decoder output (e.g., Sigmoid). If
        ``None``, no activation is applied.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128, 64),
        final_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 3:
            # Keep the model reasonably expressive by default
            raise ValueError("hidden_dims should contain at least 3 layer sizes")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.final_activation = final_activation

        # Encoder: [input -> h1 -> h2 -> h3] -> z
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU(inplace=True))
            prev = h
        self.encoder_backbone = nn.Sequential(*enc_layers)
        self.to_latent = nn.Linear(prev, latent_dim)

        # Decoder: z -> [h3 -> h2 -> h1] -> input
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU(inplace=True))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        if self.final_activation is not None:
            dec_layers.append(self.final_activation)
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent vectors ``z``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Latent batch of shape ``(batch, latent_dim)``.
        """
        h = self.encoder_backbone(x)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors ``z`` back to input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent batch of shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(batch, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the autoencoder end-to-end.

        Returns a tuple ``(reconstruction, z)``.
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def random_event_embedding(
    text: Optional[str] = None,
    *,
    latent_dim: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a random unit-norm vector in latent space.

    If ``text`` is provided, a deterministic vector is produced by hashing the
    text to seed a per-call RNG. Does not alter global RNG state.

    Parameters
    ----------
    text : Optional[str]
        Optional text used to deterministically seed the random vector.
    latent_dim : int, default 64
        Size of the latent vector to generate.
    device : Optional[torch.device]
        Device for the returned tensor. Defaults to CPU or current CUDA if set.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(latent_dim,)`` with L2 norm equal to 1.
    """
    import hashlib

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if text is None:
        v = torch.randn(latent_dim, device=device)
    else:
        # Seed a local generator from the text hash; do not affect global RNG
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="little", signed=False)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        v = torch.randn(latent_dim, generator=gen, device=device)

    norm = torch.linalg.vector_norm(v)
    if norm.item() == 0.0:
        # Extremely unlikely fallback; return standard basis e0
        v = torch.zeros(latent_dim, device=device)
        v[0] = 1.0
        return v
    return v / norm


if __name__ == "__main__":
    # Minimal demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 128
    latent_dim = 64
    batch_size = 32

    model = ManifoldAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        recon, z = model(x)

    norms = torch.linalg.vector_norm(z, dim=1)
    print(f"z shape: {z.shape}, recon shape: {recon.shape}")
    print("Latent norms (first 10):", [round(float(n), 4) for n in norms[:10]])
    e = random_event_embedding(text="toy-event", latent_dim=latent_dim, device=device)
    print(f"Random event embedding norm: {float(torch.linalg.vector_norm(e)):.4f}")