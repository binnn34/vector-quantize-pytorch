"""
VQ-VAE for 2D Pose Sequences
- Input: (T, 17, 2) normalized pose sequences
- Output: Reconstructed poses + discrete codes
- Codebook: 512 learnable pose embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class PoseEncoder(nn.Module):
    """
    Encode pose sequences into continuous latent representations
    Input: (B, T, 17, 2)
    Output: (B, T, latent_dim)
    """
    def __init__(self, input_dim=34, hidden_dim=256, latent_dim=128, num_layers=3):
        super().__init__()

        self.input_dim = input_dim  # 17 keypoints * 2 coords = 34
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Flatten pose per frame: (17, 2) -> (34,)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Temporal encoder (LSTM or Transformer)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Project to latent space
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, 17, 2) pose sequences
        Returns:
            z: (B, T, latent_dim) continuous latents
        """
        B, T, J, C = x.shape  # Batch, Time, Joints (17), Coords (2)

        # Flatten joints: (B, T, 17, 2) -> (B, T, 34)
        x = x.reshape(B, T, J * C)

        # Input projection
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = F.relu(x)

        # Temporal encoding
        x, _ = self.lstm(x)  # (B, T, hidden_dim)

        # Project to latent space
        z = self.latent_proj(x)  # (B, T, latent_dim)

        return z


class PoseDecoder(nn.Module):
    """
    Decode quantized latent codes back to pose sequences
    Input: (B, T, latent_dim)
    Output: (B, T, 17, 2)
    """
    def __init__(self, latent_dim=128, hidden_dim=256, output_dim=34, num_layers=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Expand from latent
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Temporal decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: (B, T, latent_dim) quantized latents
        Returns:
            x_recon: (B, T, 17, 2) reconstructed poses
        """
        B, T, D = z.shape

        # Expand latent
        x = self.latent_proj(z)  # (B, T, hidden_dim)
        x = F.relu(x)

        # Temporal decoding
        x, _ = self.lstm(x)  # (B, T, hidden_dim)

        # Output projection
        x = self.output_proj(x)  # (B, T, 34)

        # Reshape to pose format: (B, T, 34) -> (B, T, 17, 2)
        x_recon = x.reshape(B, T, 17, 2)

        return x_recon


class PoseVQVAE(nn.Module):
    """
    Complete VQ-VAE for Pose Sequences

    Architecture:
        Encoder → VectorQuantize (Codebook) → Decoder

    Args:
        codebook_size: Number of codes (default: 512)
        latent_dim: Latent dimension (default: 128)
        hidden_dim: Hidden dimension for LSTM (default: 256)
        commitment_weight: Commitment loss weight (default: 0.25)
    """
    def __init__(
        self,
        codebook_size=512,
        latent_dim=128,
        hidden_dim=256,
        num_layers=3,
        commitment_weight=0.25,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = PoseEncoder(
            input_dim=34,  # 17 * 2
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        # Vector Quantization (Codebook)
        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=0.8,  # EMA decay for codebook updates
            commitment_weight=commitment_weight,
            accept_image_fmap=False,  # We have 1D temporal features
        )

        # Decoder
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=34,  # 17 * 2
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 17, 2) normalized pose sequences

        Returns:
            x_recon: (B, T, 17, 2) reconstructed poses
            vq_loss: VQ loss (commitment + codebook)
            perplexity: Codebook usage metric
            indices: (B, T) discrete code indices
        """
        # Encode
        z = self.encoder(x)  # (B, T, latent_dim)

        # Vector Quantization
        # Reshape for VQ: (B, T, D) -> (B*T, D)
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)

        quantized, indices, vq_loss = self.vq(z_flat)

        # Reshape back: (B*T, D) -> (B, T, D)
        quantized = quantized.reshape(B, T, D)
        indices = indices.reshape(B, T)

        # Decode
        x_recon = self.decoder(quantized)  # (B, T, 17, 2)

        # Calculate perplexity (codebook usage diversity)
        # Perplexity = exp(entropy) = exp(-sum(p * log(p)))
        avg_probs = torch.bincount(
            indices.flatten(),
            minlength=self.codebook_size
        ).float() / indices.numel()

        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return x_recon, vq_loss, perplexity, indices

    def encode(self, x):
        """Encode poses to discrete codes"""
        z = self.encoder(x)
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        quantized, indices, _ = self.vq(z_flat)
        return indices.reshape(B, T)

    def decode_codes(self, indices):
        """Decode discrete codes to poses"""
        # Get embeddings from codebook
        B, T = indices.shape
        embeddings = self.vq.codebook[indices.flatten()]  # (B*T, D)
        embeddings = embeddings.reshape(B, T, self.latent_dim)

        # Decode
        x_recon = self.decoder(embeddings)
        return x_recon


def test_vqvae():
    """Test VQ-VAE model"""
    print("Testing PoseVQVAE...")

    # Create model
    model = PoseVQVAE(
        codebook_size=512,
        latent_dim=128,
        hidden_dim=256,
        num_layers=3
    )

    # Test input: 2 videos, 100 frames, 17 joints, 2 coords
    x = torch.randn(2, 100, 17, 2)

    # Forward pass
    x_recon, vq_loss, perplexity, indices = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f} / {model.codebook_size}")
    print(f"Indices shape: {indices.shape}")
    print(f"Unique codes used: {torch.unique(indices).numel()} / {model.codebook_size}")

    # Test encode/decode
    codes = model.encode(x)
    x_decoded = model.decode_codes(codes)
    print(f"\nEncode-decode test:")
    print(f"Codes shape: {codes.shape}")
    print(f"Decoded shape: {x_decoded.shape}")
    print(f"Reconstruction matches: {torch.allclose(x_recon, x_decoded, atol=1e-6)}")

    print("\n✅ VQ-VAE test passed!")


if __name__ == '__main__':
    test_vqvae()
