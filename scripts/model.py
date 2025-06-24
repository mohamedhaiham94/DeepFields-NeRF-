import torch
from torch import nn
import torch.nn.functional as F


class OptimizedPositionalEncoding(nn.Module):
    """
    Optimized positional encoding with vectorized operations.

    Key optimizations:
    - Precomputed frequency bands
    - Vectorized sin/cos computation
    - Efficient broadcasting
    """

    def __init__(self, L=10, include_input=True):
        """
        Initialize optimized positional encoding.

        Args:
            L: Number of frequency bands
            include_input: Whether to include the original input in the encoding
        """
        super().__init__()
        self.L = L
        self.include_input = include_input
        self.output_dim = 3 + (3 * 2 * L) if include_input else 3 * 2 * L

        # Precompute frequency multipliers as buffer
        self.register_buffer("freq_bands", 2.0 ** torch.arange(L) * torch.pi)

    def forward(self, x):
        """
        Apply optimized positional encoding.

        Args:
            x: Input coordinates of shape [..., 3]

        Returns:
            Encoded coordinates of shape [..., output_dim]
        """
        encoded = [x] if self.include_input else []

        # Vectorized computation using broadcasting
        # x[..., None] shape: [..., 3, 1]
        # freq_bands shape: [L]
        # Result shape: [..., 3, L]
        x_freq = x[..., None] * self.freq_bands

        # Compute sin and cos in one go and flatten
        encoded.extend([torch.sin(x_freq).flatten(-2), torch.cos(x_freq).flatten(-2)])

        return torch.cat(encoded, dim=-1)


class OptimizedVanillaNeRF(nn.Module):
    """
    Optimized Vanilla NeRF model with performance improvements.

    Key optimizations:
    - Optimized positional encoding
    - Improved initialization
    - Memory-efficient operations
    """

    def __init__(
        self,
        pos_L=10,
        dir_L=4,
        hidden_dim=256,
        include_input_pos=True,
        include_input_dir=True,
    ):
        """
        Initialize the NeRF model.

        Args:
            pos_L: Number of frequency bands for position encoding
            dir_L: Number of frequency bands for direction encoding
            hidden_dim: Dimension of hidden layers
            include_input_pos: Whether to include original position in encoding
            include_input_dir: Whether to include original direction in encoding
        """
        super().__init__()

        # Optimized positional encoding
        self.pos_encoding = OptimizedPositionalEncoding(
            L=pos_L, include_input=include_input_pos
        )
        self.dir_encoding = OptimizedPositionalEncoding(
            L=dir_L, include_input=include_input_dir
        )

        # Dimensions
        pos_dim = self.pos_encoding.output_dim
        dir_dim = self.dir_encoding.output_dim

        # Position processing layers
        self.pos_layers = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Skip connection layers
        self.skip_layers = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Density output
        self.density_layer = nn.Linear(hidden_dim, 1)

        # Improved initialization
        nn.init.constant_(self.density_layer.bias, 0.1)
        nn.init.normal_(self.density_layer.weight, mean=0.0, std=0.01)

        # Feature vector output
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        # Color prediction layers
        self.color_layer = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def query_density(self, pos):
        """
        Optimized density query with improved memory efficiency.
        
        Args:
            pos: Position tensor of shape [..., 3]

        Returns:
            sigma: Density values of shape [..., 1]
            features: Feature vector of shape [..., hidden_dim]
        """
        # Encode position
        pos_encoded = self.pos_encoding(pos)

        x = self.pos_layers(pos_encoded)
        x = torch.cat([x, pos_encoded], dim=-1)
        x = self.skip_layers(x)

        # Get density (sigma)
        sigma = self.density_layer(x)

        if self.training:
            # More efficient noise generation
            sigma = sigma + torch.randn_like(sigma)

        # Use in-place ReLU for memory efficiency
        # sigma = torch.relu_(sigma)
        sigma = F.relu(sigma)

        # Get features for color prediction
        features = self.feature_layer(x)

        return sigma, features

    def forward(self, pos, view):
        """Optimized forward pass."""
        # Get density and features
        sigma, features = self.query_density(pos)

        # Encode view directions
        view_encoded = self.dir_encoding(view)

        # Predict color using features and view directions
        x = torch.cat([features, view_encoded], dim=-1)
        rgb = self.color_layer(x)
        # Apply sigmoid to ensure RGB values are in [0, 1]
        rgb = torch.sigmoid(rgb)

        return rgb, sigma


def get_nerf_models(
    pos_L=10,
    dir_L=4,
    hidden_dim=256,
    include_input_pos=True,
    include_input_dir=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Create optimized coarse and fine NeRF models."""

    model_coarse = OptimizedVanillaNeRF(
        pos_L=pos_L,
        dir_L=dir_L,
        hidden_dim=hidden_dim,
        include_input_pos=include_input_pos,
        include_input_dir=include_input_dir,
    ).to(device)

    model_fine = OptimizedVanillaNeRF(
        pos_L=pos_L,
        dir_L=dir_L,
        hidden_dim=hidden_dim,
        include_input_pos=include_input_pos,
        include_input_dir=include_input_dir,
    ).to(device)

    return model_coarse, model_fine


if __name__ == "__main__":
    pos = torch.rand(10, 3)
    view = torch.rand(10, 3)

    pos_encoder = OptimizedPositionalEncoding(L=10)
    view_encoder = OptimizedPositionalEncoding(L=4)
    pos_encoding = pos_encoder(pos)
    view_encoding = view_encoder(view)
    print(f"{pos_encoding.shape=}")  # Shape (N x 63)
    print(f"{view_encoding.shape=}")  # Shape (N x 27)

    # nerf = VanillaNeRF()
    # rgb, sigma = nerf(pos, view_dir)
    # nerf.train()
    # print(f"RGB: {rgb.shape}, Density: {sigma.shape}")
