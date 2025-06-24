import torch
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F


class InstantNGP(nn.Module):

    def __init__(
        self,
        nerf_type: str = "small",
    ):
        super().__init__()
        if nerf_type == "small":
            log2_size = 15
        elif nerf_type == "medium":
            log2_size = 17
        elif nerf_type == "large":
            log2_size = 19
        else:
            raise ValueError(f"Unkown {nerf_type=}")
        
        self.pos_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_size,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
        )
        self.dir_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.dmlp = tcnn.Network(
            n_input_dims=self.pos_encoding.n_output_dims,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.cmlp = tcnn.Network(
            n_input_dims=32,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def query_density(self, pos):
        """
        Query the model for density at given positions.

        Args:
            pos: Position tensor of shape [..., 3]

        Returns:
            sigma: Density values of shape [..., 1]
            features: Feature vector of shape [..., hidden_dim]
        """
        pos_enc = self.pos_encoding(pos)
        features = self.dmlp(pos_enc)
        sigma = features[:, :1]  # Bx1
        return sigma, features

    def forward(self, pos, viewdirs):
        """
        Forward pass through the NeRF model.

        Args:
            pos: Position tensor of shape [..., 3]
            viewdirs: View direction tensor of shape [..., 3]

        Returns:
            rgb: RGB color values of shape [..., 3]
            sigma: Density values of shape [..., 1]
        """
        # Get density and features
        sigma, features = self.query_density(pos)

        # Encode view directions
        dir_encoded = self.dir_encoding(viewdirs)

        # Predict color using features and view directions
        x = torch.cat([features, dir_encoded], dim=-1)
        rgb = self.cmlp(x)

        return rgb, sigma


if __name__ == "__main__":
    B = 16
    pos = torch.rand(B, 3).cuda()
    view = torch.rand(B, 3).cuda()
    nerf = InstantNGP("medium").cuda()
    rgb, sigma = nerf(pos, view)  # Bx3
    print(f'{rgb.shape=} {sigma.shape=}')
    print(sum(p.numel() for p in nerf.parameters() if p.requires_grad) * 1e-6)
