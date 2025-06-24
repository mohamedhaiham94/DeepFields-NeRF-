import torch


class StratifiedSampler:
    def __init__(self, N_samples, device):
        self.N_samples = N_samples
        self.device = device
        self.t_vals = torch.linspace(0, 1, N_samples, device=device)

    def __call__(self, rays_o, rays_d, near, far):

        # (N_rays, N_samples)
        z_vals = near * (1.0 - self.t_vals[None, :]) + far * self.t_vals[None, :]

        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = torch.rand(z_vals.shape, device=self.device)
        z_vals = lower + (upper - lower) * t_rand  # (N_rays, N_samples)

        # Sample points along the rays
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Repeat directions to match point samples
        input_dir = rays_d[:, None].expand(pts.shape)  # (N_rays, N_samples, 3)

        # Flatten the points
        pts_flat = pts.reshape(-1, 3)  # (N, 3)
        input_dir = input_dir.reshape(-1, 3)  # (N, 3)
        return pts_flat, input_dir, z_vals

    def fine_points(
        self, rays_o, rays_d, z_samples, z_vals, near, far, N_samples, N_importance
    ):
        """
        Optimized fine point sampling.
        """
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # Sample points along the rays
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        input_dir = rays_d[:, None].expand(pts.shape)

        # Flatten the points
        pts_flat = pts.reshape(-1, 3)
        input_dir = input_dir.reshape(-1, 3)
        return pts_flat, input_dir, z_vals
