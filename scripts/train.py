import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from model import get_nerf_models
from sampling import StratifiedSampler
from dataset import OptimizedNeRFRayDataset
from instantngp import InstantNGP

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def sample_pdf(bins, weights, N_samples, device):
    """
    Sample N_samples points from a distribution defined by weights and bins
    using inverse transform sampling.
    Args:
        bins: Edges of the bins (batch_size, N_bins).
        weights: Weights for each bin (batch_size, N_bins - 1). Should be positive.
        N_samples: Number of samples to draw per batch element.
    Returns:
        samples: Sampled points (batch_size, N_samples).
    """

    # Add small value to prevent nans
    weights = weights + 1e-5

    # Normalize weights to get PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [..., num_bins+1]

    # Sample positions in CDF
    shape = list(cdf.shape[:-1]) + [N_samples]
    u = torch.rand(shape, device=weights.device)
    u = u.contiguous()
    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    inds_g = torch.stack([below, above], dim=-1)  # [..., N_samples, 2]

    # Expand cdf and bins for gathering
    gather_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_exp = cdf.unsqueeze(-2).expand(gather_shape)
    bins_exp = bins.unsqueeze(-2).expand(gather_shape)

    cdf_g = torch.gather(cdf_exp, dim=-1, index=inds_g)
    bins_g = torch.gather(bins_exp, dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def raw2outputs(rgb, sigma, z_vals, white_bg=False):
    """
    Convert raw outputs to RGB and depth maps.
    Optimized version with improved memory efficiency.
    """
    # Compute the alpha values
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dist_last = torch.full_like(dists[..., :1], 1e6)
    dists = torch.cat([dists, dist_last], -1)

    # Calculate alpha (opacity)
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
    ones = torch.ones_like(alpha[..., :1])

    # Calculate transmittance using cumulative product
    transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], -1), -1)[
        :, :-1
    ]

    # Calculate the weights
    weights = alpha * transmittance

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    acc_map = torch.sum(weights, dim=-1)

    if white_bg:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, weights


# Process in larger chunks to better utilize GPU
def process_rays_in_chunks(model, pts, view_pts, device, chunk_size=8192):
    """Process rays in chunks to optimize memory usage and GPU utilization."""
    total_points = pts.shape[0]
    all_rgb, all_sigma = [], []
    
    for i in range(0, total_points, chunk_size):
        end_idx = min(i + chunk_size, total_points)
        chunk_pts = pts[i:end_idx]
        chunk_view = view_pts[i:end_idx]
        
        with torch.amp.autocast(device_type=device.type):
            rgb_chunk, sigma_chunk = model(chunk_pts, chunk_view)
        
        all_rgb.append(rgb_chunk)
        all_sigma.append(sigma_chunk)
    
    return torch.cat(all_rgb, 0), torch.cat(all_sigma, 0)


def forward_pass(
    model_coarse,
    model_fine,
    rays_o,
    rays_d,
    t_near,
    t_far,
    image_rgb,
    sampler,
    criterion,
    device,
    white_bg
):
    # Coarse sampling
    pts, view_pts, z_vals = sampler(rays_o, rays_d, t_near, t_far)


    # USE CHUNKED PROCESSING HERE
    rgb_coarse, sigma_coarse = process_rays_in_chunks(
        model_coarse, pts, view_pts, device, chunk_size=cfg.chunk_size
    )
    
    rgb_coarse = rgb_coarse.reshape(-1, cfg.N_samples, 3)
    sigma_coarse = sigma_coarse.reshape(-1, cfg.N_samples, 1)
    
    rgb_map_coarse, weights = raw2outputs(rgb_coarse, sigma_coarse, z_vals, white_bg)

    # Fine sampling
    z_vals_mid = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], cfg.N_importance, device)
    z_samples = z_samples.detach()

    # Combine coarse and fine samples
    pts_fine, view_pts_fine, z_vals_combined = sampler.fine_points(
        rays_o,
        rays_d,
        z_samples,
        z_vals,
        t_near,
        t_far,
        cfg.N_samples,
        cfg.N_importance,
    )

    rgb_fine, sigma_fine = process_rays_in_chunks(
        model_fine, pts_fine, view_pts_fine, device, chunk_size=cfg.chunk_size
    )
    rgb_fine = rgb_fine.reshape(-1, cfg.N_importance + cfg.N_samples, 3)
    sigma_fine = sigma_fine.reshape(-1, cfg.N_importance + cfg.N_samples, 1)
    
    rgb_map_fine, _ = raw2outputs(rgb_fine, sigma_fine, z_vals_combined, white_bg)

    # Loss computation
    loss_coarse = criterion(rgb_map_coarse, image_rgb)
    loss_fine = criterion(rgb_map_fine, image_rgb)
    total_loss = loss_coarse + loss_fine

    return total_loss, loss_coarse, loss_fine


def train(cfg: OmegaConf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.9)

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / "logs")
    dataset = OptimizedNeRFRayDataset(
        data_path=cfg.rays_file, use_memmap=cfg.use_memmap
    )

    # Optimized DataLoader configuration
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,  # Enable multiprocessing
        pin_memory=True,  # Faster CPU-GPU transfer
        persistent_workers=True,  # Avoid worker respawn overhead
        prefetch_factor=2,  # Prefetch batches
    )

    if cfg.ngp:
        model_fine = model_coarse = InstantNGP(cfg.nerf_type).to(device=device)
        def get_params():
            return model_coarse.parameters()
    else:
        model_coarse, model_fine = get_nerf_models(
            pos_L=cfg.pos_L, dir_L=cfg.dir_L, hidden_dim=cfg.hidden_dim, device=device
        )
        def get_params():
            return list(model_coarse.parameters()) + list(model_fine.parameters())

    def get_num_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"NERF: {get_num_params(model_fine) * 1e-6:.3f}M")
    
    # Compile models for PyTorch 2.0+ (if available)
    # Can provide a significant speed boost
    # try:
    #     model_coarse = torch.compile(model_coarse, mode="reduce-overhead")
    #     model_fine = torch.compile(model_fine, mode="reduce-overhead")
    #     print("Models compiled for faster execution")
    # except:
    #     print("Model compilation not available, using standard models")

    optimizer = optim.AdamW(
        get_params(),
        lr=cfg.lr,
        weight_decay=1e-6,
    )

    total_steps = cfg.num_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr * 10,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    # Mixed precision setup
    scaler = torch.amp.GradScaler()

    # Efficient loss computation
    criterion = torch.nn.MSELoss(reduction="mean")

    # Initialize optimized sampler
    sampler = StratifiedSampler(cfg.N_samples, device)

    model_coarse.train()
    model_fine.train()
    start_time = time.time()
    steps_per_epoch = len(dataloader)
    
    print(f"White background: {cfg.white_bg}")
    print(f"Starting optimized training for {cfg.num_epochs} epochs")
    print(f"Total steps: {total_steps}")
    
    for epoch in range(1, cfg.num_epochs + 1):
        for idx, ray_batch in enumerate(
            tqdm(dataloader, desc="Training", total=len(dataloader))
        ):

            # Efficient data transfer to GPU
            rays_o = ray_batch["ray_origin"].to(device, non_blocking=True)
            rays_d = ray_batch["ray_direction"].to(device, non_blocking=True)
            image_rgb = ray_batch["rgb"].to(device, non_blocking=True)
            t_near = ray_batch["t_near"].to(device, non_blocking=True).unsqueeze(-1)
            t_far = ray_batch["t_far"].to(device, non_blocking=True).unsqueeze(-1)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type):
                total_loss, loss_coarse, loss_fine = forward_pass(
                    model_coarse,
                    model_fine,
                    rays_o,
                    rays_d,
                    t_near,
                    t_far,
                    image_rgb,
                    sampler,
                    criterion,
                    device,
                    cfg.white_bg
                )

            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                get_params(),
                max_norm=1.0,
            )

            with torch.no_grad():
                psnr = -10.0 * torch.log10(loss_fine)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if idx % 100 == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"| Epoch: {epoch} | Iteration: {idx} | ",
                    f"Loss: {total_loss.item():.4f} (Coarse: {loss_coarse.item():.4f}, Fine: {loss_fine.item():.4f}) | ",
                    f"PSNR: {psnr.item():.2f} | Time: {elapsed_time:.2f}s |",
                )

                step = epoch * steps_per_epoch + idx
                writer.add_scalar("Loss/Coarse", loss_coarse.item(), step)
                writer.add_scalar("Loss/Fine", loss_fine.item(), step)
                writer.add_scalar("Loss/Total", total_loss.item(), step)
                writer.add_scalar("PSNR", psnr.item(), step)
                writer.add_scalar("Scheduler Step", scheduler.get_last_lr()[0], step)

        if epoch % 2 == 0:
            # Save the model checkpoint
            checkpoint_path = checkpoint_dir / f"nerf_epoch_{epoch:06d}.pth"
            torch.save(
                {
                    "iteration": epoch,
                    "nerf_coarse_state_dict": model_coarse.state_dict(),
                    "nerf_fine_state_dict": model_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final models
    final_checkpoint_path = checkpoint_dir / "nerf_final.pth"
    torch.save(
        {
            "iteration": cfg.num_epochs,
            "nerf_coarse_state_dict": model_coarse.state_dict(),
            "nerf_fine_state_dict": model_fine.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_checkpoint_path,
    )

    total_time = time.time() - start_time
    print(f"Saved final models to {final_checkpoint_path}")
    print(f"Training completed in {total_time:.2f}s")
    print("Training completed.")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    train(cfg)
