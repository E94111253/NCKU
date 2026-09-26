import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):

    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s= 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-4, max=0.999)

    return betas


def extract(values, t, x_shape):
    batch_size = t.shape[0]
    t = t.to(values.device).long()
    out = values.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size=64, channels=3, timesteps=1000, beta_schedule="linear", loss_type="mse"):
        super().__init__()

        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.loss_type = loss_type

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def p_losses(self, x_start, t, condition, noise=None, cond_drop_prob=0.0):
 
        if noise is None:
            noise = torch.randn_like(x_start)
        if cond_drop_prob > 0:
            drop_mask = torch.rand(condition.shape[0], device=condition.device) < cond_drop_prob
            condition = condition.clone()
            condition[drop_mask] = 0.0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.model(x_noisy, t, condition)

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_noise, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, condition, guidance_scale=0.0):
       
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)

        
        pred_noise_cond = self.model(x, t, condition)
       
        if guidance_scale > 0:
            null_condition = torch.zeros_like(condition)
            pred_noise_uncond = self.model(x, t, null_condition)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        else:
            pred_noise = pred_noise_cond


        model_mean = sqrt_recip_alpha_t * (x - betas_t * pred_noise / sqrt_one_minus_alpha_bar_t)

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, condition, batch_size=None, guidance_scale=0.0, return_process=False, process_steps=8):
  
        device = condition.device

        if batch_size is None:
            batch_size = condition.shape[0]

        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)

        process_images: List[torch.Tensor] = []

        if return_process:
            save_interval = max(1, self.timesteps // process_steps)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x=x, t=t, condition=condition, guidance_scale=guidance_scale)

            if return_process and (i % save_interval == 0 or i == self.timesteps - 1 or i == 0):
                process_images.append(x.detach().cpu())

        if return_process:
            return x, process_images

        return x

    @staticmethod
    def denormalize(x: torch.Tensor):
        return (x.clamp(-1, 1) + 1) / 2


if __name__ == "__main__":
    from models import ConditionalUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNet(image_channels=3, condition_dim=24, base_channels=64, time_dim=256).to(device)

    diffusion = GaussianDiffusion(model=model, image_size=64, channels=3, timesteps=1000, beta_schedule="linear", loss_type="mse").to(device)

    x_start = torch.rand(8, 3, 64, 64).to(device)*2-1
    condition = torch.zeros(8, 24).to(device)
    condition[:, 0] = 1.0

    t = torch.randint(0, diffusion.timesteps, (8,), device=device).long()

    loss = diffusion.p_losses(x_start=x_start, t=t, condition=condition)
    # samples = diffusion.sample(condition=condition, batch_size=8, guidance_scale=0.0)

    print("Device:", device)
    print("Loss:", loss.item())
    # print("Sample shape:", samples.shape)
    # print("Sample min/max:", samples.min().item(), samples.max().item())
    # print("Denormalized min/max:", diffusion.denormalize(samples).min().item(), diffusion.denormalize(samples).max().item())

    # x = torch.rand(8, 3, 64, 64).to(device) * 2 - 1
    # t = torch.randint(0, 1000, (8,), device=device)
    # condition = torch.zeros(8, 24).to(device)
    # condition[:, 0] = 1.0

    # with torch.no_grad():
    #     y = model(x, t, condition)

    # print(torch.isnan(y).any())
    # print(y.min(), y.max())