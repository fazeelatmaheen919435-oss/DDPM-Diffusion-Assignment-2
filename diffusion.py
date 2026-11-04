import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def gather_timestep_coeff(schedule, timesteps, x_shape):
    """Picks the schedule value for each batch item and reshapes it for broadcasting."""
    b, *_ = timesteps.shape
    out = schedule.gather(-1, timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """Cosine schedule: returns alpha_t for t = 0..T-1."""
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        alphas = cosine_schedule(self.num_timesteps)                  # alpha_t
        alphas_bar = torch.cumprod(alphas, dim=0)                     # alpha_bar_t
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)   # alpha_bar_{t-1}, 1 at t=0
        betas = 1.0 - alphas

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance", betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        # posterior mean coefficients (Eq. 18): mu = coef1 * x_t + coef2 * x0_hat
        self.register_buffer(
            "posterior_mean_coef1", torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        self.register_buffer(
            "posterior_mean_coef2", torch.sqrt(alphas_bar_prev) * (1.0 - alphas) / (1.0 - alphas_bar)
        )

    def q_sample(self, x_0, t, noise):
        """x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"""
        x_t = (
            gather_timestep_coeff(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + gather_timestep_coeff(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        return x_t

    def p_losses(self, x_0, t, noise):
        """L1 loss between the true noise and the noise predicted by the U-Net."""
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(predicted_noise, noise)
        return loss

    def forward(self, x_0, noise):
        """Wrapper for p_losses: picks a random timestep per image."""
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_0, t, noise)

    def noise_like(self, shape, device):
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """One reverse step: x_t -> x_{t-1}."""
        sqrt_ab_t = gather_timestep_coeff(self.sqrt_alphas_bar, t, x.shape)
        sqrt_one_minus_ab_t = gather_timestep_coeff(self.sqrt_one_minus_alphas_bar, t, x.shape)
        coef1 = gather_timestep_coeff(self.posterior_mean_coef1, t, x.shape)
        coef2 = gather_timestep_coeff(self.posterior_mean_coef2, t, x.shape)

        predicted_noise = self.model(x, t)                               # eps_theta(x_t, t)
        x0_hat = (x - sqrt_one_minus_ab_t * predicted_noise) / sqrt_ab_t  # estimated x_0
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)                           # rectify x_0
        model_mean = coef1 * x + coef2 * x0_hat                           # posterior mean

        if t_index == 0:
            return model_mean
        else:
            posterior_var_t = gather_timestep_coeff(self.posterior_variance, t, x.shape)
            noise = self.noise_like(x.shape, x.device)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, img):
        """Runs all reverse steps, then clamps to [-1, 1] and maps to [0, 1]."""
        b = img.shape[0]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling", total=self.num_timesteps):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        img = torch.clamp(img, -1.0, 1.0)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, batch_size):
        """Starts from pure noise and generates `batch_size` images."""
        device = self.betas.device
        img = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), device)
        return self.p_sample_loop(img)
