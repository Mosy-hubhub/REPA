import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    # Added for fixed sigma/t evaluation: centralize timestep resolution so
    # deterministic evaluation can reuse persistent t values instead of
    # sampling fresh randomness inside the loss.
    def _resolve_time_input(self, images, fixed_time=None):
        if fixed_time is not None:
            if torch.is_tensor(fixed_time):
                time_input = fixed_time.to(device=images.device, dtype=images.dtype)
                if time_input.ndim == 0:
                    time_input = time_input.view(1, 1, 1, 1).expand(images.shape[0], 1, 1, 1)
                elif time_input.ndim == 1:
                    time_input = time_input.view(-1, 1, 1, 1)
                return time_input
            return torch.full(
                (images.shape[0], 1, 1, 1),
                float(fixed_time),
                device=images.device,
                dtype=images.dtype,
            )

        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return time_input.to(device=images.device, dtype=images.dtype)

    # Added for fixed sigma/t evaluation: centralize diffusion noise creation
    # so deterministic evaluation can inject saved perturbations.
    def _resolve_diffusion_noise(self, images, fixed_noise=None):
        if fixed_noise is not None:
            return fixed_noise.to(device=images.device, dtype=images.dtype)
        return torch.randn_like(images)

    # Added for fixed sigma/t evaluation: expose the denoising MSE branch as a
    # reusable function that accepts fixed time and fixed noise inputs.
    def compute_denoising_loss(self, model, images, model_kwargs=None, fixed_time=None, fixed_noise=None):
        if model_kwargs is None:
            model_kwargs = {}

        time_input = self._resolve_time_input(images, fixed_time=fixed_time)
        noises = self._resolve_diffusion_noise(images, fixed_noise=fixed_noise)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        return denoising_loss, {
            "time_input": time_input,
            "sigma_t": sigma_t,
            "model_input": model_input,
            "model_target": model_target,
            "model_output": model_output,
            "zs_tilde": zs_tilde,
        }

    # Added for fixed sigma/t evaluation: keep the REPA alignment branch
    # reusable so train/val fixed-eval can monitor the projection term and the
    # full training objective instead of only the denoising MSE.
    def compute_projection_loss(self, zs, zs_tilde):
        if zs is None or zs_tilde is None:
            raise ValueError("Projection loss requires both encoder targets and model projector outputs.")

        per_encoder_losses = []
        for z, z_tilde in zip(zs, zs_tilde):
            z_tilde = F.normalize(z_tilde, dim=-1)
            z = F.normalize(z, dim=-1)
            per_encoder_losses.append(-(z * z_tilde).sum(dim=-1).mean(dim=-1))

        return torch.stack(per_encoder_losses, dim=0).mean(dim=0)

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss
