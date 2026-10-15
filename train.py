import argparse
try:
    BooleanOptionalAction = argparse.BooleanOptionalAction
except AttributeError:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            option_strings_with_no = []
            for option_string in option_strings:
                option_strings_with_no.append(option_string)
                if option_string.startswith("--"):
                    option_strings_with_no.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=option_strings_with_no,
                dest=dest,
                nargs=0,
                default=default,
                **kwargs,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders
from utils.fixed_eval import (
    build_or_load_fixed_eval_cache,
    get_rank_positions,
    parse_float_sequence,
    sigma_to_time,
)
from utils.lr_warmup import apply_scheduled_lr
from utils.wandb_resume import get_wandb_init_kwargs
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0., noise=None):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    # Added for fixed sigma/t evaluation tracking: allow callers to reuse
    # persisted posterior noise so latent construction stays deterministic.
    if noise is None:
        noise = torch.randn_like(mean)
    else:
        noise = noise.to(device=device, dtype=mean.dtype)
    z = mean + std * noise
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999, optimizer=None, use_optimizer_z=False):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    optimizer_state = None
    if use_optimizer_z:
        if optimizer is None:
            raise ValueError("optimizer must be provided when use_optimizer_z=True")
        optimizer_state = get_base_optimizer(optimizer).state

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        source_tensor = param.data
        if use_optimizer_z:
            state = optimizer_state.get(param, None)
            if state is not None and "z" in state:
                source_tensor = state["z"].to(device=param.device, dtype=param.dtype)
        ema_params[name].mul_(decay).add_(source_tensor, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# Added for Nexus compatibility: match the per-parameter gradient
# normalization used by NexusInnerSGDPrecisionTrainer when optimizing cosine
# gradient alignment instead of raw dot-product updates.
def normalize_parameter_gradients(parameters, epsilon=1e-8):
    for param in parameters:
        if param.grad is not None:
            param.grad = param.grad / (torch.norm(param.grad) + float(epsilon))


# Added for Nexus compatibility: keep the accumulated inner-loop SGD updates in
# simple parameter-order buffers so we can roll them back before the outer
# optimizer step without introducing a second trainer abstraction.
def allocate_parameter_update_buffers(parameters):
    return [torch.zeros_like(param, requires_grad=False) for param in parameters]


# Added for Nexus compatibility: accumulate the exact temporary parameter
# displacement caused by the inner SGD steps.
def accumulate_parameter_updates(parameters, update_buffers, inner_lr):
    scale = -float(inner_lr)
    for param, update_buffer in zip(parameters, update_buffers):
        if param.grad is not None:
            update_buffer.add_(param.grad.detach(), alpha=scale)


# Added for Nexus compatibility: rollback the temporary inner-loop SGD updates
# and expose the resulting pseudo-gradient to the outer optimizer.
def rollback_parameter_updates_and_set_gradients(parameters, update_buffers, inner_lr):
    scale = float(inner_lr)
    with torch.no_grad():
        for param, update_buffer in zip(parameters, update_buffers):
            param.add_(-update_buffer)
            param.grad = (-update_buffer / scale).to(dtype=torch.float32)


# Added for Nexus compatibility: clear the persistent inner-loop update
# buffers after every completed outer optimization step.
def zero_parameter_update_buffers(update_buffers):
    for update_buffer in update_buffers:
        update_buffer.zero_()






# Added for LR scheduling: decouple the optimizer schedule step from the
# checkpoint/global step so resumed runs can restart a decay phase while
# preserving continuous wandb and checkpoint step numbering.
def get_lr_schedule_step(global_step, lr_schedule_start_step):
    return max(1, int(global_step) + 1 - int(lr_schedule_start_step))


def resolve_resume_checkpoint_path(args):
    """
    Resolve the checkpoint path for resumed training.
    """
    if args.resume_checkpoint_path:
        return os.path.expanduser(args.resume_checkpoint_path)

    if args.resume_step > 0:
        ckpt_name = f"{int(args.resume_step):07d}.pt"
        return os.path.join(args.output_dir, args.exp_name, "checkpoints", ckpt_name)

    return None


def maybe_restore_args_from_checkpoint(args):
    """
    Optionally replace the current CLI args with the args persisted inside the
    resume checkpoint. This is useful when resuming a partially completed decay
    run and wanting to keep the original schedule/run configuration without
    manually re-entering every flag.
    """
    if not getattr(args, "resume_use_checkpoint_args", False):
        return args

    resume_checkpoint_path = resolve_resume_checkpoint_path(args)
    if resume_checkpoint_path is None:
        raise ValueError(
            "--resume-use-checkpoint-args requires --resume-checkpoint-path or --resume-step."
        )
    if not os.path.isfile(resume_checkpoint_path):
        raise FileNotFoundError(
            "Resume checkpoint not found while restoring checkpoint args: "
            f"{resume_checkpoint_path}"
        )

    checkpoint = torch.load(
        resume_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "args" not in checkpoint:
        raise KeyError(
            "The resume checkpoint does not contain an 'args' entry, so "
            "--resume-use-checkpoint-args cannot restore the original run configuration."
        )

    restored_args = copy.deepcopy(checkpoint["args"])

    # Preserve only the operational CLI flags that decide which checkpoint to
    # resume from and whether wandb should intentionally start a fresh run.
    restored_args.resume_step = args.resume_step
    restored_args.resume_checkpoint_path = args.resume_checkpoint_path
    restored_args.resume_use_checkpoint_args = args.resume_use_checkpoint_args
    restored_args.force_new_wandb_run = args.force_new_wandb_run

    return restored_args


def get_base_optimizer(optimizer):
    """
    Unwrap Accelerate's optimizer wrapper when present.
    """
    return getattr(optimizer, "optimizer", optimizer)


def set_optimizer_train_mode(optimizer, train_mode):
    """
    Toggle optimizers that expose explicit train/eval parameter views, such as
    Schedule-Free optimizers. Standard PyTorch optimizers simply ignore this.
    """
    base_optimizer = get_base_optimizer(optimizer)
    method_name = "train" if train_mode else "eval"
    method = getattr(base_optimizer, method_name, None)
    if callable(method):
        method()


def get_optimizer_logged_lr(optimizer, default_lr):
    """
    Read the learning rate that should be logged to wandb. Schedule-Free
    optimizers expose the warmup-adjusted value via "scheduled_lr".
    """
    base_optimizer = get_base_optimizer(optimizer)
    param_groups = getattr(base_optimizer, "param_groups", None)
    if not param_groups:
        return float(default_lr)

    group = param_groups[0]
    scheduled_lr = group.get("scheduled_lr", None)
    if scheduled_lr is not None:
        try:
            scheduled_lr_value = float(scheduled_lr)
            if scheduled_lr_value > 0:
                return scheduled_lr_value
        except (TypeError, ValueError):
            pass

    return float(group.get("lr", default_lr))


def should_alias_schedulefree_x_to_ema(args):
    """
    Schedule-Free comparisons should use the optimizer eval-mode x-weights in
    the legacy EMA slots so downstream checkpoints/metrics stay comparable to
    methods that report EMA results only.
    """
    return args.optimizer_mode == "adamw_schedulefree"


def should_maintain_explicit_ema(args):
    """
    Standard optimizers keep a separate EMA shadow model. Schedule-Free runs use
    the optimizer eval-mode x-weights instead, so maintaining an extra EMA(z)
    copy is unnecessary.
    """
    return args.optimizer_mode != "adamw_schedulefree"


# Added for fixed sigma/t evaluation tracking: build a deterministic subset per
# process so train/test loss curves are measured on the same examples every time.
def create_fixed_subset_loader(dataset, requested_samples, batch_size, num_workers, accelerator):
    total_samples = min(int(requested_samples), len(dataset))
    local_positions = get_rank_positions(
        total_samples,
        accelerator.num_processes,
        accelerator.process_index,
    )
    subset = Subset(dataset, local_positions)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader, local_positions, total_samples


@torch.no_grad()
# Added for fixed sigma/t evaluation: reuse the exact same image preprocessing
# and encoder feature extraction path during deterministic train/val probes so
# the logged projection/total losses match the training objective definition.
def extract_encoder_features(raw_image, encoders, encoder_types, accelerator):
    zs = []
    with accelerator.autocast():
        for encoder, encoder_type in zip(encoders, encoder_types):
            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
            z = encoder.forward_features(raw_image_)
            if 'mocov3' in encoder_type:
                z = z[:, 1:]
            if 'dinov2' in encoder_type:
                z = z['x_norm_patchtokens']
            zs.append(z)
    return zs


# Added for fixed sigma/t evaluation tracking: normalize the user-facing
# sigma/t configuration into named evaluation targets for wandb curves.
def build_fixed_eval_plan(args):
    eval_plan = []

    # Added for fixed sigma/t evaluation tracking: allow the user to append a
    # deterministic random-t curve alongside specific sigma/t probe points.
    if args.fixed_eval_include_random:
        eval_plan.append({"name": "random_t", "fixed_time": None})

    if args.fixed_eval_mode == "sigma":
        for sigma_value in parse_float_sequence(args.fixed_eval_sigmas):
            eval_plan.append(
                {
                    "name": f"sigma_{sigma_value:.4f}",
                    "fixed_time": sigma_to_time(sigma_value, args.path_type),
                }
            )
    elif args.fixed_eval_mode == "t":
        for t_value in parse_float_sequence(args.fixed_eval_ts):
            eval_plan.append(
                {
                    "name": f"t_{float(t_value):.4f}",
                    "fixed_time": float(t_value),
                }
            )
    else:
        raise ValueError(f"Unsupported fixed evaluation mode: {args.fixed_eval_mode}")

    if not eval_plan:
        raise ValueError(
            "Fixed evaluation requested but no sigma/t values were provided and random_t was not enabled."
        )

    return eval_plan


def parse_int_step_list(step_values):
    """
    Parse a comma-separated checkpoint step list such as "320000,321000".
    """
    if step_values is None:
        return set()

    parsed_steps = set()
    for raw_value in str(step_values).split(","):
        value = raw_value.strip()
        if not value:
            continue
        step = int(value)
        if step <= 0:
            raise ValueError("Checkpoint steps must be positive integers.")
        parsed_steps.add(step)
    return parsed_steps




def _normalize_tracked_points_for_cache(tracked_points):
    normalized_points = []
    for tracked_point in tracked_points:
        normalized_points.append(
            {
                "name": str(tracked_point["name"]),
                "fixed_time": None if tracked_point["fixed_time"] is None else float(tracked_point["fixed_time"]),
            }
        )
    return normalized_points














def compute_training_losses(
    model,
    loss_fn,
    images,
    labels,
    zs,
    proj_coeff,
    raw_image=None,
):
    model_kwargs = dict(y=labels)
    denoising_loss, denoising_details = loss_fn.compute_denoising_loss(
        model=model,
        images=images,
        model_kwargs=model_kwargs,
    )
    projection_loss = loss_fn.compute_projection_loss(
        zs=zs,
        zs_tilde=denoising_details["zs_tilde"],
    )


    total_loss = denoising_loss + proj_coeff * projection_loss
    return {
        "denoising_loss": denoising_loss,
        "projection_loss": projection_loss,
        "total_loss": total_loss,
        "denoising_details": denoising_details,
    }


@torch.no_grad()
# Added for fixed sigma/t evaluation tracking: compute deterministic
# denoising/projection/total loss curves on persistent train/val subsets and
# aggregate them across ranks for stable wandb comparisons.
def evaluate_fixed_losses(
    model,
    loss_fn,
    dataloader,
    cache,
    cache_positions,
    tracked_points,
    accelerator,
    device,
    latents_scale,
    latents_bias,
    encoders,
    encoder_types,
    proj_coeff,
):
    metric_keys = ["denoising", "projection", "total"]
    local_sums = {
        metric_key: {item["name"]: 0.0 for item in tracked_points}
        for metric_key in metric_keys
    }
    local_counts = {item["name"]: 0 for item in tracked_points}
    cache_offset = 0

    for raw_image, moments, labels in dataloader:
        batch_size = moments.shape[0]
        batch_positions = cache_positions[cache_offset: cache_offset + batch_size]
        cache_offset += batch_size

        raw_image = raw_image.to(device)
        moments = moments.squeeze(dim=1).to(device)
        labels = labels.to(device)
        zs = extract_encoder_features(
            raw_image=raw_image,
            encoders=encoders,
            encoder_types=encoder_types,
            accelerator=accelerator,
        )
        posterior_noise = cache["posterior_noise"][batch_positions].to(device=device, dtype=moments.dtype)
        latents = sample_posterior(
            moments,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            noise=posterior_noise,
        )
        diffusion_noise = cache["diffusion_noise"][batch_positions].to(device=device, dtype=latents.dtype)
        random_t = cache["random_t"][batch_positions].to(device=device, dtype=latents.dtype)
        model_kwargs = dict(y=labels)

        for tracked_point in tracked_points:
            fixed_time = random_t if tracked_point["fixed_time"] is None else tracked_point["fixed_time"]
            with accelerator.autocast():
                denoising_loss, denoising_details = loss_fn.compute_denoising_loss(
                    model=model,
                    images=latents,
                    model_kwargs=model_kwargs,
                    fixed_time=fixed_time,
                    fixed_noise=diffusion_noise,
                )
                projection_loss = loss_fn.compute_projection_loss(
                    zs=zs,
                    zs_tilde=denoising_details["zs_tilde"],
                )
                total_loss = denoising_loss + proj_coeff * projection_loss
            local_sums["denoising"][tracked_point["name"]] += denoising_loss.detach().sum().item()
            local_sums["projection"][tracked_point["name"]] += projection_loss.detach().sum().item()
            local_sums["total"][tracked_point["name"]] += total_loss.detach().sum().item()
            local_counts[tracked_point["name"]] += denoising_loss.shape[0]

    metrics = {metric_key: {} for metric_key in metric_keys}
    for metric_key in metric_keys:
        for tracked_point in tracked_points:
            local_stats = torch.tensor(
                [[local_sums[metric_key][tracked_point["name"]], float(local_counts[tracked_point["name"]])]],
                device=device,
                dtype=torch.float64,
            )
            gathered_stats = accelerator.gather(local_stats)
            global_sum = gathered_stats[:, 0].sum().item()
            global_count = gathered_stats[:, 1].sum().item()
            metrics[metric_key][tracked_point["name"]] = global_sum / global_count if global_count > 0 else 0.0

    return metrics


def run_fixed_evaluation(
    model,
    ema,
    loss_fn,
    fixed_train_loader,
    fixed_train_cache,
    fixed_train_positions,
    fixed_test_loader,
    fixed_test_cache,
    fixed_test_positions,
    fixed_eval_plan,
    accelerator,
    device,
    latents_scale,
    latents_bias,
    encoders,
    encoder_types,
    proj_coeff,
    use_schedulefree_x_as_ema,
    step,
):
    """Run and log the deterministic fixed-eval suite at an arbitrary step."""
    model_was_training = model.training
    model.eval()
    logged_models = []
    if not use_schedulefree_x_as_ema:
        logged_models.append(("", model))
    logged_models.append(("_ema", model if use_schedulefree_x_as_ema else ema))

    metric_prefixes = {
        "denoising": ("fixed_train_mse", "fixed_test_mse"),
        "projection": ("fixed_train_proj_loss", "fixed_val_proj_loss"),
        "total": ("fixed_train_total_loss", "fixed_val_total_loss"),
    }
    fixed_eval_logs = {}
    for suffix, eval_model in logged_models:
        fixed_train_metrics = evaluate_fixed_losses(
            model=eval_model,
            loss_fn=loss_fn,
            dataloader=fixed_train_loader,
            cache=fixed_train_cache,
            cache_positions=fixed_train_positions,
            tracked_points=fixed_eval_plan,
            accelerator=accelerator,
            device=device,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            encoders=encoders,
            encoder_types=encoder_types,
            proj_coeff=proj_coeff,
        )
        fixed_val_metrics = evaluate_fixed_losses(
            model=eval_model,
            loss_fn=loss_fn,
            dataloader=fixed_test_loader,
            cache=fixed_test_cache,
            cache_positions=fixed_test_positions,
            tracked_points=fixed_eval_plan,
            accelerator=accelerator,
            device=device,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            encoders=encoders,
            encoder_types=encoder_types,
            proj_coeff=proj_coeff,
        )
        for metric_name, (train_prefix, val_prefix) in metric_prefixes.items():
            if metric_name not in fixed_train_metrics:
                continue
            fixed_eval_logs.update(
                {
                    f"{train_prefix}{suffix}/{probe_name}": metric_value
                    for probe_name, metric_value in fixed_train_metrics[metric_name].items()
                }
            )
            fixed_eval_logs.update(
                {
                    f"{val_prefix}{suffix}/{probe_name}": metric_value
                    for probe_name, metric_value in fixed_val_metrics[metric_name].items()
                }
            )


    accelerator.log(fixed_eval_logs, step=step)
    if model_was_training:
        model.train()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    if args.optimizer_mode == "adamw_schedulefree":
        if AdamWScheduleFree is None:
            raise ImportError(
                "optimizer-mode=adamw_schedulefree requires the schedulefree package. "
                "Install facebookresearch/schedule_free first."
            )
        if args.lr_schedule != "constant":
            raise ValueError(
                "AdamWScheduleFree manages its own schedule-free updates and should "
                "not be combined with external LR schedules. Use --lr-schedule=constant "
                "and control warmup with --warmup-steps instead."
            )

    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = f"{save_dir}/checkpoints"
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = dict(vars(args))
        if isinstance(args_dict.get("extra_checkpoint_steps"), set):
            args_dict["extra_checkpoint_steps"] = sorted(args_dict["extra_checkpoint_steps"])
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    maintain_explicit_ema = should_maintain_explicit_ema(args)
    ema = deepcopy(model).to(device) if maintain_explicit_ema else None
    # Added for offline/local setups: allow the training entrypoint to load the
    # Stable Diffusion VAE either from a Hugging Face repo id or from a local
    # directory supplied with --vae-path.
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    if ema is not None:
        requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.optimizer_mode == "adamw_schedulefree":
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            warmup_steps=args.warmup_steps,
            inner_momentum=args.sf_inner_momentum,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    # Added for Nexus compatibility: the stable NexusInnerSGDPrecisionTrainer
    # variant uses a temporary SGD inner loop plus the existing outer
    # optimizer. Keeping the outer optimizer unchanged minimizes integration
    # cost with REPA's checkpointing, EMA, and logging stack.
    inner_optimizer = None
    if args.optimizer_mode == "nexus-v":
        inner_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.nexus_inner_lr,
            momentum=0.0,
        )
        if args.gradient_accumulation_steps % args.nexus_inner_accumulation_steps != 0:
            raise ValueError(
                "For Nexus mode, --gradient-accumulation-steps must be divisible by "
                "--nexus-inner-accumulation-steps."
            )
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # Added for fixed sigma/t evaluation tracking: let train/test curves use a
    # deterministic subset from the training data and an optional held-out path.
    eval_data_dir = args.eval_data_dir if args.eval_data_dir is not None else args.data_dir
    eval_dataset = CustomDataset(eval_data_dir)
    fixed_eval_batch_size = args.fixed_eval_batch_size if args.fixed_eval_batch_size > 0 else local_batch_size
    fixed_eval_batch_size = max(1, fixed_eval_batch_size)
    fixed_eval_plan = build_fixed_eval_plan(args)
    fixed_train_loader, fixed_train_positions, fixed_train_size = create_fixed_subset_loader(
        train_dataset,
        args.fixed_eval_num_samples,
        fixed_eval_batch_size,
        args.num_workers,
        accelerator,
    )
    fixed_test_loader, fixed_test_positions, fixed_test_size = create_fixed_subset_loader(
        eval_dataset,
        args.fixed_eval_num_samples,
        fixed_eval_batch_size,
        args.num_workers,
        accelerator,
    )
    # Added for cross-run comparability: keep the deterministic fixed-eval
    # caches under output_dir so sibling runs (for example warmup vs decay)
    # can reuse the same persisted train/val noise configuration.
    fixed_eval_dir = os.path.join(args.output_dir, "fixed_eval")
    train_cache_path = os.path.join(fixed_eval_dir, "train_cache.pt")
    test_cache_path = os.path.join(fixed_eval_dir, "test_cache.pt")
    fixed_train_cache = None
    fixed_test_cache = None
    fixed_train_cache_status = "disabled"
    fixed_test_cache_status = "disabled"
    if args.fixed_eval_steps > 0:
        latent_shape = [4, latent_size, latent_size]
        if accelerator.is_main_process:
            _, fixed_train_cache_status = build_or_load_fixed_eval_cache(
                train_cache_path,
                fixed_train_size,
                latent_shape,
                args.fixed_eval_seed,
                return_status=True,
            )
            _, fixed_test_cache_status = build_or_load_fixed_eval_cache(
                test_cache_path,
                fixed_test_size,
                latent_shape,
                args.fixed_eval_seed + 1,
                return_status=True,
            )
        accelerator.wait_for_everyone()
        fixed_train_cache = torch.load(train_cache_path, map_location="cpu")
        fixed_test_cache = torch.load(test_cache_path, map_location="cpu")
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
        logger.info(f"Fixed eval train subset size: {fixed_train_size:,}")
        logger.info(f"Fixed eval val subset size: {fixed_test_size:,} ({eval_data_dir})")
        logger.info(f"Fixed eval train cache: {fixed_train_cache_status} ({train_cache_path})")
        logger.info(f"Fixed eval val cache: {fixed_test_cache_status} ({test_cache_path})")
        logger.info(
            "Fixed eval targets: "
            + ", ".join(point["name"] for point in fixed_eval_plan)
        )
    
    # Prepare models for training:
    if ema is not None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if ema is not None:
        ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    resume_checkpoint_path = resolve_resume_checkpoint_path(args)
    if resume_checkpoint_path is not None:
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{resume_checkpoint_path}. "
                "Pass --resume-checkpoint-path to load from another experiment directory, "
                "or make sure --exp-name and --resume-step point to an existing checkpoint."
            )
        ckpt = torch.load(
            resume_checkpoint_path,
            map_location='cpu',
            # PyTorch 2.6 changed torch.load() to default to weights_only=True.
            # Our training checkpoints also persist argparse.Namespace in "args",
            # so resumed runs must explicitly opt into full checkpoint loading.
            weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        if ema is not None:
            ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        if inner_optimizer is not None and "inner_opt" in ckpt:
            inner_optimizer.load_state_dict(ckpt["inner_opt"])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            logger.info(f"Resumed checkpoint from {resume_checkpoint_path} at step {global_step}")

    # Removed for Nexus compatibility:
    # model, optimizer, train_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader
    # )
    # Added for Nexus compatibility: prepare the optional inner optimizer only
    # when the Nexus training path is enabled.
    if inner_optimizer is not None:
        model, optimizer, inner_optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, inner_optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    set_optimizer_train_mode(optimizer, train_mode=True)

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        # Added for wandb resume: reuse the persisted run id whenever training
        # is resumed from a checkpoint so the wandb history stays continuous.
        wandb_init_kwargs = get_wandb_init_kwargs(
            save_dir=save_dir,
            run_name=f"{args.exp_name}",
            should_resume=resume_checkpoint_path is not None,
            force_new_run=args.force_new_wandb_run,
        )
        accelerator.init_trackers(
            project_name="REPA_updated",
            config=tracker_config,
            init_kwargs=wandb_init_kwargs,
        )

    # A resumed branch should expose the parent checkpoint's deterministic
    # fixed-eval metrics before its first optimizer update. This creates an
    # explicit branch origin at the restored global step in the new W&B run.
    should_run_resume_fixed_eval = (
        resume_checkpoint_path is not None
        and args.fixed_eval_steps > 0
        and global_step >= args.fixed_eval_start_step
    )
    if should_run_resume_fixed_eval:
        schedulefree_eval_context = args.optimizer_mode == "adamw_schedulefree"
        if schedulefree_eval_context:
            set_optimizer_train_mode(optimizer, train_mode=False)
        if accelerator.is_main_process:
            logger.info("Running fixed evaluation at resumed step %d before training.", global_step)
        run_fixed_evaluation(
            model=model,
            ema=ema,
            loss_fn=loss_fn,
            fixed_train_loader=fixed_train_loader,
            fixed_train_cache=fixed_train_cache,
            fixed_train_positions=fixed_train_positions,
            fixed_test_loader=fixed_test_loader,
            fixed_test_cache=fixed_test_cache,
            fixed_test_positions=fixed_test_positions,
            fixed_eval_plan=fixed_eval_plan,
            accelerator=accelerator,
            device=device,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            encoders=encoders,
            encoder_types=encoder_types,
            proj_coeff=args.proj_coeff,
            use_schedulefree_x_as_ema=should_alias_schedulefree_x_to_ema(args),
            step=global_step,
        )
        if schedulefree_eval_context:
            set_optimizer_train_mode(optimizer, train_mode=True)
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    # Added for Nexus compatibility: allocate the persistent inner-loop update
    # buffers after accelerator.prepare so they match the wrapped model
    # parameters exactly.
    nexus_trainable_params = None
    nexus_update_buffers = None
    nexus_micro_step = 0
    if args.optimizer_mode == "nexus-v":
        nexus_trainable_params = [param for param in model.parameters() if param.requires_grad]
        nexus_update_buffers = allocate_parameter_update_buffers(nexus_trainable_params)
        optimizer.zero_grad(set_to_none=True)
        inner_optimizer.zero_grad(set_to_none=True)
        
    for epoch in range(args.epochs):
        model.train()
        set_optimizer_train_mode(optimizer, train_mode=True)
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            if args.optimizer_mode == "nexus-v":
                # Added for Nexus compatibility: reproduce the stable
                # NexusInnerSGDPrecisionTrainer behavior inside REPA's existing
                # training loop instead of introducing a second trainer stack.
                loss_outputs = compute_training_losses(
                    model=model,
                    loss_fn=loss_fn,
                    images=x,
                    labels=labels,
                    zs=zs,
                    proj_coeff=args.proj_coeff,
                    raw_image=raw_image,
                )
                loss_mean = loss_outputs["denoising_loss"].mean()
                proj_loss_mean = loss_outputs["projection_loss"].mean()
                total_loss_mean = loss_outputs["total_loss"].mean()
                scaled_total_loss = total_loss_mean / args.gradient_accumulation_steps

                accelerator.backward(scaled_total_loss)
                nexus_micro_step += 1

                if nexus_micro_step % args.nexus_inner_accumulation_steps == 0:
                    if args.nexus_gradient_mode == "cosine":
                        normalize_parameter_gradients(
                            nexus_trainable_params,
                            epsilon=args.nexus_normalize_gradient_epsilon,
                        )
                    accumulate_parameter_updates(
                        nexus_trainable_params,
                        nexus_update_buffers,
                        args.nexus_inner_lr,
                    )
                    inner_optimizer.step()
                    inner_optimizer.zero_grad(set_to_none=True)

                # Added for Nexus compatibility: only complete a visible/global
                # step after the outer optimizer consumes the accumulated inner
                # SGD displacement.
                if nexus_micro_step % args.gradient_accumulation_steps == 0:
                    # Added for LR scheduling: keep the wandb/global step continuous
                    # while allowing the LR schedule to restart from a user-chosen step.
                    lr_schedule_step = get_lr_schedule_step(
                        global_step=global_step,
                        lr_schedule_start_step=args.lr_schedule_start_step,
                    )
                    # Added for LR scheduling: update the optimizer LR from the selected
                    # schedule right before the optimizer step so wandb can track it.
                    if args.optimizer_mode == "adamw_schedulefree":
                        current_lr = get_optimizer_logged_lr(optimizer, args.learning_rate)
                    else:
                        current_lr = apply_scheduled_lr(
                            optimizer=optimizer,
                            step=lr_schedule_step,
                            base_lr=args.learning_rate,
                            schedule_type=args.lr_schedule,
                            warmup_steps=args.warmup_steps,
                            decay_steps=args.lr_decay_steps if args.lr_decay_steps > 0 else args.max_train_steps,
                            min_lr=args.min_lr,
                        )

                    rollback_parameter_updates_and_set_gradients(
                        nexus_trainable_params,
                        nexus_update_buffers,
                        args.nexus_inner_lr,
                    )
                    grad_norm = accelerator.clip_grad_norm_(nexus_trainable_params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    inner_optimizer.zero_grad(set_to_none=True)
                    zero_parameter_update_buffers(nexus_update_buffers)
                    if args.optimizer_mode == "adamw_schedulefree":
                        current_lr = get_optimizer_logged_lr(optimizer, args.learning_rate)
                    if ema is not None:
                        update_ema(
                            ema,
                            model,
                            optimizer=optimizer,
                            use_optimizer_z=(args.optimizer_mode == "adamw_schedulefree"),
                        )
                    did_outer_step = True
                else:
                    did_outer_step = False
            else:
                with accelerator.accumulate(model):
                    loss_outputs = compute_training_losses(
                        model=model,
                        loss_fn=loss_fn,
                        images=x,
                        labels=labels,
                        zs=zs,
                        proj_coeff=args.proj_coeff,
                        raw_image=raw_image,
                    )
                    loss_mean = loss_outputs["denoising_loss"].mean()
                    proj_loss_mean = loss_outputs["projection_loss"].mean()
                    total_loss_mean = loss_outputs["total_loss"].mean()
                    loss = total_loss_mean

                    # Added for LR scheduling: keep the wandb/global step continuous
                    # while allowing the LR schedule to restart from a user-chosen step.
                    lr_schedule_step = get_lr_schedule_step(
                        global_step=global_step,
                        lr_schedule_start_step=args.lr_schedule_start_step,
                    )
                    # Added for LR scheduling: update the optimizer LR from the selected
                    # schedule right before the optimizer step so wandb can track it.
                    if args.optimizer_mode == "adamw_schedulefree":
                        current_lr = get_optimizer_logged_lr(optimizer, args.learning_rate)
                    else:
                        current_lr = apply_scheduled_lr(
                            optimizer=optimizer,
                            step=lr_schedule_step,
                            base_lr=args.learning_rate,
                            schedule_type=args.lr_schedule,
                            warmup_steps=args.warmup_steps,
                            decay_steps=args.lr_decay_steps if args.lr_decay_steps > 0 else args.max_train_steps,
                            min_lr=args.min_lr,
                        )
                    ## optimization
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = model.parameters()
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.sync_gradients and args.optimizer_mode == "adamw_schedulefree":
                        current_lr = get_optimizer_logged_lr(optimizer, args.learning_rate)

                    if accelerator.sync_gradients and ema is not None:
                        update_ema(
                            ema,
                            model,
                            optimizer=optimizer,
                            use_optimizer_z=(args.optimizer_mode == "adamw_schedulefree"),
                        )
                did_outer_step = accelerator.sync_gradients
            
            # Removed for Nexus compatibility:
            # if accelerator.sync_gradients:
            # Added for Nexus compatibility: keep the original side effects
            # bound to real outer-optimizer steps in both baseline and Nexus
            # modes.
            if did_outer_step:
                progress_bar.update(1)
                global_step += 1

                should_save_checkpoint = (
                    global_step >= args.checkpoint_start_step
                    and (
                        (global_step % args.checkpointing_steps == 0 and global_step > 0)
                        or global_step in args.extra_checkpoint_steps
                    )
                )
                should_sample = ( global_step % args.sampling_steps == 0 and global_step > 0 )
                should_run_fixed_eval = (
                    args.fixed_eval_steps > 0
                    and global_step >= args.fixed_eval_start_step
                    and (global_step == 1 or (global_step % args.fixed_eval_steps == 0 and global_step > 0))
                )
                optimizer_eval_context = (
                    args.optimizer_mode == "adamw_schedulefree"
                    and (should_save_checkpoint or should_sample or should_run_fixed_eval)
                )
                if optimizer_eval_context:
                    set_optimizer_train_mode(optimizer, train_mode=False)


                if should_save_checkpoint:
                    if accelerator.is_main_process:
                        alias_schedulefree_x = should_alias_schedulefree_x_to_ema(args)
                        model_state_dict = model.module.state_dict()
                        checkpoint = {
                            "model": model_state_dict,
                            "ema": model_state_dict if alias_schedulefree_x else ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            **({"inner_opt": inner_optimizer.state_dict()} if inner_optimizer is not None else {}),
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                if should_sample:
                    from samplers import euler_sampler
                    with torch.no_grad():
                        alias_schedulefree_x = should_alias_schedulefree_x_to_ema(args)
                        samples_model = euler_sampler(
                            model,
                            xT,
                            ys,
                            num_steps=50,
                            cfg_scale=4.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                        ).to(torch.float32)
                        if alias_schedulefree_x:
                            samples_ema = samples_model.clone()
                        else:
                            samples_ema = euler_sampler(
                                ema,
                                xT,
                                ys,
                                num_steps=50,
                                cfg_scale=4.0,
                                guidance_low=0.,
                                guidance_high=1.,
                                path_type=args.path_type,
                                heun=False,
                            ).to(torch.float32)
                        samples_model = vae.decode((samples_model - latents_bias) / latents_scale).sample
                        samples_ema = vae.decode((samples_ema - latents_bias) / latents_scale).sample
                        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        samples_model = (samples_model + 1) / 2.
                        samples_ema = (samples_ema + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples_model = accelerator.gather(samples_model.to(torch.float32))
                    out_samples_ema = accelerator.gather(samples_ema.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    accelerator.log(
                        {
                            "samples_model": wandb.Image(array2grid(out_samples_model)),
                            "samples_ema": wandb.Image(array2grid(out_samples_ema)),
                            "gt_samples": wandb.Image(array2grid(gt_samples)),
                        }
                    )
                    logging.info("Generated model and EMA samples.")

                if should_run_fixed_eval:
                    # Added for fixed sigma/t evaluation tracking: run deterministic
                    # train/val denoising, projection, and full-objective probes so
                    # each fixed t/random_t curve exposes the same components used
                    # by the training objective.
                    model_was_training = model.training
                    model.eval()
                    alias_schedulefree_x = should_alias_schedulefree_x_to_ema(args)
                    fixed_eval_logs = {}
                    fixed_train_metrics = evaluate_fixed_losses(
                        model=model,
                        loss_fn=loss_fn,
                        dataloader=fixed_train_loader,
                        cache=fixed_train_cache,
                        cache_positions=fixed_train_positions,
                        tracked_points=fixed_eval_plan,
                        accelerator=accelerator,
                        device=device,
                        latents_scale=latents_scale,
                        latents_bias=latents_bias,
                        encoders=encoders,
                        encoder_types=encoder_types,
                        proj_coeff=args.proj_coeff,
                    )
                    fixed_val_metrics = evaluate_fixed_losses(
                        model=model,
                        loss_fn=loss_fn,
                        dataloader=fixed_test_loader,
                        cache=fixed_test_cache,
                        cache_positions=fixed_test_positions,
                        tracked_points=fixed_eval_plan,
                        accelerator=accelerator,
                        device=device,
                        latents_scale=latents_scale,
                        latents_bias=latents_bias,
                        encoders=encoders,
                        encoder_types=encoder_types,
                        proj_coeff=args.proj_coeff,
                    )
                    if alias_schedulefree_x:
                        fixed_train_metrics_ema = fixed_train_metrics
                        fixed_val_metrics_ema = fixed_val_metrics
                    else:
                        fixed_train_metrics_ema = evaluate_fixed_losses(
                            model=ema,
                            loss_fn=loss_fn,
                            dataloader=fixed_train_loader,
                            cache=fixed_train_cache,
                            cache_positions=fixed_train_positions,
                            tracked_points=fixed_eval_plan,
                            accelerator=accelerator,
                            device=device,
                            latents_scale=latents_scale,
                            latents_bias=latents_bias,
                            encoders=encoders,
                            encoder_types=encoder_types,
                            proj_coeff=args.proj_coeff,
                        )
                        fixed_val_metrics_ema = evaluate_fixed_losses(
                            model=ema,
                            loss_fn=loss_fn,
                            dataloader=fixed_test_loader,
                            cache=fixed_test_cache,
                            cache_positions=fixed_test_positions,
                            tracked_points=fixed_eval_plan,
                            accelerator=accelerator,
                            device=device,
                            latents_scale=latents_scale,
                            latents_bias=latents_bias,
                            encoders=encoders,
                            encoder_types=encoder_types,
                            proj_coeff=args.proj_coeff,
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_train_mse/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_train_metrics["denoising"].items()
                            }
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_test_mse/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_val_metrics["denoising"].items()
                            }
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_train_proj_loss/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_train_metrics["projection"].items()
                            }
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_val_proj_loss/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_val_metrics["projection"].items()
                            }
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_train_total_loss/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_train_metrics["total"].items()
                            }
                        )
                        fixed_eval_logs.update(
                            {
                                f"fixed_val_total_loss/{metric_name}": metric_value
                                for metric_name, metric_value in fixed_val_metrics["total"].items()
                            }
                        )
                    fixed_eval_logs.update(
                        {
                            f"fixed_train_mse_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_train_metrics_ema["denoising"].items()
                        }
                    )
                    fixed_eval_logs.update(
                        {
                            f"fixed_test_mse_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_val_metrics_ema["denoising"].items()
                        }
                    )
                    fixed_eval_logs.update(
                        {
                            f"fixed_train_proj_loss_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_train_metrics_ema["projection"].items()
                        }
                    )
                    fixed_eval_logs.update(
                        {
                            f"fixed_val_proj_loss_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_val_metrics_ema["projection"].items()
                        }
                    )
                    fixed_eval_logs.update(
                        {
                            f"fixed_train_total_loss_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_train_metrics_ema["total"].items()
                        }
                    )
                    fixed_eval_logs.update(
                        {
                            f"fixed_val_total_loss_ema/{metric_name}": metric_value
                            for metric_name, metric_value in fixed_val_metrics_ema["total"].items()
                        }
                    )
                    accelerator.log(fixed_eval_logs, step=global_step)
                    if model_was_training:
                        model.train()

                if optimizer_eval_context and global_step < args.max_train_steps:
                    set_optimizer_train_mode(optimizer, train_mode=True)

                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "total_loss": accelerator.gather(total_loss_mean).mean().detach().item(),
                    "lr": current_lr,
                    "lr_schedule_step": float(lr_schedule_step),
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    set_optimizer_train_mode(optimizer, train_mode=False)
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume-checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--resume-use-checkpoint-args",
        action="store_true",
        help=(
            "Restore the original training args saved inside the resume checkpoint. "
            "Useful for resuming an interrupted decay run without manually re-entering "
            "the original exp-name/output-dir/LR schedule flags."
        ),
    )
    parser.add_argument("--force-new-wandb-run", action="store_true")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    # Added for offline/local setups: accept either a local Diffusers VAE
    # directory or a Hugging Face repo id so training can start without relying
    # on the default cache state of the machine.
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-mse")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument(
        "--checkpoint-start-step",
        type=int,
        default=0,
        help=(
            "Only save scheduled or extra checkpoints at global steps >= this value. "
            "The default 0 preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--extra-checkpoint-steps",
        type=str,
        default="",
        help=(
            "Comma-separated additional global steps to checkpoint, such as "
            "'320000,321000,322000'. These are saved in addition to the regular "
            "--checkpointing-steps interval."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--sf-inner-momentum",
        type=float,
        default=0.0,
        help="Inner gradient EMA momentum used by AdamWScheduleFree.",
    )
    # Added for optimizer experimentation: keep the original AdamW path,
    # expose AdamWScheduleFree as a direct alternative, and preserve the
    # Nexus inner-SGD outer-optimizer approximation without introducing a
    # second training entrypoint.
    parser.add_argument(
        "--optimizer-mode",
        type=str,
        default="adamw",
        choices=["adamw", "adamw_schedulefree", "nexus-v"],
    )
    parser.add_argument("--nexus-inner-lr", type=float, default=0.1)
    parser.add_argument("--nexus-inner-accumulation-steps", type=int, default=1)
    parser.add_argument("--nexus-gradient-mode", type=str, default="cosine", choices=["dot", "cosine"])
    parser.add_argument("--nexus-normalize-gradient-epsilon", type=float, default=1e-8)
    parser.add_argument("--nexus-inner-similarity-steps", type=int, default=0, help=argparse.SUPPRESS)
    # Added for LR scheduling: expose the schedule mode and its controls so users
    # can switch between constant, warmup+constant, linear decay, and cosine decay
    # without editing code.
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["warmup_constant", "decay", "cosinedecay", "constant"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    # Added for LR scheduling: let resumed runs preserve global_step while
    # restarting the LR schedule from a chosen checkpoint boundary.
    parser.add_argument("--lr-schedule-start-step", type=int, default=0)
    # Added for fixed sigma/t evaluation tracking: expose deterministic train/test
    # loss probes so users can inspect smoothed wandb curves without code edits.
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--fixed-eval-steps", type=int, default=10000)
    parser.add_argument(
        "--fixed-eval-start-step",
        type=int,
        default=0,
        help=(
            "Only run fixed evaluation at global steps >= this value. "
            "The default 0 preserves the original behavior."
        ),
    )
    parser.add_argument("--fixed-eval-num-samples", type=int, default=10000)
    parser.add_argument("--fixed-eval-batch-size", type=int, default=0)
    # Added for fixed sigma/t evaluation tracking: keep sigma and t as the two
    # primary probe modes and let random_t be attached as an extra curve.
    parser.add_argument("--fixed-eval-mode", type=str, default="sigma",
                        choices=["sigma", "t"])
    parser.add_argument("--fixed-eval-sigmas", type=str, default="0.05,0.2,0.5")
    parser.add_argument("--fixed-eval-ts", type=str, default="0.05,0.2,0.5")
    parser.add_argument("--fixed-eval-include-random", action="store_true")
    parser.add_argument("--fixed-eval-seed", type=int, default=1234)
    parser.add_argument("--fixed-gradient-num-samples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--fixed-gradient-seed", type=int, default=None, help=argparse.SUPPRESS)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.extra_checkpoint_steps = parse_int_step_list(args.extra_checkpoint_steps)

    return args

if __name__ == "__main__":
    args = parse_args()
    args = maybe_restore_args_from_checkpoint(args)
    
    main(args)
