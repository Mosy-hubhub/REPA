"""Helpers for deterministic fixed-noise evaluation curves."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch


def parse_float_sequence(raw_value: str) -> List[float]:
    if raw_value is None:
        return []
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def sigma_to_time(sigma_value: float, path_type: str) -> float:
    sigma_value = float(sigma_value)
    if path_type == "linear":
        return sigma_value
    if path_type == "cosine":
        sigma_tensor = torch.tensor(sigma_value, dtype=torch.float64).clamp(0.0, 1.0)
        return float((2.0 / torch.pi) * torch.asin(sigma_tensor))
    raise ValueError(f"Unsupported path_type for sigma conversion: {path_type}")


# Added for fixed-eval cache consistency: store and validate the cache request
# metadata so stale cache files are regenerated instead of being silently reused.
def _cache_matches_request(cache, expected_num_samples: int, expected_latent_shape: List[int], expected_seed: int) -> bool:
    meta = cache.get("meta")
    if meta is None:
        return False

    return (
        int(meta.get("num_samples", -1)) == int(expected_num_samples)
        and list(meta.get("latent_shape", [])) == list(expected_latent_shape)
        and int(meta.get("seed", -1)) == int(expected_seed)
    )


def build_or_load_fixed_eval_cache(
    cache_path: str,
    num_samples: int,
    latent_shape: List[int],
    seed: int,
    return_status: bool = False,
):
    cache_file = Path(cache_path)
    if cache_file.is_file():
        cache = torch.load(cache_file, map_location="cpu")
        # Added for fixed-eval cache consistency: rebuild legacy or mismatched
        # cache files so users do not accidentally reuse stale noise settings.
        if _cache_matches_request(cache, num_samples, latent_shape, seed):
            if return_status:
                return cache, "reused"
            return cache
        cache_status = "rebuilt"
    else:
        cache_status = "created"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    cache = {
        # Added for fixed-eval cache consistency: persist the cache request
        # metadata so future runs can validate that the cache still matches.
        "meta": {
            "num_samples": int(num_samples),
            "latent_shape": list(latent_shape),
            "seed": int(seed),
        },
        "indices": torch.arange(num_samples, dtype=torch.long),
        "posterior_noise": torch.randn((num_samples, *latent_shape), generator=generator, dtype=torch.float16),
        "diffusion_noise": torch.randn((num_samples, *latent_shape), generator=generator, dtype=torch.float16),
        "random_t": torch.rand((num_samples,), generator=generator, dtype=torch.float32),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    if return_status:
        return cache, cache_status
    return cache


def get_rank_positions(num_samples: int, num_processes: int, process_index: int) -> List[int]:
    return list(range(process_index, num_samples, num_processes))
