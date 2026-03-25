#!/usr/bin/env python3
"""
ReCogDrive Training Pipeline Debugger
======================================

A standalone script that inspects every stage of the ReCogDrive training pipeline
without requiring GPUs, distributed setup, or actual model weights. It loads cached
data, verifies tensor shapes/dtypes/ranges, simulates the collate function, and
checks model architecture consistency.

Usage:
    python scripts/debug_training_pipeline.py \
        --cache_path /path/to/recogdrive_agent_cache_dir_train \
        [--max_samples 10] \
        [--check_model] \
        [--dit_type small] \
        [--vlm_size large]

Stages:
    1. Cache Integrity   - scans cache directory, validates .gz files
    2. Sample Loading     - loads individual samples, checks tensor shapes/dtypes/values
    3. Collate Simulation - runs custom_collate_fn on a mini-batch
    4. Model Smoke Test   - (optional) instantiates model, runs a forward pass on CPU
"""

import argparse
import gzip
import logging
import math
import os
import pickle
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debug_pipeline")

# ─────────────────────────────────────────────
# Expected tensor specs (name -> (shape_suffix, dtype, value_range))
# shape_suffix uses -1 for variable/batch dimensions
# ─────────────────────────────────────────────
FEATURE_SPECS = {
    "history_trajectory": {"shape": (4, 3), "dtype": torch.float32, "range": (-500, 500)},
    "high_command_one_hot": {"shape": (3,), "dtype": torch.float32, "range": (0, 1)},
    "status_feature": {"shape": (8,), "dtype": torch.float32, "range": (-100, 100)},
    "last_hidden_state": {"shape_check": "2d_variable", "dtype": None, "range": (-100, 100)},
    "image_path_tensor": {"shape_check": "1d_variable", "dtype": torch.long, "range": (0, 256)},
}
TARGET_SPECS = {
    "trajectory": {"shape": (8, 3), "dtype": torch.float32, "range": (-500, 500)},
}

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def load_gz_pickle(path: Path) -> Dict[str, torch.Tensor]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def check_tensor(name: str, tensor: Any, spec: dict, context: str = "") -> List[str]:
    """Validate a single tensor against its spec. Returns list of issues."""
    issues = []
    prefix = f"[{context}] " if context else ""

    if not isinstance(tensor, torch.Tensor):
        issues.append(f"{prefix}{name}: expected torch.Tensor, got {type(tensor).__name__}")
        return issues

    # dtype check
    if spec.get("dtype") is not None and tensor.dtype != spec["dtype"]:
        issues.append(f"{prefix}{name}: dtype {tensor.dtype} != expected {spec['dtype']}")

    # shape check
    if "shape" in spec:
        expected = spec["shape"]
        if tensor.shape != torch.Size(expected):
            issues.append(f"{prefix}{name}: shape {tuple(tensor.shape)} != expected {expected}")
    elif spec.get("shape_check") == "2d_variable":
        if tensor.ndim != 2:
            issues.append(f"{prefix}{name}: expected 2D tensor, got ndim={tensor.ndim}")
    elif spec.get("shape_check") == "1d_variable":
        if tensor.ndim != 1:
            issues.append(f"{prefix}{name}: expected 1D tensor, got ndim={tensor.ndim}")

    # value range check
    if "range" in spec and isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        lo, hi = spec["range"]
        t_float = tensor.float()
        if torch.isnan(t_float).any():
            issues.append(f"{prefix}{name}: contains NaN values!")
        if torch.isinf(t_float).any():
            issues.append(f"{prefix}{name}: contains Inf values!")
        actual_min, actual_max = t_float.min().item(), t_float.max().item()
        if actual_min < lo or actual_max > hi:
            issues.append(
                f"{prefix}{name}: values [{actual_min:.4f}, {actual_max:.4f}] "
                f"outside expected [{lo}, {hi}]"
            )

    return issues


def format_tensor_info(name: str, tensor: Any) -> str:
    if not isinstance(tensor, torch.Tensor):
        return f"  {name}: <{type(tensor).__name__}>"
    stats = ""
    if tensor.is_floating_point() and tensor.numel() > 0:
        t = tensor.float()
        stats = (
            f" min={t.min().item():.4f} max={t.max().item():.4f} "
            f"mean={t.mean().item():.4f} std={t.std().item():.4f}"
            f" nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}"
        )
    elif tensor.numel() > 0:
        stats = f" min={tensor.min().item()} max={tensor.max().item()}"
    return f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}{stats}"


# ═════════════════════════════════════════════
# Stage 1: Cache Integrity
# ═════════════════════════════════════════════

def stage1_cache_integrity(cache_path: Path, max_samples: int) -> Tuple[List[Path], List[str]]:
    """Scan cache directory and validate structure."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Cache Integrity Check")
    logger.info("=" * 60)
    issues = []

    if not cache_path.is_dir():
        issues.append(f"Cache path does not exist: {cache_path}")
        return [], issues

    # Expected builder files
    expected_files = {"internvl_feature.gz", "trajectory_target.gz"}

    log_dirs = sorted([d for d in cache_path.iterdir() if d.is_dir()])
    logger.info(f"Found {len(log_dirs)} log directories in {cache_path}")

    valid_token_paths = []
    file_stats = Counter()
    corrupt_files = []
    empty_dirs = 0

    for log_dir in log_dirs:
        token_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir()])
        if not token_dirs:
            empty_dirs += 1
            continue

        for token_dir in token_dirs:
            gz_files = {f.name for f in token_dir.iterdir() if f.name.endswith(".gz")}
            file_stats.update(gz_files)

            missing = expected_files - gz_files
            if missing:
                issues.append(f"Missing files in {token_dir}: {missing}")
                continue

            # Quick corruption check: try loading header
            all_ok = True
            for fname in expected_files:
                fpath = token_dir / fname
                try:
                    with gzip.open(fpath, "rb") as f:
                        pickle.load(f)
                except Exception as e:
                    corrupt_files.append((fpath, str(e)))
                    all_ok = False

                if len(valid_token_paths) + len(corrupt_files) >= max_samples * 3:
                    break  # Don't scan everything, just enough to detect patterns

            if all_ok:
                valid_token_paths.append(token_dir)

            if len(valid_token_paths) >= max_samples:
                break
        if len(valid_token_paths) >= max_samples:
            break

    total_tokens_approx = sum(
        len(list(log_dir.iterdir())) for log_dir in log_dirs[:5]
    )
    logger.info(f"Scanned tokens (sampled): {len(valid_token_paths)} valid")
    logger.info(f"Empty log dirs: {empty_dirs}")
    logger.info(f"File types found: {dict(file_stats)}")

    if corrupt_files:
        for fpath, err in corrupt_files[:5]:
            issues.append(f"Corrupt file {fpath}: {err}")
        if len(corrupt_files) > 5:
            issues.append(f"... and {len(corrupt_files) - 5} more corrupt files")

    if not issues:
        logger.info("PASS: Cache structure looks valid")
    else:
        for issue in issues:
            logger.warning(f"ISSUE: {issue}")

    return valid_token_paths, issues


# ═════════════════════════════════════════════
# Stage 2: Sample Loading & Validation
# ═════════════════════════════════════════════

def stage2_sample_loading(
    token_paths: List[Path], max_samples: int
) -> Tuple[List[Tuple[dict, dict, str]], List[str]]:
    """Load individual samples and validate tensor shapes/dtypes/values."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Sample Loading & Validation")
    logger.info("=" * 60)
    issues = []
    loaded_samples = []

    # Detect mode from first sample
    first_path = token_paths[0] / "internvl_feature.gz"
    first_data = load_gz_pickle(first_path)
    has_hidden_state = "last_hidden_state" in first_data
    has_image_path = "image_path_tensor" in first_data
    mode = "cached_hidden_state" if has_hidden_state else "image_path"
    logger.info(f"Detected caching mode: {mode}")
    logger.info(f"Feature keys in cache: {sorted(first_data.keys())}")

    hidden_state_dims = set()
    hidden_state_seqlens = []

    for i, token_path in enumerate(token_paths[:max_samples]):
        token = token_path.name
        context = f"sample_{i}({token[:12]}...)"

        try:
            features = load_gz_pickle(token_path / "internvl_feature.gz")
            targets = load_gz_pickle(token_path / "trajectory_target.gz")
        except Exception as e:
            issues.append(f"{context}: Failed to load: {e}")
            continue

        # Check features
        for name, spec in FEATURE_SPECS.items():
            if name not in features:
                if name == "last_hidden_state" and not has_hidden_state:
                    continue  # Expected in image_path mode
                if name == "image_path_tensor" and not has_image_path:
                    continue  # Expected in hidden_state mode
                issues.append(f"{context}: Missing feature '{name}'")
                continue
            issues.extend(check_tensor(name, features[name], spec, context))

        # Collect hidden state statistics
        if has_hidden_state and "last_hidden_state" in features:
            hs = features["last_hidden_state"]
            if isinstance(hs, torch.Tensor) and hs.ndim == 2:
                hidden_state_dims.add(hs.shape[1])
                hidden_state_seqlens.append(hs.shape[0])

        # Check high_command_one_hot sums to 1
        if "high_command_one_hot" in features:
            hcoh = features["high_command_one_hot"]
            if isinstance(hcoh, torch.Tensor):
                s = hcoh.sum().item()
                if abs(s - 1.0) > 0.01:
                    issues.append(f"{context}: high_command_one_hot sum={s:.4f}, expected 1.0")

        # Check targets
        for name, spec in TARGET_SPECS.items():
            if name not in targets:
                issues.append(f"{context}: Missing target '{name}'")
                continue
            issues.extend(check_tensor(name, targets[name], spec, context))

        loaded_samples.append((features, targets, token))

        if i == 0:
            logger.info("First sample details:")
            for name, tensor in sorted(features.items()):
                logger.info(format_tensor_info(name, tensor))
            for name, tensor in sorted(targets.items()):
                logger.info(format_tensor_info(name, tensor))

    # Summary statistics
    if hidden_state_dims:
        logger.info(f"Hidden state embedding dims found: {hidden_state_dims}")
        if len(hidden_state_dims) > 1:
            issues.append(f"Inconsistent hidden state dims: {hidden_state_dims}")
        logger.info(
            f"Hidden state seq_len: min={min(hidden_state_seqlens)} "
            f"max={max(hidden_state_seqlens)} "
            f"mean={np.mean(hidden_state_seqlens):.1f}"
        )

    logger.info(f"Successfully loaded {len(loaded_samples)}/{min(max_samples, len(token_paths))} samples")

    if not issues:
        logger.info("PASS: All samples valid")
    else:
        for issue in issues[:20]:
            logger.warning(f"ISSUE: {issue}")
        if len(issues) > 20:
            logger.warning(f"... and {len(issues) - 20} more issues")

    return loaded_samples, issues


# ═════════════════════════════════════════════
# Stage 3: Collate Simulation
# ═════════════════════════════════════════════

def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], tuple]:
    """Replica of the training collate function for debugging."""
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack(
        [f["history_trajectory"] for f in features_list], dim=0
    ).cpu()
    high_command_one_hot = torch.stack(
        [f["high_command_one_hot"] for f in features_list], dim=0
    ).cpu()
    status_feature = torch.stack(
        [f["status_feature"] for f in features_list], dim=0
    ).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [f["last_hidden_state"] for f in features_list],
        batch_first=True,
        padding_value=0.0,
    ).clone().detach()

    trajectory = torch.stack(
        [t["trajectory"] for t in targets_list], dim=0
    ).cpu()

    features = {
        "history_trajectory": history_trajectory,
        "high_command_one_hot": high_command_one_hot,
        "last_hidden_state": last_hidden_state,
        "status_feature": status_feature,
    }
    targets = {"trajectory": trajectory}
    return features, targets, tokens_list


def stage3_collate_simulation(
    samples: List[Tuple[dict, dict, str]], batch_size: int = 4
) -> Tuple[Optional[Tuple], List[str]]:
    """Simulate the collate function on a mini-batch."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Collate Simulation")
    logger.info("=" * 60)
    issues = []

    # Check if samples have last_hidden_state (required for collate)
    has_hs = all("last_hidden_state" in s[0] for s in samples)
    if not has_hs:
        logger.info("SKIP: Samples use image_path_tensor mode, collate requires last_hidden_state")
        return None, issues

    # Take a mini-batch
    mini_batch = samples[:min(batch_size, len(samples))]
    actual_bs = len(mini_batch)
    logger.info(f"Collating mini-batch of {actual_bs} samples")

    try:
        features, targets, tokens = custom_collate_fn(mini_batch)
    except Exception as e:
        issues.append(f"Collate function failed: {e}")
        logger.error(f"Collate FAILED: {e}")
        traceback.print_exc()
        return None, issues

    # Validate batch shapes
    expected_shapes = {
        "history_trajectory": (actual_bs, 4, 3),
        "high_command_one_hot": (actual_bs, 3),
        "status_feature": (actual_bs, 8),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(features[name].shape)
        if actual != expected:
            issues.append(f"Batched {name}: shape {actual} != expected {expected}")
        else:
            logger.info(f"  {name}: {actual} OK")

    # last_hidden_state: (bs, max_seq_len, embed_dim)
    hs = features["last_hidden_state"]
    if hs.ndim != 3:
        issues.append(f"Batched last_hidden_state: expected 3D, got ndim={hs.ndim}")
    else:
        logger.info(f"  last_hidden_state: {tuple(hs.shape)} (padded)")

    # trajectory target
    traj = targets["trajectory"]
    expected_traj = (actual_bs, 8, 3)
    if tuple(traj.shape) != expected_traj:
        issues.append(f"Batched trajectory: shape {tuple(traj.shape)} != expected {expected_traj}")
    else:
        logger.info(f"  trajectory: {tuple(traj.shape)} OK")

    # Check for NaN/Inf in batched tensors
    for name, tensor in {**features, **targets}.items():
        if isinstance(tensor, torch.Tensor):
            t = tensor.float()
            if torch.isnan(t).any():
                issues.append(f"Batched {name}: contains NaN after collate!")
            if torch.isinf(t).any():
                issues.append(f"Batched {name}: contains Inf after collate!")

    if not issues:
        logger.info("PASS: Collate function works correctly")
    else:
        for issue in issues:
            logger.warning(f"ISSUE: {issue}")

    return (features, targets, tokens), issues


# ═════════════════════════════════════════════
# Stage 4: Model Architecture Smoke Test
# ═════════════════════════════════════════════

def stage4_model_smoke_test(
    collated_batch: Optional[Tuple],
    dit_type: str = "small",
    vlm_size: str = "large",
) -> List[str]:
    """Instantiate model on CPU and run a forward pass with fake data."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Model Architecture Smoke Test")
    logger.info("=" * 60)
    issues = []

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
        from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
            ReCogDriveDiffusionPlanner,
        )
    except ImportError as e:
        issues.append(f"Cannot import model classes: {e}")
        logger.error(f"Import failed: {e}")
        return issues

    # Config
    input_embedding_dim = 384 if dit_type == "small" else 1536
    try:
        cfg = make_recogdrive_config(
            dit_type,
            action_dim=3,
            action_horizon=8,
            input_embedding_dim=input_embedding_dim,
            sampling_method="ddim",
            grpo=False,
        )
        cfg.vlm_size = vlm_size
        logger.info(f"Config: dit_type={dit_type}, vlm_size={vlm_size}, "
                     f"input_embedding_dim={input_embedding_dim}")
    except Exception as e:
        issues.append(f"Config creation failed: {e}")
        return issues

    # Instantiate on CPU
    try:
        model = ReCogDriveDiffusionPlanner(cfg).cpu().float()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model instantiated: {total_params:,} total params, {trainable_params:,} trainable")
    except Exception as e:
        issues.append(f"Model instantiation failed: {e}")
        traceback.print_exc()
        return issues

    # Print submodule parameter counts
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {n_params:,} params")

    # Determine embed dim for VLM features
    vlm_embed_dim = 3584 if vlm_size == "large" else 1536

    # Create synthetic batch
    bs = 2
    seq_len = 64  # Typical VLM output length
    device = torch.device("cpu")
    dtype = torch.float32

    if collated_batch is not None:
        features, targets, tokens = collated_batch
        # Use real shapes but ensure CPU + float32
        last_hidden_state = features["last_hidden_state"][:bs].to(device).float()
        # Verify embed dim matches model expectation
        actual_embed_dim = last_hidden_state.shape[-1]
        if actual_embed_dim != vlm_embed_dim:
            issues.append(
                f"Hidden state embed_dim={actual_embed_dim} doesn't match "
                f"model expectation for vlm_size={vlm_size} (expected {vlm_embed_dim}). "
                f"Try --vlm_size={'small' if vlm_size == 'large' else 'large'}"
            )
            # Create compatible synthetic data
            last_hidden_state = torch.randn(bs, seq_len, vlm_embed_dim, device=device, dtype=dtype)
            logger.warning(f"Using synthetic hidden states with dim={vlm_embed_dim}")

        from transformers.feature_extraction_utils import BatchFeature
        action_inputs = BatchFeature(data={
            "state": torch.randn(bs, 20, device=device, dtype=dtype),
            "his_traj": torch.randn(bs, 12, device=device, dtype=dtype),
            "status_feature": torch.randn(bs, 8, device=device, dtype=dtype),
            "action": torch.randn(bs, 8, 3, device=device, dtype=dtype),
        })
    else:
        last_hidden_state = torch.randn(bs, seq_len, vlm_embed_dim, device=device, dtype=dtype)
        from transformers.feature_extraction_utils import BatchFeature
        action_inputs = BatchFeature(data={
            "state": torch.randn(bs, 20, device=device, dtype=dtype),
            "his_traj": torch.randn(bs, 12, device=device, dtype=dtype),
            "status_feature": torch.randn(bs, 8, device=device, dtype=dtype),
            "action": torch.randn(bs, 8, 3, device=device, dtype=dtype),
        })

    # Forward pass (training mode)
    model.train()
    try:
        output = model(last_hidden_state, action_inputs)
        logger.info(f"Training forward pass succeeded")
        if hasattr(output, "loss"):
            logger.info(f"  loss: {output.loss.item():.6f}")
        elif isinstance(output, dict) and "loss" in output:
            logger.info(f"  loss: {output['loss'].item():.6f}")
    except Exception as e:
        issues.append(f"Training forward pass failed: {e}")
        traceback.print_exc()

    # Inference pass
    model.eval()
    try:
        with torch.no_grad():
            output = model.get_action(last_hidden_state, action_inputs)
        pred_traj = output.get("pred_traj") if isinstance(output, dict) else getattr(output, "pred_traj", None)
        if pred_traj is not None:
            logger.info(f"Inference forward pass succeeded")
            logger.info(f"  pred_traj: shape={tuple(pred_traj.shape)}, "
                        f"min={pred_traj.min().item():.4f}, max={pred_traj.max().item():.4f}")
            expected_shape = (bs, 8, 3)
            if tuple(pred_traj.shape) != expected_shape:
                issues.append(f"pred_traj shape {tuple(pred_traj.shape)} != expected {expected_shape}")
        else:
            issues.append("Inference output missing 'pred_traj'")
    except Exception as e:
        issues.append(f"Inference forward pass failed: {e}")
        traceback.print_exc()

    if not issues:
        logger.info("PASS: Model architecture is consistent")
    else:
        for issue in issues:
            logger.warning(f"ISSUE: {issue}")

    return issues


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Debug the ReCogDrive training pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache_path", type=str, required=True,
        help="Path to the training cache directory",
    )
    parser.add_argument(
        "--max_samples", type=int, default=10,
        help="Maximum number of samples to inspect (default: 10)",
    )
    parser.add_argument(
        "--check_model", action="store_true",
        help="Run Stage 4: instantiate model and test forward pass on CPU",
    )
    parser.add_argument(
        "--dit_type", type=str, default="small", choices=["small", "large"],
        help="DiT model size preset (default: small)",
    )
    parser.add_argument(
        "--vlm_size", type=str, default="large", choices=["small", "large"],
        help="VLM size preset - determines feature_encoder input dim (default: large)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for collate simulation (default: 4)",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    all_issues = []

    logger.info("ReCogDrive Training Pipeline Debugger")
    logger.info(f"Cache path: {cache_path}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info("")

    # Stage 1
    token_paths, issues1 = stage1_cache_integrity(cache_path, args.max_samples)
    all_issues.extend(issues1)

    if not token_paths:
        logger.error("No valid token paths found. Cannot proceed.")
        sys.exit(1)

    # Stage 2
    samples, issues2 = stage2_sample_loading(token_paths, args.max_samples)
    all_issues.extend(issues2)

    if not samples:
        logger.error("No samples could be loaded. Cannot proceed.")
        sys.exit(1)

    # Stage 3
    collated, issues3 = stage3_collate_simulation(samples, args.batch_size)
    all_issues.extend(issues3)

    # Stage 4 (optional)
    if args.check_model:
        issues4 = stage4_model_smoke_test(collated, args.dit_type, args.vlm_size)
        all_issues.extend(issues4)

    # ─── Summary ───
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if not all_issues:
        logger.info("ALL STAGES PASSED - pipeline looks healthy")
    else:
        logger.warning(f"Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            logger.warning(f"  {i}. {issue}")

    sys.exit(1 if all_issues else 0)


if __name__ == "__main__":
    main()
