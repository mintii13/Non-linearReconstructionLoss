#!/usr/bin/env python3
import argparse
import importlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import get_dataset
from data.utils import get_transforms
from model import get_model
from util.net import trans_state_dict


def parse_int_list(value):
    if value is None or value == "" or value.lower() == "auto":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_cfg(config_path, checkpoint_path):
    module_path = config_path.replace("/", ".").replace("\\", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    cfg = importlib.import_module(module_path).cfg()
    cfg.mode = "test"
    cfg.dist = False
    cfg.master = True
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.model.kwargs["checkpoint_path"] = checkpoint_path
    return cfg


def load_net(cfg, checkpoint_path, device):
    cfg.model.kwargs["checkpoint_path"] = ""
    net = get_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("net", ckpt)
    state_dict = trans_state_dict(state_dict, dist=False)
    model_state = net.state_dict()
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    net.to(device)
    net.eval()
    return net


def build_loader(cfg, num_images, indices):
    _, test_set = get_dataset(cfg)
    if indices is None:
        indices = list(range(min(num_images, len(test_set))))
    else:
        indices = [idx for idx in indices if 0 <= idx < len(test_set)]
    subset = Subset(test_set, indices[:num_images])
    return DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0), indices[:num_images]


def load_external_images(cfg, image_paths):
    transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)
    imgs = []
    paths = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        imgs.append(transforms(img))
        paths.append(str(path))
    return {
        "img": torch.stack(imgs, dim=0),
        "img_path": paths,
        "cls_name": ["external"] * len(paths),
        "anomaly": torch.zeros(len(paths), dtype=torch.long),
    }


def unnormalize_image(img):
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=img.dtype, device=img.device)[:, None, None]
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=img.dtype, device=img.device)[:, None, None]
    img = (img * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()


def normalize_map(x):
    x = x.detach().float().cpu().numpy()
    x = np.nan_to_num(x)
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-8:
        lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def tokens_to_feature_map(tokens, feature_size, batch_index):
    h, w = feature_size
    if tokens is None:
        return None
    # [L, B, C] -> [C, H, W]
    return tokens[:, batch_index, :].permute(1, 0).reshape(tokens.shape[-1], h, w)


def select_channels(feature_map, requested_channels, max_channels):
    if requested_channels is not None:
        return [c for c in requested_channels if 0 <= c < feature_map.shape[0]]
    flat = feature_map.flatten(1)
    scores = flat.var(dim=1)
    k = min(max_channels, feature_map.shape[0])
    return torch.topk(scores, k=k).indices.cpu().tolist()


def save_feature_channels(out_path, original, feature_maps, channels, title, orig_feature=None, orig_channels=None):
    rows = len(feature_maps) + (1 if orig_feature is not None else 0)
    max_channel_count = max(len(channels), len(orig_channels or []))
    cols = max_channel_count + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), squeeze=False)
    row_offset = 0
    if orig_feature is not None:
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("image")
        axes[0, 0].axis("off")
        for col, channel in enumerate(orig_channels or [], start=1):
            axes[0, col].imshow(normalize_map(orig_feature[channel]), cmap="magma")
            axes[0, col].set_title(f"orig_feature\nch {channel}", fontsize=9)
            axes[0, col].axis("off")
        for col in range(1 + len(orig_channels or []), cols):
            axes[0, col].axis("off")
        row_offset = 1

    for row, (name, fmap) in enumerate(feature_maps.items(), start=row_offset):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("image")
        axes[row, 0].axis("off")
        for col, channel in enumerate(channels, start=1):
            axes[row, col].imshow(normalize_map(fmap[channel]), cmap="magma")
            axes[row, col].set_title(f"{name}\nch {channel}", fontsize=9)
            axes[row, col].axis("off")
        for col in range(1 + len(channels), cols):
            axes[row, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_anomaly_map(out_path, original, pred):
    pred_map = normalize_map(pred.squeeze(0))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original)
    axes[0].set_title("image")
    axes[0].axis("off")
    axes[1].imshow(original)
    axes[1].imshow(pred_map, cmap="jet", alpha=0.55)
    axes[1].set_title("pred")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_channel_attention_maps(out_path, channel_result, batch_index, feature_size, slots):
    if channel_result is None:
        return
    h, w = feature_size
    att = channel_result["att_weight"].detach().cpu()
    mem_dim = att.shape[-1]
    att = att.reshape(h * w, -1, mem_dim)[:, batch_index, :]
    fig, axes = plt.subplots(1, len(slots), figsize=(2.4 * len(slots), 2.4), squeeze=False)
    for i, slot in enumerate(slots):
        axes[0, i].imshow(normalize_map(att[:, slot].reshape(h, w)), cmap="viridis")
        axes[0, i].set_title(f"slot {slot}")
        axes[0, i].axis("off")
    fig.suptitle("channel memory attention over spatial tokens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_spatial_attention_table(out_path, spatial_result, batch_index, channels, slots, hidden_dim):
    if spatial_result is None:
        return
    att = spatial_result["att_weight"].detach().cpu()
    # Spatial memory attention is indexed by [B * hidden_dim, mem_dim].
    rows = [batch_index * hidden_dim + ch for ch in channels if ch < hidden_dim]
    if not rows:
        return
    table = att[rows][:, slots]
    fig, ax = plt.subplots(1, 1, figsize=(1.0 + 0.6 * len(slots), 1.5 + 0.35 * len(rows)))
    im = ax.imshow(table.numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(slots)), [str(s) for s in slots])
    ax.set_yticks(range(len(rows)), [f"ch {channels[i]}" for i in range(len(rows))])
    ax.set_xlabel("spatial memory slot")
    ax.set_title("spatial memory attention")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_slots(results, key, requested_slots, max_slots):
    if requested_slots is not None:
        return requested_slots[:max_slots]
    weights = []
    for output in results:
        res = output.get(key)
        if res is not None:
            weights.append(res["att_weight"].detach().float().cpu().mean(dim=0))
    if not weights:
        return []
    mean_weight = torch.stack(weights).mean(dim=0)
    return torch.topk(mean_weight, k=min(max_slots, mean_weight.numel())).indices.tolist()


def save_channel_memory_bank(out_path, memory, slots):
    memory = memory.detach().float().cpu()
    fig, axes = plt.subplots(len(slots), 2, figsize=(11, 1.8 * len(slots)), squeeze=False)
    for row, slot in enumerate(slots):
        vec = memory[slot]
        axes[row, 0].imshow(normalize_map(vec.unsqueeze(0)), cmap="coolwarm", aspect="auto")
        axes[row, 0].set_title(f"channel memory slot {slot} heat")
        axes[row, 0].set_yticks([])
        axes[row, 1].plot(vec.numpy(), linewidth=0.8)
        axes[row, 1].axhline(0, color="black", linewidth=0.5)
        axes[row, 1].set_title(f"slot {slot} vector")
        axes[row, 1].set_xlim(0, vec.numel() - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_spatial_memory_bank(out_path, memory, slots):
    memory = memory.detach().float().cpu()
    fig, axes = plt.subplots(1, len(slots), figsize=(2.5 * len(slots), 2.5), squeeze=False)
    for col, slot in enumerate(slots):
        axes[0, col].imshow(normalize_map(memory[slot]), cmap="magma")
        axes[0, col].set_title(f"slot {slot}")
        axes[0, col].axis("off")
    fig.suptitle("spatial memory bank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def get_model_core(net):
    return net.module if hasattr(net, "module") else net


def batch_value(values, index, default="unknown"):
    if values is None:
        return default
    if isinstance(values, torch.Tensor):
        return values[index].item()
    if isinstance(values, (list, tuple)):
        return values[index]
    return values


def main():
    parser = argparse.ArgumentParser(description="Visualize baseline memory-bank features.")
    parser.add_argument("--config", default="configs/baseline_mvtec.py")
    parser.add_argument("--checkpoint", default="runs/resfes/ckpt.pth")
    parser.add_argument("--out-dir", default="runs/resfes/memory_vis")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--indices", default=None, help="Comma-separated test-set indices, e.g. 0,1,2.")
    parser.add_argument("--images", nargs="*", default=None, help="Optional raw image paths instead of dataset samples.")
    parser.add_argument("--channels", default="auto", help="Comma-separated hidden channels or 'auto'.")
    parser.add_argument("--max-channels", type=int, default=6)
    parser.add_argument("--orig-channels", default="auto", help="Comma-separated original feature channels or 'auto'.")
    parser.add_argument("--max-orig-channels", type=int, default=None)
    parser.add_argument("--memory-slots", default="auto", help="Comma-separated memory slot ids or 'auto'.")
    parser.add_argument("--max-slots", type=int, default=6)
    parser.add_argument("--spatial-patch-size", type=int, default=None)
    parser.add_argument("--conv-memory-init", default=None, choices=["normal", "pretrained"])
    parser.add_argument("--conv-memory-pretrained-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    banks_dir = out_dir / "memory_banks"
    samples_dir.mkdir(parents=True, exist_ok=True)
    banks_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, args.checkpoint)
    if args.spatial_patch_size is not None:
        cfg.model.kwargs["spatial_patch_size"] = args.spatial_patch_size
    if args.conv_memory_init is not None:
        cfg.model.kwargs["conv_memory_init"] = args.conv_memory_init
    if args.conv_memory_pretrained_path is not None:
        cfg.model.kwargs["conv_memory_pretrained_path"] = args.conv_memory_pretrained_path
    device = torch.device(args.device)
    net = load_net(cfg, args.checkpoint, device)
    model_core = get_model_core(net)
    feature_size = model_core.net_ad.feature_size
    hidden_dim = model_core.net_ad.hidden_dim

    if args.images:
        batch = load_external_images(cfg, args.images)
        sample_ids = list(range(len(args.images)))
    else:
        indices = parse_int_list(args.indices)
        loader, sample_ids = build_loader(cfg, args.num_images, indices)
        batch = next(iter(loader))

    imgs = batch["img"].to(device, non_blocking=True)
    with torch.no_grad():
        output = net(imgs)
    output_dict = output[-1] if isinstance(output, (tuple, list)) and isinstance(output[-1], dict) else output
    if not isinstance(output_dict, dict):
        raise RuntimeError("Model did not return output_dict. Expected BaselineWrapper output.")

    results = [output_dict]
    requested_channels = parse_int_list(args.channels)
    requested_orig_channels = parse_int_list(args.orig_channels)
    requested_slots = parse_int_list(args.memory_slots)
    channel_slots = select_slots(results, "channel_result", requested_slots, args.max_slots)
    spatial_slots = select_slots(results, "spatial_result", requested_slots, args.max_slots)

    post_first = tokens_to_feature_map(output_dict["post_fusion_tokens"], feature_size, 0)
    channels = select_channels(post_first, requested_channels, args.max_channels)
    orig_first = output_dict.get("pre_sigmoid_orig")
    orig_channels = []
    if orig_first is not None:
        max_orig_channels = args.max_orig_channels or args.max_channels
        orig_channels = select_channels(orig_first[0], requested_orig_channels, max_orig_channels)

    for b in range(imgs.shape[0]):
        original = unnormalize_image(imgs[b])
        sample_id = sample_ids[b] if b < len(sample_ids) else b
        cls_name = str(batch_value(batch.get("cls_name"), b)).replace("/", "_")
        safe_name = f"sample_{b:02d}_{cls_name}_idx_{sample_id}"

        post_fusion = tokens_to_feature_map(output_dict.get("post_fusion_tokens"), feature_size, b)
        post_proj = tokens_to_feature_map(output_dict.get("post_fusion_proj"), feature_size, b)
        channel_retrieved = None
        spatial_retrieved = None
        if output_dict.get("channel_result") is not None:
            channel_retrieved = tokens_to_feature_map(output_dict["channel_result"]["output"], feature_size, b)
        if output_dict.get("spatial_result") is not None:
            spatial_retrieved = tokens_to_feature_map(output_dict["spatial_result"]["output"], feature_size, b)
        orig_feature = output_dict["pre_sigmoid_orig"][b] if output_dict.get("pre_sigmoid_orig") is not None else None

        maps = {"post_fusion": post_fusion}
        if post_proj is not None:
            maps["post_proj"] = post_proj
        if channel_retrieved is not None:
            maps["channel_query"] = channel_retrieved
        if spatial_retrieved is not None:
            maps["spatial_query"] = spatial_retrieved
        save_feature_channels(
            samples_dir / f"{safe_name}_features_after_memory.png",
            original,
            maps,
            channels,
            f"{safe_name}: features after memory query",
            orig_feature=orig_feature,
            orig_channels=orig_channels,
        )

        pred = output_dict.get("pred")
        if pred is not None:
            save_anomaly_map(samples_dir / f"{safe_name}_pred_overlay.png", original, pred[b].detach().cpu())
        save_channel_attention_maps(
            samples_dir / f"{safe_name}_channel_memory_attention.png",
            output_dict.get("channel_result"),
            b,
            feature_size,
            channel_slots,
        )
        save_spatial_attention_table(
            samples_dir / f"{safe_name}_spatial_memory_attention.png",
            output_dict.get("spatial_result"),
            b,
            channels,
            spatial_slots,
            hidden_dim,
        )

    channel_result = output_dict.get("channel_result")
    if channel_result is not None and channel_slots:
        save_channel_memory_bank(
            banks_dir / "channel_memory_bank_slots.png",
            channel_result["memory"],
            channel_slots,
        )
    spatial_result = output_dict.get("spatial_result")
    if spatial_result is not None and spatial_slots:
        save_spatial_memory_bank(
            banks_dir / "spatial_memory_bank_slots.png",
            spatial_result["memory"],
            spatial_slots,
        )

    meta_path = out_dir / "summary.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"config: {args.config}\n")
        f.write(f"sample_ids: {sample_ids}\n")
        f.write(f"class_names: {batch.get('cls_name')}\n")
        f.write(f"channels: {channels}\n")
        f.write(f"orig_channels: {orig_channels}\n")
        f.write(f"channel_memory_slots: {channel_slots}\n")
        f.write(f"spatial_memory_slots: {spatial_slots}\n")
        f.write(f"output_dir: {out_dir}\n")

    print(f"Saved visualization to: {out_dir}")
    print(f"Channels: {channels}")
    print(f"Original feature channels: {orig_channels}")
    print(f"Channel memory slots: {channel_slots}")
    print(f"Spatial memory slots: {spatial_slots}")


if __name__ == "__main__":
    main()
