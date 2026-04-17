import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from data import get_loader
from model import get_model

# Import class cfg từ config
from configs.baseline_visa import cfg as cfg_class

# Tạo instance (giống như trainer vẫn làm)
cfg = cfg_class()

def load_model(checkpoint_path, device):
    model = get_model(cfg.model.name)(**cfg.model.kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    
    if 'net' in state:
        state = state['net']
    elif 'model' in state:
        state = state['model']
    
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def compute_stats(tensor):
    t = tensor.detach().float().flatten().cpu().numpy()
    return {
        'min': np.min(t),
        'max': np.max(t),
        'mean': np.mean(t),
        'var': np.var(t)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--num_batches', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Cập nhật đường dẫn dataset
    cfg.data.root = '/home/minhtringuyen/ADer/data/visa'
    
    # Tạo data loader - SỬA: dùng cfg.data và truyền split qua args
    # Hoặc get_loader nhận dict hoặc Namespace
    loader = get_loader(cfg.data, split=args.split, distributed=False)
    # Nếu vẫn lỗi, thử cách khác:
    # from data import build_loader
    # loader = build_loader(cfg.data, args.split, distributed=False)
    
    print(f"Dataset: {cfg.data.root}, Split: {args.split}")
    print(f"Total batches: {len(loader)}")

    # Load model
    device = args.device
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # Accumulators
    backbone_stats = [[] for _ in range(4)]
    encoder_stats = []
    decoder_stats = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f'Processing {args.split} data')):
            if args.num_batches > 0 and i >= args.num_batches:
                break
            
            imgs = batch['img'].to(device)
            _, _, _, output_dict = model(imgs)

            # Backbone outputs
            backbone_feats = model.net_backbone(imgs)
            for stage_idx, feat in enumerate(backbone_feats):
                if stage_idx < 4:
                    stats = compute_stats(feat)
                    backbone_stats[stage_idx].append(stats)

            # Encoder output
            encoded = output_dict.get('pre_memory_tokens')
            if encoded is not None:
                stats = compute_stats(encoded)
                encoder_stats.append(stats)

            # Decoder output
            decoded = output_dict.get('decoded_tokens')
            if decoded is not None:
                stats = compute_stats(decoded)
                decoder_stats.append(stats)

    print("\n===== BACKBONE OUTPUT =====")
    for stage_idx, stats_list in enumerate(backbone_stats):
        if stats_list:
            avg_stats = {k: np.mean([s[k] for s in stats_list]) for k in ['min','max','mean','var']}
            print(f"Stage {stage_idx}: min={avg_stats['min']:.6f}, max={avg_stats['max']:.6f}, mean={avg_stats['mean']:.6f}, var={avg_stats['var']:.6f}")

    if encoder_stats:
        avg_enc = {k: np.mean([s[k] for s in encoder_stats]) for k in ['min','max','mean','var']}
        print("\n===== ENCODER OUTPUT (pre_memory_tokens) =====")
        print(f"min={avg_enc['min']:.6f}, max={avg_enc['max']:.6f}, mean={avg_enc['mean']:.6f}, var={avg_enc['var']:.6f}")

    if decoder_stats:
        avg_dec = {k: np.mean([s[k] for s in decoder_stats]) for k in ['min','max','mean','var']}
        print("\n===== DECODER OUTPUT (decoded_tokens) =====")
        print(f"min={avg_dec['min']:.6f}, max={avg_dec['max']:.6f}, mean={avg_dec['mean']:.6f}, var={avg_dec['var']:.6f}")

if __name__ == '__main__':
    main()