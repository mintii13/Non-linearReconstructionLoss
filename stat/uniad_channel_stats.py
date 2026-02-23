# PYTHONPATH=. python stat/uniad_channel_stats.py --use_train_data --config_path /home/minhtringuyen/ADer/configs/uniad/uniad_mvtec.py --save_dir stat_results --gpu 0

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import importlib.util
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ======================================================
# 1. SETUP PATH
# ======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from data import get_loader
    from model import get_model
except ImportError as e:
    print(f"Lỗi Import: {e}")
    sys.exit(1)

# ======================================================
# 2. LOAD CONFIG
# ======================================================
def load_config_from_path(path):
    print(f">>> Loading config: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")

    spec = importlib.util.spec_from_file_location("dynamic_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    if hasattr(mod, 'cfg'):
        return mod.cfg()
    raise AttributeError("No 'cfg' class found in config file.")

def force_set_attr(obj, key, value):
    try:
        obj[key] = value
        return
    except (TypeError, AttributeError):
        pass
    try:
        setattr(obj, key, value)
        return
    except (TypeError, AttributeError):
        pass

# Nhớ thêm dòng này ở đầu file nhé
import scipy.stats as stats

# ======================================================
# 4. TÍNH STATS THEO CHANNEL & VẼ DENSITY (TOP 5 CHUẨN NHẤT)
# ======================================================
def calculate_channel_statistics(feature_tensor, save_dir):
    print(">>> Calculating Channel-wise Statistics...")
    
    if len(feature_tensor.shape) == 4:
        B, C, H, W = feature_tensor.shape
        print(f"Tensor shape: {B} batches, {C} channels, {H}x{W} spatial.")
        features_by_channel = feature_tensor.permute(1, 0, 2, 3).reshape(C, -1).numpy()
    elif len(feature_tensor.shape) == 2:
        B, C = feature_tensor.shape
        features_by_channel = feature_tensor.t().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape: {feature_tensor.shape}")

    os.makedirs(save_dir, exist_ok=True)
    
    channel_stats = {}
    channel_scores = [] # Mảng lưu điểm "chuẩn" của từng channel
    cis = [99, 95, 90]
    
    print("=" * 60)
    print(f"{'CALCULATING STATS FOR ' + str(C) + ' CHANNELS':^60}")
    print("=" * 60)

    for ch in tqdm(range(C), desc="Processing Channels"):
        ch_data = features_by_channel[ch]
        
        # Để scipy tính nhanh hơn nếu data quá lớn, ta lấy mẫu đại diện
        sample_data = ch_data if len(ch_data) < 50000 else np.random.choice(ch_data, 50000, replace=False)
        
        # Tính Skewness và Kurtosis
        skewness = stats.skew(sample_data)
        kurt = stats.kurtosis(sample_data) # Excess kurtosis
        normality_score = abs(skewness) + abs(kurt) # Càng gần 0 càng giống phân phối chuẩn
        
        channel_scores.append((ch, normality_score))
        
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        
        ch_stat = {
            "mean": float(mean_val),
            "std": float(std_val),
            "skewness": float(skewness),
            "kurtosis": float(kurt),
            "normality_score": float(normality_score),
            "percentiles": {}
        }
        
        for ci in cis:
            tail = (100 - ci) / 2.0
            val_lower = np.percentile(ch_data, tail)
            val_upper = np.percentile(ch_data, 100 - tail)
            ch_stat["percentiles"][f"{ci}%_CI"] = {
                "min": float(val_lower),
                "max": float(val_upper)
            }
            
        channel_stats[f"channel_{ch}"] = ch_stat

    json_path = os.path.join(save_dir, 'channel_stats.json')
    with open(json_path, 'w') as f:
        json.dump(channel_stats, f, indent=4)
    print(f"\n[+] Saved full statistics to: {json_path}")

    # --- TÌM 5 CHANNEL CHUẨN NHẤT ---
    # Sắp xếp channel theo normality_score tăng dần (tốt nhất đứng đầu)
    channel_scores.sort(key=lambda x: x[1])
    selected_channels = [ch for ch, score in channel_scores[:5]]
    
    print(f">>> Top 5 most normal channels: {selected_channels}")
    for ch, score in channel_scores[:5]:
        print(f"    - Channel {ch}: Score = {score:.4f}")

    # --- VẼ BIỂU ĐỒ ---
    print(">>> Generating Density Plot for Top 5 Normal Channels...")
    plt.figure(figsize=(10, 6))
    
    for ch in selected_channels:
        data = features_by_channel[ch]
        if len(data) > 50000:
            data = np.random.choice(data, 50000, replace=False)
            
        sns.kdeplot(data, fill=True, alpha=0.3, linewidth=2, label=f'Channel {ch}')

    plt.title('Feature Density Distribution (Representative Channels)', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Mean Target')
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'feature_density_plot_best5.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved density plot to: {plot_path}")
    print("=" * 60)


# ======================================================
# 5. MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./analysis_results")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_train_data", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config_from_path(args.config_path)

    bs = args.batch_size
    force_set_attr(config, 'batch_train', bs)
    force_set_attr(config, 'batch_test_per', bs)
    force_set_attr(config, 'batch_size', bs)

    if hasattr(config, 'data'):
        force_set_attr(config.data, 'batch_size', bs)
        force_set_attr(config.data, 'batch_size_per_gpu', bs)
        force_set_attr(config.data, 'drop_last', False)

    if hasattr(config, 'trainer') and hasattr(config.trainer, 'data'):
        force_set_attr(config.trainer.data, 'batch_size', bs)
        force_set_attr(config.trainer.data, 'batch_size_per_gpu_test', bs)
        force_set_attr(config.trainer.data, 'batch_size_per_gpu', bs)

    config.dist = False
    config.world_size = 1
    config.rank = 0
    
    if hasattr(config, 'model_backbone') and hasattr(config.model_backbone, 'kwargs'):
        config.model_backbone.kwargs['pretrained'] = True

    print(f">>> Loading Data ({'Train' if args.use_train_data else 'Test'})...")
    try:
        train_loader, test_loader = get_loader(config)
    except Exception as e:
        print(f"Lỗi Loader: {e}")
        raise e

    loader = train_loader if args.use_train_data else test_loader
    print(f">>> Loader ready. Batches: {len(loader)}")

    print(">>> Initializing Model...")
    net = get_model(config.model)
    net.to(device)
    net.eval()

    print(">>> Extracting Features...")
    features = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            if isinstance(data, dict):
                imgs = data['img']
            elif isinstance(data, (list, tuple)):
                imgs = data[0]
            else:
                imgs = data
            
            imgs = imgs.to(device)
            out = net(imgs)
            
            if isinstance(out, (list, tuple)):
                feat = out[0]
            elif isinstance(out, dict) and 'feature_align' in out:
                feat = out['feature_align']
            else:
                feat = out 

            features.append(feat.cpu())

    if features:
        full_tensor = torch.cat(features, dim=0)
        print(f"Final Tensor: {full_tensor.shape}")
        calculate_channel_statistics(full_tensor, args.save_dir)
    else:
        print("No features found.")

if __name__ == "__main__":
    main()