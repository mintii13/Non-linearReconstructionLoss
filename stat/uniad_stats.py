# PYTHONPATH=. python stat/uniad_stats.py --use_train_data --config_path /home/minhtringuyen/ADer/configs/uniad/uniad_mvtec.py --save_dir stat_results --gpu 0

import os

import sys

import argparse

import numpy as np

import torch

from tqdm import tqdm

import importlib.util



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



# ======================================================

# 3. HELPER GÁN DATA AN TOÀN (QUAN TRỌNG)

# ======================================================

def force_set_attr(obj, key, value):

    """Thử gán cả kiểu dict['key'] và object.key"""

    # 1. Thử kiểu Dict

    try:

        obj[key] = value

        # print(f"Set dict: obj['{key}'] = {value}")

        return

    except (TypeError, AttributeError):

        pass

   

    # 2. Thử kiểu Object/Namespace

    try:

        setattr(obj, key, value)

        # print(f"Set attr: obj.{key} = {value}")

        return

    except (TypeError, AttributeError):

        pass



# ======================================================

# 4. TÍNH STATS

# ======================================================

def calculate_statistics(feature_tensor, save_dir):

    print(">>> Calculating Statistics...")

    all_values = feature_tensor.flatten().numpy()

   

    mean_val = np.mean(all_values)

    std_val = np.std(all_values)

    min_val = np.min(all_values)

    max_val = np.max(all_values)



    results = {

        "global_mean": float(mean_val),

        "global_std": float(std_val),

        "global_min": float(min_val),

        "global_max": float(max_val),

        "percentiles": {}

    }



    print("=" * 73)

    print(f"{'STATISTICS':^73}")

    print("=" * 73)

    print(f"Global Mean: {mean_val:.6f}")

    print(f"Global Std:  {std_val:.6f}")

    print(f"Range:       [{min_val:.6f}, {max_val:.6f}]")

    print("-" * 73)



    cis = [99, 98, 95, 90, 85, 80, 75, 70]

    # Cập nhật header mới

    print(f"{'CI':<6} | {'Min':<10} | {'Max':<10} | {'Mean (CI)':<10} | {'k':<10}")

    print("-" * 73)

   

    for ci in cis:

        tail = (100 - ci) / 2.0

        val_lower = np.percentile(all_values, tail)

        val_upper = np.percentile(all_values, 100 - tail)

       

        # Mean of CI

        mask = (all_values >= val_lower) & (all_values <= val_upper)

        filtered = all_values[mask]

        ci_mean = np.mean(filtered) if len(filtered) > 0 else 0.0



        # --- TÍNH TOÁN CỘT 'k' MỚI ---

        range_half = (val_upper - val_lower) / 2.0

       

        # Công thức k = 4 / (Max_CI - Min_CI / 2)

        # Bỏ qua trường hợp chia cho 0 (rất khó xảy ra với percentile)

        if range_half == 0:

            k_val = 0.0

        else:

            k_val = 4.0 / range_half

        # -----------------------------

       

        results["percentiles"][f"{ci}%_CI"] = {

            "min": float(val_lower),

            "max": float(val_upper),

            "mean": float(ci_mean),

            "k": float(k_val) # Thêm k vào JSON

        }

       

        # Cập nhật định dạng in

        print(f"{ci}%   | {val_lower:.6f}   | {val_upper:.6f}   | {ci_mean:.6f}   | {k_val:.6f}")

   



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



    # === FIX DATA LOADER (CRITICAL FIX) ===

    bs = args.batch_size

   

    # 1. Update root attributes

    force_set_attr(config, 'batch_train', bs)

    force_set_attr(config, 'batch_test_per', bs)

    force_set_attr(config, 'batch_size', bs) # Thêm cái này phòng trường hợp get_loader đọc trực tiếp



    # 2. Update config.data (Nơi get_loader thường đọc nhất)

    if hasattr(config, 'data'):

        force_set_attr(config.data, 'batch_size', bs)

        force_set_attr(config.data, 'batch_size_per_gpu', bs)

        # Tắt drop_last để tránh xung đột nếu batch_size vẫn lỗi (phòng hờ)

        force_set_attr(config.data, 'drop_last', False)



    # 3. Update config.trainer.data

    if hasattr(config, 'trainer') and hasattr(config.trainer, 'data'):

        force_set_attr(config.trainer.data, 'batch_size', bs)

        force_set_attr(config.trainer.data, 'batch_size_per_gpu_test', bs)

        force_set_attr(config.trainer.data, 'batch_size_per_gpu', bs)



    # 4. Disable Distributed

    config.dist = False

    config.world_size = 1

    config.rank = 0

   

    # 5. Force Pretrained Backbone

    if hasattr(config, 'model_backbone') and hasattr(config.model_backbone, 'kwargs'):

        config.model_backbone.kwargs['pretrained'] = True



    # === EXECUTE ===

    print(f">>> Loading Data ({'Train' if args.use_train_data else 'Test'})...")

    # Lấy loader

    try:

        train_loader, test_loader = get_loader(config)

    except Exception as e:

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(f"Vẫn lỗi Loader: {e}")

        print("Check lại xem trong file 'data/__init__.py' hàm get_loader đọc biến nào.")

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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

            # Xử lý input đa dạng

            if isinstance(data, dict):

                imgs = data['img']

            elif isinstance(data, (list, tuple)):

                imgs = data[0]

            else:

                imgs = data

           

            imgs = imgs.to(device)

           

            # Forward (Lấy output đầu tiên: feature_align)

            out = net(imgs)

            # UniAD trả về tuple (feature_align, feature_rec, pred) hoặc dict

            if isinstance(out, (list, tuple)):

                feat = out[0]

            elif isinstance(out, dict) and 'feature_align' in out:

                feat = out['feature_align']

            else:

                feat = out # Fallback



            features.append(feat.cpu())



    if features:

        full_tensor = torch.cat(features, dim=0)

        print(f"Final Tensor: {full_tensor.shape}")

        calculate_statistics(full_tensor, args.save_dir)

    else:

        print("No features found.")



if __name__ == "__main__":

    main()