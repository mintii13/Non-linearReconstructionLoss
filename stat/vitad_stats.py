# [Copy đè toàn bộ file stat/vitad_stats.py của bạn bằng code này]

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import importlib.util
from argparse import Namespace
import time
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode 
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

# === 1. IMPORT CẦN THIẾT TỪ PROJECT & TORCHVISION ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from model import get_model
    from timm.data.constants import IMAGENET_DEFAULT_MEAN
    from timm.data.constants import IMAGENET_DEFAULT_STD
    from torchvision.transforms import InterpolationMode 
    from data.ad_dataset import DefaultAD 
except ImportError as e:
    print(f"Lỗi Import: {e}")
    sys.exit(1)

# ======================================================
# 2. KHỞI TẠO CÁC BASE CLASS VÀ CONFIG HARDCODE (Giữ nguyên)
# ======================================================
class cfg_common:
    def __init__(self):
        self.logdir = 'logs/default'
        self.task_start_time = time.time()
class cfg_dataset_default:
    def __init__(self):
        self.data = Namespace()
        self.data.num_workers = 4
        self.data.pin_memory = True
        self.data.train_size = 1
        self.data.test_size = 1
        self.data.train_transforms = []
        self.data.test_transforms = []
        self.data.target_transforms = []
        self.data.loader_type = 'pil' 
        self.data.loader_type_target = 'pil_L' 
class cfg_model_vitad:
    def __init__(self):
        self.model = Namespace()
        self.evaluator = Namespace()
        self.optim = Namespace()
        self.loss = Namespace()
        self.trainer = Namespace()
        self.trainer.data = Namespace()
        self.logging = Namespace()

class cfg(cfg_common, cfg_dataset_default, cfg_model_vitad):
    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_vitad.__init__(self)

        self.seed = 42
        self.size = 256
        self.epoch_full = 100
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = self.epoch_full // 10
        self.batch_train = 8
        self.batch_test_per = 8
        self.lr = 1e-4 * self.batch_train / 8
        self.weight_decay = 0.0001
        self.metrics = ['mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max','mAUPRO_px', 'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1', 'mIoU_max_px',]
        self.data.type = 'DefaultAD'
        self.data.root = 'data/mvtec'
        self.data.meta = 'meta.json'
        self.data.cls_names = []
        self.data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]
        self.model_t = Namespace()
        self.model_t.name = 'vit_small_patch16_224_dino'
        self.model_t.kwargs = dict(pretrained=True, checkpoint_path='', pretrained_strict=False,strict=True, img_size=self.size, teachers=[3, 6, 9], neck=[12])
        self.model_f = Namespace()
        self.model_f.name = 'fusion'
        self.model_f.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, mul=1)
        self.model_s = Namespace()
        self.model_s.name = 'de_vit_small_patch16_224_dino'
        self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, img_size=self.size, students=[3, 6, 9], depth=9)
        self.model = Namespace()
        self.model.name = 'vitad'
        self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=self.model_t, model_f=self.model_f, model_s=self.model_s)
        self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100)
        self.optim.lr = self.lr
        self.optim.kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay, amsgrad=False)
        self.trainer.name = 'ViTADTrainer'
        self.trainer.logdir_sub = ''
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2, warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0, use_iters=True, patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1)
        self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per
        self.loss.loss_terms = [dict(type='CosLoss', name='cos', avg=False, lam=1.0),]
        self.logging.log_terms_train = [dict(name='batch_t', fmt=':>5.3f', add_name='avg'),]
        self.logging.log_terms_test = [dict(name='batch_t', fmt=':>5.3f', add_name='avg'),]
        self.dist = False
        self.world_size = 1
        self.rank = 0

# ======================================================
# 3. HÀM DATA TỰ ĐỊNH NGHĨA (THAY THẾ data/__init__.py)
# ======================================================

def get_dataset(cfg):
    """Giả lập hàm get_dataset và chỉ trả về DefaultAD"""
    if cfg.data.type == 'DefaultAD':
        train_transforms = transforms.Compose([
            transforms.Resize(size=(cfg.size, cfg.size), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=(cfg.size, cfg.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        target_transforms = transforms.Compose([
            transforms.Resize(size=(cfg.size, cfg.size), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=(cfg.size, cfg.size)),
            transforms.ToTensor(),
        ])
        
        train_set = DefaultAD(cfg, train=True, transform=train_transforms, target_transform=target_transforms)
        test_set = DefaultAD(cfg, train=False, transform=train_transforms, target_transform=target_transforms)
        return train_set, test_set
    else:
        raise NotImplementedError(f"Dataset type {cfg.data.type} not supported in script.")


def get_loader(cfg):
    """
    Giả lập hàm get_loader và FIX lỗi batch_size=None triệt để.
    """
    train_set, test_set = get_dataset(cfg)
    
    # FIX: Đảm bảo batch_size là số nguyên, lấy giá trị đã được hardcode/gán (8)
    # LƯU Ý: Đây là FIX cuối cùng cho lỗi batch_size=None
    batch_size_train = getattr(cfg.trainer.data, 'batch_size', getattr(cfg, 'batch_train', 8))
    batch_size_test = getattr(cfg.trainer.data, 'batch_size_per_gpu_test', getattr(cfg, 'batch_test_per', 8))

    # train_loader: Luôn cần drop_last=True cho training
    train_loader = DataLoader(dataset=train_set,
                                batch_size=batch_size_train, 
                                shuffle=True,
                                num_workers=cfg.data.num_workers,
                                pin_memory=cfg.data.pin_memory,
                                drop_last=True) 
    
    # test_loader: Không cần drop_last
    test_loader = DataLoader(dataset=test_set,
                                batch_size=batch_size_test,
                                shuffle=False,
                                num_workers=cfg.data.num_workers,
                                pin_memory=cfg.data.pin_memory,
                                drop_last=False) 
    
    return train_loader, test_loader

# ======================================================
# 4. HELPER FUNCTIONS & TÍNH STATS
# ======================================================
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

def calculate_statistics(feature_tensor, save_dir, layer_name="Aggregated"):
    print(f">>> Calculating Statistics for Layer: {layer_name}")
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
    print(f"{layer_name:^73}")
    print("=" * 73)
    print(f"Global Mean: {mean_val:.6f}")
    print(f"Global Std:  {std_val:.6f}")
    print(f"Range:       [{min_val:.6f}, {max_val:.6f}]")
    print("-" * 73)

    cis = [99, 98, 95, 90, 85, 80, 75, 70]
    print(f"{'CI':<6} | {'Min':<10} | {'Max':<10} | {'Mean (CI)':<10} | {'k':<10}")
    print("-" * 73)
    
    for ci in cis:
        tail = (100 - ci) / 2.0
        val_lower = np.percentile(all_values, tail)
        val_upper = np.percentile(all_values, 100 - tail)
        
        mask = (all_values >= val_lower) & (all_values <= val_upper)
        filtered = all_values[mask]
        ci_mean = np.mean(filtered) if len(filtered) > 0 else 0.0

        range_half = (val_upper - val_lower) / 2.0
        k_val = 0.0 if range_half == 0 else 4.0 / range_half
        
        results["percentiles"][f"{ci}%_CI"] = {
            "min": float(val_lower), "max": float(val_upper), 
            "mean": float(ci_mean), "k": float(k_val)
        }
        print(f"{ci}%   | {val_lower:.6f}   | {val_upper:.6f}   | {ci_mean:.6f}   | {k_val:.6f}")
    
    print("=" * 73)
    os.makedirs(save_dir, exist_ok=True)
    
    # Lưu file JSON theo tên Layer
    safe_name = layer_name.replace(" ", "_").replace(":", "")
    json_path = os.path.join(save_dir, f"vitad_target_stats_{safe_name}.json")
    with open(json_path, "w") as f:
        import json
        json.dump(results, f, indent=4)
    print(f"Saved statistics to: {json_path}")


# ======================================================
# 5. MAIN (KẾT HỢP FIX VÀ TÍNH ĐA TẦNG)
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./analysis_results")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_train_data", action="store_true")
    parser.add_argument("--config_path", type=str, default="") 
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = cfg()
    
    bs = args.batch_size
    config.batch_train = bs
    config.batch_test_per = bs
    
    if hasattr(config, 'data'):
        force_set_attr(config.data, 'batch_size', bs)
        force_set_attr(config.data, 'drop_last', False) 
    if hasattr(config, 'trainer') and hasattr(config.trainer, 'data'):
        force_set_attr(config.trainer.data, 'batch_size', bs)
        force_set_attr(config.trainer.data, 'batch_size_per_gpu_test', bs)

    print(f">>> Loading Data ({'Train' if args.use_train_data else 'Test'})...")
    try:
        # 🌟 GỌI HÀM LOADER TỰ ĐỊNH NGHĨA 🌟
        train_loader, test_loader = get_loader(config)
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"LỖI LOADER: {e}")
        raise e

    loader = train_loader if args.use_train_data else test_loader
    print(f">>> Loader ready. Batches: {len(loader)}")

    print(">>> Initializing Model...")
    net = get_model(config.model)
    net.to(device)
    net.eval()
    
    if not hasattr(net, 'net_t'):
        print("Lỗi: Class ViTAD không có thuộc tính 'net_t' (Teacher Encoder).")
        return
        
    teacher_net = net.net_t
    teacher_net.eval() 

    # 🌟 ĐỊNH NGHĨA LIST ĐỂ CHỨA FEATURE RIÊNG BIỆT CHO MỖI TẦNG (STAGE) 🌟
    # Vì output của teacher_net(imgs) là list có 3 tensor [F1, F2, F3] (theo config teachers=[3, 6, 9])
    num_stages = len(config.model_t.kwargs['teachers']) 
    all_features_per_stage = [[] for _ in range(num_stages)]
    
    print(">>> Extracting Teacher Target Features (Multi-Stage)...")
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            if isinstance(data, dict):
                imgs = data['img']
            elif isinstance(data, (list, tuple)):
                imgs = data[0]
            else:
                imgs = data
            
            imgs = imgs.to(device)
            
            feats_t, _ = teacher_net(imgs)
            
            # THU THẬP TỪNG TENSOR VÀO LIST CỦA TẦNG TƯƠNG ỨNG
            if len(feats_t) != num_stages:
                 # Trường hợp lỗi: Teacher trả về số lượng feature map không khớp với config (teachers=[3, 6, 9])
                 print(f"Warning: Teacher returned {len(feats_t)} features, expected {num_stages}.")
                 continue
                 
            for j, feat_tensor in enumerate(feats_t):
                all_features_per_stage[j].append(feat_tensor.cpu())

    # 5. TÍNH STATS RIÊNG BIỆT CHO MỖI TẦNG
    if any(all_features_per_stage):
        for j in range(num_stages):
            features_list_j = all_features_per_stage[j]
            if features_list_j:
                full_tensor_j = torch.cat(features_list_j, dim=0)
                stage_name = f"F{j+1} (Stage {config.model_t.kwargs['teachers'][j]})"
                
                # In ra shape trước khi tính toán
                print(f"\n--- {stage_name} Shape: {full_tensor_j.shape} ---")
                
                calculate_statistics(full_tensor_j, args.save_dir, layer_name=stage_name)
    else:
        print("No features found.")

if __name__ == "__main__":
    main()