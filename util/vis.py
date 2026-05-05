import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
# import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F

def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, method, root_out, dataset_name):
    # 1. Đảm bảo imgs và masks cùng kích thước pixel
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)

    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        # --- Xử lý đường dẫn lưu file ---
        parts = img_path.split('/')
        needed_parts = parts[parts.index('mvtec')+1:-1] if 'mvtec' in parts else parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = os.path.join(root_out, method, specific_root)
        os.makedirs(out_dir, exist_ok=True)
        
        img_path_save = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        # --- 1. Chuẩn bị ảnh gốc (Base Image) ---
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        
        # Denormalize: [C, H, W]
        img_rec_tensor = img * std[:, None, None] + mean[:, None, None]
        img_rec_tensor = (img_rec_tensor * 255).clamp(0, 255).type(torch.uint8)
        
        # Convert sang [H, W, C] numpy để tạo PIL Image
        img_rec_np = img_rec_tensor.cpu().numpy().transpose(1, 2, 0)
        img_rec_pil = Image.fromarray(img_rec_np)
        img_rec_pil.save(img_path_save)

        # --- 2. Xử lý Anomaly Map (Cốt lõi lỗi nằm ở đây) ---
        # Đảm bảo anomaly_map là numpy và loại bỏ các chiều thừa (Squeeze)
        if torch.is_tensor(anomaly_map):
            anomaly_map = anomaly_map.cpu().numpy()
        
        # Loại bỏ các chiều size=1 (ví dụ từ (1,1,H,W) về (H,W))
        anomaly_map = np.squeeze(anomaly_map)
        
        # Nếu vẫn còn 3 chiều (C, H, W), lấy trung bình các channel
        if anomaly_map.ndim == 3:
            anomaly_map = np.mean(anomaly_map, axis=0)

        # Normalize về 0-1
        am_max = anomaly_map.max()
        if am_max > 0:
            anomaly_map = anomaly_map / am_max
        
        # Apply Colormap (Jet) -> Kết quả (H, W, 4)
        anomaly_map_vis = cm.jet(anomaly_map)
        anomaly_map_vis = (anomaly_map_vis[:, :, :3] * 255).astype('uint8')
        
        # Convert sang PIL và Resize cho khớp ảnh gốc
        anomaly_map_pil = Image.fromarray(anomaly_map_vis)
        if anomaly_map_pil.size != img_rec_pil.size:
            anomaly_map_pil = anomaly_map_pil.resize(img_rec_pil.size, Image.BILINEAR)

        # Blend ảnh gốc với Heatmap
        img_rec_anomaly_map = Image.blend(img_rec_pil, anomaly_map_pil, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)

        # --- 3. Lưu Mask ---
        # Giả sử img_mask có dạng (1, H, W) hoặc (H, W)
        img_mask_np = np.squeeze(img_mask)
        if img_mask_np.ndim == 2:
            # Tạo ảnh grayscale 8-bit
            img_mask_pil = Image.fromarray((img_mask_np * 255).astype(np.uint8))
        else:
            # Nếu có channel, transpose về (H, W, C)
            img_mask_pil = Image.fromarray((img_mask_np * 255).astype(np.uint8).transpose(1, 2, 0))
            
        if img_mask_pil.size != img_rec_pil.size:
            img_mask_pil = img_mask_pil.resize(img_rec_pil.size, Image.NEAREST)
        
        img_mask_pil.save(mask_path)