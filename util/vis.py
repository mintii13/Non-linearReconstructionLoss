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
    # 1. Đảm bảo imgs và masks cùng kích thước (Logic cũ của bạn)
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)

    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path_save = f'{out_dir}/{img_num}_img.png'     # Đổi tên biến tránh trùng lặp
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        # --- Chuẩn bị ảnh gốc (Base Image) ---
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        img_rec = (img_rec * 255).clamp(0, 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        img_rec = Image.fromarray(img_rec)
        img_rec.save(img_path_save)

        # --- Xử lý Anomaly Map (Quan trọng: Resize về bằng ảnh gốc) ---
        # Nếu anomaly_map là (C, H, W) hoặc (H, W), ta cần đưa về (H, W) chuẩn
        if anomaly_map.ndim == 3: 
            # Nếu map có nhiều channel (ví dụ output từ feature), lấy trung bình
            anomaly_map = np.mean(anomaly_map, axis=0) 
            
        # Normalize về 0-1 để tránh lỗi cm.jet
        if anomaly_map.max() > 0:
            anomaly_map = anomaly_map / anomaly_map.max()
        
        # Apply colormap
        anomaly_map_vis = cm.jet(anomaly_map) # Kết quả ra (H, W, 4)
        anomaly_map_vis = (anomaly_map_vis[:, :, :3] * 255).astype('uint8') # Bỏ kênh Alpha, lấy RGB
        
        # Convert sang PIL Image
        anomaly_map_pil = Image.fromarray(anomaly_map_vis)

        # [FIX QUAN TRỌNG] Resize Anomaly Map cho bằng kích thước ảnh gốc
        if anomaly_map_pil.size != img_rec.size:
            anomaly_map_pil = anomaly_map_pil.resize(img_rec.size, Image.BILINEAR)

        # Blend
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map_pil, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)

        # --- Save Mask ---
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        # Resize mask nếu cần (đề phòng trường hợp hiếm)
        if img_mask.size != img_rec.size:
            img_mask = img_mask.resize(img_rec.size, Image.NEAREST)
        img_mask.save(mask_path)