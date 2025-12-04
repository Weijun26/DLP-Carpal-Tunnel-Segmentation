import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class CarpalTunnelDataset(Dataset):
    def __init__(self, root_dir, case_indices, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        
        # 【修正】移除這裡的 self.clahe 初始化，避免 Pickling Error
        # self.clahe = cv2.createCLAHE(...)  <-- 這行刪除
        
        for case_idx in case_indices:
            case_path = os.path.join(self.root_dir, str(case_idx))
            t1_path = os.path.join(case_path, 'T1')
            
            if not os.path.exists(t1_path):
                continue
                
            file_names = sorted([f for f in os.listdir(t1_path) if f.endswith('.jpg') or f.endswith('.png')])
            
            for fname in file_names:
                self.image_paths.append({
                    't1': os.path.join(case_path, 'T1', fname),
                    't2': os.path.join(case_path, 'T2', fname),
                    'mn': os.path.join(case_path, 'MN', fname),
                    'ft': os.path.join(case_path, 'FT', fname),
                    'ct': os.path.join(case_path, 'CT', fname)
                })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        
        # 1. 讀取影像
        img_t1 = cv2.imread(paths['t1'], cv2.IMREAD_GRAYSCALE)
        img_t2 = cv2.imread(paths['t2'], cv2.IMREAD_GRAYSCALE)
        
        if img_t1 is None: raise FileNotFoundError(f"Missing {paths['t1']}")

        # --- [修正] 在這裡即時建立 CLAHE 物件 ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # 應用 CLAHE
        img_t1 = clahe.apply(img_t1)
        img_t2 = clahe.apply(img_t2)

        # 2. 讀取 Mask
        h, w = img_t1.shape
        mask_mn = cv2.imread(paths['mn'], cv2.IMREAD_GRAYSCALE)
        mask_ft = cv2.imread(paths['ft'], cv2.IMREAD_GRAYSCALE)
        mask_ct = cv2.imread(paths['ct'], cv2.IMREAD_GRAYSCALE)

        # 製作 label
        label = np.zeros((h, w), dtype=np.int64)
        if mask_ct is not None: label[mask_ct > 127] = 3
        if mask_ft is not None: label[mask_ft > 127] = 2
        if mask_mn is not None: label[mask_mn > 127] = 1

        # ==========================================
        # --- 資料擴增 (Data Augmentation) ---
        # ==========================================
        
        # A. 隨機水平翻轉
        if random.random() > 0.5:
            img_t1 = cv2.flip(img_t1, 1)
            img_t2 = cv2.flip(img_t2, 1)
            label = cv2.flip(label, 1)

        # B. 隨機旋轉
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            
            img_t1 = cv2.warpAffine(img_t1, M, (w, h), flags=cv2.INTER_LINEAR)
            img_t2 = cv2.warpAffine(img_t2, M, (w, h), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)

        # ==========================================

        # 歸一化與堆疊
        img_t1 = img_t1.astype(np.float32) / 255.0
        img_t2 = img_t2.astype(np.float32) / 255.0
        image = np.stack([img_t1, img_t2], axis=0)

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()