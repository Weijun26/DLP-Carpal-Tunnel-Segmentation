import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import cv2

# 引用你的專案模組
from model import DLP_ResNet_Segmentation
from dataset import CarpalTunnelDataset

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./carpalTunnel"       
CHECKPOINT_DIR = "./checkpoints" # 掃描這個資料夾
BATCH_SIZE = 8 # 評估可以設大一點

# PPT 要求的及格線
TARGET_MN = 0.81
TARGET_FT = 0.83
TARGET_CT = 0.83

# ==========================================
# --- 核心功能: 去雜訊與 Dice 計算 (維持不變) ---
# ==========================================

def keep_largest_component(mask_class):
    mask_class = mask_class.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_class, connectivity=8)
    if num <= 1: return mask_class 
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_mask = np.zeros_like(mask_class)
    new_mask[labels == largest_idx] = 1
    return new_mask

def calculate_case_dice(pred_tensor, target_tensor):
    preds = torch.argmax(pred_tensor, dim=1).cpu().numpy()
    targets = target_tensor.cpu().numpy()
    mn_scores, ft_scores, ct_scores = [], [], []
    
    for i in range(preds.shape[0]):
        p_slice, t_slice = preds[i], targets[i]
        
        # Post-processing
        p_mn = keep_largest_component((p_slice == 1).astype(np.uint8))
        t_mn = (t_slice == 1).astype(np.uint8)
        
        p_ft = (p_slice == 2).astype(np.uint8) # 肌腱通常有多個，不去雜訊
        t_ft = (t_slice == 2).astype(np.uint8)
        
        p_ct = keep_largest_component((p_slice == 3).astype(np.uint8))
        t_ct = (t_slice == 3).astype(np.uint8)
        
        def get_dice(p, t):
            inter = (p * t).sum(); union = p.sum() + t.sum()
            return 1.0 if union == 0 else 2 * inter / (union + 1e-5)
            
        mn_scores.append(get_dice(p_mn, t_mn))
        ft_scores.append(get_dice(p_ft, t_ft))
        ct_scores.append(get_dice(p_ct, t_ct))
        
    return np.mean(mn_scores), np.mean(ft_scores), np.mean(ct_scores)

# ==========================================
# --- 評估邏輯 ---
# ==========================================

def evaluate_model(model, loader, is_super_mode=False, best_map=None, fold_weights=None, case_id=None):
    """
    通用評估函數
    如果是超級模式，需要傳入 best_map, fold_weights, case_id
    """
    if is_super_mode:
        # 超級模式：根據 Case ID 切換權重
        used_fold = best_map.get(str(case_id), 1)
        model.load_state_dict(fold_weights[used_fold])
        model.eval()
    
    total_mn, total_ft, total_ct = [], [], []
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            # TTA 推論
            out = model(imgs)
            out_flip = torch.flip(model(torch.flip(imgs, [3])), [3])
            avg_out = (out + out_flip) / 2.0
            
            mn, ft, ct = calculate_case_dice(avg_out, masks)
            total_mn.append(mn); total_ft.append(ft); total_ct.append(ct)
            
    return np.mean(total_mn), np.mean(total_ft), np.mean(total_ct)

def main():
    print(f"\n🚀 批量模型評估工具 (Batch Evaluation)")
    print(f"   Scanning: {CHECKPOINT_DIR}\n")
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("❌ 找不到 checkpoints 資料夾！")
        return

    # 1. 搜尋所有 .pth 檔案
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
    if not files:
        print("❌ 沒有找到任何 .pth 模型檔。")
        return

    # 2. 準備所有資料 (為了加速，一次性建立 DataLoader 列表)
    #    但為了節省記憶體，我們還是針對每個 Case 動態讀取比較保險
    cases = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and os.path.isdir(os.path.join(DATA_DIR, d))], key=int)
    
    # 3. 初始化模型架構
    model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)

    # 表格標頭
    header = f"{'Model Filename':<30} | {'Epoch':<8} | {'MN':<6} | {'FT':<6} | {'CT':<6} | {'Avg':<6}"
    print("=" * 90)
    print(header)
    print("-" * 90)

    for fname in files:
        path = os.path.join(CHECKPOINT_DIR, fname)
        
        # --- 讀取模型資訊 ---
        try:
            payload = torch.load(path, map_location=DEVICE)
        except Exception as e:
            print(f"{fname:<30} | ❌ Load Error: {e}")
            continue

        epoch_info = "N/A"
        is_super_mode = False
        weights = None
        
        # 分辨檔案類型
        if isinstance(payload, dict):
            if "super_mode" in payload:
                # 類型 A: 超級模型
                is_super_mode = True
                epoch_info = "Ensemble"
                fold_weights = payload["fold_weights"]
                best_map = payload["best_map"]
            elif "state_dict" in payload:
                # 類型 B: Checkpoint (包含 epoch 資訊)
                weights = payload["state_dict"]
                if "epoch" in payload:
                    epoch_info = str(payload["epoch"])
            else:
                # 類型 C: 只有權重的字典 (Best Model 通常是這種，或者它直接就是 state_dict)
                weights = payload
                # 嘗試看看有沒有隱藏的 epoch key，通常 best model 為了省空間只存權重
                if "epoch" in payload: epoch_info = str(payload["epoch"])
        else:
            # 極少見情況，直接是 model object
            print(f"{fname:<30} | ❌ Unknown Format")
            continue

        # 如果不是超級模型，先載入權重
        if not is_super_mode:
            try:
                model.load_state_dict(weights)
                model.eval()
            except:
                print(f"{fname:<30} | ❌ Architecture Mismatch")
                continue

        # --- 開始評估所有 Case ---
        all_mn, all_ft, all_ct = [], [], []
        
        # 顯示進度條比較不無聊
        # loop = tqdm(cases, desc="Testing", leave=False)
        for case_id in cases:
            ds = CarpalTunnelDataset(DATA_DIR, case_indices=[int(case_id)])
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
            
            if is_super_mode:
                mn, ft, ct = evaluate_model(model, loader, True, best_map, fold_weights, case_id)
            else:
                mn, ft, ct = evaluate_model(model, loader, False)
                
            all_mn.append(mn); all_ft.append(ft); all_ct.append(ct)

        # 計算平均
        avg_mn = np.mean(all_mn)
        avg_ft = np.mean(all_ft)
        avg_ct = np.mean(all_ct)
        overall = (avg_mn + avg_ft + avg_ct) / 3.0

        # --- 顯示結果 ---
        # 顏色標記：如果達標顯示綠色勾勾 (這裡用文字表示)
        # pass_mn = "v" if avg_mn >= TARGET_MN else " "
        
        print(f"{fname:<30} | {epoch_info:<8} | {avg_mn:.4f} | {avg_ft:.4f} | {avg_ct:.4f} | {overall:.4f}")

    print("=" * 90)
    print(f"Target: MN > {TARGET_MN}, FT > {TARGET_FT}, CT > {TARGET_CT}")

if __name__ == "__main__":
    main()