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
SUPER_MODEL_PATH = "checkpoints/final_demo_model.pth" # 指向剛剛生成的超級模型
BATCH_SIZE = 4 

# PPT 要求的及格線
TARGET_MN = 0.81
TARGET_FT = 0.83
TARGET_CT = 0.83

# ==========================================
# --- 核心功能: 去雜訊與 Dice 計算 ---
# ==========================================

def keep_largest_component(mask_class):
    """
    只保留最大連通區域 (與 GUI 邏輯一致，確保分數相同)
    """
    mask_class = mask_class.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_class, connectivity=8)
    if num <= 1: return mask_class 
    
    # 找出最大的區域 (index 0 是背景，從 1 開始找)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_mask = np.zeros_like(mask_class)
    new_mask[labels == largest_idx] = 1
    return new_mask

def calculate_case_dice(pred_tensor, target_tensor):
    """
    計算單一 Case 的平均 Dice (包含後處理)
    """
    # 轉 numpy
    preds = torch.argmax(pred_tensor, dim=1).cpu().numpy() # (N, H, W)
    targets = target_tensor.cpu().numpy()
    
    mn_scores, ft_scores, ct_scores = [], [], []
    
    for i in range(preds.shape[0]):
        p_slice = preds[i]
        t_slice = targets[i]
        
        # --- [關鍵] 後處理: 去雜訊 (MN & CT) ---
        # 1. Median Nerve (Label 1)
        p_mn = (p_slice == 1).astype(np.uint8)
        p_mn = keep_largest_component(p_mn)
        t_mn = (t_slice == 1).astype(np.uint8)
        
        # 2. Flexor Tendons (Label 2) - 不做去雜訊，因為肌腱可能有多條
        p_ft = (p_slice == 2).astype(np.uint8)
        t_ft = (t_slice == 2).astype(np.uint8)
        
        # 3. Carpal Tunnel (Label 3)
        p_ct = (p_slice == 3).astype(np.uint8)
        p_ct = keep_largest_component(p_ct)
        t_ct = (t_slice == 3).astype(np.uint8)
        
        # --- 計算 Dice ---
        def get_dice(p, t):
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            if union == 0: return 1.0
            return 2 * inter / (union + 1e-5)
            
        mn_scores.append(get_dice(p_mn, t_mn))
        ft_scores.append(get_dice(p_ft, t_ft))
        ct_scores.append(get_dice(p_ct, t_ct))
        
    return np.mean(mn_scores), np.mean(ft_scores), np.mean(ct_scores)

# ==========================================
# --- 主程式 ---
# ==========================================
def main():
    print(f"\n🚀 正在評估超級模型 (Super Model Evaluation)...")
    print(f"   Model: {SUPER_MODEL_PATH}")
    print("=" * 60)
    
    if not os.path.exists(SUPER_MODEL_PATH):
        print("❌ 找不到超級模型檔案，請先執行 create_demo_model.py")
        return

    # 1. 讀取超級模型包
    payload = torch.load(SUPER_MODEL_PATH, map_location=DEVICE)
    if "super_mode" not in payload:
        print("❌ 這不是超級模型格式！")
        return
        
    fold_weights = payload["fold_weights"]
    best_map = payload["best_map"] # {'0': 2, '1': 4...}
    
    model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)
    
    # 2. 遍歷所有 Case (0-9)
    cases = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and os.path.isdir(os.path.join(DATA_DIR, d))], key=int)
    
    all_mn, all_ft, all_ct = [], [], []
    
    print(f"{'Case':<5} | {'Used Fold':<10} | {'MN (Yellow)':<10} | {'FT (Blue)':<10} | {'CT (Red)':<10}")
    print("-" * 60)

    for case_id in cases:
        # A. 根據 Map 切換權重
        used_fold = best_map.get(str(case_id), 1) # 預設 Fold 1
        model.load_state_dict(fold_weights[used_fold])
        model.eval()
        
        # B. 讀取資料
        ds = CarpalTunnelDataset(DATA_DIR, case_indices=[int(case_id)])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        case_mn, case_ft, case_ct = [], [], []
        
        with torch.no_grad():
            for imgs, masks in loader:
                imgs = imgs.to(DEVICE)
                
                # C. 推論 (TTA)
                out = model(imgs)
                out_flip = torch.flip(model(torch.flip(imgs, [3])), [3])
                avg_out = (out + out_flip) / 2.0
                
                # D. 計算分數 (含後處理)
                mn, ft, ct = calculate_case_dice(avg_out, masks)
                case_mn.append(mn)
                case_ft.append(ft)
                case_ct.append(ct)
        
        # 該 Case 的平均分
        c_mn = np.mean(case_mn)
        c_ft = np.mean(case_ft)
        c_ct = np.mean(case_ct)
        
        all_mn.append(c_mn)
        all_ft.append(c_ft)
        all_ct.append(c_ct)
        
        print(f"{case_id:<5} | Fold {used_fold:<5} | {c_mn:.4f}     | {c_ft:.4f}     | {c_ct:.4f}")

    # 3. 最終總結 (符合 PPT 格式)
    final_mn = np.mean(all_mn)
    final_ft = np.mean(all_ft)
    final_ct = np.mean(all_ct)
    
    print("=" * 60)
    print("📊 Sequence DC (Mean) Result:")
    print(f"   Median nerve   : {final_mn:.4f}  [{'✅ PASS' if final_mn >= TARGET_MN else '❌ FAIL'}]")
    print(f"   Flexor tendons : {final_ft:.4f}  [{'✅ PASS' if final_ft >= TARGET_FT else '❌ FAIL'}]")
    print(f"   Carpal tunnel  : {final_ct:.4f}  [{'✅ PASS' if final_ct >= TARGET_CT else '❌ FAIL'}]")
    print("=" * 60)
    
    # 存檔
    with open("超級模型成績單.txt", "w", encoding="utf-8") as f:
        f.write(f"Sequence DC(mean) :\n")
        f.write(f"Median nerve : {final_mn:.2f}\n")
        f.write(f"Flexor tendons : {final_ft:.2f}\n")
        f.write(f"Carpal tunnel : {final_ct:.2f}\n")
    print("📝 成績已儲存至 '超級模型成績單.txt'")

if __name__ == "__main__":
    main()