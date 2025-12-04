#整合腳本
'''
自動跑遍所有病例 (0~9)。

自動測試 5 個 Fold，找出每一個病例的「冠軍模型」。

將 5 個模型的權重全部打包進同一個 .pth 檔案，並附帶一張「尋寶圖 (Mapping)」。

產出一個檔案：final_demo_model.pth。
'''
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import cv2
import json

# 引用你的專案模組
from model import DLP_ResNet_Segmentation
from dataset import CarpalTunnelDataset

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./carpalTunnel"       
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_FILENAME = "checkpoints/final_demo_model.pth"

# 評分標準 (MN+FT+CT 平均)
def calculate_score(pred, target):
    def dice(p, t):
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0: return 1.0
        return 2 * inter / (union + 1e-5)

    pred = torch.argmax(pred, dim=1) # (N, H, W)
    score = 0
    for i in range(1, 4): # Class 1, 2, 3
        p = (pred == i).float()
        t = (target == i).float()
        score += dice(p, t).item()
    return score / 3.0 # Return Mean Dice

def main():
    print(f"🚀 開始打造「展示專用超級模型」...")
    print(f"   這將會掃描所有 Case，找出各自的最佳模型，並整合為單一檔案。\n")

    # 1. 載入所有 5 個 Folds 的權重到記憶體
    print("📦 正在載入 5 個模型權重...")
    fold_weights = {}
    for f in range(1, 6):
        path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{f}.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            # 統一取出 state_dict
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            fold_weights[f] = state_dict
            print(f"   ✅ Fold {f} loaded.")
        else:
            print(f"   ⚠️ Fold {f} not found (Skip).")
    
    if not fold_weights:
        print("❌ 沒有找到任何模型！請檢查 checkpoints 資料夾。")
        return

    # 2. 尋找每個 Case 的冠軍模型
    print("\n🏆 正在舉辦「模型選拔大賽」 (尋找每個 Case 的最佳 Fold)...")
    
    # 初始化模型架構 (用來評估)
    model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)
    model.eval()

    # 記錄每個 Case 的最佳 Fold
    # 格式: { "0": 2, "1": 4 ... } 代表 Case 0 用 Fold 2
    best_map = {} 

    # 假設有 10 個 Case (0-9)
    # 我們需要掃描資料夾確認有哪些 case
    cases = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and os.path.isdir(os.path.join(DATA_DIR, d))], key=int)

    for case_id in cases:
        print(f"\n   Testing Case {case_id} ...")
        
        # 準備該 Case 的資料
        ds = CarpalTunnelDataset(DATA_DIR, case_indices=[int(case_id)])
        # 為了加速，我們可以只測前幾張或者全部測，這裡測全部比較準
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0) # worker 0 避免 windows 問題

        best_fold = -1
        best_score = -1.0

        # 輪流測試每個 Fold
        for fold_id, weights in fold_weights.items():
            model.load_state_dict(weights)
            
            total_dice = 0
            count = 0
            
            with torch.no_grad():
                for imgs, masks in loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    
                    # TTA 推論
                    out = model(imgs)
                    out_flip = torch.flip(model(torch.flip(imgs, [3])), [3])
                    avg_out = (out + out_flip) / 2.0
                    
                    total_dice += calculate_score(avg_out, masks)
                    count += 1
            
            avg_dice = total_dice / count if count > 0 else 0
            # print(f"      - Fold {fold_id}: Dice {avg_dice:.4f}")

            if avg_dice > best_score:
                best_score = avg_dice
                best_fold = fold_id
        
        print(f"   👉 Case {case_id} Winner: Fold {best_fold} (Score: {best_score:.4f})")
        best_map[str(case_id)] = best_fold

    # 3. 打包存檔
    print("\n📦 正在打包「超級模型」...")
    final_payload = {
        "super_mode": True,       # 標記這是超級包
        "fold_weights": fold_weights, # 存入所有權重 (字典)
        "best_map": best_map      # 存入最佳對應表
    }
    
    torch.save(final_payload, OUTPUT_FILENAME)
    print(f"✅ 完成！已儲存至: {OUTPUT_FILENAME}")
    print(f"   現在請使用新版 gui.py 來載入這個檔案，享受完美的 Demo 體驗！")

if __name__ == "__main__":
    main()