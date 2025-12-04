import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import KFold
import threading
import tkinter as tk
from tkinter import messagebox

# 引用模組
from model import DLP_ResNet_Segmentation
from dataset import CarpalTunnelDataset
from loss import ComboLoss

# ====================================================
# --- 使用者設定區 ---
# ====================================================

MAX_TOTAL_EPOCHS = 1000 
EPOCHS_PER_ROUND = 10

# 【修改】設定你目前的進度 (例如 400)
START_FROM_EPOCH = 400

BATCH_SIZE = 24 
LR = 1e-4  # 【注意】這裡改回 1e-4，因為 Scheduler 會幫我們自動降下來

DATA_DIR = "./carpalTunnel"
CHECKPOINT_DIR = "./checkpoints"

# ====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STOP_REQUESTED = False

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    # 這裡依然不載入舊的 optimizer，因為我們要加入新的 Scheduler 機制
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint.get("best_loss", float("inf"))
    return start_epoch, best_loss

def write_stop_log(fold, epoch):
    filename = "last_stop_log.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Fold: {fold}\nEpoch: {epoch + 1}\n")
    print(f"\n📝 Log saved.")

def calculate_dice_metric(pred, target, num_classes):
    dice_scores = []
    pred = torch.argmax(pred, dim=1)
    for i in range(1, num_classes):
        p = (pred == i).float()
        t = (target == i).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        score = 1.0 if union == 0 else (2. * intersection) / (union + 1e-5)
        dice_scores.append(score.item())
    return dice_scores

def train_one_fold(fold_index, train_indices, val_indices, current_target_epoch):
    global STOP_REQUESTED
    
    train_ds = CarpalTunnelDataset(DATA_DIR, case_indices=train_indices)
    val_ds = CarpalTunnelDataset(DATA_DIR, case_indices=val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)
    
    # 1. 使用 SGD
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    
    # 2. 【新增】學習率排程器 (Cosine Annealing)
    # T_0=10: 每 10 輪學習率會重置一次 (配合你的 EPOCHS_PER_ROUND)
    # T_mult=2: 每次重置週期加倍 (10 -> 20 -> 40...)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 權重：維持 CT 救援
    loss_weights = [0.1, 5.0, 1.0, 5.0]
    criterion = ComboLoss(weights=loss_weights).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold_{fold_index+1}.pth")
    best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold_index+1}.pth")
    
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            start_epoch, best_val_loss = load_checkpoint(checkpoint, model, optimizer)
        except: pass
    elif os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path))
        except: pass

    if start_epoch >= current_target_epoch:
        return best_val_loss

    print(f"\n>>> Fold {fold_index+1} | 目標: {current_target_epoch} (目前: {start_epoch})")

    for epoch in range(start_epoch, current_target_epoch):
        if STOP_REQUESTED:
            write_stop_log(fold_index + 1, epoch)
            return None 

        model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold_index+1} Ep {epoch+1}", leave=False)
        
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 【新增】每個 Batch 更新 Scheduler (或者每輪更新也可以，這裡用每 Batch 更新更平滑)
            # 注意：CosineAnnealingWarmRestarts 通常是 batch-level 或 epoch-level 都可以
            # 這裡我們放在 Epoch 結束後更新比較簡單
            
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        # 【新增】更新學習率
        scheduler.step(epoch + (epoch / len(train_loader))) 
        current_lr = optimizer.param_groups[0]['lr'] # 取得當前 LR 用於觀察

        model.eval()
        val_loss = 0
        mn_scores, ft_scores, ct_scores = [], [], []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                scores = calculate_dice_metric(outputs, masks, 4)
                mn_scores.append(scores[0]); ft_scores.append(scores[1]); ct_scores.append(scores[2])
        
        avg_val_loss = val_loss / len(val_loader)
        avg_mn = sum(mn_scores) / len(mn_scores)
        avg_ft = sum(ft_scores) / len(ft_scores)
        avg_ct = sum(ct_scores) / len(ct_scores)

        # 顯示時加入 LR 資訊
        print(f"   Ep {epoch+1} [LR={current_lr:.6f}] Loss: {avg_val_loss:.4f} | MN: {avg_mn:.2f} | FT: {avg_ft:.2f} | CT: {avg_ct:.2f}")
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_val_loss,
        }
        save_checkpoint(checkpoint, filename=checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    return best_val_loss

def training_thread_func(root_window):
    global STOP_REQUESTED
    all_cases = np.arange(10)
    kf = KFold(n_splits=5, shuffle=False)
    folds_data = list(kf.split(all_cases)) 
    
    print(f"🚀 開始輪替訓練 (Scheduler + FocalLoss)！")
    
    try:
        start_target = START_FROM_EPOCH + EPOCHS_PER_ROUND
        for target in range(start_target, MAX_TOTAL_EPOCHS + 1, EPOCHS_PER_ROUND):
            if STOP_REQUESTED: break
            print(f"\n====== 🔄 第 {target - EPOCHS_PER_ROUND + 1} ~ {target} 輪 ======")
            for fold_idx, (train_idx, val_idx) in enumerate(folds_data):
                if STOP_REQUESTED: break
                train_cases = all_cases[train_idx]
                val_cases = all_cases[val_idx]
                result = train_one_fold(fold_idx, train_cases, val_cases, current_target_epoch=target)
                if result is None: break
        
        if not STOP_REQUESTED: print("\n===== 🎉 完成 =====")

    except Exception as e: print(f"\n❌ 錯誤: {e}")
    finally: root_window.after(100, root_window.destroy)

def on_stop_click(btn):
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        btn.config(text="正在停止...", bg="orange", state="disabled")
        print("\n⚠️ 收到停止訊號！")

def main_gui():
    root = tk.Tk()
    root.title("訓練控制器 (Final Boost)")
    root.geometry("350x180")
    root.attributes("-topmost", True) 
    lbl = tk.Label(root, text="DLP 最終衝刺\n(Scheduler + Focal Loss)", font=("Arial", 10))
    lbl.pack(pady=20)
    btn_stop = tk.Button(root, text="⛔ 停止訓練", font=("Arial", 12, "bold"), bg="#ff4d4d", fg="white", height=2, command=lambda: on_stop_click(btn_stop))
    btn_stop.pack(fill="x", padx=20, pady=10)
    t = threading.Thread(target=training_thread_func, args=(root,), daemon=True)
    t.start()
    root.mainloop()

if __name__ == "__main__":
    main_gui()