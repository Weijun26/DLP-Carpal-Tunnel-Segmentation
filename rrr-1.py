# 舊版：AdamW + 固定LR + IoU評分 + [自動抓取 loss-1.py]
# 適合 從頭訓練 (From Scratch)
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
import importlib.util # 用於動態載入

from model import DLP_ResNet_Segmentation
from dataset import CarpalTunnelDataset

# --- 動態載入 loss-1.py ---
def load_loss_class(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ 找不到 {filename}！請確認它在程式同一目錄下。")
    spec = importlib.util.spec_from_file_location("dynamic_loss_module", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComboLoss

# 強制使用 loss-1.py
print(f"📦 正在載入 Loss 定義檔: loss-1.py (CE + Dice)...")
ComboLoss = load_loss_class("loss-1.py")
# -------------------------

# --- 使用者設定 ---
MAX_TOTAL_EPOCHS = 300
EPOCHS_PER_ROUND = 10
START_FROM_EPOCH = 0 
BATCH_SIZE = 12 #視 GPU 記憶體調整
LR = 1e-4  

DATA_DIR = "./carpalTunnel"
CHECKPOINT_DIR = "./checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
STOP_REQUESTED = False

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    # Optimizer 防呆載入
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as e:
        print(f"⚠️  Optimizer 狀態不相容 (可能是不同階段的模型)，已重置優化器。")
    
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint.get("best_loss", float("inf"))
    return start_epoch, best_loss

def write_stop_log(fold, epoch):
    with open("last_stop_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Fold: {fold}\nEpoch: {epoch + 1}\n")

def calculate_metrics(pred, target, num_classes):
    dice_scores = []; iou_scores = []
    pred = torch.argmax(pred, dim=1)
    for i in range(1, num_classes):
        p = (pred == i).float(); t = (target == i).float()
        intersection = (p * t).sum()
        total_area = p.sum() + t.sum()
        union_area = total_area - intersection
        
        dice = 1.0 if total_area == 0 else (2. * intersection) / (total_area + 1e-5)
        iou = 1.0 if union_area == 0 else (intersection) / (union_area + 1e-5)
        dice_scores.append(dice.item()); iou_scores.append(iou.item())
    return dice_scores, iou_scores

def train_one_fold(fold_index, train_indices, val_indices, current_target_epoch):
    global STOP_REQUESTED
    train_ds = CarpalTunnelDataset(DATA_DIR, case_indices=train_indices)
    val_ds = CarpalTunnelDataset(DATA_DIR, case_indices=val_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 初始化 Loss-1
    loss_weights = [0.1, 5.0, 2.0, 5.0] 
    criterion = ComboLoss(weights=loss_weights).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold_{fold_index+1}.pth")
    best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold_index+1}.pth")
    
    start_epoch = 0; best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        try:
            start_epoch, best_val_loss = load_checkpoint(torch.load(checkpoint_path), model, optimizer)
        except Exception as e: print(f"載入失敗: {e}")
    elif os.path.exists(best_model_path):
        try: model.load_state_dict(torch.load(best_model_path))
        except: pass

    if start_epoch >= current_target_epoch: return best_val_loss

    print(f"\n>>> Fold {fold_index+1} | 目標: {current_target_epoch} (目前: {start_epoch})")

    for epoch in range(start_epoch, current_target_epoch):
        if STOP_REQUESTED:
            write_stop_log(fold_index + 1, epoch)
            return None 

        model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold_index+1} Ep {epoch+1}", leave=False)
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs); loss = criterion(outputs, masks)
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        val_loss = 0
        mn_d, ft_d, ct_d = [], [], []
        mn_i, ft_i, ct_i = [], [], []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs); loss = criterion(outputs, masks)
                val_loss += loss.item()
                d_s, i_s = calculate_metrics(outputs, masks, 4)
                mn_d.append(d_s[0]); ft_d.append(d_s[1]); ct_d.append(d_s[2])
                mn_i.append(i_s[0]); ft_i.append(i_s[1]); ct_i.append(i_s[2])
        
        avg_val = val_loss / len(val_loader)
        print(f"   Ep {epoch+1} Loss: {avg_val:.4f}")
        print(f"     [Dice] MN: {np.mean(mn_d):.2f} | FT: {np.mean(ft_d):.2f} | CT: {np.mean(ct_d):.2f}")
        print(f"     [IoU ] MN: {np.mean(mn_i):.2f} | FT: {np.mean(ft_i):.2f} | CT: {np.mean(ct_i):.2f}")
        
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best_loss": best_val_loss}
        save_checkpoint(checkpoint, filename=checkpoint_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val; torch.save(model.state_dict(), best_model_path)

    return best_val_loss

def training_thread_func(root_window):
    global STOP_REQUESTED
    kf = KFold(n_splits=5, shuffle=False)
    folds_data = list(kf.split(np.arange(10)))
    print(f"🚀 開始輪替訓練 (AdamW + IoU + Loss-1)！")
    try:
        start_target = START_FROM_EPOCH + EPOCHS_PER_ROUND
        for target in range(start_target, MAX_TOTAL_EPOCHS + 1, EPOCHS_PER_ROUND):
            if STOP_REQUESTED: break
            print(f"\n====== 🔄 第 {target - EPOCHS_PER_ROUND + 1} ~ {target} 輪 ======")
            for f_idx, (t_idx, v_idx) in enumerate(folds_data):
                if STOP_REQUESTED: break
                if train_one_fold(f_idx, t_idx, v_idx, target) is None: break
        if not STOP_REQUESTED: print("\n===== 🎉 完成 =====")
    except Exception as e: print(f"\n❌ 錯誤: {e}")
    finally: root_window.after(100, root_window.destroy)

def on_stop_click(btn):
    global STOP_REQUESTED = True; btn.config(text="停止中...", bg="orange", state="disabled")

def main_gui():
    root = tk.Tk(); root.geometry("300x150"); root.attributes("-topmost", True)
    tk.Label(root, text="DLP 訓練控制器 (rrr-1)\n(Loss-1: CE+Dice)", font=("Arial", 12)).pack(pady=20)
    btn = tk.Button(root, text="⛔ 停止", font=("Arial", 12, "bold"), bg="#ff4d4d", fg="white", command=lambda: on_stop_click(btn))
    btn.pack(fill="x", padx=20)
    threading.Thread(target=training_thread_func, args=(root,), daemon=True).start()
    root.mainloop()

if __name__ == "__main__": main_gui()