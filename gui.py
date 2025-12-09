import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import os
import numpy as np
import cv2

# 引用你的專案模組
from model import DLP_ResNet_Segmentation

# --- 顯示用的顏色定義 (BGR 格式) ---
DISPLAY_COLORS = {
    1: (0, 255, 255),  # MN (正中神經): 黃色
    2: (255, 0, 0),    # FT (屈肌腱): 藍色
    3: (0, 0, 255)     # CT (腕隧道): 紅色
}

def cv2_imread(file_path, flags=cv2.IMREAD_COLOR):
    """
    [關鍵修復] 解決 Windows 中文路徑讀取問題的函式
    使用 numpy 讀取 raw data 再 decode，避開 cv2.imread 的路徑編碼 bug
    """
    try:
        # 使用 numpy 讀取檔案 (這步支援中文路徑)
        stream = np.fromfile(file_path, dtype=np.uint8)
        # 再用 opencv 解碼
        img = cv2.imdecode(stream, flags)
        return img
    except Exception as e:
        print(f"Read Error: {e}")
        return None

class CarpalTunnelDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DLP Final Project - Manual Loader (Chinese Path Fix)")
        self.root.geometry("1200x700")
        
        # 變數初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DLP_ResNet_Segmentation(num_classes=4).to(self.device)
        self.model.eval()
        
        self.data_dir = None
        self.loaded_checkpoint = None
        self.current_paths = []
        self.cases = []
        
        # --- 介面佈局 ---
        self.setup_ui()

    def setup_ui(self):
        # 1. 頂部控制區 (檔案選擇)
        top_frame = tk.Frame(self.root, pady=10, bg="#e0e0e0")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        btn_font = ("Arial", 10, "bold")
        
        # 選擇模型按鈕
        tk.Button(top_frame, text="📂 1. 載入模型 (.pth)", font=btn_font, command=self.select_model).pack(side=tk.LEFT, padx=10)
        self.lbl_model_status = tk.Label(top_frame, text="未載入模型", fg="red", bg="#e0e0e0")
        self.lbl_model_status.pack(side=tk.LEFT, padx=5)
        
        # 分隔線
        tk.Label(top_frame, text="|", bg="#e0e0e0").pack(side=tk.LEFT, padx=10)

        # 選擇資料夾按鈕
        tk.Button(top_frame, text="📂 2. 選擇資料集資料夾 (carpalTunnel)", font=btn_font, command=self.select_data_dir).pack(side=tk.LEFT, padx=10)
        self.lbl_data_status = tk.Label(top_frame, text="未選擇資料夾", fg="red", bg="#e0e0e0")
        self.lbl_data_status.pack(side=tk.LEFT, padx=5)

        # 2. Case 選擇區
        case_frame = tk.Frame(self.root, pady=5)
        case_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(case_frame, text="選擇 Case:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.case_var = tk.StringVar()
        self.case_combo = ttk.Combobox(case_frame, textvariable=self.case_var, state="disabled", width=10)
        self.case_combo.pack(side=tk.LEFT)
        self.case_combo.bind("<<ComboboxSelected>>", self.on_case_change)
        
        self.lbl_fold_info = tk.Label(case_frame, text="", fg="green", font=("Arial", 10, "bold"))
        self.lbl_fold_info.pack(side=tk.LEFT, padx=20)

        # 3. 圖片顯示區
        img_frame = tk.Frame(self.root)
        img_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        self.panel_t1 = self.create_image_panel(img_frame, "Original MRI (T1)", 0)
        self.panel_gt = self.create_image_panel(img_frame, "Ground Truth (Color)", 1)
        self.panel_pred = self.create_image_panel(img_frame, "Prediction Result", 2)
        
        # 4. 底部控制區
        bottom_frame = tk.Frame(self.root, pady=10, bg="#f0f0f0")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.score_label = tk.Label(bottom_frame, text="Dice: MN: 0.00 | FT: 0.00 | CT: 0.00", font=("Consolas", 14, "bold"), bg="#f0f0f0")
        self.score_label.pack(pady=5)

        slider_frame = tk.Frame(bottom_frame, bg="#f0f0f0")
        slider_frame.pack(fill=tk.X, padx=50)
        tk.Label(slider_frame, text="Slice Index:", bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.slider = tk.Scale(slider_frame, from_=0, to=19, orient=tk.HORIZONTAL, command=self.on_slider_move, length=800)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    def create_image_panel(self, parent, title, col):
        frame = tk.Frame(parent)
        frame.grid(row=0, column=col, padx=5, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        tk.Label(frame, text=title, font=("Arial", 12, "bold")).pack()
        # 預設給一個灰色背景，確保看得到框
        canvas = tk.Label(frame, bg="#333333") 
        canvas.pack(expand=True, fill=tk.BOTH)
        return canvas

    # --- 功能函數 ---

    def select_model(self):
        path = filedialog.askopenfilename(title="選擇模型權重檔", filetypes=[("PyTorch Model", "*.pth")])
        if path:
            try:
                print(f"Loading model: {path}")
                self.loaded_checkpoint = torch.load(path, map_location=self.device)
                self.lbl_model_status.config(text="✅ 模型已載入", fg="green")
                if self.data_dir:
                    self.on_case_change(None)
            except Exception as e:
                messagebox.showerror("錯誤", f"載入失敗: {e}")

    def select_data_dir(self):
        path = filedialog.askdirectory(title="選擇 carpalTunnel 資料夾")
        if path:
            self.data_dir = path
            self.cases = sorted([d for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))], key=int)
            
            if not self.cases:
                messagebox.showwarning("警告", "選擇的資料夾中沒有發現數字編號的病例資料夾 (如 0, 1, 2...)")
                return

            self.case_combo['values'] = self.cases
            self.case_combo.current(0)
            self.case_combo.config(state="readonly")
            self.lbl_data_status.config(text=f"✅ 已載入 (共 {len(self.cases)} 例)", fg="green")
            
            self.load_case_data(self.cases[0])

    def load_case_data(self, case_id):
        if not self.data_dir: return

        # 1. 自動選模邏輯
        if self.loaded_checkpoint and "best_map" in self.loaded_checkpoint:
            mapping = self.loaded_checkpoint["best_map"]
            best_fold = mapping.get(str(case_id), 1)
            
            if best_fold not in self.loaded_checkpoint["fold_weights"]:
                 best_fold = list(self.loaded_checkpoint["fold_weights"].keys())[0]

            weights = self.loaded_checkpoint["fold_weights"][best_fold]
            self.model.load_state_dict(weights)
            self.lbl_fold_info.config(text=f"使用模型: Fold {best_fold} (Auto)")
        else:
            self.lbl_fold_info.config(text="使用模型: 當前權重")

        # 2. 構建路徑
        case_path = os.path.join(self.data_dir, str(case_id))
        t1_dir = os.path.join(case_path, 'T1')
        t2_dir = os.path.join(case_path, 'T2')
        gt_dir = os.path.join(case_path, 'GT')
        
        print(f"\n--- Loading Case {case_id} ---")

        self.current_paths = []
        if os.path.exists(t1_dir):
            fnames = sorted([f for f in os.listdir(t1_dir) if f.lower().endswith(('.jpg', '.png'))])
            for f in fnames:
                self.current_paths.append({
                    't1': os.path.join(t1_dir, f),
                    't2': os.path.join(t2_dir, f),
                    'gt': os.path.join(gt_dir, f)
                })
            print(f"Found {len(self.current_paths)} images.")
        else:
            print("❌ T1 folder not found!")

        # 3. 重置滑桿
        max_idx = len(self.current_paths) - 1
        self.slider.config(to=max_idx if max_idx > 0 else 0)
        self.slider.set(0)
        
        if self.current_paths:
            self.update_view(0)
        else:
            self.panel_t1.config(image='')
            self.panel_gt.config(image='')
            self.panel_pred.config(image='')

    def on_case_change(self, event):
        if self.case_var.get():
            self.load_case_data(self.case_var.get())

    def on_slider_move(self, val):
        if self.current_paths:
            self.update_view(int(val))

    def process_gt_image(self, gt_path):
        if not os.path.exists(gt_path):
            return np.zeros((512, 512), dtype=np.uint8)

        # [修正] 使用 cv2_imread 代替 cv2.imread
        gt_color = cv2_imread(gt_path) 
        if gt_color is None: return np.zeros((512, 512), dtype=np.uint8)
        
        h, w = gt_color.shape[:2]
        mask_idx = np.zeros((h, w), dtype=np.uint8)
        
        B, G, R = gt_color[:,:,0], gt_color[:,:,1], gt_color[:,:,2]

        mask_ft = (G > 128) & (B > 128) & (R < 150)
        mask_idx[mask_ft] = 2

        mask_mn = (R > 128) & (B > 128) & (G < 150)
        mask_idx[mask_mn] = 1

        mask_ct = (B > 128) & (R < 100) & (G < 128)
        mask_idx[mask_ct] = 3
        
        return mask_idx

    def apply_overlay(self, bg_img, mask_idx):
        if bg_img is None: return None
        bg_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
        overlay = bg_bgr.copy()
        
        for cls_id, color in DISPLAY_COLORS.items():
            region = (mask_idx == cls_id).astype(np.uint8)
            if np.sum(region) > 0:
                contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bg_bgr, contours, -1, color, 2)
                overlay[region == 1] = color
        
        final = cv2.addWeighted(bg_bgr, 0.7, overlay, 0.3, 0)
        return final

    def update_view(self, idx):
        if idx >= len(self.current_paths): return
        paths = self.current_paths[idx]
        
        # [修正] 使用 cv2_imread
        img_t1 = cv2_imread(paths['t1'], cv2.IMREAD_GRAYSCALE)
        img_t2 = cv2_imread(paths['t2'], cv2.IMREAD_GRAYSCALE)
        
        if img_t1 is None:
            print(f"Still failing to read: {paths['t1']}")
            return

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        t1_c = clahe.apply(img_t1)
        t2_c = clahe.apply(img_t2)
        
        input_tensor = np.stack([t1_c, t2_c], axis=0).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_mask_idx = self.process_gt_image(paths['gt'])

        scores = {}
        for name, i in [('MN', 1), ('FT', 2), ('CT', 3)]:
            p = (pred_mask == i)
            t = (gt_mask_idx == i)
            inter = np.sum(p & t)
            union = np.sum(p) + np.sum(t)
            score = (2.0 * inter / (union + 1e-5)) if union > 0 else 1.0
            scores[name] = score
            
        self.score_label.config(text=f"Dice: MN: {scores['MN']:.2f} | FT: {scores['FT']:.2f} | CT: {scores['CT']:.2f}")

        show_t1 = Image.fromarray(img_t1)
        vis_gt = self.apply_overlay(img_t1, gt_mask_idx)
        show_gt = Image.fromarray(cv2.cvtColor(vis_gt, cv2.COLOR_BGR2RGB))
        vis_pred = self.apply_overlay(img_t1, pred_mask)
        show_pred = Image.fromarray(cv2.cvtColor(vis_pred, cv2.COLOR_BGR2RGB))
        
        disp_size = (380, 380)
        self.tk_t1 = ImageTk.PhotoImage(show_t1.resize(disp_size))
        self.tk_gt = ImageTk.PhotoImage(show_gt.resize(disp_size))
        self.tk_pred = ImageTk.PhotoImage(show_pred.resize(disp_size))
        
        self.panel_t1.config(image=self.tk_t1)
        self.panel_gt.config(image=self.tk_gt)
        self.panel_pred.config(image=self.tk_pred)

if __name__ == "__main__":
    root = tk.Tk()
    app = CarpalTunnelDemoApp(root)
    root.mainloop()