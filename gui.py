import sys
import os
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QFrame, QGridLayout, QMessageBox, QComboBox, QSizePolicy, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtCore import Qt

# 引用模型
from model import DLP_ResNet_Segmentation

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
DEFAULT_DATA_ROOT = "./carpalTunnel" 

# 下拉選單樣式 (維持白色背景)
COMBO_STYLE = """
    QComboBox {
        background-color: white;
        color: #333;
        border: 1px solid #aaa;
        border-radius: 6px;
        padding: 5px;
        font-family: "Microsoft JhengHei", Arial;
        font-size: 14px;
        font-weight: bold;
    }
    QComboBox::drop-down { border: 0px; }
    QComboBox QAbstractItemView {
        background-color: white;
        color: #333;
        selection-background-color: #d0d0d0;
        selection-color: black;
        outline: 0px;
    }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLP 期末專案 - AI 影像分割展示系統 (DLP Final Project - AI Segmentation Demo)")
        self.setGeometry(50, 50, 1450, 900) # 視窗再加大一點

        self.root_dir = DEFAULT_DATA_ROOT
        self.t1_folder = ""
        self.t2_folder = ""
        self.gt_folder = "" 
        self.image_list = []
        self.model = None
        
        # 超級模式變數
        self.is_super_mode = False
        self.cached_weights = {}
        self.best_map = {}
        
        self.init_ui()
        self.populate_model_combo()
        if os.path.exists(self.root_dir):
            self.populate_case_combo()

    # --- [輔助函式] 產生雙語 HTML 標籤文字 ---
    def get_bilingual_text(self, cn, en, cn_size=12, en_size=9, color="#333", align="left"):
        """產生 HTML 格式的雙語文字 (中文大，英文小)"""
        return f"""
        <div style='text-align: {align}; color: {color}; line-height: 120%;'>
            <span style='font-family: "Microsoft JhengHei"; font-size: {cn_size}pt; font-weight: 800;'>{cn}</span><br>
            <span style='font-family: Arial; font-size: {en_size}pt; color: #777;'>{en}</span>
        </div>
        """

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)
        main_widget.setLayout(main_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # =============================
        # 1. 左側控制面板 (Left Panel)
        # =============================
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(380) # 加寬以容納雙語
        left_panel.setStyleSheet("background-color: #f8f9fa; border-radius: 12px; border: 1px solid #dee2e6;")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_panel.setLayout(left_layout)
        
        # 0. Model
        lbl_model = QLabel(self.get_bilingual_text("0. 選擇模型", "Select Model (Checkpoint)", color="#2E86C1"))
        left_layout.addWidget(lbl_model)
        
        self.combo_model = QComboBox()
        self.combo_model.setStyleSheet(COMBO_STYLE)
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        left_layout.addWidget(self.combo_model)
        
        # 狀態顯示
        self.lbl_mode_status = QLabel("模式: 標準 (Standard)")
        self.lbl_mode_status.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_mode_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        left_layout.addWidget(self.lbl_mode_status)
        left_layout.addSpacing(10)

        # 1. Root
        lbl_root = QLabel(self.get_bilingual_text("1. 資料集路徑", "Dataset Root Folder"))
        left_layout.addWidget(lbl_root)
        
        self.btn_root = QPushButton()
        self.btn_root.setText("📂 選擇資料夾 / Select Folder") # 按鈕不支援 HTML，用斜線分隔
        self.btn_root.setStyleSheet("""
            QPushButton {
                background-color: #e2e6ea; border-radius: 6px; padding: 8px; 
                font-family: "Microsoft JhengHei"; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #dbe2ef; }
        """)
        self.btn_root.clicked.connect(self.select_root_folder)
        left_layout.addWidget(self.btn_root)
        
        self.lbl_root_status = QLabel(self.root_dir)
        self.lbl_root_status.setStyleSheet("color: #6c757d; font-size: 10px; margin-top: 2px;")
        self.lbl_root_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_root_status)
        left_layout.addSpacing(15)

        # 2. Case
        lbl_case = QLabel(self.get_bilingual_text("2. 選擇病例 (0-9)", "Select Case ID"))
        left_layout.addWidget(lbl_case)
        
        self.combo_cases = QComboBox()
        self.combo_cases.setStyleSheet(COMBO_STYLE)
        self.combo_cases.currentIndexChanged.connect(self.on_case_changed)
        left_layout.addWidget(self.combo_cases)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar {height: 6px; border: 0px; background: #e9ecef; border-radius: 3px;} QProgressBar::chunk {background: #2E86C1; border-radius: 3px;}")
        self.progress_bar.setTextVisible(False)
        left_layout.addWidget(self.progress_bar)
        left_layout.addSpacing(20)
        
        # 分隔線
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("color: #ced4da;")
        left_layout.addWidget(line)
        
        # --- Sequence Mean Score ---
        lbl_mean_title = QLabel(self.get_bilingual_text("序列平均 Dice 分數", "Sequence DC (Mean)", cn_size=13, en_size=10))
        left_layout.addWidget(lbl_mean_title)
        
        self.lbl_mean_mn = QLabel(self.get_bilingual_text("• 正中神經: -", "Median Nerve", en_size=8, color="#555"))
        self.lbl_mean_ft = QLabel(self.get_bilingual_text("• 屈肌腱: -", "Flexor Tendons", en_size=8, color="#555"))
        self.lbl_mean_ct = QLabel(self.get_bilingual_text("• 腕隧道: -", "Carpal Tunnel", en_size=8, color="#555"))
        
        for lbl in [self.lbl_mean_mn, self.lbl_mean_ft, self.lbl_mean_ct]:
            lbl.setStyleSheet("margin-left: 10px;")
            left_layout.addWidget(lbl)
        left_layout.addSpacing(10)

        # --- Current Slice Score (彩色) ---
        lbl_curr_title = QLabel(self.get_bilingual_text("當前切片 Dice 分數", "Current Slice DC", cn_size=13, en_size=10))
        left_layout.addWidget(lbl_curr_title)
        
        # 使用顏色區分
        self.lbl_curr_mn = QLabel(self.get_bilingual_text("正中神經: 0.00", "Median Nerve", color="#D4AF37")) # 金黃色
        left_layout.addWidget(self.lbl_curr_mn)
        
        self.lbl_curr_ft = QLabel(self.get_bilingual_text("屈肌腱: 0.00", "Flexor Tendons", color="#007ACC")) # 藍色
        left_layout.addWidget(self.lbl_curr_ft)
        
        self.lbl_curr_ct = QLabel(self.get_bilingual_text("腕隧道: 0.00", "Carpal Tunnel", color="#D93025")) # 紅色
        left_layout.addWidget(self.lbl_curr_ct)

        left_layout.addStretch()
        content_layout.addWidget(left_panel)

        # =============================
        # 2. 右側圖片展示區 (Right Panel)
        # =============================
        right_panel = QWidget()
        right_layout = QGridLayout()
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        right_layout.setContentsMargins(10, 0, 10, 0)
        right_layout.setSpacing(25)
        right_panel.setLayout(right_layout)

        self.view_input = QLabel(); self.view_gt = QLabel(); self.view_pred = QLabel()
        IMG_DISP_SIZE = 340 
        for lbl in [self.view_input, self.view_gt, self.view_pred]:
            lbl.setFixedSize(IMG_DISP_SIZE, IMG_DISP_SIZE)
            lbl.setStyleSheet("border: 2px solid #555; background-color: black; border-radius: 8px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setScaledContents(True)

        # 圖片標題 (置中雙語)
        lbl_t1 = QLabel(self.get_bilingual_text("原始 T1 影像", "Original T1 Input", align="center"))
        lbl_gt = QLabel(self.get_bilingual_text("真實標註 (GT)", "Ground Truth", align="center"))
        lbl_ai = QLabel(self.get_bilingual_text("AI 預測結果 (Ours)", "AI Prediction", align="center"))

        right_layout.addWidget(lbl_t1, 0, 0); right_layout.addWidget(self.view_input, 1, 0)
        right_layout.addWidget(lbl_gt, 0, 1); right_layout.addWidget(self.view_gt, 1, 1)
        right_layout.addWidget(lbl_ai, 0, 2); right_layout.addWidget(self.view_pred, 1, 2)

        content_layout.addWidget(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # =============================
        # 3. 下半部滑桿 (Slider)
        # =============================
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #dee2e6; border-radius: 4px; }
            QSlider::handle:horizontal { background: #2E86C1; width: 20px; margin: -6px 0; border-radius: 10px; }
            QSlider::sub-page:horizontal { background: #5DADE2; border-radius: 4px; }
        """)
        
        self.lbl_progress = QLabel("0/0")
        self.lbl_progress.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        lbl_slice = QLabel(self.get_bilingual_text("切片索引:", "Slice Index:", cn_size=11, en_size=8, align="right"))
        
        slider_layout.addWidget(lbl_slice)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_progress)
        main_layout.addLayout(slider_layout)

    # --- 邏輯功能 (維持不變，僅微調文字顯示) ---

    def populate_model_combo(self):
        self.combo_model.clear()
        if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR); return
        models = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
        if not models:
            self.combo_model.addItem("未找到模型 (No models found)")
            self.combo_model.setEnabled(False)
        else:
            self.combo_model.addItems(models)
            self.combo_model.setEnabled(True)
            for i, name in enumerate(models):
                if "final_demo" in name:
                    self.combo_model.setCurrentIndex(i); return
            for i, name in enumerate(models):
                if "fold_4" in name or "best" in name:
                    self.combo_model.setCurrentIndex(i)

    def on_model_changed(self):
        filename = self.combo_model.currentText()
        if not filename or "No models" in filename: return
        path = os.path.join(CHECKPOINT_DIR, filename)
        self.load_model_from_path(path)
        if self.image_list:
            self.update_weights_for_current_case()
            self.calculate_sequence_mean()
            self.run_segmentation(self.slider.value())

    def load_model_from_path(self, path):
        try:
            self.model = DLP_ResNet_Segmentation(num_classes=4).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            if "super_mode" in checkpoint and checkpoint["super_mode"]:
                self.is_super_mode = True
                self.cached_weights = checkpoint["fold_weights"]
                self.best_map = checkpoint["best_map"]
                self.lbl_mode_status.setText("模式: ⚡ 超級演示 (Super Demo Mode)")
                self.lbl_mode_status.setStyleSheet("color: #28B463; font-weight: bold; font-size: 12px;")
                first_fold = list(self.cached_weights.keys())[0]
                self.model.load_state_dict(self.cached_weights[first_fold])
            else:
                self.is_super_mode = False
                state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
                self.lbl_mode_status.setText("模式: 標準 (Standard)")
                self.lbl_mode_status.setStyleSheet("color: gray; font-size: 11px;")
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{e}")
            self.model = None

    def update_weights_for_current_case(self):
        if not self.is_super_mode: return
        case_id = self.combo_cases.currentText()
        if case_id in self.best_map:
            best_fold = self.best_map[case_id]
            if best_fold in self.cached_weights:
                self.model.load_state_dict(self.cached_weights[best_fold])

    def select_root_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if path:
            self.root_dir = path
            self.lbl_root_status.setText(path)
            self.populate_case_combo()

    def populate_case_combo(self):
        self.combo_cases.blockSignals(True)
        self.combo_cases.clear()
        if os.path.exists(self.root_dir):
            subs = sorted([d for d in os.listdir(self.root_dir) if d.isdigit() and os.path.isdir(os.path.join(self.root_dir, d))], key=int)
            self.combo_cases.addItems(subs)
        self.combo_cases.blockSignals(False)
        if self.combo_cases.count() > 0: self.on_case_changed()

    def on_case_changed(self):
        case_id = self.combo_cases.currentText()
        if not case_id: return
        self.t1_folder = os.path.join(self.root_dir, case_id, "T1")
        self.t2_folder = os.path.join(self.root_dir, case_id, "T2")
        self.gt_folder = os.path.join(self.root_dir, case_id)
        if os.path.exists(self.t1_folder):
            self.image_list = sorted([f for f in os.listdir(self.t1_folder) if f.endswith(('.png', '.jpg'))])
            self.slider.setMaximum(len(self.image_list) - 1)
            self.slider.setValue(0); self.slider.setEnabled(True)
            self.update_weights_for_current_case()
            self.calculate_sequence_mean()
            self.run_segmentation(0)
        else:
            self.image_list = []; self.slider.setEnabled(False)

    def on_slider_changed(self, val):
        if not self.image_list: return
        self.lbl_progress.setText(f"{val+1}/{len(self.image_list)}")
        self.run_segmentation(val)

    def calculate_sequence_mean(self):
        if not self.image_list or self.model is None: return
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        scores_mn, scores_ft, scores_ct = [], [], []
        total = len(self.image_list)
        with torch.no_grad():
            for i, fname in enumerate(self.image_list):
                pred = self.predict_mask(fname)
                if pred is None: continue
                gt = self.get_mask_from_gt(fname)
                scores_mn.append(self.calculate_dice(pred, gt, 1))
                scores_ft.append(self.calculate_dice(pred, gt, 2))
                scores_ct.append(self.calculate_dice(pred, gt, 3))
                self.progress_bar.setValue(int((i+1)/total * 100))
        
        avg_mn = sum(scores_mn)/len(scores_mn) if scores_mn else 0
        avg_ft = sum(scores_ft)/len(scores_ft) if scores_ft else 0
        avg_ct = sum(scores_ct)/len(scores_ct) if scores_ct else 0
        
        self.lbl_mean_mn.setText(self.get_bilingual_text(f"• 正中神經: {avg_mn:.2f}", "Median Nerve", en_size=8, color="#555"))
        self.lbl_mean_ft.setText(self.get_bilingual_text(f"• 屈肌腱: {avg_ft:.2f}", "Flexor Tendons", en_size=8, color="#555"))
        self.lbl_mean_ct.setText(self.get_bilingual_text(f"• 腕隧道: {avg_ct:.2f}", "Carpal Tunnel", en_size=8, color="#555"))

    def predict_mask(self, fname):
        p1 = os.path.join(self.t1_folder, fname); p2 = os.path.join(self.t2_folder, fname)
        if not os.path.exists(p1) or not os.path.exists(p2): return None
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE); img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        i1_c = clahe.apply(img1); i2_c = clahe.apply(img2)
        inp = np.stack([i1_c, i2_c], axis=0).astype(np.float32) / 255.0
        inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(DEVICE)
        out = self.model(inp_t)
        out_flip = torch.flip(self.model(torch.flip(inp_t, [3])), [3])
        avg_out = (out + out_flip) / 2.0
        pred_mask = torch.argmax(avg_out, dim=1).cpu().numpy()[0]
        
        mask_mn = self.keep_largest_component((pred_mask == 1).astype(np.uint8))
        mask_ct = self.keep_largest_component((pred_mask == 3).astype(np.uint8))
        final_pred = pred_mask.copy()
        final_pred[pred_mask == 1] = 0; final_pred[pred_mask == 3] = 0
        final_pred[mask_mn == 1] = 1; final_pred[mask_ct == 1] = 3
        return final_pred

    def keep_largest_component(self, mask_class):
        mask_class = mask_class.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_class, connectivity=8)
        if num <= 1: return mask_class 
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_mask = np.zeros_like(mask_class)
        new_mask[labels == largest_idx] = 1
        return new_mask

    def get_mask_from_gt(self, filename):
        h, w = 512, 512; final = np.zeros((h, w), dtype=np.uint8)
        for name, label_id in [('CT', 3), ('FT', 2), ('MN', 1)]:
            p = os.path.join(self.gt_folder, name, filename)
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is not None: final[m > 127] = label_id
        return final

    def draw_nice_overlay(self, img_gray, mask):
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay = np.zeros_like(img_rgb)
        colors = { 1: (255, 255, 0), 2: (0, 100, 255), 3: (255, 0, 0) }
        for cid, col in colors.items(): overlay[mask == cid] = col
        output = img_rgb.copy()
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        for cid, col in colors.items():
            bin_mask = (mask == cid).astype(np.uint8)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, col, 2)
        return output

    def calculate_dice(self, pred, target, class_id):
        p = (pred == class_id).astype(np.float32)
        t = (target == class_id).astype(np.float32)
        inter = (p*t).sum(); union = p.sum() + t.sum()
        if union == 0: return 1.0
        return 2*inter / (union + 1e-5)

    def run_segmentation(self, idx):
        if not self.image_list or self.model is None: return
        fname = self.image_list[idx]
        final_pred = self.predict_mask(fname)
        if final_pred is None: return
        gt_mask = self.get_mask_from_gt(fname)
        d1 = self.calculate_dice(final_pred, gt_mask, 1)
        d2 = self.calculate_dice(final_pred, gt_mask, 2)
        d3 = self.calculate_dice(final_pred, gt_mask, 3)
        
        self.lbl_curr_mn.setText(self.get_bilingual_text(f"正中神經: {d1:.2f}", "Median Nerve", color="#D4AF37"))
        self.lbl_curr_ft.setText(self.get_bilingual_text(f"屈肌腱: {d2:.2f}", "Flexor Tendons", color="#007ACC"))
        self.lbl_curr_ct.setText(self.get_bilingual_text(f"腕隧道: {d3:.2f}", "Carpal Tunnel", color="#D93025"))
        
        p1 = os.path.join(self.t1_folder, fname)
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        vis_input = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        vis_gt = self.draw_nice_overlay(img1, gt_mask)
        vis_pred = self.draw_nice_overlay(img1, final_pred)
        self.show_on_lbl(vis_input, self.view_input)
        self.show_on_lbl(vis_gt, self.view_gt)
        self.show_on_lbl(vis_pred, self.view_pred)

    def show_on_lbl(self, img, lbl):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c*w, QImage.Format.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())