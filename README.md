
# 專案名稱：基於深度學習的腕隧道症候群 MRI 影像分割

  # (Carpal Tunnel Syndrome Segmentation)

## 📌 專案簡介 (Project Overview)

本專案旨在利用深度學習技術，自動從手腕 MRI 影像（T1-weighted 與 T2-weighted）中分割出關鍵組織，以輔助腕隧道症候群 (CTS) 的診斷 。

目標是精確分割以下三個區域，並達到指定的 Dice Coefficient (DC) 標準 ：

  正中神經 (Median Nerve, MN)：Target DC \> 0.81
  
  屈肌腱 (Flexor Tendons, FT)：Target DC \> 0.83
  
  腕隧道 (Carpal Tunnel, CT)：Target DC \> 0.83
  

本系統包含完整的訓練流程、特徵工程、兩階段優化策略，以及一個基於 PyQt6 開發的圖形化介面 (GUI)，可即時展示分割結果與評估數據。

----------------------------------------------------------------------

## 🌟 核心技術與亮點 (Key Features)

  * **模型架構 (Model Architecture)**：採用 **ResNet-Based Encoder** 結合 **Attention Gate Decoder** 的 U-Net 架構。利用 Attention 機制讓模型聚焦於細微的組織邊緣，並透過 ResNet 提取深層特徵。
  * **兩階段訓練策略 (Two-Stage Training Strategy)**：
    1.  **Stage 1 (Warm-up)**：使用 AdamW 優化器搭配 Cross Entropy + Dice Loss，快速收斂並學習基礎特徵。
    2.  **Stage 2 (Fine-tuning)**：切換至 SGD 優化器搭配 Scheduler，並引入 **Focal Loss** 進行 Hard Mining，專注於解決難以分割的邊緣細節。
  * **超級模型整合 (Ensemble & Selection)**：`create_demo_model.py` 腳本會自動掃描 5-Fold 交叉驗證的結果，針對每一個病例 (Case) 自動挑選表現最好的模型權重，組合成一個「超級模型 (Super Model)」。
  * **強大的前處理與後處理**：
      * **Pre-processing**：應用 CLAHE (對比度受限自適應直方圖均衡化) 增強 MRI 影像對比。
      * **Post-processing**：包含測試時增強 (TTA, Test Time Augmentation) 與連通區域過濾 (Connected Components) 去除雜訊。

-----

## 📂 檔案結構說明 (File Structure)

| 檔案名稱 | 類型 | 功能說明 |
| :--- | :--- | :--- |
| **`model.py`** | 核心架構 | 定義 `DLP_ResNet_Segmentation` 模型，包含 Attention Block 與 ResNet Encoder。 |
| **`dataset.py`** | 資料處理 | 負責讀取 T1/T2 雙模態影像，執行 CLAHE 增強與 Data Augmentation (翻轉、旋轉)。 |
| **`loss-1.py`** | 損失函數 | **(Stage 1)** 定義 `CrossEntropy + Dice`，用於初期穩定訓練。 |
| **`loss-2.py`** | 損失函數 | **(Stage 2)** 定義 `Focal Loss + Dice`，用於後期微調，加強難分類樣本權重。 |
| **`rrr-1.py`** | 訓練腳本 | **(Stage 1)** 使用 **AdamW** 優化器，適合從零開始訓練 (From Scratch)。 |
| **`rrr-2.py`** | 訓練腳本 | **(Stage 2)** 使用 **SGD + CosineScheduler**，適合接續訓練 (Resume) 突破瓶頸。 |
| **`create_demo_model.py`** | 工具腳本 | 自動評估所有 Fold 的模型，為每個 Case 挑選最佳權重並打包成 `final_demo_model.pth`。 |
| **`lll.py`** | 評估腳本 | 計算模型在各器官的 Dice Score，並驗證是否達到 PPT 要求標準。 |
| **`gui.py`** | 使用者介面 | 基於 PyQt6 的視覺化展示程式，可載入超級模型並顯示分割結果與覆蓋圖 (Overlay)。 |

-----

## 🚀 訓練策略指南 (Training Strategy)

本專案建議採用以下流程以達到最佳效果：

### 第一階段：基礎訓練 (Epoch 0 - 300)

  * **目的**：讓模型快速學習什麼是背景、什麼是器官。
  * **配置**：將 `rrr-1.py` 設為訓練腳本，並引用 `loss-1.py`。
  * **優化器**：AdamW (收斂速度快)。
  * **Loss**：Cross Entropy (0.2) + Dice (0.8)。

### 第二階段：進階微調 (Epoch 300+)

  * **時機**：當 Loss 卡住降不下去，或特定類別 (如 Carpal Tunnel) 分數無法達標時。
  * **配置**：切換至 `rrr-2.py`，引用 `loss-2.py`，並設定 `START_FROM_EPOCH` 接續上一階段的權重。
  * **優化器**：SGD + Momentum + Scheduler (穩定性高，適合擠出最後的準確度)。
  * **Loss**：Focal Loss (0.3) + Weighted Dice (0.7)。Focal Loss 能有效挖掘難題 (Hard Mining)。

-----

## 💻 安裝與執行 (Installation & Usage)

### 1\. 環境需求

請確保已安裝 Python 3.8 或以上版本。

建議先建立虛擬環境，接著使用以下指令一次安裝所有套件：

```bash
pip install -r requirements.txt
```

⚠️ 注意 (Note regarding GPU)： 如果您需要使用 NVIDIA 顯卡進行加速 (CUDA)，建議先至 PyTorch 官網 查詢適合您顯卡版本的指令

例如：

```bash
# 例如：安裝支援 CUDA 11.8 的 PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 接著再安裝其餘套件
pip install -r requirements.txt
```
### 2\. 準備資料

資料下載：https://drive.google.com/drive/folders/1Qp2Mhn3A8tZQ2_6Y1_EvPuPohHYiq0oD?usp=sharing

請將資料集放置於 `carpalTunnel/` 資料夾中，結構如下：

```text
carpalTunnel/
├── 0/
│   ├── T1/
│   ├── T2/
│   ├── CT/ (Ground Truth)
│   ├── FT/ (Ground Truth)
│   └── MN/ (Ground Truth)
├── 1/
...
```

### 3\. 執行訓練
執行前建議將檔名改為rrr.py(rrr 與 loss 同步更改)

```bash
# 第一階段
python rrr-1.py

# 第二階段 (需修改程式碼中的 import loss 與 START_FROM_EPOCH)
python rrr-2.py
```

### 4\. 建立 Demo 模型

訓練完成後，執行此腳本來整合最佳權重：

```bash
python create_demo_model.py
```

這將產生 `checkpoints/final_demo_model.pth`。

### 5\. 啟動 GUI 展示

```bash
python gui.py
```

在介面中選擇 `final_demo_model.pth` 以及資料集路徑，即可開始瀏覽分割結果。

-----

## 📊 評估指標 (Evaluation)

評估標準採用 **Dice Coefficient (DC)**：
$$DC = \frac{2 |A \cap B|}{|A| + |B|}$$
[cite_start]其中 A 為 Ground Truth，B 為預測結果 [cite: 56]。

GUI 介面將即時顯示：

1.  **Sequence DC (Mean)**：該病例所有切片的平均分數。
2.  **Current Slice DC**：當前檢視切片的分數。

-----
## 📥 模型權重下載 (Download Pre-trained Models)

如果您不想從頭訓練，可以直接下載我們訓練好的模型權重，即可直接運行 GUI 進行展示。

**安裝步驟：**
1. 下載下方列表中的 `.pth` 檔案。
2. 在專案根目錄下建立一個名為 `checkpoints` 的資料夾。
3. 將所有 `.pth` 檔案放入 `checkpoints/` 資料夾中。

| 檔案名稱 (Filename) | 類型 | 描述 (Description) | 下載連結 (Download) |
| :--- | :--- | :--- | :--- |
| **`final_demo_model.pth`** | 🏆 **推薦** | **超級模型 (Super Model)**<br>整合了針對每個 Case 的最佳權重，GUI 展示專用。 | https://drive.google.com/file/d/1JirbLCHzAl_S0pwQYMcC60pxyODtxKQt/view?usp=sharing |
| `best_model_fold_1.pth` | Best Model | Fold 1 驗證分數最高的模型 | https://drive.google.com/file/d/1BpmiYH6Xqdljtd7fntj2r3PZ1rbO_yHb/view?usp=sharing |
| `best_model_fold_2.pth` | Best Model | Fold 2 驗證分數最高的模型 | https://drive.google.com/file/d/1ARc2VtKVHhDreZFbEgArdCzIV0oHE9CR/view?usp=sharing |
| `best_model_fold_3.pth` | Best Model | Fold 3 驗證分數最高的模型 | https://drive.google.com/file/d/1DqHDMl7j2NXPqWONW7lYdMyIwRC8PoZP/view?usp=sharing |
| `best_model_fold_4.pth` | Best Model | Fold 4 驗證分數最高的模型 | https://drive.google.com/file/d/1-YIRCAeztMtnsiktzQDID-tD-SyFWbOl/view?usp=sharing |
| `best_model_fold_5.pth` | Best Model | Fold 5 驗證分數最高的模型 | https://drive.google.com/file/d/1FSZnBGbMxzod1BHdQyqY4zLlb_m0w0h7/view?usp=sharing |
| `checkpoint_fold_1.pth `| Checkpoint | (開發者用) 訓練中斷點備份，包含優化器狀態。 | https://drive.google.com/file/d/1jzjbARWhlo2rMCAxI7_2B1RjJUESvial/view?usp=sharing |
| `checkpoint_fold_2.pth` | Checkpoint | (開發者用) 訓練中斷點備份，包含優化器狀態。 | https://drive.google.com/file/d/1bUL3-983nz-mmhGi6EbY0hICvOOsG-cD/view?usp=sharing |
| `checkpoint_fold_3.pth` | Checkpoint | (開發者用) 訓練中斷點備份，包含優化器狀態。 | https://drive.google.com/file/d/1a1PtVbOhRPFLzbAfznKg8JgaaAfjzkgz/view?usp=sharing |
| `checkpoint_fold_4.pth` | Checkpoint | (開發者用) 訓練中斷點備份，包含優化器狀態。 | https://drive.google.com/file/d/1GIj1XGG1xdp5MC1cHmvFl6umimtFHpcY/view?usp=sharing |
| `checkpoint_fold_5.pth` | Checkpoint | (開發者用) 訓練中斷點備份，包含優化器狀態。 | https://drive.google.com/file/d/1vmBTPKi59lfrpMxu7IP7I1XY5nRgQbim/view?usp=sharing |

> **⚠️ 注意**：若只要執行 GUI 展示，僅下載 `final_demo_model.pth` 即可；若要重現 `create_demo_model.py` 的整合過程，則需要下載所有的 `best_model_fold_*.pth`。


-----

### 📝 作者

[95326/戴煒駿]
DLP 2025 Final Project

-----

