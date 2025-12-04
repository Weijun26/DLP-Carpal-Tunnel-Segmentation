#舊版：CrossEntropy + Dice
#用途：標準訓練用，適合從頭開始打基礎。

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super(ComboLoss, self).__init__()
        self.weights = weights
        # [舊版] 使用標準 Cross Entropy
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().cuda() if weights else None)
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        # 1. Cross Entropy Loss
        ce = self.ce_loss(input, target)
        
        # 2. Dice Loss
        input_soft = F.softmax(input, dim=1)
        C = input_soft.shape[1]
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        dice_total = 0
        count = 0
        for i in range(C):
            dice_val = self.dice_loss(input_soft[:, i], target_one_hot[:, i])
            w = self.weights[i] if self.weights else 1.0
            dice_total += dice_val * w
            count += 1
            
        dice_avg = dice_total / count
        
        # 3. 混合: 0.2 CE + 0.8 Dice (Dice 權重較高)
        return 0.2 * ce + 0.8 * dice_avg