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

class FocalLoss(nn.Module):
    """
    Focal Loss: 專注於難分類的樣本 (Hard Mining)
    """
    def __init__(self, gamma=2.0, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, input, target):
        # Cross Entropy
        logpt = F.cross_entropy(input, target, weight=self.weights, reduction='none')
        pt = torch.exp(-logpt)
        # Focal Term: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * logpt
        return focal_loss.mean()

class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super(ComboLoss, self).__init__()
        w_tensor = torch.tensor(weights).float().cuda() if weights else None
        
        # [新版] 使用 Focal Loss
        self.focal_loss = FocalLoss(gamma=2.0, weights=w_tensor)
        self.dice_loss = DiceLoss()
        self.weights = weights

    def forward(self, input, target):
        # 1. Focal Loss
        focal = self.focal_loss(input, target)
        
        # 2. Weighted Dice Loss
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
        
        # 3. 混合: 0.3 Focal + 0.7 Dice
        return 0.3 * focal + 0.7 * dice_avg