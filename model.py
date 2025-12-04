import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Attention Gate (新增模組) ---
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (粗糙特徵), x: skip connection (細節特徵)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --- ResNet Basic Block (維持不變) ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# --- ResNet Encoder (維持不變) ---
class ResNetEncoder(nn.Module):
    def __init__(self, block, layers):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)     
        x1 = self.maxpool(x0)  
        x2 = self.layer1(x1)   
        x3 = self.layer2(x2)   
        x4 = self.layer3(x3)   
        x5 = self.layer4(x4)   
        return x0, x1, x2, x3, x4, x5

# --- Attention ResUNet (升級版) ---
class DLP_ResNet_Segmentation(nn.Module):
    def __init__(self, num_classes=4):
        super(DLP_ResNet_Segmentation, self).__init__()
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3])
        
        # Decoder + Attention
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=256, F_l=256, F_int=128) # 新增 Attention
        self.conv1 = self._conv_block(512, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64) # 新增 Attention
        self.conv2 = self._conv_block(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32) # 新增 Attention
        self.conv3 = self._conv_block(128, 64)  
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32) # 新增 Attention
        self.conv4 = self._conv_block(128, 64) 

        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder + Attention
        d5 = self.up1(x5)       
        x4 = self.att1(g=d5, x=x4) # 應用 Attention
        d5 = torch.cat([d5, x4], dim=1)
        d5 = self.conv1(d5)
        
        d4 = self.up2(d5)       
        x3 = self.att2(g=d4, x=x3) # 應用 Attention
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.conv2(d4)
        
        d3 = self.up3(d4)       
        x2 = self.att3(g=d3, x=x2) # 應用 Attention
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv3(d3)
        
        d2 = self.up4(d3)       
        x0 = self.att4(g=d2, x=x0) # 應用 Attention (Skip x0)
        d2 = torch.cat([d2, x0], dim=1) 
        d2 = self.conv4(d2)
        
        d1 = self.up5(d2)       
        out = self.final_conv(d1)
        return out