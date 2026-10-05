import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, pool=2, dropout=0.2):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        out = self.pool(out)
        out = self.drop(out)
        return out

class CNNFeatureExtractor(nn.Module):
    """
    1D-CNN that processes a single or multi-channel window (e.g. 1 x 8000 or 6 x 8000)
    and extracts a compact feature representation (e.g. 64-dim).
    """
    def __init__(self, in_channels=1, feat_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        self.layer1 = ConvBlock1D(in_channels, 16, kernel_size=15, pool=4)  # 8000 -> 2000
        self.layer2 = ConvBlock1D(16, 32, kernel_size=11, pool=4)          # 2000 -> 500
        self.layer3 = ConvBlock1D(32, 64, kernel_size=7, pool=4)           # 500 -> 125
        self.layer4 = ConvBlock1D(64, feat_dim, kernel_size=5, pool=5)     # 125 -> 25
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, x):
        # x: (B, C, L)
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        pooled = self.global_pool(h).squeeze(-1) # (B, feat_dim)
        feat = F.relu(self.fc(pooled))
        return feat

class CNNClassifier(nn.Module):
    """
    Baseline 2: CNN-only end-to-end classifier.
    """
    def __init__(self, in_channels=6, num_classes=5, feat_dim=64):
        super(CNNClassifier, self).__init__()
        self.backbone = CNNFeatureExtractor(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.det_head = nn.Linear(feat_dim, 1) # Binary detection
        
    def forward(self, x):
        feat = self.backbone(x)
        cls_logits = self.classifier(feat)
        det_logit = self.det_head(feat).squeeze(-1)
        return cls_logits, det_logit, feat
