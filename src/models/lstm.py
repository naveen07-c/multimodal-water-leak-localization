from src.models.cnn import CNNFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalLSTM(nn.Module):
    """
    LSTM sequence aggregator over consecutive window features.
    Input: (B, seq_len, feat_dim)
    Output: (B, hidden_dim) temporal representation.
    """
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, bidirectional=True, dropout=0.2):
        super(TemporalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        out_dim = hidden_dim * self.num_directions
        self.attn_fc = nn.Linear(out_dim, 1)
        self.proj = nn.Linear(out_dim, hidden_dim)
        
    def forward(self, x_seq):
        # x_seq: (B, T, D)
        lstm_out, _ = self.lstm(x_seq) # (B, T, out_dim)
        
        # Temporal Attention Pooling
        attn_weights = F.softmax(self.attn_fc(lstm_out), dim=1) # (B, T, 1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1) # (B, out_dim)
        
        out = F.relu(self.proj(pooled)) # (B, hidden_dim)
        return out, attn_weights

class CNNLSTMClassifier(nn.Module):
    """
    Baseline 3: CNN-LSTM Temporal Model.
    """
    def __init__(self, in_channels=6, num_classes=5, cnn_dim=64, lstm_dim=64):
        super(CNNLSTMClassifier, self).__init__()
        self.cnn = CNNFeatureExtractor(in_channels=in_channels, feat_dim=cnn_dim)
        self.lstm = TemporalLSTM(input_dim=cnn_dim, hidden_dim=lstm_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.det_head = nn.Linear(lstm_dim, 1)
        
    def forward(self, x_seq):
        # x_seq: (B, T, C, L)
        B, T, C, L = x_seq.shape
        x_flat = x_seq.view(B * T, C, L)
        cnn_feats = self.cnn(x_flat) # (B*T, cnn_dim)
        cnn_seq = cnn_feats.view(B, T, -1) # (B, T, cnn_dim)
        
        temporal_feat, attn_w = self.lstm(cnn_seq)
        cls_logits = self.classifier(temporal_feat)
        det_logit = self.det_head(temporal_feat).squeeze(-1)
        
        return cls_logits, det_logit, temporal_feat, attn_w
