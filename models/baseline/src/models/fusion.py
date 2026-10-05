import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalAttentionFusion(nn.Module):
    """
    Attention-Based Fusion Layer across Acoustic, Vibration, Pressure, and Graph modalities.
    Input:
      f_acoustic: (B, D) from Hydrophones (H1, H2)
      f_vib:      (B, D) from Accelerometers (A1, A2)
      f_press:    (B, D) from Pressure Sensors (P1, P2)
      f_graph:    (B, D) from GNN spatial features
    Output:
      fused: (B, D)
      modality_weights: (B, 4) dynamic attention distribution
    """
    def __init__(self, embed_dim=64, num_modalities=4):
        super(MultimodalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Attention scoring network
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, f_acoustic, f_vib, f_press, f_graph):
        # Stack modalities: (B, 4, D)
        modalities = torch.stack([f_acoustic, f_vib, f_press, f_graph], dim=1)
        
        # Compute dynamic importance weights
        scores = self.attn_net(modalities).squeeze(-1) # (B, 4)
        weights = F.softmax(scores, dim=1)            # (B, 4)
        
        # Weighted combination: (B, D)
        fused = torch.sum(modalities * weights.unsqueeze(-1), dim=1)
        fused = F.relu(self.proj_out(fused))
        
        return fused, weights
