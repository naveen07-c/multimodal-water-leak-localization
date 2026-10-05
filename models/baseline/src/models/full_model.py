import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dae import ConvDAE1D
from src.models.cnn import CNNFeatureExtractor
from src.models.lstm import TemporalLSTM
from src.models.gnn import PipelineGNN
from src.models.fusion import MultimodalAttentionFusion

class RobustWaterLeakSystem(nn.Module):
    """
    Proposed End-to-End Architecture:
    Denoising Autoencoder -> CNN -> LSTM -> GNN -> Attention-Based Multimodal Fusion -> Prediction Heads
    
    Inputs:
      x_seq: (B, T, 6, L) Sequence of T multimodal window slices, each with 6 sensor channels and length L.
      adj_norm: (6, 6) Symmetrically normalized pipe network graph adjacency.
      use_dae: Whether to pass inputs through DAE before CNN.
      sensor_mask: Optional boolean/float mask of shape (B, 6) to simulate sparse sensor availability.
    """
    def __init__(
        self,
        num_sensors=6,
        window_len=8000,
        feat_dim=64,
        num_classes=5,
        use_dae=True,
        dae_pretrained_path=None
    ):
        super(RobustWaterLeakSystem, self).__init__()
        self.num_sensors = num_sensors
        self.feat_dim = feat_dim
        self.use_dae = use_dae
        
        # 1. Denoising Autoencoders (Separate for Acoustic, Vib, Press or Shared 1D Conv)
        self.dae = ConvDAE1D(in_channels=1, hidden_channels=[16, 32, 64])
        
        # 2. Per-sensor CNN feature extractors
        # Modality grouping: 0:P1, 1:A1, 2:H1, 3:H2, 4:A2, 5:P2
        self.cnn_nodes = nn.ModuleList([
            CNNFeatureExtractor(in_channels=1, feat_dim=feat_dim)
            for _ in range(num_sensors)
        ])
        
        # 3. Temporal LSTM for sequence modeling
        self.lstm = TemporalLSTM(input_dim=feat_dim * num_sensors, hidden_dim=feat_dim)
        
        # 4. Spatial GNN for topological network reasoning
        self.gnn = PipelineGNN(in_features=feat_dim, hidden_dim=feat_dim, gnn_dim=feat_dim)
        
        # 5. Modality Stream Projectors
        self.proj_acoustic = nn.Linear(feat_dim * 2, feat_dim) # H1, H2
        self.proj_vib = nn.Linear(feat_dim * 2, feat_dim)      # A1, A2
        self.proj_press = nn.Linear(feat_dim * 2, feat_dim)    # P1, P2
        
        # 6. Attention Multimodal Fusion
        self.fusion = MultimodalAttentionFusion(embed_dim=feat_dim, num_modalities=4)
        
        # 7. Final Prediction Heads
        # Head A: Leak Classification (5-class: NL, CC, GL, LC, OL)
        self.head_cls = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Head B: Binary Leak Detection (0: No-leak, 1: Leak)
        self.head_det = nn.Linear(feat_dim, 1)
        
        # Head C: Spatial Leak Localization (Middle Pipe / Sensor Node Scores)
        self.head_loc = nn.Linear(feat_dim, num_sensors)
        
        # Head D: Leak Flow Severity
        self.head_sev = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_seq, adj_norm, sensor_mask=None, apply_dae=True):
        """
        x_seq: (B, T, 6, L) where T is temporal sequence length (e.g. 5 or 10).
        sensor_mask: (B, 6) float tensor where 0.0 means missing/masked sensor.
        """
        B, T, num_s, L = x_seq.shape
        
        # Step 1: Optional DAE Denoising
        if self.use_dae and apply_dae:
            with torch.no_grad():
                x_flat = x_seq.view(B * T * num_s, 1, L)
                # Mini-chunk DAE to save peak VRAM
                x_clean_chunks = []
                for chunk_idx in range(0, x_flat.shape[0], 32):
                    chunk = x_flat[chunk_idx:chunk_idx+32]
                    x_clean_chunks.append(self.dae(chunk))
                x_clean = torch.cat(x_clean_chunks, dim=0)
                x_in = x_clean.view(B, T, num_s, L)
        else:
            x_in = x_seq
            
        # Step 2: Extract CNN features per sensor node
        # For each time step t in sequence
        node_feats_list = [] # Will store (B, T, 6, feat_dim)
        for s_idx in range(self.num_sensors):
            # Extract signal for sensor s across batch & time: (B*T, 1, L)
            s_sig = x_in[:, :, s_idx, :].contiguous().view(B * T, 1, L)
            s_feat = self.cnn_nodes[s_idx](s_sig) # (B*T, feat_dim)
            s_feat = s_feat.view(B, T, self.feat_dim)
            
            # Apply sensor mask if sparse deployment simulation
            if sensor_mask is not None:
                mask_s = sensor_mask[:, s_idx].view(B, 1, 1) # (B, 1, 1)
                s_feat = s_feat * mask_s
                
            node_feats_list.append(s_feat)
            
        # Stack nodes: (B, T, 6, feat_dim)
        node_feats = torch.stack(node_feats_list, dim=2)
        
        # Step 3: Spatial GNN on current/latest time step (or average across sequence)
        # Latest time step: (B, 6, feat_dim)
        curr_node_feats = node_feats[:, -1, :, :] # (B, 6, feat_dim)
        f_graph, node_scores, node_attn = self.gnn(curr_node_feats, adj_norm) # f_graph: (B, feat_dim)
        
        # Step 4: Modality feature projections (using latest time step)
        # Node indices: 0:P1, 1:A1, 2:H1, 3:H2, 4:A2, 5:P2
        f_press = F.relu(self.proj_press(torch.cat([curr_node_feats[:, 0], curr_node_feats[:, 5]], dim=-1)))
        f_vib = F.relu(self.proj_vib(torch.cat([curr_node_feats[:, 1], curr_node_feats[:, 4]], dim=-1)))
        f_acoustic = F.relu(self.proj_acoustic(torch.cat([curr_node_feats[:, 2], curr_node_feats[:, 3]], dim=-1)))
        
        # Step 5: Temporal LSTM modeling
        # Flatten all sensors per time step: (B, T, 6*feat_dim)
        seq_flat = node_feats.view(B, T, self.num_sensors * self.feat_dim)
        f_temporal, temporal_attn = self.lstm(seq_flat) # (B, feat_dim)
        
        # Step 6: Attention Multimodal Fusion (combines Acoustic, Vib, Press, Graph)
        f_fused, modality_weights = self.fusion(f_acoustic, f_vib, f_press, f_graph)
        
        # Combine fused spatial-modal representation with temporal LSTM representation
        f_final = F.relu(f_fused + f_temporal)
        
        # Step 7: Prediction Heads
        cls_logits = self.head_cls(f_final)
        det_logit = self.head_det(f_final).squeeze(-1)
        loc_scores = self.head_loc(f_final) # (B, 6)
        sev_pred = self.head_sev(f_final).squeeze(-1)
        
        return {
            'cls_logits': cls_logits,
            'det_logit': det_logit,
            'loc_scores': loc_scores,
            'sev_pred': sev_pred,
            'f_final': f_final,
            'f_temporal': f_temporal,
            'f_graph': f_graph,
            'f_fused': f_fused,
            'modality_weights': modality_weights,
            'node_scores': node_scores,
            'temporal_attn': temporal_attn
        }
