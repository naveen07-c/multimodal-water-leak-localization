from src.models.cnn import CNNFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Spatially normalized Graph Convolution layer: H' = sigma(A_norm * H * W)
    Supports batched node feature tensors (B, num_nodes, in_features).
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, adj_norm):
        # x: (B, N, in_features)
        # adj_norm: (N, N) or (B, N, N)
        h = self.linear(x) # (B, N, out_features)
        if adj_norm.dim() == 2:
            out = torch.einsum('ij,bjk->bik', adj_norm, h)
        else:
            out = torch.bmm(adj_norm, h)
        out = out + self.bias
        return out

class PipelineGNN(nn.Module):
    """
    Graph Neural Network operating on the 6-node pipe network topology.
    Input: Node features (B, 6, node_dim)
    Output:
      spatial_feat: (B, gnn_dim) Graph-level representation
      node_logits: (B, 6) Node-level localization/fault scores
    """
    def __init__(self, in_features=64, hidden_dim=64, gnn_dim=64, num_layers=2, dropout=0.2):
        super(PipelineGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.layers.append(GraphConvLayer(in_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(6))
        
        for _ in range(num_layers - 1):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(6))
            
        self.proj = nn.Linear(hidden_dim, gnn_dim)
        self.drop = nn.Dropout(dropout)
        
        # Node attention for graph readout
        self.node_attn = nn.Linear(hidden_dim, 1)
        
        # Localization head (node-level score)
        self.loc_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_feats, adj_norm):
        # node_feats: (B, 6, in_features)
        # adj_norm: (6, 6)
        h = node_feats
        for conv, bn in zip(self.layers, self.bns):
            h = F.relu(bn(conv(h, adj_norm)))
            h = self.drop(h)
            
        # Node-level localization score
        node_scores = self.loc_head(h).squeeze(-1) # (B, 6)
        
        # Graph Readout (Attention-weighted sum over 6 nodes)
        attn_weights = F.softmax(self.node_attn(h), dim=1) # (B, 6, 1)
        graph_pooled = torch.sum(h * attn_weights, dim=1)  # (B, hidden_dim)
        spatial_feat = F.relu(self.proj(graph_pooled))     # (B, gnn_dim)
        
        return spatial_feat, node_scores, attn_weights

class GNNClassifier(nn.Module):
    """
    Baseline 4: Graph-only classifier.
    """
    def __init__(self, num_nodes=6, in_feat_dim=64, num_classes=5, gnn_dim=64):
        super(GNNClassifier, self).__init__()
        self.gnn = PipelineGNN(in_features=in_feat_dim, hidden_dim=gnn_dim, gnn_dim=gnn_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(gnn_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.det_head = nn.Linear(gnn_dim, 1)
        
    def forward(self, node_feats, adj_norm):
        spatial_feat, node_scores, node_attn = self.gnn(node_feats, adj_norm)
        cls_logits = self.classifier(spatial_feat)
        det_logit = self.det_head(spatial_feat).squeeze(-1)
        return cls_logits, det_logit, node_scores, spatial_feat
