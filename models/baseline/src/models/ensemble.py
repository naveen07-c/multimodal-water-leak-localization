import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

class EnsembleDecisionLayer:
    """
    Weighted Ensemble Decision Layer calibrated on validation probabilities.
    Combines:
      P1: CNN-LSTM predictions
      P2: GNN predictions
      P3: Multimodal Attention Fusion predictions
    """
    def __init__(self):
        self.weights = np.array([1.0/3, 1.0/3, 1.0/3], dtype=np.float32)
        
    def fit_weights(self, p_lstm, p_gnn, p_fusion, y_true):
        """
        Finds optimal simplex weights [w1, w2, w3] minimizing multi-class cross-entropy on validation data.
        """
        probs_list = [p_lstm, p_gnn, p_fusion]
        
        def loss_fn(w):
            w = np.array(w)
            w = w / np.sum(w)
            p_blend = sum(w[i] * probs_list[i] for i in range(3))
            p_blend = np.clip(p_blend, 1e-7, 1.0 - 1e-7)
            # Categorical cross entropy
            n_samples = len(y_true)
            loss = -np.sum(np.log(p_blend[np.arange(n_samples), y_true])) / n_samples
            return loss
            
        init_w = [0.33, 0.33, 0.34]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
        
        res = minimize(loss_fn, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            w = np.array(res.x)
            self.weights = w / np.sum(w)
        print(f"[Ensemble] Optimized weights: CNN-LSTM={self.weights[0]:.3f}, GNN={self.weights[1]:.3f}, Fusion={self.weights[2]:.3f}")
        return self.weights
        
    def predict_proba(self, p_lstm, p_gnn, p_fusion):
        w = self.weights
        return w[0]*p_lstm + w[1]*p_gnn + w[2]*p_fusion
        
    def predict(self, p_lstm, p_gnn, p_fusion):
        p_final = self.predict_proba(p_lstm, p_gnn, p_fusion)
        return np.argmax(p_final, axis=1)
