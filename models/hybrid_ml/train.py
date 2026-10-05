import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import shap
import json
import joblib

root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'

# Helper to load data
def load_data(dataset_type='mendeley'):
    if dataset_type == 'mendeley':
        train = np.load(proc_dir / 'train_data.npz')
        test = np.load(proc_dir / 'test_data.npz')
        # Flatten windows (N, 6, 8000) -> (N, 48000)
        X_train = train['windows'].reshape(train['windows'].shape[0], -1)
        X_test = test['windows'].reshape(test['windows'].shape[0], -1)
        y_train = train['y_det']
        y_test = test['y_det']
    else:
        train = np.load(proc_dir / 'train_data_hk.npz')
        test = np.load(proc_dir / 'test_data_hk.npz')
        # (N, 1, 8000) -> (N, 8000)
        X_train = train['windows'].reshape(train['windows'].shape[0], -1)
        X_test = test['windows'].reshape(test['windows'].shape[0], -1)
        y_train = train['y_det']
        y_test = test['y_det']
        
    return X_train, y_train, X_test, y_test

def run_model1(dataset_type):
    print(f"\n--- Running Model 1 (Transparent Hybrid ML) on {dataset_type} dataset ---")
    X_train, y_train, X_test, y_test = load_data(dataset_type)
    
    # Feature engineering: instead of using raw 48000 or 8000 features, extract statistical features to make Random Forest feasible
    def extract_stats(X):
        # X is (N, length)
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        max_v = np.max(X, axis=1, keepdims=True)
        min_v = np.min(X, axis=1, keepdims=True)
        energy = np.sum(X**2, axis=1, keepdims=True)
        return np.concatenate([mean, std, max_v, min_v, energy], axis=1)

    X_train_feat = extract_stats(X_train)
    X_test_feat = extract_stats(X_test)
    
    print(f"Features extracted: shape {X_train_feat.shape}")
    
    # Train
    # Finetune (increase n_estimators for better efficiency/accuracy)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train_feat, y_train)
    
    # Evaluate
    preds = clf.predict(X_test_feat)
    preds_proba = clf.predict_proba(X_test_feat)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    
    auc = roc_auc_score(y_test, preds_proba)
    cm = confusion_matrix(y_test, preds).tolist()
    
    print(f"[{dataset_type}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # SHAP Explainability
    # For a Random Forest, we can use TreeExplainer
    explainer = shap.TreeExplainer(clf)
    # Explain on a subset to save time
    shap_values = explainer.shap_values(X_test_feat[:100])
    
    results = {
        'model': 'models/hybrid_ml',
        'dataset': dataset_type,
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
        'confusion_matrix': cm
    }
    
    out_dir = root_dir / 'models/hybrid_ml'
    joblib.dump(clf, out_dir / f'model_{dataset_type}.pkl')
    with open(out_dir / f'results_{dataset_type}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    run_model1('mendeley')
    run_model1('hk')
