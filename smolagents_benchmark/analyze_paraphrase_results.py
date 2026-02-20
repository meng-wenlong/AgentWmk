import json
from sklearn.metrics import roc_auc_score
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_metrics(pos_file, neg_file, name):
    """Compute mean value and AUC for a dataset"""
    pos_data = load_json(pos_file)
    neg_data = load_json(neg_file)

    # Extract values
    pos_values = [item['value'] for item in pos_data.values()]
    neg_values = [item['value'] for item in neg_data.values()]

    # Compute mean
    pos_mean = np.mean(pos_values)
    neg_mean = np.mean(neg_values)

    # Prepare for AUC computation
    # Positive samples get label 1, negative samples get label 0
    y_true = [1] * len(pos_values) + [0] * len(neg_values)
    y_scores = pos_values + neg_values

    # Compute AUC
    auc = roc_auc_score(y_true, y_scores)

    print(f"\n{name}:")
    print(f"  Positive mean: {pos_mean:.4f}")
    print(f"  Negative mean: {neg_mean:.4f}")
    print(f"  AUC: {auc:.4f}")

    return {
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'auc': auc
    }

if __name__ == "__main__":
    base_path = "/zju_wck/mwl/CAW/smolagents_benchmark/res"

    datasets = [
        ("network-pos-paraphrase.json", "network-neg.json", "Network (Paraphrase)"),
        ("visit-pos-paraphrase.json", "visit-neg.json", "Visit (Paraphrase)"),
        ("version-pos-paraphrase.json", "version-neg.json", "Version (Paraphrase)"),
        ("task-pos-paraphrase.json", "task-neg.json", "Task (Paraphrase)")
    ]

    print("="*60)
    print("Paraphrase Results - Metrics Summary")
    print("="*60)

    results = {}
    for pos_file, neg_file, name in datasets:
        pos_path = f"{base_path}/{pos_file}"
        neg_path = f"{base_path}/{neg_file}"
        results[name] = compute_metrics(pos_path, neg_path, name)

    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    print(f"{'Dataset':<25} {'Pos Mean':>12} {'Neg Mean':>12} {'AUC':>12}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['pos_mean']:>12.4f} {metrics['neg_mean']:>12.4f} {metrics['auc']:>12.4f}")
    print("="*60)
