from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics
