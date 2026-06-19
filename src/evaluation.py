from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def evaluate_model(name, y_true, y_pred, y_prob=None, train_time=None, infer_time=None):
    print(f"\n========================================")
    print(f"{name} Evaluation Metrics")
    print(f"========================================")
    if train_time: print(f"Training Time  : {train_time:.4f} seconds")
    if infer_time: print(f"Inference Time : {infer_time:.4f} seconds")
    print(f"Accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score       : {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"ROC-AUC        : {roc_auc_score(y_true, y_prob):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("========================================\n")