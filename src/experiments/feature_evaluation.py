import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model

def main():
    # 1. Load and Preprocess Data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading and preprocessing data from {data_path}...\n")
    df = preprocessing.preprocess(data_path)
    y = df['label'].values

    feature_modes = ["handcrafted", "tfidf", "hybrid"]
    results = []

    # 2. Iterate through each feature extraction mode
    for mode in feature_modes:
        print(f"\n--- Extracting and Training on '{mode.upper()}' features ---")
        
        # Build features using the specific mode
        X_mode, feature_names_mode, _ = features.build_features(df, mode=mode, fit_tfidf=True)
        print(f"Number of features: {X_mode.shape[1]}")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mode, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Model (Using Random Forest as the baseline for this experiment)
        rf_model = get_random_forest()

        start_train = time.time()
        rf_model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Predict & Evaluate
        start_infer = time.time()
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - start_infer

        # Detailed evaluation printout
        evaluate_model(
            name=f"RF with {mode.upper()} features", 
            y_true=y_test, 
            y_pred=y_pred, 
            y_prob=y_prob, 
            train_time=train_time, 
            infer_time=infer_time
        )

        # Save metrics for the final summary table
        results.append({
            "Feature Mode": mode.capitalize(),
            "Feature Count": X_mode.shape[1],
            "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
            "Precision": f"{precision_score(y_test, y_pred):.4f}",
            "Recall": f"{recall_score(y_test, y_pred):.4f}",
            "F1-Score": f"{f1_score(y_test, y_pred):.4f}",
            "ROC-AUC": f"{roc_auc_score(y_test, y_prob):.4f}",
            "Train Time (sec)": f"{train_time:.2f}",
            "Infer Time (sec)": f"{infer_time:.4f}"
        })

    # 3. Display the final comparison table
    results_df = pd.DataFrame(results)
    print("\n=========================================================================================")
    print(results_df.to_string(index=False))
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()