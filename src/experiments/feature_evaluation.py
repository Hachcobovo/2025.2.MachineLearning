import re
import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model
from src.explainability import plot_feature_importance, plot_shap_summary, analyze_false_positives_negatives

def main():
    # Load Data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading and preprocessing data from {data_path}...\n")
    df = pd.read_csv(data_path, low_memory=False)
    y = df['label'].values

    print("Splitting dataset into train and test sets...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Store raw test texts for the evaluation function
    raw_test_texts = df_test['full_text'].values

    feature_modes = ["handcrafted", "tfidf", "hybrid"]
    results = []

    # Iterate through each feature extraction mode
    for mode in feature_modes:
        print(f"\n--- Extracting and Training on '{mode.upper()}' features ---")

        X_train, feature_names_mode, tfidf_vec = features.build_features(df_train, mode=mode, fit_tfidf=True)
        X_test, _, _ = features.build_features(df_test, mode=mode, fit_tfidf=False, tfidf_vectorizer=tfidf_vec)
        
        print(f"Number of training features: {X_train.shape[1]}")

        # Train Model (Using Random Forest as the baseline for this experiment)
        rf_model = get_random_forest()

        start_train = time.time()
        rf_model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Evaluate (Predictions and inference time happen internally now)
        res = evaluate_model(
            model=rf_model, 
            X_test=X_test, 
            y_test=y_test, 
            model_name=f"RF with {mode.upper()} features", 
            training_time=train_time
        )

        # Feature Importance
        plot_feature_importance(rf_model, feature_names_mode, top_n=10)
        
        # SHAP Summary (Sample 500 rows to prevent memory overload from TF-IDF)
        clean_feature_names = [re.sub(r'[\[\]<>]', '_', name) for name in feature_names_mode]
        print(f"\nGenerating SHAP summary for {mode.upper()} features...")
        sample_indices = np.random.choice(X_test.shape[0], min(100, X_test.shape[0]), replace=False)
        X_test_sample = X_test[sample_indices]
        plot_shap_summary(rf_model, X_test_sample, clean_feature_names)

        print(f"\nEvaluating Misclassifications for {mode.upper()} features:")
        analyze_false_positives_negatives(
            model=rf_model,
            X_test=X_test,
            y_test=y_test,
            raw_texts=raw_test_texts,
            num_samples=3
        )

        # Save metrics for the final summary table
        results.append({
            "Feature Mode": mode.capitalize(),
            "Feature Count": X_train.shape[1],
            "Accuracy": f"{res['Accuracy']:.4f}",
            "Precision": f"{res['Precision']:.4f}",
            "Recall": f"{res['Recall']:.4f}",
            "F1-Score": f"{res['F1']:.4f}",
            "ROC-AUC": f"{res['ROC-AUC']:.4f}" if res['ROC-AUC'] else "N/A",
            "Train Time (sec)": f"{res['Training_time(s)']:.2f}",
            "Infer Time (sec)": f"{res['Inference_time(s)']:.4f}"
        })

    results_df = pd.DataFrame(results)
    print("\n=========================================================================================")
    print(results_df.to_string(index=False))
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()