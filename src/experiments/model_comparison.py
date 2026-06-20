import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest, get_xgboost
from src.evaluation import evaluate_model
from src.evaluation import plot_roc_curve
from src.explainability import plot_shap_summary, analyze_false_positives_negatives
import scipy.sparse as sp
import numpy as np

def main():

    # Load Data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading and preprocessing data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    y = df['label'].values

    print("\nSplitting dataset into train and test sets...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nExtracting features (Hybrid mode)...")
    X_train, feature_names, tfidf_vectorizer = features.build_features(df_train, mode="hybrid", fit_tfidf=True)
    X_test, _, _ = features.build_features(df_test, mode="hybrid", fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer)
    
    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Testing feature matrix shape: {X_test.shape}")

    print("\nInitializing XGBoost Model...")
    xgb_model = get_xgboost()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    print("Training XGBoost...")
    start_train_time = time.time()
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=10)
    train_time_xgb = time.time() - start_train_time
    print(f"Training completed in {train_time_xgb:.4f} seconds.")
    
    print("\nEvaluating XGBoost...")
    res_xgb = evaluate_model(
        model=xgb_model, 
        X_test=X_test, 
        y_test=y_test, 
        model_name="XGBoost", 
        training_time=train_time_xgb
    )
    for k, v in res_xgb.items(): print(f"{k}: {v}")

    print("\nInitializing Random Forest Model...")
    rf_model = get_random_forest()
    print("Training Random Forest...")
    start_train_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time_rf = time.time() - start_train_time
    print(f"Training completed in {train_time_rf:.4f} seconds.")
    
    print("\nEvaluating Random Forest...")
    res_rf = evaluate_model(
        model=rf_model, 
        X_test=X_test, 
        y_test=y_test, 
        model_name="Random Forest", 
        training_time=train_time_rf
    )
    for k, v in res_rf.items(): print(f"{k}: {v}")

    plot_roc_curve(
        models={"XGBoost": xgb_model, "Random Forest": rf_model},
        X_test=X_test,
        y_test=y_test
    )

    # SHAP Summary Plot
    sample_indices = np.random.choice(X_test.shape[0], min(500, X_test.shape[0]), replace=False)
    X_test_sample = X_test[sample_indices]
    print("\nSHAP summary for XGBoost")
    plot_shap_summary(xgb_model, X_test_sample, feature_names)
    print("\nSHAP summary for Random Forest")
    plot_shap_summary(rf_model, X_test_sample, feature_names)

    # Analyze False Positives / False Negatives
    raw_test_texts = df_test['full_text'].values
    print("\nFalse Positives / False Negatives for XGBoost")
    analyze_false_positives_negatives(model=xgb_model, X_test=X_test, y_test=y_test, raw_texts=raw_test_texts, num_samples=3)
    print("\nFalse Positives / False Negatives for Random Forest")
    analyze_false_positives_negatives(model=rf_model, X_test=X_test, y_test=y_test, raw_texts=raw_test_texts, num_samples=3)

if __name__ == "__main__":
    main()