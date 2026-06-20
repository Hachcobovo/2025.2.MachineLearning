import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model
from src.explainability import plot_shap_summary, analyze_false_positives_negatives

def main():
    print("=========================================================")
    print("Class Imbalance Simulation")
    print("=========================================================\n")

    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading data from {data_path}...\n")
    df = pd.read_csv(data_path, low_memory=False)
    y = df['label'].values

    print("Splitting dataset into train and test sets...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Store raw test texts for the analysis step
    raw_test_texts = df_test['full_text'].fillna("").values

    print("Extracting features (Hybrid mode)...")
    X_train, feature_names, tfidf_vec = features.build_features(df_train, mode="hybrid", fit_tfidf=True)
    X_test, _, _ = features.build_features(df_test, mode="hybrid", fit_tfidf=False, tfidf_vectorizer=tfidf_vec)


    n_normal_train = sum(y_train == 0)
    target_attacks_train = int(n_normal_train * 0.05)

    undersampler = RandomUnderSampler(
        sampling_strategy={0: n_normal_train, 1: target_attacks_train},
        random_state=42
    )
    X_train_imb, y_train_imb = undersampler.fit_resample(X_train, y_train)

    print(f"Original Training Attacks: {sum(y_train == 1)}")
    print(f"Imbalanced Training Attacks: {sum(y_train_imb == 1)} (Simulating 5% rare attacks)")

    n_normal_test = sum(y_test == 0)
    target_attacks_test = int(n_normal_test * 0.05)
    
    test_undersampler = RandomUnderSampler(
        sampling_strategy={0: n_normal_test, 1: target_attacks_test},
        random_state=42
    )
    X_test_imb, y_test_imb = test_undersampler.fit_resample(X_test, y_test)

    imb_indices = test_undersampler.sample_indices_
    raw_test_texts_imb = raw_test_texts[imb_indices]


    # Train Model 1: Ignorant of Imbalance
    print("\nTraining Standard Model on Imbalanced Data...")
    rf_imbalanced = get_random_forest()
    start_train = time.time()
    rf_imbalanced.fit(X_train_imb, y_train_imb)
    train_time_imb = time.time() - start_train

    # Train Model 2: Aware of Imbalance
    print("Training 'Balanced' Model to fix Imbalance...")
    rf_fixed = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    start_train = time.time()
    rf_fixed.fit(X_train_imb, y_train_imb)
    train_time_fixed = time.time() - start_train

    # Evaluation
    res1 = evaluate_model(
        model=rf_imbalanced, 
        X_test=X_test_imb, 
        y_test=y_test_imb, 
        model_name="1. Standard Model (No Class Weight)",
        training_time=train_time_imb
    )
    for k, v in res1.items(): print(f"{k}: {v}")
    
    print("-" * 50)

    res2 = evaluate_model(
        model=rf_fixed, 
        X_test=X_test_imb, 
        y_test=y_test_imb, 
        model_name="2. Fixed Model (Class Weight = Balanced)",
        training_time=train_time_fixed
    )
    for k, v in res2.items(): print(f"{k}: {v}")


    sample_indices = np.random.choice(X_test_imb.shape[0], min(500, X_test_imb.shape[0]), replace=False)
    X_test_sample = X_test_imb[sample_indices]
    print("\n--- SHAP Summary: Standard Imbalanced Model ---")
    plot_shap_summary(rf_imbalanced, X_test_sample, feature_names)
    print("\n--- SHAP Summary: Balanced Model ---")
    plot_shap_summary(rf_fixed, X_test_sample, feature_names)

    print("\nMISCLASSIFICATION COMPARISON: Imbalanced vs Balanced")
    print("\n1. Standard Imbalanced Model")
    analyze_false_positives_negatives(
        model=rf_imbalanced, 
        X_test=X_test_imb, 
        y_test=y_test_imb, 
        raw_texts=raw_test_texts_imb,
        num_samples=2
    )
    print("\n2. Class-Weighted Balanced Model")
    analyze_false_positives_negatives(
        model=rf_fixed, 
        X_test=X_test_imb, 
        y_test=y_test_imb, 
        raw_texts=raw_test_texts_imb,
        num_samples=2
    )

if __name__ == "__main__":
    main()