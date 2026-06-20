import sys
import os
import time
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model

def main():
    print("=========================================================")
    print("Class Imbalance Simulation")
    print("=========================================================\n")

    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading data from {data_path}...\n")
    df = preprocessing.load_raw_csv(data_path)
    
    print("Extracting features (Hybrid mode)...")
    X, feature_names, _ = features.build_features(df, mode="hybrid", fit_tfidf=True)
    y = df['label'].values

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


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


    # ---------------------------------------------------------
    # TRAIN AND EVALUATE MODELS
    # ---------------------------------------------------------
    
    # Train Model 1: Ignorant of Imbalance
    print("\nTraining Standard Model on Imbalanced Data...")
    rf_imbalanced = get_random_forest()
    
    start_train = time.time()
    rf_imbalanced.fit(X_train_imb, y_train_imb)
    train_time_imb = time.time() - start_train
    
    start_infer = time.time()
    y_pred_imb = rf_imbalanced.predict(X_test_imb)
    y_prob_imb = rf_imbalanced.predict_proba(X_test_imb)[:, 1] # Needed for ROC-AUC
    infer_time_imb = time.time() - start_infer

    # Train Model 2: Aware of Imbalance (Using class_weight)
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
    
    start_infer = time.time()
    y_pred_fixed = rf_fixed.predict(X_test_imb)
    y_prob_fixed = rf_fixed.predict_proba(X_test_imb)[:, 1]
    infer_time_fixed = time.time() - start_infer


    # Evaluation
    print("\n" + "="*50)
    print("EXPERIMENT 4 RESULTS (Real-World Imbalance)")
    print("="*50)

    evaluate_model(
        name="1. Standard Model (No Class Weight)", 
        y_true=y_test_imb, 
        y_pred=y_pred_imb, 
        y_prob=y_prob_imb,
        train_time=train_time_imb,
        infer_time=infer_time_imb
    )
    
    evaluate_model(
        name="2. Fixed Model (Class Weight = Balanced)", 
        y_true=y_test_imb, 
        y_pred=y_pred_fixed, 
        y_prob=y_prob_fixed,
        train_time=train_time_fixed,
        infer_time=infer_time_fixed
    )

if __name__ == "__main__":
    main()