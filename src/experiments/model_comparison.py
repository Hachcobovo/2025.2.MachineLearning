import sys
import os
import time
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest, get_xgboost
from src.evaluation import evaluate_model
from src.evaluation import plot_roc_curve

def main():

    # 1. Load and Preprocess Data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading and preprocessing data from {data_path}...")
    df = preprocessing.preprocess(data_path)

    # 2. Feature Extraction
    print("\nExtracting features (Hybrid mode)...")
    X, feature_names, tfidf_vectorizer = features.build_features(df, mode="hybrid", fit_tfidf=True)
    y = df['label'].values
    print(f"Feature matrix shape: {X.shape}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 4. Train and Evaluate XGBoost
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 5. Train and Evaluate Random Forest
    # ---------------------------------------------------------
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

if __name__ == "__main__":
    main()