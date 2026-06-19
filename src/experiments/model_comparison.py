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

def main():

    # 1. Load and Preprocess Data
    # Assuming your data is stored in the root 'data' folder
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
    print("Splitting dataset into train and test sets (80/20)...")
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
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    train_time_xgb = time.time() - start_train_time
    print(f"Training completed in {train_time_xgb:.4f} seconds.")

    print("Making predictions on test set...")
    start_inference_time = time.time()
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    infer_time_xgb = time.time() - start_inference_time

    # Use our evaluation module
    evaluate_model(
        name="XGBoost", 
        y_true=y_test, 
        y_pred=y_pred_xgb, 
        y_prob=y_prob_xgb, 
        train_time=train_time_xgb, 
        infer_time=infer_time_xgb
    )

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

    print("Making predictions on test set...")
    start_inference_time = time.time()
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    infer_time_rf = time.time() - start_inference_time

    # Use our evaluation module
    evaluate_model(
        name="Random Forest", 
        y_true=y_test, 
        y_pred=y_pred_rf, 
        y_prob=y_prob_rf, 
        train_time=train_time_rf, 
        infer_time=infer_time_rf
    )

if __name__ == "__main__":
    main()