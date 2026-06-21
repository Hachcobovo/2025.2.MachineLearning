"""
scripts/train_models.py
------------------------
Train Random Forest + XGBoost trên hybrid features (handcrafted + TF-IDF)
và lưu model + tfidf vectorizer ra thư mục models/ để app/streamlit_app.py
load lại dùng cho demo.

KHÔNG sửa bất kỳ file nào trong src/ — chỉ import và gọi lại.

Cách chạy:
    python scripts/train_models.py
    python scripts/train_models.py --data data/processed/processed.csv

Output:
    models/tfidf_vectorizer.joblib
    models/random_forest.joblib
    models/xgboost.joblib
    models/feature_names.joblib   (list feature names, dùng cho hiển thị/SHAP nếu cần)
    models/metrics.json           (kết quả evaluate trên test set, để hiển thị trong UI)
"""
import sys
import os
import json
import argparse
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src import features
from src.models import get_random_forest, get_xgboost
from src.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train RF + XGBoost cho demo")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(project_root, "data", "processed", "processed.csv"),
        help="Đường dẫn tới processed.csv (output của src/preprocessing.py)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(project_root, "models"),
        help="Thư mục lưu model đã train",
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("=========================================================")
    print("Training models cho Demo System")
    print("=========================================================\n")

    print(f"Loading data từ {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)
    y = df["label"].values

    print("Splitting train/test (80/20, stratified) ...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Extracting hybrid features (handcrafted + TF-IDF) ...")
    X_train, feature_names, tfidf_vec = features.build_features(
        df_train, mode="hybrid", fit_tfidf=True
    )
    X_test, _, _ = features.build_features(
        df_test, mode="hybrid", fit_tfidf=False, tfidf_vectorizer=tfidf_vec
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

    metrics = {}

    # ---- Random Forest ----
    print("Training Random Forest ...")
    rf_model = get_random_forest()
    start = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start
    res_rf = evaluate_model(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        model_name="Random Forest",
        training_time=rf_train_time,
    )
    metrics["random_forest"] = {k: v for k, v in res_rf.items() if k != "Model"}
    print(f"  -> Accuracy={res_rf['Accuracy']:.4f}  F1={res_rf['F1']:.4f}  "
          f"ROC-AUC={res_rf['ROC-AUC']:.4f}\n")

    # ---- XGBoost ----
    print("Training XGBoost ...")
    xgb_model = get_xgboost()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    start = time.time()
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    xgb_train_time = time.time() - start
    res_xgb = evaluate_model(
        model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost",
        training_time=xgb_train_time,
    )
    metrics["xgboost"] = {k: v for k, v in res_xgb.items() if k != "Model"}
    print(f"  -> Accuracy={res_xgb['Accuracy']:.4f}  F1={res_xgb['F1']:.4f}  "
          f"ROC-AUC={res_xgb['ROC-AUC']:.4f}\n")

    # ---- Save artifacts ----
    print(f"Saving artifacts vào {args.models_dir} ...")
    joblib.dump(tfidf_vec, os.path.join(args.models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(rf_model, os.path.join(args.models_dir, "random_forest.joblib"))
    joblib.dump(xgb_model, os.path.join(args.models_dir, "xgboost.joblib"))
    joblib.dump(feature_names, os.path.join(args.models_dir, "feature_names.joblib"))

    with open(os.path.join(args.models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nHoàn tất. Các file đã lưu:")
    for fname in [
        "tfidf_vectorizer.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
        "feature_names.joblib",
        "metrics.json",
    ]:
        print(f"  - {os.path.join(args.models_dir, fname)}")

    print("\nGiờ bạn có thể chạy demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()