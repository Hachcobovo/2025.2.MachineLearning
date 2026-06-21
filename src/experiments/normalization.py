import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model
from src.explainability import plot_shap_summary, analyze_false_positives_negatives

def main():
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    raw_df = preprocessing.normalize_columns(df)

    # Fill missing values BEFORE concatenation to avoid wiping out data
    raw_df = raw_df.fillna("")

    # Create a raw full text column bypassing the lowercase and URL decoding
    raw_df["full_text"] = raw_df["method"] + " " + raw_df["url"] + " " + raw_df["body"]
    y_raw = raw_df['label'].values

    print("\nSplitting dataset into train and test sets...")
    df_train, df_test, y_train, y_test = train_test_split(
        raw_df, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # Extract features (Fit on Train, Transform on Test)
    print("Extracting features from un-normalized text (TF-IDF)...")
    X_train_raw, _, tfidf_vec = features.build_features(df_train, mode="tfidf", fit_tfidf=True)
    X_test_raw, _, _ = features.build_features(df_test, mode="tfidf", fit_tfidf=False, tfidf_vectorizer=tfidf_vec)

    # Train Model on Raw Data (Random Forest)
    print("\nInitializing and Training Random Forest on Raw Data...")
    rf_raw = get_random_forest()
    
    start_train = time.time()
    rf_raw.fit(X_train_raw, y_train)
    train_time = time.time() - start_train

    # Evaluate using the centralized evaluation module
    print("\nEvaluating model on the test set...")
    res = evaluate_model(
        model=rf_raw, 
        X_test=X_test_raw, 
        y_test=y_test, 
        model_name="Random Forest (No Preprocessing / Un-normalized)", 
        training_time=train_time
    )
    
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    print("\nSHAP summary for Un-normalized Model")
    clean_feature_names_raw = [re.sub(r'[\[\]<>]', '_', name) for name in feature_names_raw]
    sample_indices = np.random.choice(X_test_raw.shape[0], min(100, X_test_raw.shape[0]), replace=False)
    X_test_raw_sample = X_test_raw[sample_indices]
    plot_shap_summary(rf_raw, X_test_raw_sample, clean_feature_names_raw)
    
    raw_texts = df_test['full_text'].values
    analyze_false_positives_negatives(model=rf_raw, X_test=X_test_raw, y_test=y_test, raw_texts=raw_texts, num_samples=5)

if __name__ == "__main__":
    main()