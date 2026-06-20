import sys
import os
import time
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src import preprocessing
from src import features
from src.models import get_random_forest
from src.evaluation import evaluate_model

def main():
    # 1. Load data WITHOUT applying the clean_text() function
    data_path = os.path.join(project_root, 'data', 'processed', 'processed.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'processed', 'processed.csv')
        
    print(f"Loading raw data from {data_path}...")
    raw_df = preprocessing.load_raw_csv(data_path)
    raw_df = preprocessing.normalize_columns(raw_df)

    # Fill missing values BEFORE concatenation to avoid wiping out data
    raw_df.fillna("", inplace=True)

    # Create a raw full text column bypassing the lowercase and URL decoding
    raw_df["full_text"] = raw_df["method"] + " " + raw_df["url"] + " " + raw_df["body"]
    y_raw = raw_df['label'].values

    print("\nSplitting dataset into train and test sets...")
    df_train, df_test, y_train, y_test = train_test_split(
        raw_df, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # 2. Extract features (Fit on Train, Transform on Test)
    print("Extracting features from un-normalized text (TF-IDF)...")
    X_train_raw, _, tfidf_vec = features.build_features(df_train, mode="tfidf", fit_tfidf=True)
    X_test_raw, _, _ = features.build_features(df_test, mode="tfidf", fit_tfidf=False, tfidf_vectorizer=tfidf_vec)

    # 3. Train Model on Raw Data (Random Forest)
    print("\nInitializing and Training Random Forest on Raw Data...")
    rf_raw = get_random_forest()
    
    start_train = time.time()
    rf_raw.fit(X_train_raw, y_train)
    train_time = time.time() - start_train

    # Predict
    print("Making predictions on the test set...")
    start_infer = time.time()
    y_pred_raw = rf_raw.predict(X_test_raw)
    y_prob_raw = rf_raw.predict_proba(X_test_raw)[:, 1]
    infer_time = time.time() - start_infer

    # 4. Evaluate using the centralized evaluation module
    evaluate_model(
        name="Random Forest (No Preprocessing / Un-normalized)", 
        y_true=y_test, 
        y_pred=y_pred_raw, 
        y_prob=y_prob_raw, 
        train_time=train_time, 
        infer_time=infer_time
    )

if __name__ == "__main__":
    main()