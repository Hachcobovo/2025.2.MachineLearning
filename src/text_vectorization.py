"""
text_vectorization.py
---------------------
Text feature pipeline for HTTP Attack Detection.
Provides:
!. Handcrafted features only
2. TF-IDF only: Character-level TF-IDF
3. Hybrid: Handcrafted features + Character TF-IDF
"""
import pandas as pd
from scipy.sparse import (hstack,csr_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from features import (extract_handcrafted,HANDCRAFTED_FEATURE_NAMES)
# TF-IDF Vectorizer
def create_tfidf_vectorizer(max_features=10000,ngram_range=(3,5)):
    return TfidfVectorizer(analyzer="char",ngram_range=ngram_range,max_features=max_features,min_df=2,sublinear_tf=True,lowercase=True)
# Extract TF-IDF
def extract_tfidf(train_df,test_df,vectorizer=None,fit=True,max_features=10000):
    """
    Fit TF-IDF on train data only.
    Parameters
    ----------
    train_df: training dataframe
    test_df: testing dataframe
    vectorizer: fitted vectorizer for inference
    fit: True during training, False during testing

    Returns
    -------
    X_train
    X_test
    feature_names
    vectorizer
    """
    train_text = (train_df["full_text"].fillna("").astype(str).tolist())
    test_text = (test_df["full_text"].fillna("").astype(str).tolist())
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer(max_features=max_features)
    if fit:
        X_train = vectorizer.fit_transform(train_text)
    else:
        X_train = vectorizer.transform(train_text)
    X_test = vectorizer.transform(test_text)
    names = ["tfidf_" + x for x in vectorizer.get_feature_names_out()]
    return (X_train,X_test,names,vectorizer)
# Build Feature Sets
def build_text_features(train_df,test_df,mode="hybrid",vectorizer=None,fit_tfidf=True,max_features=10000):
    """
    Modes:
    handcrafted: security/statistical features only
    tfidf: character TF-IDF only
    hybrid: handcrafted + TF-IDF
    """
    mode = mode.lower()
    # Handcrafted
    X_train_hc = extract_handcrafted(train_df)
    X_test_hc = extract_handcrafted(test_df)
    if mode == "handcrafted":
        return (X_train_hc,X_test_hc,HANDCRAFTED_FEATURE_NAMES,None)
    # TF-IDF
    (X_train_tfidf,X_test_tfidf,tfidf_names,vectorizer) = extract_tfidf(train_df,test_df,vectorizer,fit_tfidf,max_features)
    if mode == "tfidf":
        return (X_train_tfidf,X_test_tfidf,tfidf_names,vectorizer)
    # Hybrid
    if mode == "hybrid":
        X_train = hstack([csr_matrix(X_train_hc),X_train_tfidf])
        X_test = hstack([csr_matrix(X_test_hc),X_test_tfidf])
        names = (HANDCRAFTED_FEATURE_NAMES + tfidf_names)
        return (X_train,X_test,names,vectorizer)
    raise ValueError("mode must be handcrafted, tfidf, or hybrid")
# Compare Experiments
def compare_feature_sets(train_df,test_df):
    results = {}
    for mode in ["handcrafted","tfidf","hybrid"]:
        X_train, X_test, names, _ = build_text_features(train_df,test_df,mode=mode)
        results[mode] = {"train_samples":X_train.shape[0],"test_samples":X_test.shape[0],"features":X_train.shape[1]}
    return pd.DataFrame(results).T
# Test
if __name__ == "__main__":
    from preprocessing import preprocess
    from sklearn.model_selection import train_test_split
    df = preprocess("csic_database.csv")
    train_df, test_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df["label"])
    print(compare_feature_sets(train_df,test_df))
    X_train, X_test, names, vec = build_text_features(train_df,test_df,mode="hybrid")
    print("\nHybrid:",X_train.shape)
    print("First features:",names[:10])