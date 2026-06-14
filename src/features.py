"""
features.py
-----------
3 chế độ trích xuất đặc trưng:
  A) handcrafted  - statistical + security + HTTP features
  B) tfidf        - TF-IDF unigram + bigram
  C) hybrid       - A + B (concat)

Dùng:
    X, feature_names = build_features(df, mode="hybrid")
"""

import re
import math
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# A. HANDCRAFTED FEATURES
# ---------------------------------------------------------------------------

# --- Security keyword lists ---
SQL_KEYWORDS = [
    "select", "union", "insert", "update", "delete", "drop", "create",
    "alter", "exec", "execute", "xp_", "sp_", "declare", "cast", "convert",
    "char(", "nchar", "varchar", "0x", "or 1=1", "and 1=1", "--", "/*", "*/",
    "benchmark", "sleep(", "waitfor", "information_schema", "sysobjects",
]

XSS_KEYWORDS = [
    "<script", "</script>", "javascript:", "onerror=", "onload=", "onclick=",
    "onmouseover=", "onfocus=", "alert(", "document.cookie", "eval(",
    "src=", "href=javascript", "<img", "<iframe", "<body", "<svg",
    "expression(", "vbscript:",
]

CMD_KEYWORDS = [
    ";", "|", "&&", "||", "`", "$(", "cat ", "ls ", "pwd",
    "/etc/passwd", "/etc/shadow", "cmd.exe", "powershell",
    "wget ", "curl ", "nc ", "bash", "/bin/sh", "python -c",
]

PATH_KEYWORDS = [
    "../", "..\\", "%2e%2e", "%252e", "....//", ".../",
    "/etc/", "/proc/", "/var/", "c:\\windows", "c:/windows",
    "boot.ini", "win.ini",
]

ENCODED_PATTERNS = re.compile(r'(%[0-9a-fA-F]{2})+')


def _entropy(s: str) -> float:
    """Shannon entropy của chuỗi."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    total = len(s)
    return -sum((v / total) * math.log2(v / total) for v in freq.values())


def _keyword_count(text: str, keywords: list) -> int:
    return sum(1 for kw in keywords if kw in text)


def _encoded_char_count(text: str) -> int:
    """Đếm số lần xuất hiện của percent-encoded sequences."""
    return len(ENCODED_PATTERNS.findall(text))


def extract_handcrafted(df: pd.DataFrame) -> np.ndarray:
    """
    Trả về numpy array shape (n_samples, n_features).
    Thứ tự features được ghi trong HANDCRAFTED_FEATURE_NAMES.
    """
    records = []

    for _, row in df.iterrows():
        url   = str(row.get("url", ""))
        query = str(row.get("query", ""))
        body  = str(row.get("body", ""))
        method = str(row.get("method", ""))
        full  = url + " " + query + " " + body

        # ---- Statistical Features ----
        url_len      = len(url)
        query_len    = len(query)
        body_len     = len(body)
        num_params   = query.count("=")
        num_digits   = sum(c.isdigit() for c in full)
        num_special  = sum(not c.isalnum() for c in full)
        entropy_url  = _entropy(url)
        entropy_body = _entropy(body)

        # ---- Security Features ----
        sql_count  = _keyword_count(full, SQL_KEYWORDS)
        xss_count  = _keyword_count(full, XSS_KEYWORDS)
        cmd_count  = _keyword_count(full, CMD_KEYWORDS)
        path_count = _keyword_count(full, PATH_KEYWORDS)
        encoded_cnt = _encoded_char_count(full)

        # ---- HTTP Features ----
        method_get  = 1 if method == "get"  else 0
        method_post = 1 if method == "post" else 0

        records.append([
            url_len, query_len, body_len, num_params,
            num_digits, num_special, entropy_url, entropy_body,
            sql_count, xss_count, cmd_count, path_count, encoded_cnt,
            method_get, method_post,
        ])

    return np.array(records, dtype=np.float32)


HANDCRAFTED_FEATURE_NAMES = [
    "url_len", "query_len", "body_len", "num_params",
    "num_digits", "num_special", "entropy_url", "entropy_body",
    "sql_count", "xss_count", "cmd_count", "path_count", "encoded_cnt",
    "method_get", "method_post",
]


# ---------------------------------------------------------------------------
# B. TF-IDF FEATURES
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",       # character n-grams – tốt hơn cho HTTP
        ngram_range=(2, 4),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )


# ---------------------------------------------------------------------------
# C. MAIN BUILD FUNCTION
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   mode: str = "hybrid",
                   tfidf_vectorizer=None,
                   fit_tfidf: bool = True,
                   max_tfidf_features: int = 5000):
    """
    Parameters
    ----------
    df               : DataFrame đã qua preprocessing
    mode             : "handcrafted" | "tfidf" | "hybrid"
    tfidf_vectorizer : truyền vào nếu muốn dùng vectorizer đã fit (inference)
    fit_tfidf        : True khi train, False khi inference
    max_tfidf_features : số features TF-IDF tối đa

    Returns
    -------
    X                : dense np.ndarray hoặc scipy sparse matrix
    feature_names    : list[str]
    tfidf_vectorizer : đối tượng đã fit (dùng lại khi inference)
    """
    mode = mode.lower()

    if mode == "handcrafted":
        X = extract_handcrafted(df)
        return X, HANDCRAFTED_FEATURE_NAMES, None

    # TF-IDF
    texts = df["full_text"].fillna("").tolist()

    if tfidf_vectorizer is None:
        tfidf_vectorizer = build_tfidf_vectorizer(max_tfidf_features)

    if fit_tfidf:
        X_tfidf = tfidf_vectorizer.fit_transform(texts)
    else:
        X_tfidf = tfidf_vectorizer.transform(texts)

    tfidf_names = [f"tfidf_{f}" for f in tfidf_vectorizer.get_feature_names_out()]

    if mode == "tfidf":
        return X_tfidf, tfidf_names, tfidf_vectorizer

    # Hybrid: dense handcrafted + sparse tfidf → sparse concat
    X_hc = extract_handcrafted(df)
    from scipy.sparse import csr_matrix
    X_hybrid = hstack([csr_matrix(X_hc), X_tfidf])
    feature_names = HANDCRAFTED_FEATURE_NAMES + tfidf_names

    return X_hybrid, feature_names, tfidf_vectorizer


if __name__ == "__main__":
    # Quick smoke test
    sample = pd.DataFrame({
        "url":       ["http://example.com/login?id=1 union select 1,2--", "http://example.com/index.php"],
        "query":     ["id=1 union select 1,2--", "page=home"],
        "body":      ["", "user=admin&pass=1234"],
        "method":    ["get", "post"],
        "full_text": ["get login id=1 union select 1,2--", "post index.php page=home user=admin pass=1234"],
        "label":     [1, 0],
    })
    X, names, vec = build_features(sample, mode="hybrid")
    print(f"Hybrid feature shape: {X.shape}")
    print(f"First 5 handcrafted features: {names[:5]}")