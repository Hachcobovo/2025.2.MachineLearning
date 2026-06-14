"""
preprocessing.py
----------------
Đọc CSIC 2010 HTTP Dataset (CSV), parse và làm sạch từng HTTP request.

Cột đầu ra (DataFrame):
    method, url, query, body, headers_raw, label (0=Normal, 1=Attack)
"""

import re
import urllib.parse
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Load raw CSV
# ---------------------------------------------------------------------------

def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    Đọc file CSV gốc của CSIC 2010.
    Dataset thường có dạng: mỗi dòng là một HTTP request thô
    hoặc đã tách sẵn các cột (tùy phiên bản download).

    Hàm này tự detect format và chuẩn hóa về DataFrame chuẩn.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[load] Loaded {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# 2. Chuẩn hóa tên cột
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    # Các tên cột phổ biến trong các phiên bản khác nhau của CSIC
    "Method":   "method",
    "method":   "method",
    "URL":      "url",
    "url":      "url",
    "Uri":      "url",
    "uri":      "url",
    "Body":     "body",
    "body":     "body",
    "Payload":  "body",
    "payload":  "body",
    "Headers":  "headers_raw",
    "headers":  "headers_raw",
    "Label":    "label",
    "label":    "label",
    "Class":    "label",
    "class":    "label",
    "Type":     "label",
    "type":     "label",
    "classification": "label",
    "Classification": "label",
    "content": "body",
    "Content": "body",
}

NORMAL_LABELS  = {"normal", "0", "legit", "legitimate", "benign"}
ATTACK_LABELS  = {"anomalous", "attack", "1", "malicious", "sql", "xss",
                  "command", "path", "traversal"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột về tên chuẩn, thêm cột thiếu bằng giá trị rỗng."""
    df = df.rename(columns=COLUMN_MAP)

    for col in ["method", "url", "query", "body", "headers_raw"]:
        if col not in df.columns:
            df[col] = ""

    # Encode label thành 0/1
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df["label"] = df["label"].apply(
            lambda x: 0 if x in NORMAL_LABELS else 1
        )
    else:
        raise ValueError("Không tìm thấy cột nhãn (label/Label/Class/Type).")

    return df


# ---------------------------------------------------------------------------
# 3. Parse URL → tách path và query string
# ---------------------------------------------------------------------------

def parse_url(url_str: str):
    """Trả về (path, query_string) từ URL thô."""
    if not isinstance(url_str, str) or not url_str.strip():
        return "", ""
    try:
        parsed = urllib.parse.urlparse(url_str)
        path  = parsed.path  or ""
        query = parsed.query or ""
        return path, query
    except Exception:
        return str(url_str), ""


# ---------------------------------------------------------------------------
# 4. Làm sạch text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    - Lowercase
    - URL decode (xử lý %20, %3D, ...)
    - Loại bỏ null bytes và ký tự điều khiển
    - Chuẩn hóa khoảng trắng
    """
    if not isinstance(text, str):
        return ""

    # URL decode nhiều lần (double encoding attacks)
    for _ in range(3):
        try:
            decoded = urllib.parse.unquote_plus(text)
            if decoded == text:
                break
            text = decoded
        except Exception:
            break

    text = text.lower()
    # Loại ký tự null và control chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# 5. Pipeline tổng hợp
# ---------------------------------------------------------------------------

def preprocess(csv_path: str, output_dir: str = None) -> pd.DataFrame:
    """
    Pipeline đầy đủ:
      load → normalize_columns → parse_url → clean_text → lưu processed CSV
    """
    df = load_raw_csv(csv_path)
    df = normalize_columns(df)

    # Parse URL thành path + query
    url_parsed = df["url"].apply(parse_url)
    df["url_path"]  = url_parsed.apply(lambda x: x[0])
    df["url_query"] = url_parsed.apply(lambda x: x[1])

    # Nếu chưa có cột query riêng, lấy từ url_query
    df["query"] = df.apply(
        lambda r: r["url_query"] if not r.get("query") else r["query"],
        axis=1
    )

    # Làm sạch tất cả text fields
    for col in ["url", "url_path", "url_query", "query", "body", "method"]:
        df[col] = df[col].apply(clean_text)

    # Ghép text tổng hợp cho TF-IDF
    df["full_text"] = (
        df["method"] + " " +
        df["url_path"] + " " +
        df["query"] + " " +
        df["body"]
    ).apply(clean_text)

    # Xử lý missing values
    df.fillna("", inplace=True)

    print(f"[preprocess] Normal: {(df.label==0).sum():,} | Attack: {(df.label==1).sum():,}")

    if output_dir:
        out = Path(output_dir) / "processed.csv"
        df.to_csv(out, index=False)
        print(f"[preprocess] Saved → {out}")

    return df


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "csic_database.csv"
    df = preprocess(csv_path, output_dir="data/processed")
    print(df[["method", "url_path", "query", "body", "label", "full_text"]].head(3))