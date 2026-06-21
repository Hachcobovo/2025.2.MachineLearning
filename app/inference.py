"""
app/inference.py
-----------------
Logic suy luận cho demo: nhận một HTTP request thô (method, url, body),
chạy qua đúng pipeline preprocessing + feature extraction đã có trong
src/preprocessing.py và src/features.py (KHÔNG sửa 2 file đó), rồi
predict bằng model đã train sẵn (random_forest.joblib / xgboost.joblib).

Ngoài ra cung cấp hàm detect_attack_indicators() để liệt kê các "dấu hiệu
tấn công" (SQLi/XSS/Command Injection/Path Traversal/Encoded payload...)
được phát hiện trong request, dựa trên đúng các keyword list đã định nghĩa
sẵn trong src/features.py.
"""
import os
import sys

import joblib
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src import preprocessing
from src import features as feat_module

MODELS_DIR = os.path.join(project_root, "models")

# Các nhóm keyword đã có sẵn trong src/features.py — tái sử dụng để show
# "attack indicators" cho người dùng, không định nghĩa lại danh sách mới.
INDICATOR_GROUPS = {
    "SQL Injection": feat_module.SQL_KEYWORDS,
    "XSS": feat_module.XSS_KEYWORDS,
    "Command Injection": feat_module.CMD_KEYWORDS,
    "Path Traversal": feat_module.PATH_KEYWORDS,
}


class ModelBundle:
    """Gói model + vectorizer load 1 lần, dùng lại nhiều lần."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.models = {}
        self.metrics = {}
        self._load()

    def _load(self):
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        names_path = os.path.join(self.models_dir, "feature_names.joblib")
        rf_path = os.path.join(self.models_dir, "random_forest.joblib")
        xgb_path = os.path.join(self.models_dir, "xgboost.joblib")
        metrics_path = os.path.join(self.models_dir, "metrics.json")

        missing = [p for p in [vec_path, names_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Chưa tìm thấy model artifacts. Hãy chạy "
                "`python scripts/train_models.py` trước.\n"
                f"Thiếu: {missing}"
            )

        self.tfidf_vectorizer = joblib.load(vec_path)
        self.feature_names = joblib.load(names_path)

        if os.path.exists(rf_path):
            self.models["Random Forest"] = joblib.load(rf_path)
        if os.path.exists(xgb_path):
            self.models["XGBoost"] = joblib.load(xgb_path)

        if not self.models:
            raise FileNotFoundError(
                "Không tìm thấy model nào (random_forest.joblib / xgboost.joblib) "
                "trong models/. Hãy chạy scripts/train_models.py trước."
            )

        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @property
    def available_models(self):
        return list(self.models.keys())


def _build_single_row_df(method: str, url: str, body: str) -> pd.DataFrame:
    """
    Đóng gói 1 request thô thành DataFrame 1 dòng với đúng các cột mà
    src/preprocessing.normalize_columns() và build_features() mong đợi.
    Gắn label tạm = 0 vì normalize_columns() yêu cầu có cột label.
    """
    raw_df = pd.DataFrame([{
        "method": method,
        "url": url,
        "body": body,
        "label": "normal",  # placeholder bắt buộc, không ảnh hưởng inference
    }])
    return raw_df


def preprocess_single_request(method: str, url: str, body: str) -> pd.DataFrame:
    """
    Chạy đúng pipeline của src/preprocessing.py cho 1 request:
    normalize_columns -> parse_url -> clean_text -> full_text
    (sao chép lại đúng các bước trong preprocessing.preprocess(),
    nhưng không qua bước load_raw_csv vì input đến từ form, không từ CSV).
    """
    raw_df = _build_single_row_df(method, url, body)
    df = preprocessing.normalize_columns(raw_df)

    url_parsed = df["url"].apply(preprocessing.parse_url)
    df["url_path"] = url_parsed.apply(lambda x: x[0])
    df["url_query"] = url_parsed.apply(lambda x: x[1])
    df["query"] = df.apply(
        lambda r: r["url_query"] if not r.get("query") else r["query"], axis=1
    )

    for col in ["url", "url_path", "url_query", "query", "body", "method"]:
        df[col] = df[col].apply(preprocessing.clean_text)

    df["full_text"] = (
        df["method"] + " " + df["url_path"] + " " + df["query"] + " " + df["body"]
    ).apply(preprocessing.clean_text)

    df.fillna("", inplace=True)
    return df


def detect_attack_indicators(full_text: str) -> dict:
    """
    Trả về dict {nhóm_tấn_công: [keyword tìm thấy, ...]} dựa trên
    các keyword list đã có sẵn trong src/features.py.
    """
    found = {}
    for group_name, keywords in INDICATOR_GROUPS.items():
        hits = [kw for kw in keywords if kw in full_text]
        if hits:
            found[group_name] = hits

    encoded_hits = feat_module._encoded_char_count(full_text)
    if encoded_hits > 0:
        found["Encoded Payload (percent-encoding)"] = [
            f"{encoded_hits} chuỗi percent-encoded phát hiện"
        ]
    return found


def predict(bundle: ModelBundle, model_name: str, method: str, url: str, body: str):
    """
    Chạy full pipeline cho 1 request và trả về kết quả dự đoán.

    Returns
    -------
    dict gồm:
        prediction        : "Malicious" hoặc "Normal"
        probability        : xác suất là Malicious (float 0-1), None nếu model không hỗ trợ
        attack_indicators  : dict các dấu hiệu tấn công tìm thấy trong request
        full_text           : text đã chuẩn hóa (để hiển thị debug)
        feature_count      : số chiều của feature vector
    """
    df_single = preprocess_single_request(method, url, body)
    full_text = df_single.loc[0, "full_text"]

    X, _, _ = feat_module.build_features(
        df_single,
        mode="hybrid",
        fit_tfidf=False,
        tfidf_vectorizer=bundle.tfidf_vectorizer,
    )

    model = bundle.models[model_name]
    pred_label = int(model.predict(X)[0])

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])

    indicators = detect_attack_indicators(full_text)

    return {
        "prediction": "Malicious" if pred_label == 1 else "Normal",
        "probability": prob,
        "attack_indicators": indicators,
        "full_text": full_text,
        "feature_count": X.shape[1],
    }