import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
from tensorflow.keras.models import load_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.inference_pipeline import run_inference
from src.utils.telegram_notifier import notify_admin_async

st.set_page_config(
    page_title="NASA SMAP - Giám sát Sức khỏe & Viễn trắc Tàu vũ trụ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Tinh chỉnh lại CSS để nhường chỗ cho thanh Sidebar của Streamlit */
        div.block-container {
            padding: 0rem !important;
        }
        iframe {
            border: none;
            width: 100vw;
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(ROOT_DIR, "index.html")
json_path = os.path.join(ROOT_DIR, "scenario_data.json")

LSTM_PATH = os.path.join(ROOT_DIR, "..", "models", "final", "lstm.keras")
IF_PATH = os.path.join(ROOT_DIR, "..", "models", "final", "isolation_forest.pkl")

@st.cache_resource(show_spinner="Đang khởi động Lõi Suy luận AI...")
def load_ai_cores():
    lstm = load_model(LSTM_PATH) if os.path.exists(LSTM_PATH) else None
    if_model = joblib.load(IF_PATH) if os.path.exists(IF_PATH) else None
    return lstm, if_model

def process_uploaded_file(uploaded_file, lstm_model, if_model):
    print(f"==================================================")
    print(f"AI ĐANG BẮT ĐẦU XỬ LÝ FILE: {uploaded_file.name}")
    try:
        if uploaded_file.name.endswith('.npy'):
            series = np.load(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)
            series = df.values
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file: {e}")
        return None

    if len(series) > 500:
        series = series[:500]

    labels, residuals, _ = run_inference(
        series=series,
        lstm_model=lstm_model,
        if_model=if_model
    )

    actual_full = series[:, 0].tolist() if series.ndim > 1 else series.reshape(-1).tolist()
    
    window_cut = len(actual_full) - len(residuals)
    actual = actual_full[window_cut:] 
    
    res_list = residuals.tolist()
    predicted = [float(a - r) if a > 0 else float(a + r) for a, r in zip(actual, res_list)]
    anomalies = [1 if lbl == -1 else 0 for lbl in labels]
    
    total_anomalies = sum(anomalies)
    print(f"TRẠNG THÁI: Nhận diện được {total_anomalies} điểm lỗi.")
    
    if total_anomalies > 0:
        print("Phát hiện lỗi! Đang kích hoạt Telegram...")
        max_err = max(res_list)
        notify_admin_async(uploaded_file.name, total_anomalies, max_err)
    else:
        print("File an toàn (0 lỗi). Không bắn cảnh báo.")
        
    print(f"==================================================")


    uploaded_data = {
        "actual": actual,
        "predicted": predicted,
        "residuals": res_list,
        "anomalies": anomalies,
        "scores": res_list,
        "filename": uploaded_file.name
    }
    
    return uploaded_data

def main():
    lstm_model, if_model = load_ai_cores()

    with st.sidebar:
        st.title("AI Diagnostics")
        st.write("Nạp luồng viễn trắc mới để chạy suy luận thời gian thực qua LSTM & Isolation Forest.")
        uploaded_file = st.file_uploader("Nạp file .csv hoặc .npy", type=['csv', 'npy'])

    with open(html_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    with open(json_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)

    uploaded_scenario_name = "null"

    if uploaded_file is not None:
        with st.spinner("AI đang phân tích luồng viễn trắc..."):
            live_data = process_uploaded_file(uploaded_file, lstm_model, if_model)
            if live_data:
                scenario_data["uploaded"] = live_data
                uploaded_scenario_name = "'uploaded'"

    json_str = json.dumps(scenario_data)
    html_content = html_template.replace("{{SCENARIO_DATA_JSON}}", json_str)
    html_content = html_content.replace("{{UPLOADED_SCENARIO_NAME}}", uploaded_scenario_name)

    components.html(html_content, height=980, scrolling=True)

if __name__ == "__main__":
    main()