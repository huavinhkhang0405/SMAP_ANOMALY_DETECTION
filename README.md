# 🛰️ Time Series: Anomaly Detection with LSTM & Isolation Forest

This project combines an LSTM forecasting model with Isolation Forest to detect anomalous points in the NASA SMAP telemetry dataset.

**Authors:** Hứa Vĩnh Khang & Lưu Trí Kiệt

---

## 📂 Project Structure

```text
SMAP_Anomaly_Detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI (Automated Testing)
│
├── app/
│   └── app.py                      # Streamlit Web UI Dashboard
│
├── data/
│   ├── processed/                  # Preprocessed arrays (.npy)
│   └── raw/                        # Raw NASA SMAP telemetry dataset
│
├── models/
│   ├── checkpoints/                # Model weights saved during training epochs
│   └── final/                      # Production-ready assets (lstm.keras, isolation_forest.pkl)
│
├── report/                         # Generated plots, evaluation metrics, and charts
│
├── src/                            # Core source code package
│   ├── data/                       # Data loading and preprocessing logic
│   ├── model/                      # Model training and architecture setups
│   ├── pipeline/
│   │   └── inference_pipeline.py   # Main end-to-end evaluation flow
│   ├── utils/
│   │   └── residual_features.py    # Rolling statistical features extraction
│   └── config.py                   # Centralized hyperparameters & paths
│
├── tests/
│   └── test_contracts.py           # Pytest data shape & integrity validation
│
├── .gitignore                      # Cache, venv, data, and model exclusions
├── pytest.ini                      # Pytest path configuration mapping
├── README.md                       # Project documentation (Portfolio profile)
├── requirements.txt                # Fixed-version python dependencies
└── train.py                        # Training execution pipeline script
```

## 🔍 1. Workflow Overview
```
    Time-Series Data 
            ↓ 
    LSTM Forecasting 
            ↓ 
    Residual Calculation 
            ↓ 
    Rolling Features 
            ↓ 
    Isolation Forest 
            ↓ 
    Anomaly Detection
```

## ⚙️ 2. Environment Setup
### 1. Clone repository
    git clone https://github.com/huavinhkhang0405/SMAP_ANOMALY_DETECTION.git
    cd SMAP_Anomaly_Detection

### 2. Create virtual environment
#### Windows
    python -m venv venv
    .\venv\Scripts\activate

#### Linux/Mac
    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies
    python.exe -m pip install --upgrade pip
    python.exe -m pip install -r requirements.txt

## ▶️ 3. Run Project
### 1. Train the model
    python train.py

### 2. Run Streamlit Web Dashboard
    streamlit run app/app.py

## 3. Testing
### Run unit tests
    python -m pytest tests/


## Data Contract
### LSTM Input
    Shape: (samples, 100, 1)

### LSTM Output
    Shape: (samples, 1)

### Residual
    residual = np.abs(y_true - y_pred)

## 🚀 Tech Stack
- TensorFlow
- Scikit-learn
- Streamlit
- Plotly
- NumPy / Pandas


## SPRINTS
| Sprint  | Main focus                                                         | Owner     | Deliverables                                              |
| ----- | ------------------------------------------------------------------- | --------- | --------------------------------------------------------- |
| Sprint 1 | Light EDA + Freeze Data Contract + Sliding Window                  | Khang     | EDA report, config.py, create_sequences(), verified shape |
| Sprint 2 | Build, train, and save the LSTM model                              | Khang     | lstm.keras, actual vs predicted plot                      |
| Sprint 3 | Residual engineering + Rolling Features + Isolation Forest + Metrics | Kiệt    | isolation_forest.pkl, anomaly plots, evaluation metrics   |
| Sprint 4 | Build inference_pipeline.py and Streamlit UI                       | Kiệt      | Dashboard running with final models                       |
| Sprint 5 | Full system integration, remove Mock Data, bug fixes               | Entire team | Full pipeline stable                                    |
| Sprint 6 | Write report, slides, demo rehearsal                               | Entire team | PDF report, slides, backup demo video                   |
