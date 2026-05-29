# =========================
# DATA CONFIG
# =========================

WINDOW_SIZE = 100
N_FEATURES = 1
PREDICTION_DIM = 1

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# =========================
# LSTM CONFIG
# =========================

LSTM_UNITS = 64
DROPOUT_RATE = 0.2

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 0.001

# =========================
# ANOMALY DETECTION
# =========================

ROLLING_WINDOW = 10

IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.02
RANDOM_STATE = 42

# =========================
# PATHS
# =========================

MODEL_PATH = "models/final/lstm.keras"
IF_MODEL_PATH = "models/final/isolation_forest.pkl"

# =========================
# DATA PATHS
# =========================

RAW_DATA_ROOT = "data/raw/NASA_SMAP/data/data"
PROCESSED_DIR = "data/processed"
REPORT_DIR = "report"
DEFAULT_CHANNEL = "P-1"

Y_TRUE_PATH = "data/processed/y_true.npy"
Y_PRED_PATH = "data/processed/y_pred.npy"
RESIDUALS_PATH = "data/processed/residuals.npy"
RESIDUAL_FEATURES_PATH = "data/processed/residual_features.npy"


