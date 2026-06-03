# Data config
WINDOW_SIZE = 100
N_FEATURES = 1
PREDICTION_DIM = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# LSTM config
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Anomaly detection (Isolation Forest)
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.089
RANDOM_STATE = 42
# Threshold calibration
IF_THRESHOLD_STRATEGY = "pr_curve"
IF_FIXED_THRESHOLD = 0.0
IF_PERCENTILE_THRESHOLD = 91.1
IF_FBETA = 1.5

# Temporal feature engineering
ROLLING_WINDOWS = [5, 10, 20, 50]
USE_DIFF_FEATURES = True
USE_ZSCORE_FEATURE = True

# Post-processing temporal smoothing
IF_VOTE_WINDOW = 21
IF_VOTE_RATIO = 0.50
IF_MIN_CLUSTER_LEN = 5
IF_MERGE_GAP = 15

# Grid search space for smoothing
GS_VOTE_WINDOWS = [5, 11, 15, 21]
GS_VOTE_RATIOS = [0.3, 0.4, 0.5]
GS_MERGE_GAPS = [5, 10, 15]

# Legacy
ROLLING_WINDOW = 10

# Paths
MODEL_PATH = "models/final/lstm.keras"
IF_MODEL_PATH = "models/final/isolation_forest.pkl"
IF_THRESHOLD_PATH = "models/final/if_threshold.npy"

# Data paths
RAW_DATA_ROOT = "data/raw/NASA_SMAP/data/data"
PROCESSED_DIR = "data/processed"
REPORT_DIR = "report"
DEFAULT_CHANNEL = "P-1"
DEMO_2 = "E-1"
DEMO_3 = "S-1"
Y_TRUE_PATH = "data/processed/y_true.npy"
Y_PRED_PATH = "data/processed/y_pred.npy"
RESIDUALS_PATH = "data/processed/residuals.npy"
RESIDUAL_FEATURES_PATH = "data/processed/residual_features.npy"
VAL_LABELS_PATH = "data/processed/y_val.npy"
VAL_FEATURES_PATH = "data/processed/residual_features_val.npy"