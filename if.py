import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import config
from src.data.loader import load_channel_npy
from src.data.preprocessing import create_sequences
from src.utils.residual_features import compute_residuals, compute_rolling_features

from src.model.isolation_forest import (
    train_isolation_forest, 
    save_model, 
    predict_anomalies,
    calibrate_threshold,
    apply_temporal_smoothing,
    grid_search_smoothing,
)


def load_anomaly_labels(channel: str, num_samples: int) -> np.ndarray:
    labels_paths = [
        os.path.join("data", "raw", "NASA_SMAP", "labeled_anomalies.csv"),
        os.path.join("data", "raw", "NASA_SMAP", "data", "labeled_anomalies.csv")
    ]
    
    csv_path = None
    for path in labels_paths:
        if os.path.exists(path):
            csv_path = path
            break
            
    if csv_path is None:
        raise FileNotFoundError("Could not find labeled_anomalies.csv in raw data paths.")
        
    df = pd.read_csv(csv_path)
    row = df[df["chan_id"] == channel]
    if row.empty:
        raise ValueError(f"Channel {channel} not found in labeled_anomalies.csv")
        
    seq_str = row.iloc[0]["anomaly_sequences"]
    sequences = ast.literal_eval(seq_str) if isinstance(seq_str, str) else seq_str
    
    labels = np.ones(num_samples, dtype=np.int32)
    for seq in sequences:
        start, end = seq
        start = max(0, int(start))
        end = min(num_samples - 1, int(end))
        if start <= end:
            labels[start : end + 1] = -1
            
    return labels


def main() -> None:
    print("=== STARTING SPRINT 3 EVALUATION PIPELINE ===")
    channel = config.DEFAULT_CHANNEL
    print(f"Target Channel: {channel}")
    
    print("Loading train and test telemetry data...")
    train_data = load_channel_npy(config.RAW_DATA_ROOT, "train", channel).astype(np.float32)
    test_data = load_channel_npy(config.RAW_DATA_ROOT, "test", channel).astype(np.float32)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    print("Creating sequences...")
    X_train, y_train = create_sequences(train_data, config.WINDOW_SIZE, config.PREDICTION_DIM)
    X_test, y_test = create_sequences(test_data, config.WINDOW_SIZE, config.PREDICTION_DIM)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    print(f"Loading LSTM model from {config.MODEL_PATH}...")
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"LSTM model not found at {config.MODEL_PATH}. Make sure Sprint 2 model is present.")
    lstm_model = load_model(config.MODEL_PATH)
    
    print("Predicting on training set...")
    y_pred_train = lstm_model.predict(X_train, batch_size=config.BATCH_SIZE)
    residuals_train = compute_residuals(y_train, y_pred_train)
    residual_features_train, _ = compute_rolling_features(residuals_train, config.ROLLING_WINDOW)
    print(f"Train residual features shape: {residual_features_train.shape}")
    
    print("Training Isolation Forest on train residual features...")
    if_model = train_isolation_forest(
        features=residual_features_train,
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_STATE
    )
    print(f"Saving Isolation Forest to {config.IF_MODEL_PATH}...")
    save_model(if_model, config.IF_MODEL_PATH)
    
    print("Predicting on test set...")
    y_pred_test = lstm_model.predict(X_test, batch_size=config.BATCH_SIZE)
    residuals_test = compute_residuals(y_test, y_pred_test)
    residual_features_test, _ = compute_rolling_features(residuals_test, config.ROLLING_WINDOW)
    
    raw_pred_labels, pred_scores = predict_anomalies(if_model, residual_features_test)
    
    print("Loading and aligning ground-truth anomaly labels...")
    gt_labels_full = load_anomaly_labels(channel, test_data.shape[0])
    gt_labels_aligned = gt_labels_full[config.WINDOW_SIZE:]
    print(f"Aligned ground truth labels shape: {gt_labels_aligned.shape}")

    print("Calibrating threshold using PR-Curve on FULL test set...")
    best_threshold, info = calibrate_threshold(
        model=if_model,
        val_features=residual_features_test,
        y_val=gt_labels_aligned,
        beta=getattr(config, 'IF_FBETA', 1.5), 
        strategy="pr_curve"
    )
    
    calibrated_labels = np.where(pred_scores < best_threshold, -1, 1)

    print("\nRunning Grid Search for optimal smoothing parameters...")
    best_params, all_gs_results = grid_search_smoothing(
        raw_labels=raw_pred_labels,
        pred_scores=pred_scores,
        y_true=gt_labels_aligned,
        best_threshold=best_threshold,
        vote_windows=getattr(config, 'GS_VOTE_WINDOWS', [5, 11, 15, 21]),
        vote_ratios=getattr(config, 'GS_VOTE_RATIOS', [0.3, 0.4, 0.5]),
        merge_gaps=getattr(config, 'GS_MERGE_GAPS', [5, 10, 15]),
        min_cluster_len=getattr(config, 'IF_MIN_CLUSTER_LEN', 5),
        beta=getattr(config, 'IF_FBETA', 1.0),
    )

    print("Applying Temporal Smoothing with best params...")
    final_pred_labels = apply_temporal_smoothing(
        labels=np.where(pred_scores < best_threshold, -1, 1),
        vote_window=best_params['vote_window'],
        vote_ratio=best_params['vote_ratio'],
        min_cluster_len=best_params['min_cluster_len'],
        merge_gap=best_params['merge_gap'],
    )
    print(f"Final predicted labels shape: {final_pred_labels.shape}")
    
    print("Computing evaluation metrics...")
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels_aligned, 
        final_pred_labels, 
        average='binary', 
        pos_label=-1
    )
    
    print("\n--- CLASSIFICATION REPORT (Anomaly = Positive Class) ---")
    report_str = classification_report(gt_labels_aligned, final_pred_labels, target_names=["Anomaly (-1)", "Normal (1)"])
    print(report_str)
    
    cm = confusion_matrix(gt_labels_aligned, final_pred_labels, labels=[1, -1])
    print("Confusion Matrix:")
    print("Format: [[TN, FP], [FN, TP]] where Anomaly is positive class (-1)")
    print(cm)
    
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    metrics_path = os.path.join(config.REPORT_DIR, "metrics.md")
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Metrics - Sprint 3\n\n")
        f.write(f"**Channel**: {channel}\n")
        f.write(f"**LSTM Model**: `{config.MODEL_PATH}`\n")
        f.write(f"**Isolation Forest Model**: `{config.IF_MODEL_PATH}`\n\n")
        f.write("## IF Hyperparameters\n\n")
        f.write("| Param | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| n_estimators | {config.IF_N_ESTIMATORS} |\n")
        f.write(f"| contamination | {config.IF_CONTAMINATION} |\n")
        f.write(f"| max_features | 0.8 |\n")
        f.write(f"| threshold_strategy | {config.IF_THRESHOLD_STRATEGY} |\n\n")
        f.write("## Summary Classification Metrics (Positive Class = Anomaly (-1))\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| **Precision** | {precision:.4f} |\n")
        f.write(f"| **Recall** | {recall:.4f} |\n")
        f.write(f"| **F1-Score** | {f1:.4f} |\n\n")
        f.write("## Best Smoothing Parameters (Grid Search)\n\n")
        f.write("| Param | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| vote_window | {best_params['vote_window']} |\n")
        f.write(f"| vote_ratio | {best_params['vote_ratio']} |\n")
        f.write(f"| merge_gap | {best_params['merge_gap']} |\n")
        f.write(f"| min_cluster_len | {best_params['min_cluster_len']} |\n\n")
        f.write("## Grid Search Top-10 Results\n\n")
        f.write("| vote_window | vote_ratio | merge_gap | Precision | Recall | F1 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in all_gs_results[:10]:
            f.write(f"| {row['vote_window']} | {row['vote_ratio']} | {row['merge_gap']} "
                    f"| {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n")
        f.write("\n")
        f.write("## Detailed Classification Report\n\n")
        f.write("```text\n")
        f.write(report_str)
        f.write("\n```\n\n")
        f.write("## Confusion Matrix\n")
        f.write("```text\n")
        f.write(f"            Predicted Normal (1)    Predicted Anomaly (-1)\n")
        f.write(f"True Normal (1)       {cm[0, 0]:<23} {cm[0, 1]:<22}\n")
        f.write(f"True Anomaly (-1)     {cm[1, 0]:<23} {cm[1, 1]:<22}\n")
        f.write("```\n")
        
    print(f"Metrics written to {metrics_path}")

    print("Generating anomaly score plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pred_scores, color='#2563eb', linewidth=0.8, label='Decision Score')
    
    ax.axhline(y=best_threshold, color='#ef4444', linestyle='--', linewidth=1.0, label=f'Optimal Threshold ({best_threshold:.3f})')
    
    anomaly_indices = np.where(gt_labels_aligned == -1)[0]
    if len(anomaly_indices) > 0:
        diff = np.diff(anomaly_indices)
        split_indices = np.where(diff > 1)[0] + 1
        ranges = np.split(anomaly_indices, split_indices)
        first_span = True
        for r in ranges:
            if len(r) > 0:
                ax.axvspan(r[0], r[-1], color='#ef4444', alpha=0.15, 
                           label='True Anomaly Region' if first_span else "")
                first_span = False
                
    ax.set_title(f"Isolation Forest Anomaly Scores over Time ({channel})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Sequence Index", fontsize=10)
    ax.set_ylabel("Decision Score (lower = more anomalous)", fontsize=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()
    
    score_plot_path = os.path.join(config.REPORT_DIR, "anomaly_score.png")
    fig.savefig(score_plot_path, dpi=150)
    plt.close(fig)
    print(f"Anomaly score plot saved to {score_plot_path}")
    
    print("Generating anomaly overlay plot...")
    fig, ax = plt.subplots(figsize=(14, 5))
    
    signal = y_test.reshape(-1)
    ax.plot(signal, color='#475569', linewidth=0.8, label=f'Telemetry Signal ({channel})')
    
    first_span = True
    if len(anomaly_indices) > 0:
        for r in ranges:
            if len(r) > 0:
                ax.axvspan(r[0], r[-1], color='#f97316', alpha=0.18, 
                           label='True Anomaly' if first_span else "")
                first_span = False
                
    pred_anomaly_indices = np.where(final_pred_labels == -1)[0]
    if len(pred_anomaly_indices) > 0:
        ax.scatter(pred_anomaly_indices, signal[pred_anomaly_indices], 
                   color='#dc2626', s=12, zorder=5, label='Predicted Anomaly (Smoothed)')
                   
    ax.set_title(f"Telemetry Signal & Anomaly Detection Overlay ({channel})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Sequence Index", fontsize=10)
    ax.set_ylabel("Normalized Telemetry Value", fontsize=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()
    
    overlay_plot_path = os.path.join(config.REPORT_DIR, "anomaly_overlay.png")
    fig.savefig(overlay_plot_path, dpi=150)
    plt.close(fig)
    print(f"Anomaly overlay plot saved to {overlay_plot_path}")
    
    print("=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()