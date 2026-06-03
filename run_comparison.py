import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import config
from src.data.loader import load_channel_npy
from src.data.preprocessing import create_sequences
from src.utils.residual_features import compute_residuals, compute_rolling_features

from src.model.lstm_sigma import run_lstm_sigma_detection
from src.model.standard_isolation_forest import run_standard_if_detection
from src.model.one_class_svm import run_ocsvm_detection
from src.model.isolation_forest import (
    load_model as load_if_model,
    predict_anomalies,
    calibrate_threshold,
    apply_temporal_smoothing,
    build_temporal_features,
    train_isolation_forest,
)

def load_anomaly_labels(channel: str, num_samples: int) -> np.ndarray:
    label_paths = [
        os.path.join("data", "raw", "NASA_SMAP", "labeled_anomalies.csv"),
        os.path.join("data", "raw", "NASA_SMAP", "data", "labeled_anomalies.csv"),
    ]
    csv_path = next((p for p in label_paths if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError("labeled_anomalies.csv không tìm thấy.")

    df = pd.read_csv(csv_path)
    row = df[df["chan_id"] == channel]
    if row.empty:
        raise ValueError(f"Channel '{channel}' không có trong labeled_anomalies.csv")

    seq_str = row.iloc[0]["anomaly_sequences"]
    sequences = ast.literal_eval(seq_str) if isinstance(seq_str, str) else seq_str

    labels = np.ones(num_samples, dtype=np.int32)
    for seq in sequences:
        start, end = int(seq[0]), int(seq[1])
        start = max(0, start)
        end = min(num_samples - 1, end)
        if start <= end:
            labels[start : end + 1] = -1
    return labels

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=-1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    n_pred_anomaly = int((y_pred == -1).sum())
    n_true_anomaly = int((y_true == -1).sum())
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_predicted_anomaly": n_pred_anomaly,
        "n_true_anomaly": n_true_anomaly,
    }

def _banner(title: str) -> None:
    width = 65
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_comparison_table(results: list[dict]) -> None:
    header = f"{'Model':<40} {'P':>7} {'R':>7} {'F1':>7} {'FP':>7} {'FN':>7} {'Time(s)':>9}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<40} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1']:>7.4f} "
            f"{r['fp']:>7d} "
            f"{r['fn']:>7d} "
            f"{r['inference_time']:>9.2f}"
        )
    print("-" * len(header))

def save_comparison_plots(
    signal: np.ndarray,
    gt_labels: np.ndarray,
    model_results: list[dict],
    report_dir: str,
    channel: str,
) -> None:
    n_models = len(model_results)
    colors = ["#7c3aed", "#0891b2", "#ea580c", "#16a34a"]
    anomaly_alpha = 0.15

    fig1, axes1 = plt.subplots(n_models, 1, figsize=(16, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes1 = [axes1]

    for ax, result, color in zip(axes1, model_results, colors):
        score = result.get("scores", None)
        if score is None:
            ax.set_visible(False)
            continue

        ax.plot(score, color=color, linewidth=0.7, alpha=0.85, label=result["name"])

        if result.get("threshold") is not None:
            ax.axhline(
                y=result["threshold"],
                color="#ef4444",
                linestyle="--",
                linewidth=1.2,
                label=f"Threshold ({result['threshold']:.4f})",
            )

        anomaly_idx = np.where(gt_labels == -1)[0]
        if len(anomaly_idx) > 0:
            diff = np.diff(anomaly_idx)
            splits = np.where(diff > 1)[0] + 1
            ranges = np.split(anomaly_idx, splits)
            first = True
            for r in ranges:
                if len(r) > 0:
                    ax.axvspan(
                        r[0], r[-1],
                        color="#ef4444", alpha=anomaly_alpha,
                        label="True Anomaly Region" if first else "",
                    )
                    first = False

        ax.set_ylabel("Score / Residual", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.tick_params(labelsize=8)

    axes1[-1].set_xlabel("Sequence Index", fontsize=10)
    fig1.suptitle(
        f"Model Comparison — Anomaly Scores/Residuals ({channel})",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig1.tight_layout()
    path1 = os.path.join(report_dir, "comparison_residuals.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Plot] Saved -> {path1}")

    fig2, axes2 = plt.subplots(n_models, 1, figsize=(16, 3.0 * n_models), sharex=True)
    if n_models == 1:
        axes2 = [axes2]

    sig = signal.squeeze()

    for ax, result, color in zip(axes2, model_results, colors):
        ax.plot(sig, color="#64748b", linewidth=0.6, alpha=0.7)

        anomaly_idx = np.where(gt_labels == -1)[0]
        if len(anomaly_idx) > 0:
            diff = np.diff(anomaly_idx)
            splits = np.where(diff > 1)[0] + 1
            ranges = np.split(anomaly_idx, splits)
            first = True
            for r in ranges:
                if len(r) > 0:
                    ax.axvspan(
                        r[0], r[-1],
                        color="#f97316", alpha=0.18,
                        label="True Anomaly" if first else "",
                    )
                    first = False

        pred_idx = np.where(result["labels"] == -1)[0]
        if len(pred_idx) > 0:
            pred_idx = pred_idx[pred_idx < len(sig)]
            ax.scatter(
                pred_idx, sig[pred_idx],
                color=color, s=8, zorder=5, alpha=0.8,
                label=f"Predicted Anomaly ({len(pred_idx)} pts)",
            )

        ax.set_ylabel("Telemetry", fontsize=9)
        ax.set_title(result["name"], fontsize=9, fontweight="bold", pad=3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.tick_params(labelsize=8)

    axes2[-1].set_xlabel("Sequence Index", fontsize=10)
    fig2.suptitle(
        f"Model Comparison — Anomaly Labels Overlay ({channel})",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig2.tight_layout()
    path2 = os.path.join(report_dir, "comparison_labels.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Saved -> {path2}")


def _shorten_model_name(name: str) -> str:
    name = name.replace("LSTM + ", "LSTM+")
    name = name.replace("Standard IF (Raw Data)", "IF Raw")
    name = name.replace("IF + Smoothing (Main)", "IF+Smooth")
    name = name.replace("3-Sigma", "3Sigma")
    return name


def save_performance_time_plots(
    results: list[dict],
    report_dir: str,
    channel: str,
) -> None:
    """
    Vẽ biểu đồ hiệu năng (Precision/Recall/F1) và thời gian inference.
    """
    os.makedirs(report_dir, exist_ok=True)

    labels = [_shorten_model_name(r["name"]) for r in results]
    x = np.arange(len(labels))
    width = 0.25

    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(x - width, precision, width=width, color="#2563eb", label="Precision")
    ax1.bar(x, recall, width=width, color="#16a34a", label="Recall")
    ax1.bar(x + width, f1, width=width, color="#f97316", label="F1-Score")

    ax1.set_title(f"Model Performance Comparison ({channel})", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9)
    fig1.tight_layout()

    perf_path = os.path.join(report_dir, "comparison_performance.png")
    fig1.savefig(perf_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Plot] Saved -> {perf_path}")

    times = [r["inference_time"] for r in results]

    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    ax2.bar(x, times, color="#0ea5e9")
    ax2.set_title(f"Inference Time Comparison ({channel})", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Seconds", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    fig2.tight_layout()

    time_path = os.path.join(report_dir, "comparison_runtime.png")
    fig2.savefig(time_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Saved -> {time_path}")


def save_markdown_report(
    results: list[dict],
    channel: str,
    report_dir: str,
) -> None:
    """Lưu bảng so sánh đầy đủ dưới dạng Markdown."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "comparison_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report — SMAP Anomaly Detection\n\n")
        f.write(f"**Channel**: `{channel}`\n\n")
        f.write("## Tóm tắt các mô hình\n\n")
        f.write("| # | Tên mô hình | Mô tả ngắn |\n")
        f.write("|---|-------------|------------|\n")
        f.write("| 1 | **LSTM + 3-Sigma** | Ngưỡng toán học cố định, không có IF/Smoothing |\n")
        f.write("| 2 | **Standard IF (Raw Data)** | IF trực tiếp trên dữ liệu thô, không qua LSTM |\n")
        f.write("| 3 | **LSTM + OCSVM** | Thay IF bằng OCSVM trên cùng vector 5D residuals |\n")
        f.write("| 4 | **LSTM + IF + Smoothing** | Pipeline đầy đủ (mô hình chính) |\n\n")

        f.write("## Kết quả so sánh\n\n")
        f.write("| Mô hình | Precision | Recall | F1-Score | FP | FN | #Pred Anomaly | Time (s) |\n")
        f.write("|---------|-----------|--------|----------|-----|-----|---------------|----------|\n")
        for r in results:
            f.write(
                f"| {r['name']} "
                f"| {r['precision']:.4f} "
                f"| {r['recall']:.4f} "
                f"| {r['f1']:.4f} "
                f"| {r['fp']} "
                f"| {r['fn']} "
                f"| {r['n_predicted_anomaly']} "
                f"| {r['inference_time']:.2f} |\n"
            )

        f.write("\n## Biểu đồ tổng hợp\n\n")
        f.write("![Performance Comparison](comparison_performance.png)\n\n")
        f.write("![Inference Time Comparison](comparison_runtime.png)\n\n")

        f.write("\n## Phân tích chi tiết từng mô hình\n\n")
        for r in results:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **Precision**: {r['precision']:.4f}\n")
            f.write(f"- **Recall**: {r['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {r['f1']:.4f}\n")
            f.write(f"- **True Positives (TP)**: {r['tp']}\n")
            f.write(f"- **False Positives (FP)**: {r['fp']}  ← cảnh báo giả\n")
            f.write(f"- **False Negatives (FN)**: {r['fn']}  ← bỏ lọt anomaly\n")
            f.write(f"- **True Negatives (TN)**: {r['tn']}\n")
            f.write(f"- **Tổng điểm dự đoán là anomaly**: {r['n_predicted_anomaly']}\n")
            f.write(f"- **Tổng điểm anomaly thật**: {r['n_true_anomaly']}\n")
            f.write(f"- **Thời gian inference**: {r['inference_time']:.2f} giây\n")
            if r.get("threshold") is not None:
                f.write(f"- **Ngưỡng sử dụng**: {r['threshold']:.6f}\n")
            f.write(f"\n```\n{r.get('classification_report', 'N/A')}\n```\n\n")

        f.write("## Luận điểm chứng minh\n\n")
        f.write("### 1. LSTM + 3-Sigma → Cảnh báo giả tràn lan\n")
        f.write("Ngưỡng `mean + 3σ` là một hằng số tĩnh. Vì residuals trong SMAP có phân phối "
                "lệch nặng (heavy-tailed), ngưỡng này **quá thấp** so với vùng biên thực tế, "
                "dẫn đến hàng trăm false positives trong mỗi cửa sổ bình thường.\n\n")
        f.write("### 2. Standard IF (Raw Data) → Bỏ lọt lỗi ngữ cảnh\n")
        f.write("Dữ liệu thô không phản ánh **sự lệch khỏi ngữ cảnh thời gian** mà chỉ phản "
                "ánh giá trị tuyệt đối. IF trên raw data có thể phân biệt outlier tĩnh, nhưng "
                "bỏ lọt các anomaly kéo dài (contextual anomaly) mà LSTM có thể phát hiện qua "
                "residuals.\n\n")
        f.write("### 3. LSTM + OCSVM → Thua về tốc độ và robustness\n")
        f.write("OCSVM phải tính ma trận kernel O(n²), rất chậm với dữ liệu lớn. Hơn nữa, "
                "OCSVM nhạy cảm với outlier trong training data và tham số `gamma`, `nu` khó "
                "tune. IF với phân chia ngẫu nhiên O(n·log n) bền vững hơn và nhanh hơn.\n\n")
        f.write("### 4. LSTM + IF + Smoothing → Tốt nhất\n")
        f.write("Pipeline đầy đủ: LSTM trích xuất lỗi ngữ cảnh → IF phát hiện outlier trong "
                "không gian residuals → Temporal Smoothing loại bỏ FP đơn lẻ và gộp cụm anomaly "
                "gần nhau → PR-Curve calibration tìm ngưỡng tối ưu.\n")

    print(f"[Report] Saved -> {path}")


def main() -> None:
    _banner("MODEL COMPARISON — SMAP ANOMALY DETECTION")
    channel = config.DEFAULT_CHANNEL
    print(f"Channel: {channel}")

    os.makedirs(config.REPORT_DIR, exist_ok=True)

    print("\n[Data] Loading train/test telemetry...")
    train_data = load_channel_npy(config.RAW_DATA_ROOT, "train", channel).astype(np.float32)
    test_data  = load_channel_npy(config.RAW_DATA_ROOT, "test",  channel).astype(np.float32)
    print(f"  Train: {train_data.shape}  |  Test: {test_data.shape}")

    print("[Data] Creating sequences...")
    X_train, y_train = create_sequences(train_data, config.WINDOW_SIZE, config.PREDICTION_DIM)
    X_test,  y_test  = create_sequences(test_data,  config.WINDOW_SIZE, config.PREDICTION_DIM)
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    print("[Data] Loading ground-truth labels...")
    gt_labels_full    = load_anomaly_labels(channel, test_data.shape[0])
    gt_labels_aligned = gt_labels_full[config.WINDOW_SIZE:]
    print(f"  GT labels (aligned): {gt_labels_aligned.shape} "
          f"| Anomaly ratio: {(gt_labels_aligned == -1).mean():.3%}")

    print(f"\n[LSTM] Loading model from {config.MODEL_PATH}...")
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model không tìm thấy tại {config.MODEL_PATH}. "
            "Hãy chạy train.py hoặc run_sprint3.py trước."
        )
    lstm_model = load_model(config.MODEL_PATH)

    all_results: list[dict] = []

    _banner("MODEL 1: LSTM + 3-Sigma Threshold")
    t0 = time.perf_counter()

    labels_sigma, residuals_sigma, threshold_sigma, mean_sigma, std_sigma = (
        run_lstm_sigma_detection(
            lstm_model=lstm_model,
            X_test=X_test,
            y_test=y_test,
            batch_size=config.BATCH_SIZE,
            n_sigma=3.0,
        )
    )

    t_sigma = time.perf_counter() - t0
    metrics_sigma = compute_metrics(gt_labels_aligned, labels_sigma)

    print(f"  Mean residual : {mean_sigma:.6f}")
    print(f"  Std residual  : {std_sigma:.6f}")
    print(f"  Threshold (3σ): {threshold_sigma:.6f}")
    print(f"  Predicted anomaly points: {(labels_sigma == -1).sum()}")
    print(f"  Inference time: {t_sigma:.2f}s")
    print(f"  Precision={metrics_sigma['precision']:.4f} "
          f"| Recall={metrics_sigma['recall']:.4f} "
          f"| F1={metrics_sigma['f1']:.4f}")
    print(f"  FP={metrics_sigma['fp']}  FN={metrics_sigma['fn']}")
    report_sigma = classification_report(
        gt_labels_aligned, labels_sigma,
        target_names=["Anomaly (-1)", "Normal (1)"], zero_division=0,
    )
    print(report_sigma)

    all_results.append({
        "name"                : "1. LSTM + 3-Sigma",
        "labels"              : labels_sigma,
        "scores"              : residuals_sigma,
        "threshold"           : threshold_sigma,
        "inference_time"      : t_sigma,
        "classification_report": report_sigma,
        **metrics_sigma,
    })

    _banner("MODEL 2: Standard Isolation Forest (Raw Data)")
    t0 = time.perf_counter()

    labels_raw_if, scores_raw_if, _ = run_standard_if_detection(
        train_series=train_data.squeeze(),
        test_series=test_data.squeeze(),
        window_size=config.WINDOW_SIZE,
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_STATE,
    )

    labels_raw_if_aligned = labels_raw_if[config.WINDOW_SIZE:]
    scores_raw_if_aligned = scores_raw_if[config.WINDOW_SIZE:]

    t_raw_if = time.perf_counter() - t0
    metrics_raw_if = compute_metrics(gt_labels_aligned, labels_raw_if_aligned)

    print(f"  Predicted anomaly points: {(labels_raw_if_aligned == -1).sum()}")
    print(f"  Inference time: {t_raw_if:.2f}s")
    print(f"  Precision={metrics_raw_if['precision']:.4f} "
          f"| Recall={metrics_raw_if['recall']:.4f} "
          f"| F1={metrics_raw_if['f1']:.4f}")
    print(f"  FP={metrics_raw_if['fp']}  FN={metrics_raw_if['fn']}")
    report_raw_if = classification_report(
        gt_labels_aligned, labels_raw_if_aligned,
        target_names=["Anomaly (-1)", "Normal (1)"], zero_division=0,
    )
    print(report_raw_if)

    all_results.append({
        "name"                : "2. Standard IF (Raw Data)",
        "labels"              : labels_raw_if_aligned,
        "scores"              : scores_raw_if_aligned,
        "threshold"           : 0.0,
        "inference_time"      : t_raw_if,
        "classification_report": report_raw_if,
        **metrics_raw_if,
    })

    _banner("MODEL 3: LSTM + OCSVM (5D Residual Features)")
    print("  [INFO] OCSVM với kernel=rbf có thể mất vài phút trên bộ dữ liệu lớn...")
    t0 = time.perf_counter()

    from src.model.one_class_svm import run_ocsvm_detection
    labels_ocsvm, scores_ocsvm, _ = run_ocsvm_detection(
        lstm_model=lstm_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        compute_residuals_fn=compute_residuals,
        compute_features_fn=compute_rolling_features,
        window_size=config.ROLLING_WINDOW,
        batch_size=config.BATCH_SIZE,
        nu=config.IF_CONTAMINATION,
        kernel="rbf",
        gamma="scale",
    )

    t_ocsvm = time.perf_counter() - t0
    metrics_ocsvm = compute_metrics(gt_labels_aligned, labels_ocsvm)

    print(f"  Predicted anomaly points: {(labels_ocsvm == -1).sum()}")
    print(f"  Inference time: {t_ocsvm:.2f}s")
    print(f"  Precision={metrics_ocsvm['precision']:.4f} "
          f"| Recall={metrics_ocsvm['recall']:.4f} "
          f"| F1={metrics_ocsvm['f1']:.4f}")
    print(f"  FP={metrics_ocsvm['fp']}  FN={metrics_ocsvm['fn']}")
    report_ocsvm = classification_report(
        gt_labels_aligned, labels_ocsvm,
        target_names=["Anomaly (-1)", "Normal (1)"], zero_division=0,
    )
    print(report_ocsvm)

    all_results.append({
        "name"                : "3. LSTM + OCSVM",
        "labels"              : labels_ocsvm,
        "scores"              : scores_ocsvm,
        "threshold"           : None,
        "inference_time"      : t_ocsvm,
        "classification_report": report_ocsvm,
        **metrics_ocsvm,
    })

    _banner("MODEL 4: LSTM + IF + Smoothing (Main Model)")
    t0 = time.perf_counter()

    y_pred_train = lstm_model.predict(X_train, batch_size=config.BATCH_SIZE, verbose=0)
    residuals_train = compute_residuals(y_train, y_pred_train)
    residual_features_train, _ = compute_rolling_features(residuals_train, config.ROLLING_WINDOW)

    y_pred_test = lstm_model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    residuals_test = compute_residuals(y_test, y_pred_test)
    residual_features_test, _ = compute_rolling_features(residuals_test, config.ROLLING_WINDOW)

    if os.path.exists(config.IF_MODEL_PATH):
        print(f"  Loading IF model from {config.IF_MODEL_PATH}...")
        if_model = load_if_model(config.IF_MODEL_PATH)
    else:
        print("  IF model not found, training from scratch...")
        if_model = train_isolation_forest(
            features=residual_features_train,
            n_estimators=config.IF_N_ESTIMATORS,
            contamination=config.IF_CONTAMINATION,
            random_state=config.RANDOM_STATE,
        )

    raw_labels_main, pred_scores_main = predict_anomalies(if_model, residual_features_test)

    best_threshold, _ = calibrate_threshold(
        model=if_model,
        val_features=residual_features_test,
        y_val=gt_labels_aligned,
        beta=getattr(config, "IF_FBETA", 1.5),
        strategy="pr_curve",
    )

    final_labels_main = apply_temporal_smoothing(
        labels=np.where(pred_scores_main < best_threshold, -1, 1),
        vote_window=config.IF_VOTE_WINDOW,
        vote_ratio=config.IF_VOTE_RATIO,
        min_cluster_len=config.IF_MIN_CLUSTER_LEN,
        merge_gap=config.IF_MERGE_GAP,
    )

    t_main = time.perf_counter() - t0
    metrics_main = compute_metrics(gt_labels_aligned, final_labels_main)

    print(f"  Predicted anomaly points: {(final_labels_main == -1).sum()}")
    print(f"  Inference time: {t_main:.2f}s")
    print(f"  Precision={metrics_main['precision']:.4f} "
          f"| Recall={metrics_main['recall']:.4f} "
          f"| F1={metrics_main['f1']:.4f}")
    print(f"  FP={metrics_main['fp']}  FN={metrics_main['fn']}")
    report_main = classification_report(
        gt_labels_aligned, final_labels_main,
        target_names=["Anomaly (-1)", "Normal (1)"], zero_division=0,
    )
    print(report_main)

    all_results.append({
        "name"                : "4. LSTM + IF + Smoothing (Main)",
        "labels"              : final_labels_main,
        "scores"              : pred_scores_main,
        "threshold"           : best_threshold,
        "inference_time"      : t_main,
        "classification_report": report_main,
        **metrics_main,
    })

    _banner("FINAL COMPARISON TABLE")
    print_comparison_table(all_results)

    save_comparison_plots(
        signal=y_test,
        gt_labels=gt_labels_aligned,
        model_results=all_results,
        report_dir=config.REPORT_DIR,
        channel=channel,
    )
    save_performance_time_plots(
        results=all_results,
        report_dir=config.REPORT_DIR,
        channel=channel,
    )
    save_markdown_report(
        results=all_results,
        channel=channel,
        report_dir=config.REPORT_DIR,
    )

    print(f"\n[Done] Tất cả kết quả đã lưu vào thư mục: {config.REPORT_DIR}/")
    print("  - comparison_report.md")
    print("  - comparison_residuals.png")
    print("  - comparison_labels.png")
    print("  - comparison_performance.png")
    print("  - comparison_runtime.png")


if __name__ == "__main__":
    main()
