"""
isolation_forest.py
-------------------
Full pipeline cho Isolation Forest với các cải tiến:
  1. Temporal feature engineering (multi-scale rolling stats, diff, z-score)
  2. Calibrated threshold qua Precision-Recall curve trên validation set
  3. Post-processing temporal smoothing (majority vote + merge + filter)
"""

import os
from typing import Tuple, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve


def build_temporal_features(
    residuals: np.ndarray,
    rolling_windows: List[int] = [5, 10, 20, 50],
    use_diff: bool = True,
    use_zscore: bool = True,
) -> np.ndarray:
    """
    Xây dựng feature vector phong phú từ chuỗi residuals.

    Features được tạo ra:
      - rolling mean + rolling std tại mỗi window size trong rolling_windows
      - diff bậc 1, diff bậc 2                          (nếu use_diff=True)
      - global z-score                                   (nếu use_zscore=True)

    Args:
        residuals   : array (N,) hoặc (N, 1) — residuals từ LSTM
        rolling_windows : list các window size để tính rolling stats
        use_diff    : có thêm rate-of-change features không
        use_zscore  : có thêm global z-score không

    Returns:
        feature_matrix : array (N, n_features)
    """
    r = np.array(residuals).squeeze()
    N = len(r)
    feats: List[np.ndarray] = []

    for w in rolling_windows:
        kernel = np.ones(w) / w

        roll_mean = np.convolve(r, kernel, mode="same")

        roll_sq_mean = np.convolve(r ** 2, kernel, mode="same")
        roll_std = np.sqrt(np.maximum(roll_sq_mean - roll_mean ** 2, 0.0))

        feats.append(roll_mean)
        feats.append(roll_std)

    if use_diff:
        diff1 = np.concatenate([[0.0], np.diff(r)])
        diff2 = np.concatenate([[0.0, 0.0], np.diff(r, n=2)])
        feats.extend([diff1, diff2])

    if use_zscore:
        z = (r - r.mean()) / (r.std() + 1e-8)
        feats.append(z)

    feature_matrix = np.column_stack(feats)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix


def train_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.089,
    max_samples: str | int = "auto",
    max_features: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Fit một Isolation Forest Pipeline (StandardScaler → IsolationForest).

    Args:
        features      : (n_samples, n_features) — residual features sau khi qua
                        build_temporal_features()
        n_estimators  : số cây (200 ổn định hơn mặc định 100)
        contamination : tỉ lệ anomaly ước tính — nên đặt = actual ratio
        max_samples   : số samples mỗi cây lấy ("auto" = min(256, n))
        max_features  : tỉ lệ features mỗi cây sử dụng (0.8 = 80%, tăng
                        tính đa dạng và giảm correlation giữa các cây)
        random_state  : seed

    Returns:
        Fitted sklearn Pipeline (scaler + iforest)
    """
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    model_pipeline.fit(features)
    return model_pipeline


def save_model(model: Pipeline, path: str) -> None:
    """Lưu fitted pipeline vào disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"[IF] Model saved -> {path}")


def load_model(path: str) -> Pipeline:
    """Load pipeline từ disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def save_threshold(threshold: float, path: str) -> None:
    """Lưu calibrated threshold vào file .npy."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, np.array(threshold))
    print(f"[IF] Threshold saved -> {path}  (value={threshold:.6f})")


def load_threshold(path: str) -> float:
    """Load threshold từ file .npy."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}")
    return float(np.load(path))


def calibrate_threshold(
    model: Pipeline,
    val_features: np.ndarray,
    y_val: np.ndarray,
    beta: float = 1.0,
    strategy: str = "pr_curve",
    fixed_threshold: float = 0.0,
    percentile: float = 91.1,
) -> Tuple[float, dict]:
    """
    Tìm threshold tối ưu trên validation set theo một trong ba chiến lược:

      - "pr_curve"   : tối đa hoá F-beta score trên Precision-Recall curve
      - "fixed"      : dùng fixed_threshold cố định (mặc định 0.0)
      - "percentile" : cắt tại percentile thứ N của phân phối score

    Args:
        model         : fitted Pipeline (scaler + iforest)
        val_features  : (n_val, n_features) — features của validation set
        y_val         : (n_val,) — nhãn thật, -1=anomaly, 1=normal
        beta          : F-beta beta parameter (1.0 = F1)
        strategy      : "pr_curve" | "fixed" | "percentile"
        fixed_threshold : threshold khi strategy="fixed"
        percentile    : percentile khi strategy="percentile"

    Returns:
        threshold : float — giá trị decision_score để cắt
        info      : dict — precision, recall, f_beta tại threshold đó
    """
    scores = model.decision_function(val_features)
    y_binary = (y_val == -1).astype(int)

    if strategy == "fixed":
        threshold = fixed_threshold
        preds = (scores < threshold).astype(int)
        tp = int(((preds == 1) & (y_binary == 1)).sum())
        fp = int(((preds == 1) & (y_binary == 0)).sum())
        fn = int(((preds == 0) & (y_binary == 1)).sum())
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-8)
        info = {"precision": p, "recall": r, f"f{beta}": fb, "strategy": strategy}

    elif strategy == "percentile":
        threshold = float(np.percentile(scores, 100.0 - percentile))
        preds = (scores < threshold).astype(int)
        tp = int(((preds == 1) & (y_binary == 1)).sum())
        fp = int(((preds == 1) & (y_binary == 0)).sum())
        fn = int(((preds == 0) & (y_binary == 1)).sum())
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-8)
        info = {"precision": p, "recall": r, f"f{beta}": fb, "strategy": strategy}

    else:  # "pr_curve" — mặc định
        neg_scores = -scores
        precisions, recalls, thresholds_pr = precision_recall_curve(y_binary, neg_scores)

        f_betas = (
            (1 + beta**2) * precisions * recalls
            / (beta**2 * precisions + recalls + 1e-8)
        )

        best_idx = int(np.argmax(f_betas[:-1]))
        threshold = float(-thresholds_pr[best_idx])

        info = {
            "precision": float(precisions[best_idx]),
            "recall": float(recalls[best_idx]),
            f"f{beta}": float(f_betas[best_idx]),
            "strategy": strategy,
            "best_idx": best_idx,
            "n_thresholds_evaluated": len(thresholds_pr),
        }

    print(
        f"[IF] Calibrated threshold = {threshold:.6f}  "
        f"P={info['precision']:.3f}  R={info['recall']:.3f}  "
        f"F{beta}={info[f'f{beta}']:.3f}  [{strategy}]"
    )
    return threshold, info


def predict_anomalies(
    model: Pipeline,
    features: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dự đoán nhãn anomaly và trả về decision scores.

    Args:
        model     : fitted Pipeline
        features  : (n_samples, n_features)
        threshold : nếu None → dùng threshold mặc định của sklearn (0.0)
                    nếu có  → cắt thủ công theo calibrated threshold

    Returns:
        labels : (n_samples,)  — -1=anomaly, 1=normal
        scores : (n_samples,)  — decision score, thấp hơn = anomaly hơn
    """
    scores = model.decision_function(features)

    if threshold is None:
        labels = model.predict(features)
    else:
        labels = np.where(scores < threshold, -1, 1)

    return labels, scores


def _get_anomaly_clusters(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Trả về list các (start, end) — exclusive — của mỗi cụm anomaly liên tiếp.
    Ví dụ: labels=[1,1,-1,-1,-1,1,1] → [(2, 5)]
    """
    clusters: List[Tuple[int, int]] = []
    in_cluster = False
    start = 0

    for i, v in enumerate(labels):
        if v == -1 and not in_cluster:
            in_cluster, start = True, i
        elif v != -1 and in_cluster:
            clusters.append((start, i))
            in_cluster = False

    if in_cluster:
        clusters.append((start, len(labels)))

    return clusters


def apply_temporal_smoothing(
    labels: np.ndarray,
    vote_window: int = 15,
    vote_ratio: float = 0.40,
    min_cluster_len: int = 5,
    merge_gap: int = 10,
) -> np.ndarray:
    """
    Post-process raw IF labels qua 3 bước:

      Bước 1 — Majority vote sliding window:
        Điểm i được nhãn -1 nếu ≥ vote_ratio của cửa sổ vote_window là -1.
        Giúp loại bỏ false positive đơn lẻ và kéo dài vùng anomaly thật.

      Bước 2 — Merge nearby clusters:
        Hai cụm anomaly cách nhau ≤ merge_gap điểm được gộp thành 1 cụm.
        Giúp nối liền các cụm bị ngắt quãng bởi vài normal points.

      Bước 3 — Filter short clusters:
        Loại bỏ các cụm ngắn hơn min_cluster_len điểm.
        Giúp loại bỏ spike false positive còn sót sau bước 1.

    Args:
        labels          : (n_samples,) raw labels từ predict_anomalies()
        vote_window     : kích thước cửa sổ majority vote
        vote_ratio      : ngưỡng tỉ lệ để nhãn là anomaly
        min_cluster_len : độ dài tối thiểu của cụm anomaly
        merge_gap       : khoảng cách tối đa để gộp 2 cụm

    Returns:
        smoothed_labels : (n_samples,) labels sau post-processing
    """
    N = len(labels)
    smoothed = labels.copy().astype(int)
    half_w = vote_window // 2
    anomaly_flag = (labels == -1).astype(float)

    for i in range(N):
        start_i = max(0, i - half_w)
        end_i   = min(N, i + half_w + 1)
        window_ratio = anomaly_flag[start_i:end_i].mean()
        smoothed[i] = -1 if window_ratio >= vote_ratio else 1

    clusters = _get_anomaly_clusters(smoothed)
    for k in range(len(clusters) - 1):
        curr_end   = clusters[k][1]
        next_start = clusters[k + 1][0]
        gap = next_start - curr_end
        if gap <= merge_gap:
            smoothed[curr_end:next_start] = -1

    clusters = _get_anomaly_clusters(smoothed)
    for start_c, end_c in clusters:
        if (end_c - start_c) < min_cluster_len:
            smoothed[start_c:end_c] = 1

    return smoothed


def grid_search_smoothing(
    raw_labels: np.ndarray,
    pred_scores: np.ndarray,
    y_true: np.ndarray,
    best_threshold: float,
    vote_windows: List[int] = [5, 11, 15, 21],
    vote_ratios: List[float] = [0.3, 0.4, 0.5],
    merge_gaps: List[int] = [5, 10, 15],
    min_cluster_len: int = 5,
    beta: float = 1.0,
) -> Tuple[dict, List[dict]]:
    """
    Grid search tự động để tìm bộ tham số temporal smoothing tốt nhất.

    Quét qua tổ hợp vote_window × vote_ratio × merge_gap và chọn bộ
    tham số tối đa hoá F-beta score trên tập đánh giá.

    Args:
        raw_labels      : (n,) nhãn thô từ IF (trước khi smoothing)
        pred_scores     : (n,) decision scores từ IF
        y_true          : (n,) nhãn thật -1=anomaly, 1=normal
        best_threshold  : threshold đã calibrate để re-apply lên scores
        vote_windows    : list các giá trị vote_window cần thử
        vote_ratios     : list các giá trị vote_ratio cần thử
        merge_gaps      : list các giá trị merge_gap cần thử
        min_cluster_len : độ dài tối thiểu cluster (giữ cố định)
        beta            : F-beta beta (1.0 = F1)

    Returns:
        best_params : dict chứa bộ tham số tốt nhất
        all_results : list toàn bộ kết quả, sắp xếp theo f_score giảm dần
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Dùng calibrated_labels (đã áp threshold) làm đầu vào smoothing
    calibrated_labels = np.where(pred_scores < best_threshold, -1, 1)

    all_results: List[dict] = []
    total_combos = len(vote_windows) * len(vote_ratios) * len(merge_gaps)
    print(f"[GridSearch] Quét {total_combos} tổ hợp tham số smoothing...")

    for vw in vote_windows:
        for vr in vote_ratios:
            for mg in merge_gaps:
                smoothed = apply_temporal_smoothing(
                    labels=calibrated_labels,
                    vote_window=vw,
                    vote_ratio=vr,
                    min_cluster_len=min_cluster_len,
                    merge_gap=mg,
                )
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true,
                    smoothed,
                    average="binary",
                    pos_label=-1,
                    zero_division=0,
                )
                fb = (
                    (1 + beta**2) * precision * recall
                    / (beta**2 * precision + recall + 1e-8)
                )
                all_results.append({
                    "vote_window": vw,
                    "vote_ratio": vr,
                    "merge_gap": mg,
                    "min_cluster_len": min_cluster_len,
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    f"f{beta}": round(float(fb), 4),
                })

    # Sắp xếp theo F-beta giảm dần
    all_results.sort(key=lambda x: x[f"f{beta}"], reverse=True)

    best_params = all_results[0]

    print(f"[GridSearch] [OK] Bo tham so tot nhat:")
    print(f"  vote_window={best_params['vote_window']}, "
          f"vote_ratio={best_params['vote_ratio']}, "
          f"merge_gap={best_params['merge_gap']}")
    print(f"  => Precision={best_params['precision']:.4f}  "
          f"Recall={best_params['recall']:.4f}  "
          f"F{beta}={best_params[f'f{beta}']:.4f}")

    return best_params, all_results