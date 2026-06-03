# Evaluation Metrics - Sprint 3

**Channel**: P-1
**LSTM Model**: `models/final/lstm.keras`
**Isolation Forest Model**: `models/final/isolation_forest.pkl`

## IF Hyperparameters

| Param | Value |
| --- | --- |
| n_estimators | 200 |
| contamination | 0.089 |
| max_features | 0.8 |
| threshold_strategy | pr_curve |

## Summary Classification Metrics (Positive Class = Anomaly (-1))

| Metric | Value |
| --- | --- |
| **Precision** | 0.5610 |
| **Recall** | 0.4288 |
| **F1-Score** | 0.4860 |

## Best Smoothing Parameters (Grid Search)

| Param | Value |
| --- | --- |
| vote_window | 21 |
| vote_ratio | 0.5 |
| merge_gap | 15 |
| min_cluster_len | 5 |

## Grid Search Top-10 Results

| vote_window | vote_ratio | merge_gap | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| 21 | 0.5 | 15 | 0.5610 | 0.4288 | 0.4860 |
| 21 | 0.5 | 10 | 0.5492 | 0.4088 | 0.4687 |
| 21 | 0.5 | 5 | 0.5443 | 0.4008 | 0.4617 |
| 21 | 0.4 | 5 | 0.3846 | 0.4594 | 0.4187 |
| 15 | 0.5 | 15 | 0.4397 | 0.4274 | 0.4335 |
| 21 | 0.4 | 10 | 0.3775 | 0.4594 | 0.4144 |
| 21 | 0.4 | 15 | 0.3718 | 0.4594 | 0.4110 |
| 15 | 0.5 | 10 | 0.4280 | 0.4075 | 0.4175 |
| 15 | 0.5 | 5 | 0.4348 | 0.3995 | 0.4164 |
| 11 | 0.5 | 15 | 0.3623 | 0.4274 | 0.3922 |

## Detailed Classification Report

```text
              precision    recall  f1-score   support

Anomaly (-1)       0.56      0.43      0.49       751
  Normal (1)       0.95      0.97      0.96      7654

    accuracy                           0.92      8405
   macro avg       0.75      0.70      0.72      8405
weighted avg       0.91      0.92      0.91      8405

```

## Confusion Matrix
```text
            Predicted Normal (1)    Predicted Anomaly (-1)
True Normal (1)       7402                    252                   
True Anomaly (-1)     429                     322                   
```
