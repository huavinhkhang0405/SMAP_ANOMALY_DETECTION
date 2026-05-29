# Evaluation Metrics - Sprint 3

**Channel**: P-1
**LSTM Model**: `models/final/lstm.keras`
**Isolation Forest Model**: `models/final/isolation_forest.pkl`

## Summary Classification Metrics (Positive Class = Anomaly (-1))

| Metric | Value |
| --- | --- |
| **Precision** | 0.2585 |
| **Recall** | 0.1212 |
| **F1-Score** | 0.1650 |

## Detailed Classification Report

```text
              precision    recall  f1-score   support

Anomaly (-1)       0.26      0.12      0.17       751
  Normal (1)       0.92      0.97      0.94      7654

    accuracy                           0.89      8405
   macro avg       0.59      0.54      0.55      8405
weighted avg       0.86      0.89      0.87      8405

```

## Confusion Matrix
```text
               Predicted Normal (1)    Predicted Anomaly (-1)
True Normal (1)       7393                    261                   
True Anomaly (-1)     660                     91                    
```
