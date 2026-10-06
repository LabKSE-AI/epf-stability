"""Point-forecast errors over the 48-hour horizon of one origin."""
import numpy as np


def metrics48(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE48, MAE48 and sMAPE48 [%].

    sMAPE uses the denominator |y| + |ŷ|; hours where both are exactly zero contribute 0.
    sMAPE is unstable when prices are close to zero or negative, therefore scale-free
    comparisons in the paper rely on error ratios to a benchmark (rMAE, rRMSE) instead.
    """
    e = y_pred - y_true
    denom = np.abs(y_true) + np.abs(y_pred)
    return {"rmse48": float(np.sqrt(np.mean(e ** 2))),
            "mae48": float(np.mean(np.abs(e))),
            "smape48": float(np.mean(np.where(denom > 0, 2 * np.abs(e) / np.where(denom > 0, denom, 1), 0.0)) * 100)}
