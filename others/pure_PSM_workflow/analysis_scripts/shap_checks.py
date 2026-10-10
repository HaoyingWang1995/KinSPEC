"""Independent local-accuracy checks for freshly computed TreeSHAP batches."""

def checked_contributions(booster, matrix, checks):
    import numpy as np
    contributions = booster.predict(matrix, pred_contribs=True)
    margin = booster.predict(matrix, output_margin=True)
    if contributions.shape != (matrix.num_row(), matrix.num_col() + 1):
        raise ValueError("Unexpected SHAP dimensions or missing bias column")
    if not np.isfinite(contributions).all() or not np.isfinite(margin).all():
        raise ValueError("Non-finite SHAP values or raw predictions")
    total = contributions.sum(axis=1, dtype=np.float64)
    error = float(np.abs(total - margin).max())
    if not np.allclose(total, margin, rtol=1e-4, atol=5e-4):
        raise ValueError(f"TreeSHAP fails raw-margin additivity: max error {error:.3e}")
    checks["rows"] = checks.get("rows", 0) + matrix.num_row()
    checks["batches"] = checks.get("batches", 0) + 1
    checks["max_margin_error"] = max(checks.get("max_margin_error", 0), error)
    return contributions
