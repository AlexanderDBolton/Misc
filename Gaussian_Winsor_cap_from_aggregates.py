import numpy as np
import pandas as pd
from scipy.stats import norm

def robust_lognormal_cap(
    df,
    trim_frac=0.10,      # fraction of highest-variance days to trim
    percentile=0.995     # desired cap percentile (e.g. 0.99, 0.995, 0.999)
):
    """
    Estimate a robust Winsor cap assuming lognormal baseline,
    using only daily aggregates: n, sum_y, sum_yy.
    """

    # ---- Step 1: Compute daily mean and variance ----
    n = df["n"].to_numpy()
    sum_y = df["sum_y"].to_numpy()
    sum_yy = df["sum_yy"].to_numpy()

    m = sum_y / n
    v = sum_yy / n - m**2

    # Remove degenerate days
    mask = (m > 0) & (v > 0)
    m = m[mask]
    v = v[mask]
    n = n[mask]

    # ---- Step 2: Daily implied lognormal parameters ----
    cv2 = v / m**2

    sigma2_log = np.log(1 + cv2)
    mu_log = np.log(m) - 0.5 * sigma2_log

    # ---- Step 3: Trim highest-variance days ----
    cutoff = np.quantile(sigma2_log, 1 - trim_frac)
    keep = sigma2_log <= cutoff

    sigma2_trim = sigma2_log[keep]
    mu_trim = mu_log[keep]
    n_trim = n[keep]

    # ---- Step 4: Robust aggregation (weighted median alternative possible) ----
    sigma2_hat = np.median(sigma2_trim)
    mu_hat = np.median(mu_trim)

    sigma_hat = np.sqrt(sigma2_hat)

    # ---- Step 5: Compute cap ----
    z = norm.ppf(percentile)
    cap = np.exp(mu_hat + z * sigma_hat)

    return {
        "cap": cap,
        "mu_log_hat": mu_hat,
        "sigma_log_hat": sigma_hat,
        "n_days_used": len(sigma2_trim),
        "n_days_total": len(df)
    }
