import numpy as np
from scipy.stats import norm

def did_threshold_from_pre(
    sigma_u,
    sigma_e,
    n_weekday_total,
    n_weekend_total,
    holdout_fraction,
    n_pre_weeks,
    n_post_weeks,
    alpha=0.05
):
    sigma_u2 = sigma_u**2
    sigma_e2 = sigma_e**2

    # --- Split traffic by group ---
    def split(n_total):
        n_holdout = n_total * holdout_fraction
        n_treat = n_total * (1 - holdout_fraction)
        return n_holdout, n_treat

    n_wd_h, n_wd_t = split(n_weekday_total)
    n_we_h, n_we_t = split(n_weekend_total)

    # --- Total number of days ---
    n_pre_days  = 7 * n_pre_weeks
    n_post_days = 7 * n_post_weeks
    n_days = n_pre_days + n_post_days

    # --- Construct day-level features ---
    # Weekend indicator (5 weekdays + 2 weekends repeated)
    weekend_pattern = np.array([0]*5 + [1]*2)
    weekend = np.tile(weekend_pattern, n_pre_weeks + n_post_weeks)

    # Post indicator
    post = np.concatenate([
        np.zeros(n_pre_days),
        np.ones(n_post_days)
    ])

    # Repeat each day twice (holdout + treatment)
    weekend = np.repeat(weekend, 2)
    post = np.repeat(post, 2)

    # Group indicator: 0,1,0,1,...
    group = np.tile([0,1], n_days)

    # Interaction
    interaction = group * post

    # --- Sample sizes per row ---
    n_day_total = np.where(
        weekend == 0,
        n_weekday_total,
        n_weekend_total
    )

    n_holdout = n_day_total * holdout_fraction
    n_treat = n_day_total * (1 - holdout_fraction)

    n = np.where(group == 0, n_holdout, n_treat)

    # --- Variance and weights ---
    V = sigma_u2 + sigma_e2 / n
    w = 1.0 / V

    # --- Design matrix ---
    X = np.column_stack([
        np.ones_like(group),   # intercept
        group,
        post,
        interaction,
        weekend
    ])

    # --- Compute GLS covariance ---
    W = np.diag(w)
    XtWX = X.T @ W @ X
    cov_beta = np.linalg.inv(XtWX)

    se_interaction = np.sqrt(cov_beta[3, 3])

    # One-sided threshold
    z = abs(norm.ppf(alpha))
    tau = z * se_interaction

    return tau, se_interaction


# Original version with appends
import numpy as np

def did_threshold_from_pre(
    sigma_u,
    sigma_e,
    n_weekday_total,
    n_weekend_total,
    holdout_fraction,
    n_pre_weeks,
    n_post_weeks,
    alpha=0.05
):
    """
    Compute design-based threshold tau for DiD interaction.

    Parameters
    ----------
    sigma_u : float
        Random day standard deviation (from pre regression).
    sigma_e : float
        Within-day standard deviation (from pre regression).
    n_weekday_total : float
        Average total observations per weekday (both groups combined).
    n_weekend_total : float
        Average total observations per weekend day.
    holdout_fraction : float
        Fraction of traffic in holdout group (e.g. 0.1 for 1:9 split).
    n_pre_weeks : int
    n_post_weeks : int
    alpha : float
        One-sided false discovery rate (default 0.05).

    Returns
    -------
    tau : float
        Threshold for declaring drop.
    se_interaction : float
        Design-based SE of interaction.
    """

    # --- Convert to variances ---
    sigma_u2 = sigma_u ** 2
    sigma_e2 = sigma_e ** 2

    # --- Group-level sample sizes ---
    def split_groups(n_total):
        n_holdout = n_total * holdout_fraction
        n_treat = n_total * (1 - holdout_fraction)
        return n_holdout, n_treat

    n_wd_h, n_wd_t = split_groups(n_weekday_total)
    n_we_h, n_we_t = split_groups(n_weekend_total)

    # --- Build design rows ---
    rows = []

    # Helper to add a day
    def add_day(is_post, is_weekend, n_h, n_t):
        # holdout
        rows.append((0, is_post, is_weekend, n_h))
        # treatment
        rows.append((1, is_post, is_weekend, n_t))

    # --- Pre-period ---
    for _ in range(n_pre_weeks):
        for _ in range(5):  # weekdays
            add_day(is_post=0, is_weekend=0, n_h=n_wd_h, n_t=n_wd_t)
        for _ in range(2):  # weekends
            add_day(is_post=0, is_weekend=1, n_h=n_we_h, n_t=n_we_t)

    # --- Post-period ---
    for _ in range(n_post_weeks):
        for _ in range(5):
            add_day(is_post=1, is_weekend=0, n_h=n_wd_h, n_t=n_wd_t)
        for _ in range(2):
            add_day(is_post=1, is_weekend=1, n_h=n_we_h, n_t=n_we_t)

    # --- Construct X and W ---
    X = []
    W = []

    for treat, post, weekend, n in rows:
        interaction = treat * post

        # columns: intercept, group, post, interaction, weekend
        X.append([1, treat, post, interaction, weekend])

        V = sigma_u2 + sigma_e2 / n
        W.append(1.0 / V)

    X = np.array(X)
    W = np.diag(W)

    # GLS covariance
    XtWX = X.T @ W @ X
    cov_beta = np.linalg.inv(XtWX)

    # Interaction is column 3
    var_interaction = cov_beta[3, 3]
    se_interaction = np.sqrt(var_interaction)

    # One-sided threshold
    z = abs(np.quantile(np.random.normal(size=1000000), alpha))  # approx Φ⁻¹(alpha)
    # but we know closed form:
    from scipy.stats import norm
    z = abs(norm.ppf(alpha))

    tau = z * se_interaction

    return tau, se_interaction
