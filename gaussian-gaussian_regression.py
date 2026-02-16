import numpy as np
import pandas as pd
from scipy.optimize import minimize

def did_random_intercept_mle(df):

    n = df["n"].to_numpy()
    sum_y = df["sum_y"].to_numpy()
    sum_yy = df["sum_yy"].to_numpy()

    ybar = sum_y / n
    ss_within = sum_yy - (sum_y ** 2) / n

    # Design matrix
    X = np.column_stack([
        np.ones(len(df)),
        df["group"].to_numpy(),
        df["post"].to_numpy(),
        (df["group"] * df["post"]).to_numpy()
    ])

    def neg_profile_loglik(theta):
        log_tau, log_eta = theta
        tau = np.exp(log_tau)
        eta = np.exp(log_eta)

        # weights
        w = (tau * eta * n) / (tau + eta * n)

        # Weighted least squares solution for beta
        WX = X * w[:, None]
        XtWX = X.T @ WX
        XtWy = X.T @ (w * ybar)
        beta_hat = np.linalg.solve(XtWX, XtWy)

        # residuals for daily means
        mean_resid = ybar - X @ beta_hat

        # between-day quadratic term
        between = -0.5 * np.sum(w * mean_resid**2)

        # within-day quadratic term
        within = -0.5 * eta * np.sum(ss_within)

        # log determinant term
        log_det = -0.5 * np.sum(np.log(1 + (eta / tau) * n))

        # constants
        const = (
            -0.5 * np.sum(n) * np.log(2 * np.pi)
            + 0.5 * np.sum(n) * np.log(eta)
        )

        loglik = const + log_det + within + between

        return -loglik

    # initial guess
    init = np.log([1.0, 1.0])

    result = minimize(
        neg_profile_loglik,
        init,
        method="L-BFGS-B"
    )

    # extract final estimates
    tau_hat = np.exp(result.x[0])
    eta_hat = np.exp(result.x[1])

    # compute final beta
    w = (tau_hat * eta_hat * n) / (tau_hat + eta_hat * n)
    WX = X * w[:, None]
    XtWX = X.T @ WX
    XtWy = X.T @ (w * ybar)
    beta_hat = np.linalg.solve(XtWX, XtWy)

    cov_beta = np.linalg.inv(XtWX)

    return {
        "beta": beta_hat,
        "cov_beta": cov_beta,
        "tau": tau_hat,
        "eta": eta_hat,
        "converged": result.success
    }

std errors
np.sqrt(np.diag(cov_beta))

