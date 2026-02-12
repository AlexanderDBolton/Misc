df["n"]          # n_d
df["ybar"]       # \bar{y}_d
df["ss_within"]  # sum_i (y_id - \bar{y}_d)^2

import numpy as np
from scipy.optimize import minimize

def mle_gaussian_gaussian(df):

    n = df["n"].to_numpy()
    ybar = df["ybar"].to_numpy()
    ss_within = df["ss_within"].to_numpy()
    D = len(df)

    def neg_profile_loglik(theta):
        # theta = (log_tau, log_eta) - using logs for numerical stability
        log_tau, log_eta = theta
        tau = np.exp(log_tau)
        eta = np.exp(log_eta)

        # weights for profiled mu
        w = (eta * tau * n) / (tau + n * eta)

        mu_hat = np.sum(w * ybar) / np.sum(w)

        # log determinant term
        log_det_term = -0.5 * np.sum(np.log(1 + (eta / tau) * n))

        # within-day quadratic
        within_term = -0.5 * eta * np.sum(ss_within)

        # between-day quadratic
        between_term = -0.5 * np.sum(
            (eta * tau * n / (tau + n * eta)) * (ybar - mu_hat) ** 2
        )

        # constant terms
        const_term = (
            -0.5 * np.sum(n) * np.log(2 * np.pi)
            + 0.5 * np.sum(n) * np.log(eta)
        )

        loglik = const_term + log_det_term + within_term + between_term

        return -loglik  # minimize negative log likelihood

    # reasonable starting values
    # init = np.log([1.0, 1.0])
    tau_init = 1.0 / np.var(ybar, ddof=1)
    eta_init = 1.0 / (np.sum(ss_within) / np.sum(n - 1))
    init = np.log([tau_init, eta_init])

    result = minimize(
        neg_profile_loglik,
        init,
        method="L-BFGS-B"
    )

    log_tau_hat, log_eta_hat = result.x
    tau_hat = np.exp(log_tau_hat)
    eta_hat = np.exp(log_eta_hat)

    # compute final mu_hat
    w = (eta_hat * tau_hat * n) / (tau_hat + n * eta_hat)
    mu_hat = np.sum(w * ybar) / np.sum(w)

    return {
        "mu": mu_hat,
        "tau": tau_hat,
        "eta": eta_hat,
        "converged": result.success,
        "opt_result": result
    }
