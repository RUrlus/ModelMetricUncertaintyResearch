import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import erfinv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LogisticGenerator:
    """Generates binary classification models based on LogisticRegression.

    Parameters
    ----------

    We use the same X; y for the test set

    """

    def __init__(
        self,
        betas=None,
        noise_sigma=1.0,
    ):
        """Initialise the class."""
        # statefull random number generator
        self.betas = betas or np.array((0.5, 1.2))
        self.noise_sigma = noise_sigma

    def fit_transform(
        self,
        train_samples=10000,
        test_samples=10000,
        holdout_samples=10000,
        n_sets=10000,
        betas=None,
        noise_sigma=None,
        enable_noise=False,
        random_state=None,
    ):
        """Generate binary classification models."""
        betas = betas or self.betas
        noise_sigma = noise_sigma or self.noise_sigma

        # ensure consistent behaviour for samples
        self._gen = np.random.default_rng(random_state)

        # train samples
        n_train = train_samples
        n_test = test_samples
        n_holdout = holdout_samples
        # total samples
        t_samples = n_train + n_test + n_holdout * n_sets
        # holdout cutoff
        test_idx = n_train + n_test
        # allocate memory
        X = np.ones((t_samples, 2))
        # create random uniform draws
        X[:, 1] = self._gen.uniform(-10., 10.1, size=t_samples)

        # -- X sets --
        X_train = X[:n_train, :]
        X_test = X[n_train:test_idx, :]
        X_holdout = X[test_idx:, :]

        # compute linear estimate based on true coefficients
        linear_estimates = X.dot(betas)

        # the ground truth probability
        gt_proba = expit(linear_estimates)
        gt_proba_train = gt_proba[:n_train].copy()
        gt_proba_test = gt_proba[n_train:test_idx].copy()
        gt_proba_holdout = gt_proba[test_idx:].copy()
        del gt_proba

        # add gaussian noise to the samples
        if enable_noise:
            self._with_noise_sigma = noise_sigma
            linear_estimates += self._gen.normal(0, noise_sigma, t_samples)
        else:
            self._with_noise_sigma = None

        # compute probabilities
        proba = expit(linear_estimates)

        # -- y sets --
        # sample labels based on proba
        y = self._gen.binomial(1, proba)
        # train set
        y_train = y[:n_train].copy()[:, None]
        # test set
        y_test = y[n_train:test_idx].copy()[:, None]
        # hold out sets
        y_holdout = y[test_idx:].reshape(n_holdout, n_sets).copy()
        # free memory
        del y

        # train the LogisticRegression
        model = LogisticRegression(penalty='none')
        self.fit_ = model.fit(X_train, y_train.flatten())
        proba_train = self.fit_.predict_proba(X_train)[:, 1][:, None]
        proba_test = self.fit_.predict_proba(X_test)[:, 1][:, None]
        if n_holdout > 0:
            proba_holdout = (
                self.fit_
                .predict_proba(X_holdout)
                [:, 1]
                .reshape(n_holdout, n_sets)
            )
        else:
            proba_holdout = np.empty(0)
        return {
            'ground_truth' : {
                'train': gt_proba_train,
                'test': gt_proba_test,
                'holdout': gt_proba_holdout
            },
            'train': {'y': y_train, 'proba': proba_train, 'X': X_train},
            'test': {'y': y_test, 'proba': proba_test, 'X': X_test},
            'holdout': {'y': y_holdout, 'proba': proba_holdout, 'X': X_holdout},
        }
