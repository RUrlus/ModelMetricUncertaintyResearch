import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import erfinv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import namedtuple


class LogisticGenerator:
    """Generates LogisticRegression model with holdout sets.

    The data is generated from a Logistic process

    X ~ Uniform(-10, 10) with an intercept
    L = X.dot(betas)
    L_noisy = L + e where e ~ Normal(0, noise_sigma)
    where e = 0 when noise_sigma is None
    P = sigmoid(L_noisy)
    y ~ Bernoulli(P)

    """

    def __init__(self):
        """Initialise the class."""
        self._betas = np.array((0.5, 1.2))

    def fit_transform(
        self,
        train_samples=10000,
        test_samples=10000,
        holdout_samples=10000,
        n_sets=10000,
        betas=None,
        noise_sigma=None,
        random_state=None,
        n_jobs=-1,
    ):
        """Generate observations from a Logistic process and fit and predict
        a LogisticRegression model on it.

        The data is generated from a Logistic process

        X ~ Uniform(-10, 10) with an intercept
        L = X.dot(betas)
        L_noisy = L + e where e ~ Normal(0, noise_sigma)
        where e = 0 when noise_sigma is None
        P = sigmoid(L_noisy)
        y ~ Bernoulli(P)

        Parameters
        ----------
        train_samples : int, default=10000
            number of observations in the training set used to train
            a LogisticRegression
        test_samples : int, default=10000
            number of observations in the test set for which to run
            predictions using the trained LogisticRegression
        holdout_samples : int, default=10000
            number of observations in the holdout sets
        n_sets : int, default=10000
            number of holdout sets each containing ``holdout_samples``,
            ``holdout['y']`` will be of shape (holdout_samples, n_sets)
        betas : tuple[float], default=None
            Beta coefficients used to generate the data
        noise_sigma : float, default=None
            std. dev of the Normally distributed noise added to the linear
            estimates
        random_state : int, np.random.default_rng, default=None
            seed for the random state
        n_jobs : int, default=-1
            number of processes used to fit the LogisticRegression

        Returns
        -------
        dict[str, dict]
            returns a dictionary containing four dictionaries:
                * 'ground_truth' which contains the ground truth probabilities
                for train, test and holdout.
                * 'train' which contains the labels `y`, probabilities `proba`
                and `X`
                * 'test' which contains the labels `y`, probabilities `proba`
                and `X`
                * 'holdout' which contains the labels `y`, probabilities `proba`
                and `X`

        """
        betas = betas or self._betas

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
        gt_proba_train = expit(linear_estimates[:n_train])
        gt_proba_test = expit(linear_estimates[n_train:test_idx])
        gt_proba_holdout = expit(linear_estimates[test_idx:])

        # add gaussian noise to the samples
        if noise_sigma is not None:
            linear_estimates += self._gen.normal(0, noise_sigma, t_samples)

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
        model = LogisticRegression(penalty='none', n_jobs=-1)
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
