# File to generate prediction intervals using single-run methods

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from mmur.simulators import LAlgorithm


def bstrp_cm(cm, n_draws=1, random_state=None):
    """Bootstrap a confusion matrix, equivalent to bootstrapping the test set and then computing the confusion matrices (given a deterministic predictor). To obtain a collection of confusion matrices approximating the confusion matrix' distribution.

    Parameters
    ----------
    cm : np.ndarray of shape (4,)
        Confusion matrix
    n_draws : int, optional
        number of bootstrap samples, by default 1
    random_state : int, optional
        Set to a value for reproducible results, by default None

    Returns
    -------
    np.ndarray of shape (n_draws, 4)
        Estimated distribution of confusion matrices
    """
    rng = np.random.default_rng(random_state)
    n = cm.sum()
    p = cm/n
    return rng.multinomial(n, p, size=n_draws)


def kfold_cm(X, y, model_class, model_kwargs={}, n_splits=5, random_state=None):
    """Performs k-fold cross validation and computes the confusion matrix for each split.

    Parameters
    ----------
    X : np.ndarray of shape (n, k)
        Array of feature values
    y : np.ndarray of shape (n)
        Contains true labels 0 or 1
    model_class : object
        sklearn class defining the model implemented, for instance LogisticRegression 
    model_kwargs : dict, optional
        contains the model parameters as keys and their values as dict values, by default {}
    n_splits : int, optional
        number of splits/folds to evaluate, by default 5
    random_state : _type_, optional
        _description_, by default None

    Returns
    -------
    np.ndarray of shape (n_splits,4)
        the confusion matrices corresponding to the splits
    """
    rng = np.random.default_rng(random_state)
    fold_seed = rng.integers(low=0, high=np.iinfo(
        np.uint32).max)  # TODO: seeding
    kf = KFold(n_splits, shuffle=True, random_state=fold_seed)

    cms = np.zeros((n_splits, 4), dtype=int)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = model_class(**model_kwargs).fit(X_train, y_train)
        cms[i, :] = confusion_matrix(y_test, clf.predict(X_test)).flatten()

    return cms


# TODO: Multinomial CM with Dirichlet prior, need MMU for this


def pred_iv(values, q_range=0.95):
    return lower, upper


if __name__ == '__main__':
    model = LogisticRegression
    X = np.random.uniform(size=(1000, 2))
    y = np.random.choice([0, 1], size=1000)
    kfold_cm(X, y, model)
