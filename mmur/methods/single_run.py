# File to generate prediction intervals using single-run methods

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import mmu

def bstrp_cm(cm, n_draws=1000, random_state=None):
    """Bootstrap a confusion matrix, equivalent to bootstrapping the test set and then computing the confusion matrices (given a deterministic predictor). To obtain a collection of confusion matrices approximating the confusion matrix' distribution.

    Parameters
    ----------
    cm : np.ndarray of shape (4,) or (N,4)
        Confusion matrix or array of `N` confusion matrices
    n_draws : int, optional
        Number of bootstrap samples, by default 1000
    random_state : int, optional
        Set to a value for reproducible results, by default None
+-------------  
    Returns
    -------
    np.ndarray of shape (n_draws, 4)
        Estimated distribution of confusion matrices
    """
    rng = np.random.default_rng(random_state)

    # Given a single confusion matrix
    if len(cm.shape) == 1:
        n = cm.sum()
        p = cm/n
        return rng.multinomial(n, p, size=n_draws)

    # For multiple confusion matrices
    elif len(cm.shape) == 2:
        n = cm.sum(axis=1)
        p = cm/n.reshape(-1,1)
        cm_bs = np.swapaxes(rng.multinomial(n,p,size = (n_draws,cm.shape[0])),0,1)
        # Swap axes, such that the first axis corresponds to the number of input confusion matrices
        return cm_bs

    else:
        raise ValueError('Parameter ``cm`` is of wrong shape, should be of shape (4,) or (N,4)')

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

#TODO: K-fold cross val in parallel

def kfold_cv_cm(X, y, model_class, model_kwargs={}, n_splits=5, random_state=None):
    n_obs = len(y)
    if n_obs % n_splits != 0:
        raise ValueError('The number of observations is not divisible by the number of splits')

    rng = np.random.default_rng(random_state)
    fold_seed = rng.integers(low=0, high=np.iinfo(
        np.uint32).max)  # TODO: seeding
    kf = KFold(n_splits, shuffle=True, random_state=fold_seed)

    n_test_obs = int(len(y)/n_splits)
    y_hat = np.zeros((n_splits,n_test_obs),dtype=int)
    y_true = np.zeros((n_splits,n_test_obs),dtype=int)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = model_class(**model_kwargs).fit(X_train, y_train)

        y_hat[i,:] = clf.predict(X_test)
        y_true[i,:] = y_test

    return mmu.confusion_matrices(y_true,y_hat,obs_axis=0)


def pred_iv(values, q_range=0.95):
    return lower, upper


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    X = rng.normal(size = (100,2))
    y = rng.binomial(1,0.4,size=(100))
    cv_cm = kfold_cv_cm(X, y, LogisticRegression, model_kwargs={'penalty':'none'}, n_splits=5, random_state=None)
    print(cv_cm)