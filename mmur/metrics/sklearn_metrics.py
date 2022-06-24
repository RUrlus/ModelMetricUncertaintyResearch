"""Fix sklearn precision_recall_curve()"""
import numpy as np
from sklearn.utils.multiclass import type_of_target

from sklearn.utils import assert_all_finite
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize
from sklearn.utils._encode import _encode, _unique
from sklearn.metrics._base import (
    _average_binary_score,
    _average_multiclass_ovo_score,
    _check_pos_label_consistency,
)
from sklearn.metrics._ranking import _binary_clf_curve


def precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    probas_pred : ndarray of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    precision : ndarray of shape (n_thresholds + 1,)
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : ndarray of shape (n_thresholds + 1,)
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : ndarray of shape (n_thresholds,)
        Increasing thresholds on the decision function used to compute
        precision and recall. n_thresholds <= len(np.unique(probas_pred)).

    See Also
    --------
    PrecisionRecallDisplay.from_estimator : Plot Precision Recall Curve given
        a binary classifier.
    PrecisionRecallDisplay.from_predictions : Plot Precision Recall Curve
        using predictions from a binary classifier.
    average_precision_score : Compute average precision from prediction scores.
    det_curve: Compute error rates for different probability thresholds.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    # print(last_ind)
    sl = slice(last_ind, None, -1)
    #sl = slice(len(tps), None, -1)
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]

