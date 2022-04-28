import numpy as np
import scipy.stats as sts


def pr_uni_err_prop(conf_mat, alpha=0.95):
    """Propogate the Poisson errors over the confusion matrix linearly to
    marginal confidence intervals over precision and recall.

    Parameters
    ----------
    conf_mat : np.ndarray[int]
        the confusion matrix
    alpha : float, default=0.95
        concentration of confidence interval

    Returns
    -------
    precision : np.ndarray[float]
        array of the same shape as conf_mat where the first column is the
        metric, the second std dev of the metric and the last two the lower and
        upper bound of the ``alpha`` percent confidence interval
    recall : np.ndarray[float]
        array of the same shape as conf_mat where the first column is the
        metric, the second std dev of the metric and the last two the lower and
        upper bound of the ``alpha`` percent confidence interval

    """
    alpha_ = (1 - alpha) / 2
    q = (alpha_, 1 - alpha_)
    if conf_mat.ndim == 1:
        conf_mat = conf_mat[None, :]
    fp = conf_mat[:, 1]
    fn = conf_mat[:, 2]
    tp = conf_mat[:, 3]

    tp_fp = tp + fp
    tp_fn = tp + fn

    precision = tp / (tp_fp)
    recall = tp / (tp_fn)

    tp_fn_pw2 = np.power(tp_fn, 2)
    tp_fp_pw2 = np.power(tp_fp, 2)

    # Partial derivatives
    recall_d_tp = fn / tp_fn_pw2
    recall_d_fn = -tp / tp_fn_pw2
    precision_d_tp = fp / tp_fp_pw2
    precision_d_fp = -tp / tp_fp_pw2

    tp_var = tp.copy()
    tp_var[tp == 0] = 1
    fn_var = fn.copy()
    fn_var[fn == 0] = 1
    fp_var = fp.copy()
    fp_var[fp == 0] = 1

    # Variance and covariance
    recall_var = np.power(recall_d_tp, 2) * tp_var + np.power(recall_d_fn, 2) * fn_var
    precision_var = np.power(precision_d_tp, 2) * tp_var + np.power(precision_d_fp, 2) * fp_var

    recall_out = np.empty(conf_mat.shape, dtype=float)
    precision_out = np.empty(conf_mat.shape, dtype=float)

    recall_out[:, 0] = recall
    recall_out[:, 1] = np.sqrt(recall_var)
    precision_out[:, 0] = precision
    precision_out[:, 1] = np.sqrt(precision_var)

    for i in range(recall_out.shape[0]):
        dist = sts.norm(precision_out[i, 0], precision_out[i, 1])
        recall_out[i, 2:] = dist.ppf(q)
        dist = sts.norm(precision_out[i, 0], precision_out[i, 1])
        precision_out[i, 2:] = dist.ppf(q)

    return precision_out, recall_out
