"""Metrics to assess performance on classification task given scores

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Noel Dawe <noel@dawe.me>
# License: BSD 3 clause

from __future__ import division

import numpy as np

from sklearn.utils import check_consistent_length
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from numpy import unique


def _dcg_sample_scores(y_true, y_score, k=None, log_basis=2):
    """Compute Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray, shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, optional (default=None)
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    log_basis : float, optional (default=2)
        Basis of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).

    Returns
    -------
    discounted_cumulative_gain : ndarray, shape (n_samples,)
        The DCG score for each sample.

    See also
    --------
    ndcg_score :
        The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.

    """
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_basis))
    if k is not None:
        discount[k:] = 0
    discount_cumsum = np.cumsum(discount)
    cumulative_gains = [_tie_averaged_dcg(y_t, y_s, discount_cumsum)
                        for y_t, y_s in zip(y_true, y_score)]
    return np.asarray(cumulative_gains)


def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
    _, inv, counts = unique(
        - y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.zeros(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()


def _check_dcg_target_type(y_true):
    y_type = type_of_target(y_true)
    supported_fmt = ("multilabel-indicator", "continuous-multioutput",
                     "multiclass-multioutput")
    if y_type not in supported_fmt:
        raise ValueError(
            "Only {} formats are supported. Got {} instead".format(
                supported_fmt, y_type))


def dcg_score(y_true, y_score, k=None, log_basis=2, sample_weight=None):
    """Compute Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray, shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, optional (default=None)
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    log_basis : float, optional (default=2)
        Basis of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).

    sample_weight : ndarray, shape (n_samples,), optional (default=None)
        Sample weights. If None, all samples are given the same weight.

    Returns
    -------
    discounted_cumulative_gain : float
        The averaged sample DCG scores.

    See also
    --------
    ndcg_score :
        The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.

    References
    ----------
    `Wikipedia entry for Discounted Cumulative Gain
        <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

    Jarvelin, K., & Kekalainen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM Transactions on
    Information Systems (TOIS), 20(4), 422-446.

    Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
    A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
    Annual Conference on Learning Theory (COLT 2013)

    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.

    Examples
    --------
    >>> from sklearn.metrics import dcg_score
    >>> rng = np.random.RandomState(0)
    >>> y_true = rng.randint(4, size=(3, 5))
    >>> y_score = rng.randn(15).reshape((3, 5))
    >>> dcg_score(y_true, y_score) # doctest: +ELLIPSIS
    5.724...
    >>> dcg_score(y_true, y_true) # doctest: +ELLIPSIS
    6.385...
    >>> dcg_score(y_true, y_score, k=2) # doctest: +ELLIPSIS
    3.384...
    >>> dcg_score(y_true, y_true, k=2) # doctest: +ELLIPSIS
    4.682...

    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    _check_dcg_target_type(y_true)
    return np.average(
        _dcg_sample_scores(y_true, y_score, k=k, log_basis=log_basis),
        weights=sample_weight)


def _ndcg_sample_scores(y_true, y_score, k=None):
    """Compute Normalized Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount. Then divide by the best possible
    score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
    0 and 1.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray, shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, optional (default=None)
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    Returns
    -------
    normalized_discounted_cumulative_gain : ndarray, shape (n_samples,)
        The NDCG score for each sample (float in [0., 1.]).

    See also
    --------
    dcg_score : Discounted Cumulative Gain (not normalized).

    """
    gain = _dcg_sample_scores(y_true, y_score, k)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k)
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain


def ndcg_score(y_true, y_score, k=None, sample_weight=None):
    """Compute Normalized Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount. Then divide by the best possible
    score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
    0 and 1.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray, shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, optional (default=None)
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    sample_weight : ndarray, shape (n_samples,), optional (default=None)
        Sample weights. If None, all samples are given the same weight.

    Returns
    -------
    normalized_discounted_cumulative_gain : float in [0., 1.]
        The averaged NDCG scores for all samples.

    See also
    --------
    dcg_score : Discounted Cumulative Gain (not normalized).

    References
    ----------
    `Wikipedia entry for Discounted Cumulative Gain
        <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

    Jarvelin, K., & Kekalainen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM Transactions on
    Information Systems (TOIS), 20(4), 422-446.

    Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
    A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
    Annual Conference on Learning Theory (COLT 2013)

    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.

    Examples
    --------
    >>> from sklearn.metrics import ndcg_score
    >>> rng = np.random.RandomState(0)
    >>> y_true = rng.randint(4, size=(3, 5))
    >>> y_score = rng.randn(15).reshape((3, 5))
    >>> ndcg_score(y_true, y_score) # doctest: +ELLIPSIS
    0.877...
    >>> ndcg_score(y_true, y_score, k=2) # doctest: +ELLIPSIS
    0.734...
    >>> # Score for a perfect ranking is 1.0
    >>> ndcg_score(y_true, y_true)
    1.0
    >>> ndcg_score(y_true, y_true, k=2)
    1.0

    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    _check_dcg_target_type(y_true)
    gain = _ndcg_sample_scores(y_true, y_score, k=k)
    return np.average(gain, weights=sample_weight)