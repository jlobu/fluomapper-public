import collections
from numbers import Integral
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import comb

import torch
import torch.nn as nn


class PolynomialFeaturesTorch(nn.Module):
    """
    Note, the following is copy pasted from the PolynomialFeatures
    implementation in sklearn version 1.2. and some changes in order
    to accomodate for torch.tensors. The original implementation can be found here
    https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/preprocessing/_polynomial.py#L154

    Generate polynomial and interaction features.
    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    Read more in the :ref:`User Guide <polynomial_features>`.
    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        If a single int is given, it specifies the maximal degree of the
        polynomial features. If a tuple `(min_degree, max_degree)` is passed,
        then `min_degree` is the minimum and `max_degree` is the maximum
        polynomial degree of the generated features. Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.
    interaction_only : bool, default=False
        If `True`, only interaction features are produced: features that are
        products of at most `degree` *distinct* input features, i.e. terms with
        power of 2 or higher of the same input feature are excluded:
            - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
            - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.
    include_bias : bool, default=True
        If `True` (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. `'F'` order is faster to
        compute, but may slow down subsequent estimators.
        .. versionadded:: 0.21
    Attributes
    ----------
    powers_ : ndarray of shape (`n_output_features_`, `n_features_in_`)
        `powers_[i, j]` is the exponent of the jth input in the ith output.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.
    See Also
    --------
    SplineTransformer : Transformer that generates univariate B-spline bases
        for features.
    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.
    See :ref:`examples/linear_model/plot_polynomial_interpolation.py
    <sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])
    >>> poly = PolynomialFeatures(interaction_only=True)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.],
           [ 1.,  2.,  3.,  6.],
           [ 1.,  4.,  5., 20.]])
    """

    def __init__(
            self, degree=2, n_features_in=None, *, interaction_only=False, include_bias=True, order="C"
    ):
        super(PolynomialFeaturesTorch, self).__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.n_features_in_ = n_features_in

    @staticmethod
    def _combinations(
            n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        comb = combinations if interaction_only else combinations_w_r
        start = max(1, min_degree)
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        if include_bias:
            iter = chain(comb(range(n_features), 0), iter)
        return iter

    @staticmethod
    def _num_combinations(
            n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        """Calculate number of terms in polynomial expansion
        This should be equivalent to counting the number of terms returned by
        _combinations(...) but much faster.
        """

        if interaction_only:
            combinations = sum(
                [
                    comb(n_features, i, exact=True)
                    for i in range(max(1, min_degree), min(max_degree, n_features) + 1)
                ]
            )
        else:
            combinations = comb(n_features + max_degree, max_degree, exact=True) - 1
            if min_degree > 0:
                d = min_degree - 1
                combinations -= comb(n_features + d, d, exact=True) - 1

        if include_bias:
            combinations += 1

        return combinations

    def get_powers(self):
        """Exponent for each of the inputs in the output."""
        #check_is_fitted(self)
        assert self.is_fitted()

        combinations = self._combinations(
            n_features=self.n_features_in_,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        return np.vstack(
            [np.bincount(c, minlength=self.n_features_in_) for c in combinations]
        )

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features is None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        powers = self.get_powers()
        #input_features = _check_feature_names_in(self, input_features)
        assert input_features is None or len(input_features) == self.n_features_in_
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    "%s^%d" % (input_features[ind], exp)
                    if exp != 1
                    else input_features[ind]
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.asarray(feature_names, dtype=object)

    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        _, n_features = X.shape

        assert self.n_features_in_ is None or self.n_features_in_ == n_features

        if self.n_features_in_ is None:
            self.n_features_in_ = n_features

        if isinstance(self.degree, Integral):
            if self.degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting degree to zero and include_bias to False would result in"
                    " an empty output array."
                )

            self._min_degree = 0
            self._max_degree = self.degree
        elif (
                isinstance(self.degree, collections.abc.Iterable) and len(self.degree) == 2
        ):
            self._min_degree, self._max_degree = self.degree
            if not (
                    isinstance(self._min_degree, Integral)
                    and isinstance(self._max_degree, Integral)
                    and self._min_degree >= 0
                    and self._min_degree <= self._max_degree
            ):
                raise ValueError(
                    "degree=(min_degree, max_degree) must "
                    "be non-negative integers that fulfil "
                    "min_degree <= max_degree, got "
                    f"{self.degree}."
                )
            elif self._max_degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting both min_degree and max_degree to zero and include_bias to"
                    " False would result in an empty output array."
                )
        else:
            raise ValueError(
                "degree must be a non-negative int or tuple "
                "(min_degree, max_degree), got "
                f"{self.degree}."
            )

        self.n_output_features_ = self._num_combinations(
            n_features=n_features,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        # We also record the number of output features for
        # _max_degree = 0
        self._n_out_full = self._num_combinations(
            n_features=n_features,
            min_degree=0,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        return self

    def transform(self, X):
        """Transform data to polynomial features.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        XP : {ndarray, sparse matrix} of shape (n_samples, NP)
            The matrix of features, where `NP` is the number of polynomial
            features generated from the combination of inputs.
        """

        n_samples, n_features = X.shape

        # Do as if _min_degree = 0 and cut down array after the
        # computation, i.e. use _n_out_full instead of n_output_features_.
        XP = torch.empty(
            (n_samples, self._n_out_full), dtype=X.dtype,
            device=X.device
        )

        # What follows is a faster implementation of:
        # for i, comb in enumerate(combinations):
        #     XP[:, i] = X[:, comb].prod(1)
        # This implementation uses two optimisations.
        # First one is broadcasting,
        # multiply ([X1, ..., Xn], X1) -> [X1 X1, ..., Xn X1]
        # multiply ([X2, ..., Xn], X2) -> [X2 X2, ..., Xn X2]
        # ...
        # multiply ([X[:, start:end], X[:, start]) -> ...
        # Second optimisation happens for degrees >= 3.
        # Xi^3 is computed reusing previous computation:
        # Xi^3 = Xi^2 * Xi.

        # degree 0 term
        if self.include_bias:
            XP[:, 0] = 1
            current_col = 1
        else:
            current_col = 0

        if self._max_degree == 0:
            return XP

        # degree 1 term
        XP[:, current_col: current_col + n_features] = X.clone()
        index = list(range(current_col, current_col + n_features))
        current_col += n_features
        index.append(current_col)

        # loop over degree >= 2 terms
        for _ in range(2, self._max_degree + 1):
            new_index = []
            end = index[-1]

            for feature_idx in range(n_features):
                start = index[feature_idx]
                new_index.append(current_col)

                if self.interaction_only:
                    start += index[feature_idx + 1] - index[feature_idx]

                next_col = current_col + end - start
                if next_col <= current_col:
                    break

                XP[:, current_col:next_col] = XP[:, start:end].clone() * X[:, feature_idx: feature_idx + 1].clone()

                # XP[:, start:end] are terms of degree d - 1
                # that exclude feature #feature_idx.
                # np.multiply(
                #    XP[:, start:end],
                #    X[:, feature_idx : feature_idx + 1],
                #    out=XP[:, current_col:next_col],
                #    casting="no",
                # )

                current_col = next_col

            new_index.append(current_col)
            index = new_index

        if self._min_degree > 1:
            n_XP, n_Xout = self._n_out_full, self.n_output_features_
            if self.include_bias:
                Xout = torch.empty(
                    shape=(n_samples, n_Xout), dtype=XP.dtype, order=self.order
                )
                Xout[:, 0] = 1
                Xout[:, 1:] = XP[:, n_XP - n_Xout + 1:]
            else:
                Xout = XP[:, n_XP - n_Xout:].clone()

            XP = Xout

        return XP

    def is_fitted(self):
        return hasattr(self, "_n_out_full")
