########################################################################
#
# Functions for diversifying an investment portfolio.
#
# The main idea is to calculate a so-called "Full Exposure" of each
# asset, which takes into account the asset's correlation with other
# assets in the portfolio. We then want to find new asset-weights whose
# Full Exposure is equal to the originally desired asset-weights.
#
# For example, if we want Asset A to have weight 0.1 and Asset B to have
# weight 0.2 in the portfolio, but the two assets are also correlated
# with a factor 0.5, then we have a Full Exposure for each asset that
# is higher than their weights. A simple way of calculating the Full
# Exposure of Asset A is its weight 0.1 plus the correlation of 0.5
# multiplied with Asset B's weight of 0.2, so the Full Exposure of
# Asset A is 0.1 + 0.5 * 0.2 = 0.2, and likewise the Full Exposure
# of Asset B is 0.2 + 0.5 * 0.1 = 0.25. We then want to find new asset-
# weights so that the Full Exposure of Asset A is equal to the original
# desired weight of only 0.1, and the Full Exposure of Asset B is equal
# to its original desired weight of only 0.2.
#
# Note that the calculation of the Full Exposure is more sophisticated
# than in the example above, because it needs to satisfy several other
# requirements, as described in the paper referenced further below.
#
# We provide two methods here for finding the new asset-weights:
#
# - `optimize_weights` which tries to find new asset-weights that
#   minimize the Mean Squared Error (MSE) between the original asset-
#   weights and the Full Exposure of the new weights. This uses a
#   common optimization method such as L-BFGS-B, which works for small
#   portfolios but is extremely slow for large portfolios.
#
# - `adjust_weights` is a custom-made algorithm for this problem,
#   which is both much faster and is also capable of finding much more
#   precise asset-weights that give a much lower MSE between the
#   original asset-weights and the Full Exposure of the new weights.
#
# All this is explained in more detail in the following paper:
# - M.E.H. Pedersen, "Simple Portfolio Optimization That Works!", 2021.
#   https://ssrn.com/abstract=3942552
#
########################################################################
#
# This file is part of FinanceOps:
#
# https://github.com/Hvass-Labs/FinanceOps
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2021 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.optimize import minimize
from functools import partial

########################################################################
# Helper functions.


def _fillna(weights_org, corr, weights_guess=None):
    """
    Ensure the weights and correlations do not have NaN-values (Not-a-Number)
    by filling with 0.0 and setting the correlation-matrix diagonal to 1.0

    This makes a copy of the data.

    :param weights_org:
        Array with the originally desired portfolio weights.

    :param corr:
        Matrix of floats with the correlation-matrix.

    :param weights_guess:
        Array with portfolio weights for starting guess.

    :return:
        New array with portfolio weights.
        New correlation-matrix.
        New array with portfolio weights guess.
    """
    # Copy arrays and fill NaN-values with 0.0
    weights_org = np.nan_to_num(weights_org, nan=0.0, copy=True)
    corr = np.nan_to_num(corr, nan=0.0, copy=True)

    if weights_guess is not None:
        weights_guess = np.nan_to_num(weights_guess, nan=0.0, copy=True)

    # Fill diagonal of correlation-matrix with 1.0
    np.fill_diagonal(corr, val=1.0)

    return weights_org, corr, weights_guess


def _check_pandas_index(weights_org, corr, weights_guess=None):
    """
    If the arguments are Pandas Series or DataFrames, then check that their
    indices and columns have matching names, otherwise throw a `ValueError`.

    This is because Pandas can automatically align the data when doing math
    operations on the data, but we will be using Numpy in these algorithms,
    so the data would get corrupted if index and column names don't match.

    The time-usage is around 0.1 milli-seconds when `len(weights_org) == 1000`.

    :param weights_org:
        Array with the original asset-weights.

    :param corr:
        Matrix with the correlations between assets.

    :param weights_guess:
        Array with a better starting guess for the adjusted asset-weights.

    :raises:
        `ValueError` if the args have mis-matched Pandas index and column.

    :return:
        None
    """
    # Booleans whether the args are Pandas data-types.
    is_pandas_org = isinstance(weights_org, (pd.Series, pd.DataFrame))
    is_pandas_corr = isinstance(corr, pd.DataFrame)
    is_pandas_guess = isinstance(weights_guess, (pd.Series, pd.DataFrame))

    # Error message.
    msg = 'Mismatch in index / column names for Pandas data.'

    # Check weights_org and corr.
    if is_pandas_org and is_pandas_corr:
        if not (weights_org.index.equals(corr.index) and
                weights_org.index.equals(corr.columns)):
            raise ValueError(msg)

    # Check weights_org and weights_guess.
    if is_pandas_org and is_pandas_guess:
        if not weights_org.index.equals(weights_guess.index):
            raise ValueError(msg)

    # Check weights_guess and corr.
    # This is only necessary if weights_org is not a Pandas data-type,
    # otherwise we would already know that weights_org matches corr and
    # weights_org matches weights_guess, therefore weights_guess matches corr.
    if (not is_pandas_org) and is_pandas_guess and is_pandas_corr:
        if not (weights_guess.index.equals(corr.index) and
                weights_guess.index.equals(corr.columns)):
            raise ValueError(msg)


@jit
def _find_weight_problem(weights_org, weights_new):
    """
    Helper-function for the `_check_weights` function which returns the index
    of the first problem for the portfolio weights. Runs fast with Numba Jit.

    :param weights_new:
        Array with the new asset-weights.

    :param weights_org:
        Array with the original asset-weights.

    :return:
        `None` if no problems are found.
         Otherwise an integer with the index of the first problem.
    """
    # Number of weights.
    n = len(weights_new)

    # For each weight index.
    for i in range(n):
        # Get the weights.
        w_new = weights_new[i]
        w_org = weights_org[i]

        # Check if there is a problem and then return the corresponding index.
        # We must ensure the weight signs are equal and magnitudes are valid.
        # But because np.sign(0.0)==0.0 the check for signs is a bit awkward.
        if (np.sign(w_new) != 0.0 and np.sign(w_new) != np.sign(w_org)) or \
                (np.abs(w_new) > np.abs(w_org)):
            return i

    # No problems were found.
    return None


def _check_weights(weights_org, weights_new):
    """
    Check that the original and new portfolio weights are consistent. They must
    have the same sign, and the absolute values of the new weights must be
    smaller than the absolute values of the original weights:

    (1)     sign(weights_new[i]) == sign(weights_org[i])
    (2)     abs(weights_new[i]) <= abs(weights_org[i])

    This function only takes 3.5 micro-seconds to run for 1000 weights using a
    Numba Jit implementation. A Numpy implementation would be much slower. But
    it must be split into two functions, because Numba Jit does not properly
    support the string operations used to generate the exception.

    :param weights_new:
        Array with the new asset-weights.

    :param weights_org:
        Array with the original asset-weights.

    :raises:
        `RuntimeError` if the weights are inconsistent.

    :return:
        None
    """
    # Get index of the first problem / inconsistency of the weights.
    idx = _find_weight_problem(weights_org=weights_org,
                               weights_new=weights_new)

    # If a problem was found then raise an exception.
    if idx is not None:
        msg = f'Checking the weights failed at: i={idx}, ' + \
              f'weights_new[i]={weights_new[idx]:.2e}, ' + \
              f'weights_org[i]={weights_org[idx]:.2e}'
        raise RuntimeError(msg)


def _check_convergence(max_abs_dif, tol):
    """
    Check the adjusted portfolio weights have converged, so the Full Exposure
    of the portfolio weights are sufficiently close to the original weights.

    :param max_abs_dif:
        Float with max absolute difference between the Full Exposure and the
        original portfolio weights.

    :param tol:
        Tolerance level for the max abs difference.

    :raises:
        `RuntimeError` if the weights are inconsistent.

    :return:
        None
    """
    if max_abs_dif > tol:
        msg = 'Weights did not converge: ' + \
              f'max_abs_dif={max_abs_dif:.2e}, tol={tol:.2e}'
        raise RuntimeError(msg)


########################################################################
# Full Exposure.

@jit(parallel=False)
def full_exposure(weights, corr):
    """
    Calculate the so-called "Full Exposure" for each asset, which measures
    the entire portfolio's exposure to each asset both directly through the
    asset weights themselves, but also indirectly through their correlations
    with other assets in the portfolio.

    There are different ways of defining the Full Exposure, as explained in
    the paper referenced above. This particular formula is Eq.(38) in the
    paper referenced above, which was found to work well in practice.

    The function is decorated with Numba Jit, which means it compiles into
    super-fast machine-code the first time it is run. This function is the most
    expensive part of the diversification method because it has time-complexity
    O(n^2) where n is the number of assets in the portfolio. Implementing it
    with for-loops instead of Numpy arrays, means that it avoids new memory
    allocations for large n^2 matrices, so the machine-code is very fast.

    For large portfolios of e.g. 1000 assets or more, this can run even faster
    by using `@jit(parallel=True)` as the function decorator. But for smaller
    portfolios of only e.g. 100 assets, the parallelization overhead makes it
    run a bit slower, so you have to turn on the parallelism manually.

    Note that the arguments must be Python lists or Numpy arrays and cannot be
    Pandas Series and DataFrames, because Numba Jit does not support Pandas.

    :param weights:
        Array with the portfolio weights for the assets.

    :param corr:
        Correlation matrix for the assets. The element in the i'th row
        and j'th column is the correlation between assets i and j.

    :returns:
        Array with the Full Exposure of each asset.
    """
    # Number of assets in the portfolio.
    n = len(weights)

    # Initialize an empty array for the results.
    full_exp = np.empty(shape=n, dtype=np.float64)

    # For each asset i in the portfolio.
    # Note the use of prange() instead of range() which instructs Numba Jit
    # to parallelize this loop, but only if @jit(parallel=True) was used,
    # otherwise this just becomes the ordinary Python range().
    for i in prange(n):
        # Portfolio weight of asset i.
        w_i = weights[i]

        # Initialize the sum of correlated exposures.
        sum_corr_exp = 0.0

        # For each other asset j in the portfolio.
        for j in range(n):
            # Portfolio weight of asset j.
            w_j = weights[j]

            # Correlation between assets i and j.
            c = corr[i, j]

            # Product of the two asset weights and their correlation.
            prod = w_i * w_j * c

            # If the product is positive then the correlation is deemed "bad"
            # and must be included in the calculation of the Full Exposure,
            # so the two portfolio weights can be adjusted accordingly.
            if prod > 0.0:
                # Multiply with the correlation again, because otherwise the
                # square-root calculated below would amplify the correlation.
                # Because this can result in a negative number, we also need
                # to take the absolute value.
                sum_corr_exp += np.abs(prod * c)

        # Calculate and save the Full Exposure for asset i.
        full_exp[i] = np.sign(w_i) * np.sqrt(sum_corr_exp)

    return full_exp


def _full_exposure_numpy(weights, corr):
    """
    Calculate the so-called "Full Exposure" for each asset, which measures
    the entire portfolio's exposure to each asset both directly through the
    asset weights themselves, but also indirectly through their correlations
    with other assets in the portfolio.

    This implements Eq.(38) from the paper referenced above. This version uses
    Numpy array and matrix operations. It is much slower than the function
    `full_exposure`, because this function needs to allocate new memory for
    temporary arrays and matrices.

    It is highly recommended that you use the function `full_exposure` instead.
    This function is mainly provided for educational purposes.

    :param weights:
        Array with the portfolio weights for the assets.

    :param corr:
        Correlation matrix for the assets. The element in the i'th row
        and j'th column is the correlation between assets i and j.

    :returns:
        Array with the Full Exposure of each asset.
    """
    # Product of correlation matrix, weights and transposed weights.
    corr_weights = corr * weights * np.expand_dims(weights, axis=1)

    # Don't include negative correlations in the calculation of FE.
    # For negative asset-weights this becomes a bit complicated,
    # but reduces to using the sign of the elements in corr_weights.
    # This is explained in Section 8.3 in the paper linked above.
    use = (np.sign(corr_weights) > 0.0).astype(np.float64)
    # This has created a matrix of 0 and 1 values, so multiply with
    # corr_weights to eliminate the elements we don't want to use.
    corr_weights *= use

    # Multiply with the elements of the correlation-matrix again,
    # so when we take the square-root next, we don't over-estimate the
    # effect of correlation. This can create negative values so we
    # need to take the absolute values to ensure the result is positive.
    corr_weights = np.abs(corr_weights * corr)

    # The elements of the corr_weights matrix are all positive.
    # So we can sum each row, take the square-root, and then
    # restore the proper sign from the weights. This gives an
    # array with the Full Exposure of each asset.
    full_exp = np.sign(weights) * np.sqrt(np.sum(corr_weights, axis=1))

    return full_exp


########################################################################
# Mean Squared Error.

def mse(weights_new, weights_org, corr):
    """
    Mean Squared Error (MSE) between the original asset-weights
    and the Full Exposure of the new asset-weights.

    When the MSE value is zero, it means that the Full Exposure of
    the new asset-weights are equal to the original asset-weights.

    :param weights_org:
        Array with the original asset-weights.

    :param weights_new:
        Array with the new asset-weights.

    :param corr:
        Matrix with the correlations between assets.

    :return:
        Float with the MSE value.
    """

    # Calculate the Full Exposure of the new asset-weights.
    full_exp = full_exposure(weights=weights_new, corr=corr)

    # Calculate and return the Mean Squared Error.
    return np.mean((full_exp - weights_org) ** 2)


########################################################################
# Optimize weights using generic optimizer.

def optimize_weights(weights_org, corr, weights_guess=None,
                     fillna=True, method='L-BFGS-B', **kwargs):
    """
    Find new asset-weights that minimize the Mean Squared Error (MSE) between
    the original asset-weights and the Full Exposure of the new asset-weights.

    This function uses a generic optimizer which is about 1-2000x slower than
    the custom algorithm in the function `adjust_weights`. So it is highly
    recommended that you use the function `adjust_weights` instead of this!

    :param weights_org:
        Array with the original asset-weights.

    :param corr:
        Matrix with the correlations between assets.

    :param weights_guess:
        Array with a better starting guess for the adjusted asset-weights.

    :param fillna:
        Boolean whether to fill NaN-values (Not-a-Number) in `weights_org`
        and `corr` with 0.0, and fill the diagonal of `corr` with 1.0 values.

    :param method:
        String with the method-name used for the optimization.
        This string is just passed to scipy's `minimize` method.

    :param kwargs:
        Extra keyword arguments passed to scipy's `minimize` method.

    :return:
        Array with the optimized asset-weights.
    """

    # Ensure the weights and correlation-matrix do not have NaN-values?
    if fillna:
        # This copies the data.
        weights_org, corr, weights_guess = \
            _fillna(weights_org=weights_org, corr=corr,
                    weights_guess=weights_guess)

    # Function to be minimized. This is the MSE between the
    # original weights and the Full Exposure of the new weights.
    _fun = partial(mse, weights_org=weights_org, corr=corr)

    # Search-space boundaries for the optimization method.
    # This ensures the new asset-weights are between zero and the
    # original asset-weights. It is a bit complicated because it
    # needs to support both positive and negative weights.
    bounds = list(zip(np.minimum(weights_org, 0), np.maximum(weights_org, 0)))

    # Use the original weights if no starting guess was provided.
    if weights_guess is None:
        weights_guess = weights_org

    # Perform the optimization using SciPy.
    result = minimize(fun=_fun, x0=weights_guess,
                      bounds=bounds, method=method, **kwargs)

    # Get the new portfolio weights.
    weights_new = result.x

    # Check that the original and new portfolio weights are consistent.
    _check_weights(weights_org=weights_org, weights_new=weights_new)

    return weights_new


########################################################################
# Adjust weights using custom algorithm.


def _update_weights_vec(weights_org, weights_new, corr, step_size):
    """
    Helper-function for the function `adjust_weights` which performs a single
    update of the portfolio weights. This is the vectorized version which uses
    Numpy to update all the weights simultaneously.

    This algorithm is described in Section 8.7 of the paper linked above.

    :param weights_org:
        Numpy array with the original portfolio weights.

    :param weights_new:
        Numpy array with the adjusted portfolio weights. Updated in-place.

    :param corr:
        Numpy array with the correlation matrix.

    :param step_size:
        Float between 0.0 and 1.0 for the step-size.

    :return:
        Float with the max absolute difference between the Full Exposure
        and the original portfolio weights. This is used to abort the
        algorithm's for-loop when sufficiently good weights have been found.
    """
    # Full Exposure of the current asset-weights.
    full_exp = full_exposure(weights=weights_new, corr=corr)

    # Difference between the Full Exposure of the current
    # asset-weights and the original asset-weights. This is
    # how much each asset is over-weighted due to correlated
    # exposure to other assets, when using the new weights.
    weights_dif = full_exp - weights_org

    # Max absolute difference between Full Exposure and original weights.
    # Used to abort the algorithm's for-loop when solution has been found.
    max_abs_dif = np.max(np.abs(weights_dif))

    # Ignore Divide-By-Zero in case the Full Exposure is zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Adjustment for each asset-weight by the appropriate
        # proportion of its Full Exposure, because all other
        # asset-weights will also be adjusted simultaneously,
        # so we would over-adjust if we used weights_dif directly.
        # Without this the algorithm may not converge and is
        # unstable so the new weights can approach infinity.
        weights_adj = weights_dif * weights_new / full_exp

        # Fill NaN (Not-a-Number) with zero in case of
        # Divide-By-Zero if the Full Exposure is zero.
        weights_adj = np.nan_to_num(weights_adj)

    # Update asset-weights. This updates the array in-place.
    weights_new -= weights_adj * step_size

    return max_abs_dif


@jit(parallel=False)
def _update_weights_elm(weights_org, weights_new, corr):
    """
    Helper-function for the function `adjust_weights` which performs a single
    update of the portfolio weights. This is the element-wise version which
    uses for-loops instead of Numpy to update the weights.

    This algorithm is described in Section 8.8 of the paper linked above.

    :param weights_org:
        Numpy array with the original portfolio weights.

    :param weights_new:
        Numpy array with the adjusted portfolio weights. Updated in-place.

    :param corr:
        Numpy array with the correlation matrix.

    :return:
        Float with the max absolute difference between the Full Exposure
        and the original portfolio weights. This is used to abort the
        algorithm's for-loop when sufficiently good weights have been found.
    """
    # Number of portfolio weights.
    n = len(weights_org)

    # Init. max abs difference between the Full Exposure and original weights.
    max_abs_dif = 0.0

    # For each asset i in the portfolio.
    # Note the use of prange() instead of range() which instructs Numba Jit
    # to parallelize this loop, but only if @jit(parallel=True) was used,
    # otherwise this just becomes the ordinary Python for-loop using range().
    # Also note there is a "race condition" when this loop is run in parallel,
    # because the weights_new array is both read and written inside the loop,
    # but the algorithm can handle this for the same reason that it converges
    # to the correct solution, as was proven in the paper referenced above.
    for i in prange(n):
        # The new and original portfolio weights of asset i.
        w_new_i = weights_new[i]
        w_org_i = weights_org[i]

        # First we need to calculate the Full Exposure of asset i.

        # Initialize the sum of correlated exposures.
        sum_corr_exp = 0.0

        # For each other asset j in the portfolio.
        for j in range(n):
            # Portfolio weight of asset j.
            w_new_j = weights_new[j]

            # Correlation between assets i and j.
            c = corr[i, j]

            # Product of the two asset weights and their correlation.
            prod = w_new_i * w_new_j * c

            # If the product is positive then the correlation is deemed "bad"
            # and must be included in the calculation of the Full Exposure,
            # so the two portfolio weights can be adjusted accordingly.
            if prod > 0.0:
                # Multiply with the correlation again, because otherwise the
                # square-root calculated below would amplify the correlation.
                # Because this can result in a negative number, we also need
                # to take the absolute value.
                sum_corr_exp += np.abs(prod * c)

        # Full Exposure for asset i.
        full_exp_i = np.sign(w_new_i) * np.sqrt(sum_corr_exp)

        # If the Full Exposure is non-zero.
        if full_exp_i != 0.0:
            # Update the portfolio weight for asset i.
            weights_new[i] *= w_org_i / full_exp_i

        # Update max abs difference between Full Exposure and original weight.
        abs_dif = np.abs(full_exp_i - w_org_i)
        if abs_dif > max_abs_dif:
            max_abs_dif = abs_dif

    return max_abs_dif


@jit(parallel=False)
def _update_weights_inv(weights_org, weights_new, corr):
    """
    Helper-function for the function `adjust_weights` which performs a single
    update of the portfolio weights. This is the inverse version which uses
    the mathematical inverse of the function for the Full Exposure.

    This algorithm is described in Section 8.6 of the paper linked above.

    Although this converges in fewer iterations than the other algorithms in
    `_update_weights_elm` and `_update_weights_vec`, this needs to do more
    calculations when using the Full Exposure to detect convergence, so this
    implementation is actually slower than the two other algorithm variants.

    :param weights_org:
        Numpy array with the original portfolio weights.

    :param weights_new:
        Numpy array with the adjusted portfolio weights. Updated in-place.

    :param corr:
        Numpy array with the correlation matrix.

    :return:
        Float with the max absolute difference between the Full Exposure
        and the original portfolio weights. This is used to abort the
        algorithm's for-loop when sufficiently good weights have been found.
    """
    # Number of portfolio weights.
    n = len(weights_org)

    # Init. max abs difference between the Full Exposure and original weights.
    max_abs_dif = 0.0

    # For each asset i in the portfolio.
    # Note the use of prange() instead of range() which instructs Numba Jit
    # to parallelize this loop, but only if @jit(parallel=True) was used,
    # otherwise this just becomes the ordinary Python range().
    for i in prange(n):
        # The new and original portfolio weights of asset i.
        w_new_i = weights_new[i]
        w_org_i = weights_org[i]

        # Note that we only need the Full Exposure for detecting convergence,
        # and not for updating the portfolio weights when using this algorithm.

        # Initialize the sum used to calculate the Full Exposure.
        sum_wi_wj_c = 0.0

        # Initialize the sum used to update the portfolio weights.
        sum_wj_c = 0.0

        # For each other asset j in the portfolio.
        for j in range(n):
            # Portfolio weight of asset j.
            w_new_j = weights_new[j]

            # Correlation between assets i and j.
            c = corr[i, j]

            # Product of weight for asset j and correlation between i and j.
            prod_wj_c = w_new_j * c

            # Product of both weights and their correlation.
            prod_wi_wj_c = w_new_i * prod_wj_c

            # If the product is positive then the correlation is deemed "bad"
            # and must be included in this calculation so the two portfolio
            # weights can be adjusted accordingly.
            if prod_wi_wj_c > 0.0:
                # Multiply with the correlation again, because otherwise the
                # square-root calculated below would amplify the correlation.
                # Because this can result in a negative number, we also need
                # to take the absolute value.
                sum_wi_wj_c += np.abs(prod_wi_wj_c * c)

                # Add to the sum used to update the portfolio weight.
                # This should not be added when asset index j==i.
                if i != j:
                    sum_wj_c += np.abs(prod_wj_c * c)

        # Full Exposure for asset i.
        full_exp_i = np.sign(w_new_i) * np.sqrt(sum_wi_wj_c)

        # Update portfolio weight for asset i.
        # This uses the positive solution to a 2nd degree polynomial.
        # It implements Eq.(46) in the paper linked above.
        weights_new[i] = np.sign(w_org_i) * \
            (-sum_wj_c + np.sqrt(sum_wj_c ** 2 + 4 * w_org_i ** 2)) / 2

        # Update max abs difference between Full Exposure and original weight.
        abs_dif = np.abs(full_exp_i - w_org_i)
        if abs_dif > max_abs_dif:
            max_abs_dif = abs_dif

    return max_abs_dif


def adjust_weights(weights_org, corr, weights_guess=None, fillna=True, log=None,
                   variant='inv', step_size=1.0, max_iter=100, tol=1e-3):
    """
    Find new asset-weights that minimize the Mean Squared Error (MSE) between
    the original asset-weights and the Full Exposure of the new asset-weights.

    This uses a custom algorithm for this particular problem. It is much faster
    than the `optimize_weights` function, especially for very large portfolios.

    For a portfolio of 1000 assets it only takes 20 milli-seconds to run this,
    depending on the CPU, arguments, and the weights and correlation matrix.
    Removing some of the options and overhead in the function can significantly
    improve the speed. But Numba Jit cannot improve the speed of this function.

    :param weights_org:
        Array with the originally desired asset-weights for the portfolio.
        These can be either positive or negative and they need not sum to 1.
        This data can either be a Pandas Series or Numpy array.

    :param corr:
        Matrix with the correlations between assets. These are assumed to be
        between -1 and 1. This can either be a Pandas DataFrame or Numpy array.

    :param weights_guess:
        Array with a better starting guess for the adjusted asset-weights.
        If you are calling this function with `weights_org` and `corr` being
        nearly identical on each call, then you might save computation time
        by passing the last weights that were output from this function as
        the arg `weights_guess` the next time you call this function. This
        may reduce the number of iterations needed for convergence.

    :param fillna:
        Boolean whether to fill NaN-values (Not-a-Number) in `weights_org`
        and `corr` with 0.0, and fill the diagonal of `corr` with 1.0 values.

    :param log:
        If this is a list-like object then it will have its function `append`
        called after each iteration with the new weights, so you can print
        them later. This is useful for debugging and other demonstrations.

    :param variant:
        String for the algorithm variant used to update the portfolio weights:
        - 'vec' is for vectorized update of all weights simultaneously.
        - 'elm' is for element-wise update of all the weights.
        - 'inv' is for using the mathematical inverse to update the weights.

    :param step_size:
        Float between 0.0 and 1.0 for the convergence speed of the algorithm.
        Values closer to 0.0 are slower and values closer to 1.0 are faster.
        There does not appear to be any difference in the results obtained,
        so you might as well leave this to its default value of 1.0.

    :param max_iter:
        Max iterations of the algorithm.

    :param tol:
        Stop the algorithm when asset-weight adjustments are smaller than this
        tolerance level.

    :return:
        Array with the adjusted asset-weights.
    """
    # Which algorithm variant to use for updating the portfolio weights?
    if variant == 'vec':
        # Function for vectorized weight-update.
        _update_weights = partial(_update_weights_vec, step_size=step_size)
    elif variant == 'elm':
        # Function for element-wise weight-update.
        _update_weights = _update_weights_elm
    elif variant == 'inv':
        # Function for weight-update using mathematical inverse of Full Exp.
        _update_weights = _update_weights_inv
    else:
        # Raise exception on invalid arg.
        msg = f'Invalid arg: variant=\'{variant}\''
        raise ValueError(msg)

    # If using Pandas data-types, ensure their index and column names match.
    _check_pandas_index(weights_org=weights_org, corr=corr,
                        weights_guess=weights_guess)

    # Convert weights_org from Pandas to Numpy.
    if isinstance(weights_org, (pd.Series, pd.DataFrame)):
        # Save the Pandas index for later use with the return-data.
        index = weights_org.index

        # Convert Pandas to Numpy. This may or may not be a copy of the data.
        # Note: Flatten is necessary if it is a Pandas DataFrame.
        weights_org = weights_org.to_numpy().flatten()
    else:
        # This is used to indicate that the input was not Pandas data.
        index = None

    # Convert weights_guess from Pandas to Numpy.
    if isinstance(weights_guess, (pd.Series, pd.DataFrame)):
        # This may or may not be a copy of the data.
        # Note: Flatten is necessary if it is a Pandas DataFrame.
        weights_guess = weights_guess.to_numpy().flatten()

    # Convert correlation matrix from Pandas to Numpy.
    if isinstance(corr, pd.DataFrame):
        # This may or may not be a copy of the data.
        corr = corr.to_numpy()

    # Ensure the weights and correlation-matrix do not have NaN-values.
    if fillna:
        # This copies the data.
        weights_org, corr, weights_guess = \
            _fillna(weights_org=weights_org, corr=corr,
                    weights_guess=weights_guess)

    # Select a starting point for the new adjusted weights.
    # The arrays are copied so we don't modify the argument data.
    # It is possible that the data was already copied above, so there
    # is a slight redundancy here, but it makes the code easier to read.
    if weights_guess is not None:
        # Use the guessed weights as the starting point.
        # In case the guessed weight is zero, use the original weight,
        # otherwise the weight-adjustment would always get stuck in zero.
        # This should create a new numpy array so there is no need to copy.
        weights_new = np.where(weights_guess != 0.0, weights_guess, weights_org)
    else:
        # Use the original weights as the starting point.
        weights_new = np.copy(weights_org)

    # Log the initial weights?
    if log is not None:
        # Array is copied because the update iterates on the same array, so
        # the entire log would be filled with the same values if not copied.
        log.append(weights_new.copy())

    # Repeat for a number of iterations or until convergence
    # which breaks out of the for-loop further below.
    for i in range(max_iter):
        # Update the array weights_new inplace.
        max_abs_dif = _update_weights(weights_org=weights_org,
                                      weights_new=weights_new, corr=corr)

        # Log the updated weights?
        if log is not None:
            # Array is copied because the update iterates on the same array, so
            # the entire log would be filled with the same values if not copied.
            log.append(weights_new.copy())

        # Abort the for-loop when converged to a solution.
        if max_abs_dif < tol:
            break

    # Check that the we have converged to a sufficiently good solution.
    _check_convergence(max_abs_dif=max_abs_dif, tol=tol)

    # Check that the original and new portfolio weights are consistent.
    _check_weights(weights_org=weights_org, weights_new=weights_new)

    # If the input weights_org was Pandas data, then also output Pandas data.
    if index is not None:
        weights_new = pd.Series(data=weights_new, index=index)

    return weights_new


########################################################################
# Other useful functions.

def log_to_dataframe(weights_org, corr, log):
    """
    Convert the log from `adjust_weights` to a Pandas DataFrame which shows
    the iterations of the adjusted portfolio weights and their Full Exposure.

    :param weights_org:
        Array with the originally desired portfolio weights.

    :param corr:
        Matrix of floats with the correlation-matrix.

    :param log:
        List of numpy arrays with portfolio weights. This is obtained by first
        passing the list as the `log` arg in the `adjust_weights` function.

    :return:
        Pandas DataFrame
    """
    # Convert log to numpy array.
    log_weights = np.array(log)

    # Get the number of iterations and assets in the log.
    num_iterations, num_assets = log_weights.shape

    # Initialize log for the Full Exposure.
    log_full_exp = []

    # Initialize log for the Mean Squared Error (MSE).
    log_mse = []

    # For each array of adjusted weights in the log.
    for weights_new in log_weights:
        # Calculate and the Full Exposure of the logged weights.
        fe = full_exposure(weights=weights_new, corr=corr)
        # Save the results.
        log_full_exp.append(fe)

        # Calculate the Mean Squared Error (MSE).
        _mse = mse(weights_new=weights_new, weights_org=weights_org, corr=corr)
        # Save the results.
        log_mse.append(_mse)

    # Combine the arrays of adjusted weights and Full Exposure, so that:
    # 1st column is for 1st weights, 2nd column is for 1st Full Exposure.
    # 3rd column is for 2nd weights, 4th column is for 2nd Full Exposure.
    data = np.dstack((log_weights, log_full_exp)).reshape(num_iterations, -1)

    # Generate names for the columns.
    names = []
    for i in range(1, num_assets + 1):
        names.append(f'Weight {i}')
        names.append(f'Full Exp. {i}')

    # Index for the rows.
    index = pd.Series(data=list(range(0, num_iterations)), name='Iteration')

    # Create Pandas DataFrame with the data.
    df = pd.DataFrame(data=data, columns=names, index=index)

    # Append a column for the Mean Squared Error (MSE).
    df['MSE'] = log_mse

    return df


########################################################################
