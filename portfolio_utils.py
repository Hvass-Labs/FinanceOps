########################################################################
#
# Various utility functions for investment portfolios.
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
from numba import jit

########################################################################


def check_normalized_weights(weights_norm, cash, tol=1e-9):
    """
    Check if the normalized portfolio weights and cash are valid, and raise
    an exception if they are invalid. This is only for positive weights.

    This takes 0.5 milli-second to compute for portfolios with 200 assets
    and 2500 time-steps (corresponding to 10 years of daily time-steps).

    :param weights_norm:
        Pandas DataFrame with normalized portfolio weights for assets.

    :param cash:
        Pandas Series with the portfolio's cash-weights for each time-step.

    :param tol:
        Float with the tolerance-level used in floating-point comparisons.

    :raises:
        `RuntimeError` if an error is found.

    :return:
        None
    """
    # Convert Pandas to Numpy arrays for speed.
    weights_norm_array = weights_norm.to_numpy()
    cash_array = cash.to_numpy()

    # Sum the normalized weights for each time-step.
    weights_norm_sum = np.sum(weights_norm_array, axis=1)

    # Boolean mask whether there is an error in each time-step.
    # Note: Because these are floating-points it is possible that there are
    # small rounding errors so we use a tolerance level in the comparisons.
    err_mask = (weights_norm_sum < 0.0 - tol) | \
               (weights_norm_sum > 1.0 + tol) | \
               (cash_array < 0.0 - tol) | \
               (cash_array > 1.0 + tol) | \
               (~np.isclose(weights_norm_sum + cash_array, 1.0))

    # If there is any significant error then raise exception.
    if np.any(err_mask):
        msg = f'Checking the normalized weights failed: ' + \
              f'weights_norm_sum={weights_norm_sum[err_mask]}, ' + \
              f'cash={cash_array[err_mask]}'
        raise RuntimeError(msg)


def normalize_weights(weights, check_result=False):
    """
    Normalize a portfolio's asset-weights so they sum to max 1.

    If the sum of asset-weights for a time-step are less than 1,
    then they are not changed. But if the sum of asset-weights are
    greater than 1, then all the asset-weights for that time-step
    are decreased to make those asset-weights sum to 1.

    This function only supports so-called "long" portfolios where
    all asset-weights are zero or positive.

    :param weights:
        Pandas DataFrame with the asset-weights.
        Rows are for the time-steps. Columns are for the assets.

    :param check_result:
        Boolean whether to check the results are valid.

    :raises:
        `RuntimeError` if an error is found in the results, and the
        arg `check_result` is True.

    :return:
        weights_norm: Pandas DataFrame with normalized weights.
        cash: Pandas Series with cash-fraction of the portfolio.
    """
    # Ensure all weights are non-negative aka. "long-only" portfolios.
    # This is the fastest way of checking it for Pandas DataFrames.
    assert weights.to_numpy().min() >= 0.0

    # Sum the asset-weights for each time-step.
    weights_sum = weights.sum(axis=1)

    # Cash-position for each time-step.
    # This is zero if the weights sum to more than 1.
    cash = np.maximum(0.0, 1.0 - weights_sum)

    # The scaling factor for each time-step to make the
    # asset-weights for that time-step sum to 1.
    # This is a fast calculation and also avoids division-by-zero
    # in case weights_sum == 0.0
    weights_scale = np.where(weights_sum > 1.0, 1.0 / weights_sum, 1.0)

    # Scale all the stock-weights for each time-step according
    # to the scaling factor to make the weights sum to 1.
    weights_norm = weights.mul(weights_scale, axis=0)

    # Check the results are valid?
    if check_result:
        check_normalized_weights(weights_norm=weights_norm, cash=cash)

    return weights_norm, cash


def weighted_returns(returns, weights, cash):
    """
    Calculate a portfolio's cumulative weighted returns. The assets
    are weighted at each time-step using the given weights, and
    a part of the portfolio can be held in cash (with zero return).

    :param returns:
        Pandas DataFrame with the asset-returns. These are +1
        so e.g. 1.05 is a +5% return and 0.9 is a -10% return.
        Rows are for the time-steps. Columns are for the assets.

    :param weights:
        Pandas DataFrame with the asset-weights.
        Rows are for the time-steps. Columns are for the assets.

    :param cash:
        Pandas Series with the cash-fraction of the portfolio.

    :return:
        Pandas Series with the cumulative portfolio returns.
    """
    # Weighted returns for individual assets at each time-step.
    # This is a DataFrame. Rows are time-steps. Columns are assets.
    weighted_rets = weights * returns

    # The portfolio's return for each time-step.
    # This is a Pandas Series.
    port_rets = weighted_rets.sum(axis=1)

    # Cumulative portfolio returns.
    # This is a Pandas Series.
    port_cum_rets = (port_rets + cash).cumprod()

    return port_cum_rets


########################################################################

@jit
def fix_correlation_matrix(corr):
    """
    Fix a correlation matrix so it is symmetrical, limited between -1 and 1,
    and the diagonal elements are all 1. The upper-triangle is copied to the
    lower-triangle. The data is updated inplace.

    :param corr:
        Numpy 2-dim array for the correlation matrix which is updated inplace.

    :return:
        The same Numpy array as the `corr` arg.
    """
    # Number of rows and columns.
    n = len(corr)

    # For each row and column.
    for i in range(n):
        for j in range(i + 1, n):
            # Get the correlation value.
            c = corr[i, j]

            #  Ensure the correlation value is valid.
            if np.isnan(c):
                # NaN (Not-a-Number) value is set to zero.
                c = 0.0
            elif c > 1.0:
                # Clip the value if it is higher than 1.0
                c = 1.0
            elif c < -1.0:
                # Clip the value if it is lower than -1.0
                c = -1.0

            # Update the matrix inplace.
            corr[i, j] = corr[j, i] = c

        # Ensure the diagonal is 1.
        corr[i, i] = 1.0

    return corr


@jit
def check_correlation_matrix(corr, tol=1e-9):
    """
    Check that a numpy array is a valid correlation matrix:

    - It must be matrix-shaped.
    - Its elements must be between -1 and 1.
    - The diagonal must be 1.
    - The matrix must be symmetrical.

    The checks allow for small floating point rounding errors.

    :param corr:
        Numpy 2-dim array for the correlation matrix.
        Note: It is NOT checked that it is a valid Numpy array, because that
        kind of type-checking is not supported inside a Numba Jit function.

    :param tol:
        Float with the error tolerance in the float comparisons.

    :raises:
        `ValueError` if the `corr` arg is an invalid correlation matrix.

    :return:
        None
    """
    # Assume `corr` is a valid Numpy array, because we cannot check its type
    # inside a Numba Jit function using e.g. isinstance(corr, np.ndarray).

    # Check it is matrix-shaped.
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix is not matrix-shaped.')

    # Number of rows and columns.
    n = corr.shape[0]

    # For each row in the correlation matrix.
    for i in range(n):
        # Check the diagonal is 1.
        if np.abs(corr[i, i] - 1.0) > tol:
            raise ValueError('Correlation matrix diagonal is not 1.')

        # For each relevant column in the correlation matrix.
        for j in range(i + 1, n):
            # Check the correlations are between -1 and 1.
            if (corr[i, j] < -1.0 - tol) or (corr[i, j] > 1.0 + tol):
                msg = 'Correlation matrix has element outside range [-1,1].'
                raise ValueError(msg)

            # Check the matrix is symmetrical.
            if np.abs(corr[i, j] - corr[j, i]) > tol:
                raise ValueError('Correlation matrix is not symmetrical.')


########################################################################
