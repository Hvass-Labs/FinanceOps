########################################################################
#
# Various utility functions.
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

########################################################################


def linear_map(x, x_lo, x_hi, y_lo, y_hi, clip=False):
    """
    Linear map of input `x` to output `y` where:

    - `x == x_lo` => `y == y_lo`
    - `x == x_hi` => `y == y_hi`

    Output `y` can optionally be clipped between `y_lo` and `y_hi`.

    :param x: Array compatible with numpy.
    :param x_lo: Scalar value. See function description.
    :param x_hi: Scalar value. See function description.
    :param y_lo: Scalar value. See function description.
    :param y_hi: Scalar value. See function description.
    :param clip:
        Clip output. If set to the boolean value `True` or the string 'both',
        then the output is clipped between both `y_lo` and `y_hi`.
        If set to the string 'lo' then only clip the lower bound using `y_lo`.
        If set to the string 'hi' then only clip the upper bound using `y_hi`.
    :return: Array of same size as input `x`.
    """

    # Parameters for the linear mapping.
    a = (y_hi - y_lo) / (x_hi - x_lo)
    b = y_lo - a * x_lo

    # Linear mapping.
    y = a * x + b

    # Optional clipping.
    if clip == 'both' or clip is True:
        y = np.clip(y, y_lo, y_hi)
    elif clip == 'lo':
        y = np.maximum(y, y_lo)
    elif clip == 'hi':
        y = np.minimum(y, y_hi)

    return y


########################################################################
