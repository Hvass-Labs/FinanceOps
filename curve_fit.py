########################################################################
#
# Classes for curve-fitting data.
#
########################################################################
#
# This file is part of FinanceOps:
#
# https://github.com/Hvass-Labs/FinanceOps
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2018 by Magnus Erik Hvass Pedersen
#
########################################################################

from scipy.optimize import curve_fit

########################################################################


class CurveFit:
    """
    Base-class for curve-fitting.
    """

    def __init__(self, x=None, y=None):
        """
        Pass numpy-arrays as the x and y args for fitting.

        :param x: Optional numpy array with input-values.
        :param y: Optional numpy array with output-values.
        """
        if x is not None and y is not None:
            self.fit(x=x, y=y)

    def _f(self, x, *args, **kwargs):
        """Function to be fitted. Override this!"""
        raise NotImplementedError()

    def predict(self, x):
        """
        Use the fitted function to predict new output-values.
        Call fit() before calling predict().

        :param x: Numpy array with input-values.
        :return: Predicted output-values.
        """
        return self._f(x, *self.params)

    def fit(self, x, y):
        """
        Fit the function parameters to the given data.
        Call this before predict().

        :param x: Numpy array with input-values.
        :param y: Numpy array with output-values.
        :return: Nothing.
        """
        self.params, self.covar = curve_fit(self._f, x, y)


class CurveFitLinear(CurveFit):
    """
    Linear curve-fitting: y = a * x + b
    First call fit() then predict().
    """

    def __init__(self, *args, **kwargs):
        """
        Pass numpy-arrays as the x and y args for fitting.

        :param x: Optional numpy array with input-values.
        :param y: Optional numpy array with output-values.
        """
        CurveFit.__init__(self, *args, **kwargs)

    def _f(self, x, a, b):
        """Linear function to be fitted."""
        return a * x + b


class CurveFitReciprocal(CurveFit):
    """
    Reciprocal curve-fitting: y = a / x + b
    First call fit() then predict().
    """

    def __init__(self, *args, **kwargs):
        """
        Pass numpy-arrays as the x and y args for fitting.

        :param x: Optional numpy array with input-values.
        :param y: Optional numpy array with output-values.
        """
        CurveFit.__init__(self, *args, **kwargs)

    def _f(self, x, a, b):
        """Reciprocal function to be fitted."""
        return a / x + b


########################################################################
