########################################################################
#
# Classes for optimizing and allocating portfolios of stocks.
#
# This version is for SINGLE-OBJECTIVE optimization. It has a lot of
# code in common with portfolio_multi.py but it is probably easier to
# read, understand and modify the code when it is all in one file.
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

import numpy as np
from scipy.optimize import differential_evolution as minimize

########################################################################
# Private helper-functions.


def _sigmoid(x):
    """
    Sigmoid function that smoothly limits values between 0.0 and 1.0

    :param x: Numpy array with float values that are to be limited.
    :return: Numpy array with float values between 0.0 and 1.0
    """
    return 1.0 / (1.0 + np.exp(-x))


########################################################################
# Public classes.


class Model:
    """
    Base-class for a portfolio model providing functions for doing
    the optimization and calculating the returns from using the model.

    This version is for SINGLE-OBJECTIVE optimization.
    """

    def __init__(self, signals_train, daily_rets_train, min_weights, max_weights):
        """
        Create object instance and run optimization of the portfolio model.

        :param signals_train: 2-d numpy array with signals.
        :param daily_rets_train: 2-d numpy array with daily returns.
        :param min_weights: 1-d numpy array with min stock weights.
        :param max_weights: 1-d numpy array with max stock weights.
        """

        # Copy args.
        self.signals_train = signals_train
        self.daily_rets_train = daily_rets_train
        self.min_weights = min_weights
        self.max_weights = max_weights

        # Number of stocks.
        self.num_stocks = self.signals_train.shape[1]

        assert signals_train.shape == daily_rets_train.shape
        assert self.num_stocks == len(min_weights) == len(max_weights)

        # Optimize the portfolio allocation model.
        self._optimize()

    def get_weights(self, signals):
        """
        Map the signals to stock-weights.

        :param signals: 2-d numpy array with signals for the stocks.
        :return: (weights: 2-d numpy array, weights_cash: 1-d numpy array)
        """
        raise NotImplementedError

    @property
    def _bounds(self):
        """Parameter bounds for the model that is going to be optimized."""
        raise NotImplementedError

    def _set_parameters(self, params):
        """
        Unpack and set the parameters for the portfolio allocation model.

        :param params: 1-d numpy array with the model parameters.
        :return: None.
        """
        raise NotImplementedError

    def value(self, daily_rets, signals=None):
        """
        Calculate the portfolio value when rebalancing the portfolio daily.
        The stock-weights are calculated from the given signals.

        :param daily_rets: 2-d numpy array with daily returns for the stocks.
        :param signals: 2-d numpy array with daily signals for the stocks.
        :return: 1-d numpy array with the cumulative portfolio value.
        """

        # Map the signals to stock-weights.
        weights, weights_cash = self.get_weights(signals=signals)

        # Calculate the weighted daily returns of the stocks.
        weighted_daily_rets = np.sum(daily_rets * weights, axis=1) + weights_cash

        # Accumulate the weighted daily returns to get the portfolio value.
        value = np.cumprod(weighted_daily_rets)

        # Normalize so it starts at 1.0
        value /= value[0]

        return value

    def _optimize(self):
        """
        Optimize the portfolio model's parameters using SciPy.

        This is the SINGLE-OBJECTIVE version that uses the DE optimizer.

        :return: None.
        """
        self.optimize_result = minimize(func=self._fitness,
                                        bounds=self._bounds,
                                        maxiter=300,
                                        polish=True)

    def _limit_weights(self, weights):
        """
        Limit stock-weights between self.min_weights and self.max_weights.
        Also ensure the stock-weights sum to 1.0 or less.

        :param weights: 2-d numpy array with stock-weights between 0.0 and 1.0
        :return: 2-d numpy array with limited stock-weights.
        """

        # We could just clip the weights, but if they were created from
        # e.g. a sigmoid-mapping then a hard-clip would destroy the softness.
        # So we assume weights are between 0.0 and 1.0 so we can rescale them.
        # We do not assert this because there could be tiny floating-point
        # rounding errors that are unimportant and would then cause a crash.

        # Scale weights to be between min and max.
        weights = weights * (self.max_weights - self.min_weights) + self.min_weights

        # Ensure sum(weights) <= 1
        weights_sum = np.sum(weights, axis=1)
        mask = (weights_sum > 1.0)
        weights[mask, :] /= weights_sum[mask, np.newaxis]

        # Recalculate the weight-sum for each day.
        weights_sum = np.sum(weights, axis=1)

        # If the sum of stock-weights for a day is less than 1.0
        # then let the remainder be the cash-weight.
        weights_cash = 1.0 - weights_sum

        return weights, weights_cash

    def _fitness(self, params):
        """
        Calculate the fitness-value that is to be minimized.

        This should be a good measure of performance for the portfolio model.

        :param params: Parameters for the portfolio-model.
        :return: Float for the fitness-value to be minimized.
        """

        # Set the model parameters received from the optimizer.
        self._set_parameters(params=params)

        # Calculate the cumulative portfolio value using the training-data.
        # This uses the portfolio-model with the parameters we have just set,
        # so we can evaluate how well those parameters perform.
        value = self.value(daily_rets=self.daily_rets_train,
                           signals=self.signals_train)

        # If you just want to optimize the return over the whole period,
        # you can just return the last value divided by the first value.
        # Note that the value is negated because we are doing minimization.
        # return - value[-1] / value[0]

        # Annualized returns for all 1-year investment periods.
        rets_1year = value[365:] / value[:-365]

        # Annualized returns for all 5-year investment periods.
        years = 5
        days = int(365.25 * years)
        rets_5years = (value[days:] / value[:-days]) ** (1 / years)

        # Use the mean-log returns for 5-year returns as the main fitness value.
        # This fitness-measure is also known as the Kelly Criterion.
        fitness = np.mean(np.log(rets_5years))

        # We can penalize the main fitness value to shape the returns in
        # different ways. For example, if more than 7% of the 1-year returns
        # are losses, then we severely penalize the fitness, so that we
        # strongly prefer portfolio-models with few annual losses, but this
        # may cause the long-term returns to suffer.
        prob = np.sum(rets_1year < 1.0) / len(rets_1year)
        if prob > 0.07:
            fitness -= 100

        # Note the fitness-value is negated because we are doing minimization.
        return -fitness


class EqualWeights(Model):
    """
    Portfolio model where the stock-weights are always equal.
    """

    def __init__(self, num_stocks, use_cash=False):
        """
        Create object instance.

        This is a special case because the portfolio-model is so simple.
        We also don't call Model.__init__() because the model should not
        be optimized.

        :param num_stocks:
            Number of stocks in the portfolio.

        :param use_cash:
            Boolean whether to use cash as an equal part of the portfolio.
        """

        # Copy args.
        self.num_stocks = num_stocks
        self.use_cash = use_cash

    def get_weights(self, signals=None):
        """
        Get the stock-weights for the portfolio-model.

        :param signals: Ignored.
        :return: (weights: 2-d numpy array, weights_cash: 1-d numpy array)
        """

        if self.use_cash:
            # Stocks and cash get equal weights.
            weight = 1.0 / (self.num_stocks + 1)
            weights_cash = weight
        else:
            # Only use stocks and no cash in the portfolio.
            weight = 1.0 / self.num_stocks
            weights_cash = 0.0

        # Create a 2-dim array with the equal stock-weights,
        # so it can easily be multiplied and broadcast with daily returns.
        weights = np.full(shape=(1, self.num_stocks), fill_value=weight)

        return weights, weights_cash


class FixedWeights(Model):
    """
    Portfolio model where the stock-weights are always held fixed,
    but the best stock-weights are found through optimization.

    This version is for SINGLE-OBJECTIVE optimization.
    """

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    @property
    def _bounds(self):
        """Parameter bounds for the portfolio-model."""

        # We want to find the best fixed weights between 0.0 and 1.0
        return [(0.0, 1.0)] * self.num_stocks

    def _set_parameters(self, params):
        """
        Unpack and set the parameters for the portfolio model.

        :param params: 1-d numpy array with the model parameters.
        :return: None.
        """

        # The parameters are actually the raw stock-weights between 0.0 and 1.0
        # which are then limited between min_weights and max_weights.
        self._weights, self._weights_cash = self._limit_weights(weights=params[np.newaxis, :])

    def get_weights(self, signals=None):
        """
        Get the stock-weights for the portfolio-model.

        :param signals: Ignored.
        :return: (weights: 2-d numpy array, weights_cash: 1-d numpy array)
        """

        return self._weights, self._weights_cash


class AdaptiveWeights(Model):
    """
    Portfolio model where the stock-weights are mapped from predictive signals
    using the basic function: weight = sigmoid(a * signal + b) so we want to
    find the parameters a and b that result in the best performance according
    to the fitness function in Model._fitness().

    This version is for SINGLE-OBJECTIVE optimization.
    """

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    @property
    def _bounds(self):
        """Parameter bounds for the portfolio-model."""

        # We want to find the a and b parameters for each stock.
        # We allow both a and b to be between e.g. -10.0 and 10.0
        k = 10.0
        return [(-k, k)] * 2 * self.num_stocks

    def _set_parameters(self, params):
        """
        Unpack and set the parameters for the portfolio model.

        :param params: 1-d numpy array with the model parameters.
        :return: None.
        """
        self._a = params[0:self.num_stocks]
        self._b = params[self.num_stocks:]

    def get_weights(self, signals):
        """
        Get the stock-weights for the portfolio-model.

        :param signals: 2-d numpy array with signals for the stocks.
        :return: (weights: 2-d numpy array, weights_cash: 1-d numpy array)
        """

        # Linear mapping.
        weights = signals * self._a + self._b

        # Use sigmoid-function to softly limit between 0.0 and 1.0
        weights = _sigmoid(weights)

        # Limit the weights between min_weights and max_weights.
        weights, weights_cash = self._limit_weights(weights=weights)

        return weights, weights_cash


########################################################################
