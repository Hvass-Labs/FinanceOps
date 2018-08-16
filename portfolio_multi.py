########################################################################
#
# Classes for optimizing and allocating portfolios of stocks.
#
# This version is for MULTI-OBJECTIVE optimization. It is very
# similar to portfolio.py except that it uses another optimizer
# and has some minor modifications. Parts of the source-code could
# have been reused from portfolio.py but it is probably easier to
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
import pygmo as pg  # This has the NSGA-2 optimizer.

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

    This version is for MULTI-OBJECTIVE optimization.
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
    def bounds(self):
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
        Optimize the portfolio model's parameters.

        This is the MULTI-OBJECTIVE version that uses the NSGA-2 optimizer.

        :return: None.
        """

        class Problem:
            """
            Wrapper for the Model-class that connects it with
            the optimizer. This is necessary because the optimizer
            creates a deep-copy of the problem-object passed to it,
            so it does not work when passing the Model-object directly.
            """

            def __init__(self, model):
                """
                :param model: Object-instance of the Model-class.
                """
                self.model = model

            def fitness(self, params):
                """Calculate and return the fitness for the given parameters."""
                return self.model.fitness(params=params)

            def get_bounds(self):
                """Get boundaries of the search-space."""
                return self.model.bounds

            def get_nobj(self):
                """Get number of fitness-objectives."""
                return self.model.num_objectives

        # Create a problem-instance.
        problem = Problem(model=self)

        # Create an NSGA-2 Multi-Objective optimizer.
        optimizer = pg.algorithm(pg.nsga2(gen=500))

        # Create a population of candidate solutions.
        population = pg.population(prob=problem, size=200)

        # Optimize the problem.
        population = optimizer.evolve(population)

        # Save the best-found parameters and fitnesses for later use.
        self.best_parameters = population.get_x()
        self.best_fitness = population.get_f()

        # Sorted index for the fitnesses.
        idx_sort = np.argsort(self.best_fitness[:, 0])

        # Sort the best-found parameters and fitnesses.
        self.best_parameters = self.best_parameters[idx_sort]
        self.best_fitness = self.best_fitness[idx_sort]

    def use_best_parameters_max_return(self):
        """
        Use the best found model-parameters that maximize the mean return.
        """

        # The parameters are already sorted according to max return,
        # so get the first set of parameters in the list.
        params = self.best_parameters[0]

        # Use these parameters as the model's active parameters.
        self._set_parameters(params=params)

    def use_best_parameters_min_prob_loss(self):
        """
        Use the best found model-parameters that minimize the probability of loss.
        """

        # The parameters are already sorted so the lowest prob. of loss
        # are at the end of the list.
        params = self.best_parameters[-1]

        # Use these parameters as the model's active parameters.
        self._set_parameters(params=params)

    def use_best_parameters(self, max_prob_loss=1.0):
        """
        Use the best found model-parameters that maximize the mean return
        while having a probability of loss that is less than that given.

        :param max_prob_loss: Max allowed probability of loss.
        """

        try:
            # The parameters are already sorted, so we get the parameters with
            # the highest mean and probability of loss below the given limit.
            # This is a little cryptic to understand, but you can try and print
            # the array self.best_fitness to understand how it works.
            idx = np.min(np.argwhere(self.best_fitness[:, 1] <= max_prob_loss))
            params = self.best_parameters[idx]

            # Use these parameters as the model's active parameters.
            self._set_parameters(params=params)
        except ValueError:
            # Print error-message if the probability of loss was too low.
            msg = "Error: max_prob_loss is too low! Must be higher than {0:.3f}"
            min_prob_loss = np.min(self.best_fitness[:, 1])
            print(msg.format(min_prob_loss))

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

    @property
    def num_objectives(self):
        """Number of fitness objectives to optimize."""
        return 2

    def fitness(self, params):
        """
        Calculate the MULTIPLE fitness-objectives that are to be minimized.

        :param params: Parameters for the portfolio-model.
        :return: (fitness1, fitness2) tuple with two floats.
        """

        # Set the model parameters received from the optimizer.
        self._set_parameters(params=params)

        # Calculate the cumulative portfolio value using the training-data.
        # This uses the portfolio-model with the parameters we have just set,
        # so we can evaluate how well those parameters perform.
        value = self.value(daily_rets=self.daily_rets_train,
                           signals=self.signals_train)

        # Portfolio returns for all 1-year periods.
        rets_1year = value[365:] / value[:-365]

        # Mean return for all 1-year periods.
        mean_return = np.mean(rets_1year) - 1.0

        # Portfolio returns for all 3-month periods.
        rets_3month = value[90:] / value[:-90]

        # Probability of loss for all 3-month periods.
        prob_loss = np.sum(rets_3month < 1.0) / len(rets_3month)

        # Fitness objectives.
        # Note the fitness-value is negated because we are doing minimization.
        fitness1 = -mean_return
        fitness2 = prob_loss

        return [fitness1, fitness2]


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

    This version is for MULTI-OBJECTIVE optimization.
    """

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    @property
    def bounds(self):
        """Parameter bounds for the portfolio-model."""

        # We want to find the best fixed weights between 0.0 and 1.0
        lo = np.zeros(self.num_stocks, dtype=np.float)
        hi = np.ones(self.num_stocks, dtype=np.float)

        return lo, hi

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
    to the fitness function in Model.fitness().

    This version is for MULTI-OBJECTIVE optimization.
    """

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    @property
    def bounds(self):
        """Parameter bounds for the portfolio-model."""

        # We want to find the a and b parameters for each stock.
        # We allow both a and b to be between e.g. -10.0 and 10.0
        k = 10.0
        lo = [-k] * self.num_stocks * 2
        hi = [k] * self.num_stocks * 2

        return lo, hi

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
