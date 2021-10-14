########################################################################
#
# Various statistical functions.
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
from scipy.stats import norm

########################################################################


def normal_prob_loss(mean, std):
    """
    Probability of loss for a normal distribution.

    :param mean: Float with mean.
    :param std: Float with standard deviation.
    :return: Float with probability of loss.
    """
    return norm.cdf(0.0, loc=mean, scale=std)


def normal_prob_less_than(mean1, std1, mean2, std2):
    """
    Probability that X1 < X2 for two independent and normal-distributed random
    variables X1 and X2, where X1 ~ N(mean1, std1^2) and X2 ~ N(mean2, std2^2)

    :param mean1: Float with mean for X1.
    :param std1: Float with std.dev. for X1.
    :param mean2: Float with mean for X2.
    :param std2: Float with std.dev. for X2.
    :return: Float with probability that X1 < X2.
    """
    # The difference of two normal random variables is also a random normal
    # variable with the following mean and std.dev. so we can simply calculate
    # the probability of that new random variable being less than zero.
    mean = mean1 - mean2
    std = np.sqrt(std1 ** 2 + std2 ** 2)
    return norm.cdf(0.0, loc=mean, scale=std)


########################################################################
