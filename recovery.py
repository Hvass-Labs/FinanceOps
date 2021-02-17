########################################################################
#
# Functions for calculating the number of days before a stock
# has recovered from losses and associated probabilities.
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

from data_keys import *
import numpy as np

########################################################################
# Public functions.


def recovery_days(tot_ret):
    """
    Given a time-series with the Total Return of a stock,
    create an array of integers of the same length as the
    time-series, which tells us how many time-steps before
    the time-series increased above that level. This is
    used to measure how many days it took for a stock to
    recover from a temporary loss.

    Note that an integer value of zero means that the
    time-series never reached the given value again.

    The algorithm has linear time-complexity because it uses
    a stack to keep track of the previously seen elements.

    :param tot_ret: Pandas Series with Total Return.
    :return: Numpy array of integers.
    """

    # Init array of integers for the number of recovery-days.
    rec_days = np.zeros(len(tot_ret), dtype=np.int)

    # Init the stack holding previously seen elements.
    # The stack holds tuples: (idx:int, val:float)
    stack = []

    # For each element in the Total Return time-series.
    for cur_idx in range(len(tot_ret)):
        # Value of the current element in the time-series.
        cur_val = tot_ret.iloc[cur_idx]

        # Remove all previous elements from the stack whose value
        # is less than the current element from the time-series.
        while len(stack) > 0 and stack[-1][1] < cur_val:
            # Pop a previously seen element from the stack.
            last_idx, last_val = stack.pop()

            # Set the number of recovery-days for that element.
            rec_days[last_idx] = cur_idx - last_idx

        # Push the current element and its index to the stack.
        stack.append((cur_idx, cur_val))

    return rec_days


def prob_recovery(df, start_date, end_date, num_days):
    """
    Calculate the probability of recovering losses within the
    given number of days.

    For each day in the Total Return series, we first calculate
    the number of days until the Total Return has increased,
    that is, until any temporary losses have been recovered.

    Then we calculate the probability for those recovery-days
    being less than a given number of days.

    :param df: Pandas DataFrame with TOTAL_RETURN data.
    :param start_date: Only use the Total Return from this date.
    :param end_date: Only use the Total Return to this date.
    :param num_days: List of ints for the number of recovery-days.
    :return: List of probabilities.
    """

    def prob(days):
        # Calculate the probability of rec_days < days using a boolean mask.
        mask = (rec_days <= days)
        return np.sum(mask) / len(mask)

    # Only use a part of the time-series.
    tot_ret = df[TOTAL_RETURN][start_date:end_date].dropna()

    # Calculate the recovery-days for all days in the time-series.
    rec_days = recovery_days(tot_ret=tot_ret)

    # Calculate the probabilities of recovering losses within
    # the given number of days.
    probs = [prob(days=days) for days in num_days]

    return probs


########################################################################
