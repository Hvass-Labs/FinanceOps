########################################################################
#
# Functions for calculating Total Return, Annualized Returns, etc.
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

import pandas as pd
from data_keys import *

########################################################################
# Public functions.


def total_return(df):
    """
    Calculate the "Total Return" of a stock when dividends are
    reinvested in the stock.

    The formula is:

    Total_Return[t] = Total_Return[t-1] * (Dividend[t] + Share_Price[t]) / Share_Price[t-1]

    :param df:
        Pandas data-frame assumed to contain SHARE_PRICE and DIVIDEND.
    :return:
        Pandas series with the Total Return.
    """

    # Copy the relevant data so we don't change it.
    df2 = df[[SHARE_PRICE, DIVIDEND]].copy()

    # Fill NA-values in the Dividend-column with zeros.
    df2[DIVIDEND].fillna(0, inplace=True)

    # Calculate the daily Total Return.
    tot_ret_daily = (df2[DIVIDEND] + df2[SHARE_PRICE]) / df2[SHARE_PRICE].shift(1)

    # Calculate the cumulative Total Return.
    tot_ret = tot_ret_daily.cumprod()

    # Replace the first row's NA with 1.0
    tot_ret.values[0] = 1.0

    return tot_ret


def annualized_returns(series, years):
    """
    Calculate the annualized returns for all possible
    periods of the given number of years.

    For example, given the Total Return of a stock we want
    to know the annualized returns of all holding-periods
    of 10 years.

    :param series:
        Pandas series e.g. with the Total Return of a stock.
        Assumed to be daily data.
    :param years:
        Number of years in each period.
    :return:
        Pandas series of same length as the input series. Each
        day has the annualized return of the period starting
        that day and for the given number of years. The end of
        the series has NA for the given number of years.
    """

    # Number of days to shift data. All years have 365 days
    # except leap-years which have 366 and occur every 4th year.
    # So on average a year has 365.25 days.
    days = int(years * 365.25)

    # Calculate annualized returns for all periods of this length.
    # Note: It is important we have daily (interpolated) data,
    # otherwise the series.shift(365) would shift much more than
    # a year, if the data only contains e.g. 250 days per year.
    ann_return = (series.shift(-days) / series) ** (1 / years) - 1.0

    return ann_return


def prepare_ann_returns(df, years, key=PSALES, subtract=None):
    """
    Prepare annualized returns e.g. for making a scatter-plot.
    The x-axis is given by the key (e.g. PSALES) and the y-axis
    would be the annualized returns.

    :param df:
        Pandas DataFrame with columns named key and TOTAL_RETURN.
    :param years:
        Number of years for annualized returns.
    :param key:
        Name of the data-column for x-axis e.g. PSALES or PBOOK.
    :param subtract:
        Pandas Series to be subtracted from ann-returns
        to adjust for e.g. growth in sales-per-share.
    :return:
        (x, y) Pandas Series with key and adjusted ANN_RETURN.
    """

    # Create a new data-frame so we don't modify the original.
    # We basically just use this to sync the data we are
    # interested in for the common dates and avoid NA-data.
    df2 = pd.DataFrame()

    # Copy the key-data e.g. PSALES.
    df2[key] = df[key]

    # Calculate all annualized returns for all periods of
    # the given number of years using the Total Return.
    ann_return = annualized_returns(series=df[TOTAL_RETURN],
                                    years=years)

    if subtract is None:
        # Add the ann-returns to the new data-frame.
        df2[ANN_RETURN] = ann_return
    else:
        # Calculate all annaulized returns for the series
        # that must be subtracted e.g. sales-per-share.
        ann_return_subtract = annualized_returns(series=subtract,
                                                 years=years)

        # Subtract the ann. returns for the total return
        # and the adjustment (e.g. sales-per-share).
        # Then add the result to the new data-frame.
        df2[ANN_RETURN] = ann_return - ann_return_subtract

    # Drop all rows with NA.
    df2.dropna(axis=0, how='any', inplace=True)

    # Retrieve the relevant data.
    x = df2[key]
    y = df2[ANN_RETURN]

    return x, y


def bond_annualized_returns(df, num_years):
    """
    Calculate the annualized returns from investing and reinvesting in a bond.

    This results in a list of Pandas Series with the annualized returns for
    [1, 2, ..., max_years] investment years.

    For example ann_returns[0] are for 1-year investment periods and
    ann_returns[9] are for 10-year periods.

    :param df:
        Pandas DataFrame with BOND_YIELD data for 1-year maturity.
    :param num_years:
        Max number of investment years.
    :return:
        List of Pandas Series.
    """

    # The idea is to repeatedly shift the bond-yields
    # and update a cumulative product so as to get the
    # compounded return through the years.

    # Init the list of annualized returns. For 1-year
    # investment periods these are just the bond-yields.
    ann_returns = [df[BOND_YIELD].copy()]

    # Init the cumulative product of bond-yields,
    # which will be updated when reinvesting in the
    # bonds through the years.
    cum_prod = df[BOND_YIELD].copy() + 1.0

    # Init the bond-yields shifted one year.
    # These will be shifted 365 steps for each year.
    shifted = cum_prod.copy()

    # For increasing number of investment years.
    # The bond-yields were used as the 1st year above.
    for years in range(2, num_years + 1):
        # Shift the bond-yields one year.
        # Note leap-years are not taken into account so
        # there will be a slight drift for longer periods,
        # but it probably only causes a very small error.
        shifted = shifted.shift(-365)

        # Accumulate the bond-yields so cum_prod holds the
        # cumulative return from reinvesting in the bonds.
        cum_prod *= shifted

        # Remove NA from the end of the series.
        cum_prod.dropna(inplace=True)

        # Calculate the annualized returns.
        ann_ret = cum_prod ** (1 / years) - 1.0

        # Rename the data-column.
        ann_ret.rename(ANN_RETURN)

        # Add to the list of annualized returns for all years.
        ann_returns.append(ann_ret)

    return ann_returns


########################################################################
