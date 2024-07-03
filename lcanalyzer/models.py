"""Module containing models representing lightcurves.

The Model layer is responsible for the 'business logic' part of the software.

The lightcurves are saved in a table (2D array) where each row corresponds to a single observation. 
Depending on the dataset (LSST or Kepler), a table can contain observations of a single or several objects, 
in a single or different bands.
"""

import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle


def load_dataset(filename):
    """Load a table from CSV file.

    :param filename: The name of the .csv file to load
    :returns: pd.DataFrame with the data from the file.
    """
    return pd.read_csv(filename)


def mean_mag(data, mag_col):
    """Calculate the mean magnitude of a lightcurve periods

    :param data: The data frame
    :param mag_col: a string with the name of the column for calculating the mean value
    :returns: A float with the mean value of the column.
    """
    return data[mag_col].mean()


def max_mag(data, mag_col):
    """Calculate the max magnitude of a lightcurve periods

    :param data: pd.DataFrame with the magnitudes
    :param mag_col: a string with the name of the column for calculating the max value
    :returns: A float with the max value of the column.
    """
    return data[mag_col].max()


def min_mag(data, mag_col):
    """Calculate the min magnitude of a lightcurve periods

    :param data: pd.DataFrame with observed magnitudes for a single source
    :param mag_col: a string with the name of the column for calculating the min value
    :returns: A float with the min value of the column.
    """
    return data[mag_col].min()


def calc_stats(lc, bands, mag_col):
    # Calculate max, mean and min values for all bands of a light curve
    stats = {}
    for b in bands:
        stat = {}
        stat["max"] = max_mag(lc[b], mag_col)
        stat["mean"] = mean_mag(lc[b], mag_col)
        stat["min"] = min_mag(lc[b], mag_col)
        stats[b] = stat
    return pd.DataFrame.from_records(stats)


def normalize_lc(df, mag_col):
    # Normalize a single light curve
    min = min_mag(df, mag_col)
    max = max_mag((df - min), mag_col)
    lc = (df[mag_col] - min) / max
    lc = lc.fillna(0)
    return lc
