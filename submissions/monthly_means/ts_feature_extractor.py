import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the monthly averages of the ice_area.

        Corresponding to the month to predict.
        The code could be simplified but in this way it is general, can be
        used for the other variables as well.
        """
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = np.arange(X_ds.attrs['n_burn_in'], len(X_ds['time']))
        # We convert the Dataset into a 4D DataArray
        X_xr = X_ds.to_array()
        # We compute the mean over the lat and lon axes
        mean_array = np.mean(X_xr, axis=(2, 3)).values
        # We group the 8 monthly series into 8 x 12 monthly groups of series
        monthly_groups = mean_array.reshape(
            (mean_array.shape[0], 12, -1), order='F')
        # We compute cumulative means in each group
        monthly_means = np.cumsum(monthly_groups, axis=2)\
            / (1. + np.arange(monthly_groups.shape[2]))
        # We repeat each mean 12 times
        monthly_means_per_month = np.repeat(monthly_means, 12, axis=2)
        # We pad m 0s to the series corresponding to month m, no single-line
        # operation for this
        for j in range(monthly_means_per_month.shape[0]):
            for m in range(12):
                monthly_means_per_month[j, m] = np.roll(
                    monthly_means_per_month[j, m], m)
                monthly_means_per_month[j, m, :m] = 0
        # We reshape and transpose it into one vector per month
        monthly_ice_area_mean = monthly_means_per_month[0]
        # At each month t we use the running mean correponting to month t - 8
        X_array = np.array(
            [monthly_ice_area_mean[(t + X_ds.n_lookahead - 12) % 12][t]
             for t in range(monthly_ice_area_mean.shape[1])])
        # We slice the valid range
        X_array = X_array[valid_range].reshape(-1, 1)
        return X_array
