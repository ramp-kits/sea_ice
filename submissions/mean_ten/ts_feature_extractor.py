import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        self.window_size = 10

    def transform(self, X_ds):
        """Compute the vector of input variables in window of a given size.

        Compute the vector of input variables at times
        [t, t-1, ... t-window_size+1] then concatenate. Spatial variables
        will be averaged along lat and lon coordinates.
        """
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = np.arange(X_ds.attrs['n_burn_in'], len(X_ds['time']))
        # We convert the Dataset into a 4D DataArray
        X_xr = X_ds.to_array()
        # We compute the mean over the lat and lon axes
        mean_xr = np.mean(X_xr, axis=(2, 3))
        mean_array_transposed = mean_xr.values.T
        # We concatenate the past window_size means
        mean_array_c = np.concatenate(
            [np.roll(mean_array_transposed, i)
             for i in range(self.window_size)], axis=1)
        # We slice the valid range
        X_array = mean_array_c[valid_range]
        return X_array
