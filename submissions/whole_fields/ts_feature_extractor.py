import numpy as np


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the vector of input variables at time t.

        Spatial variables will be concatenated.
        """
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        # We convert the Dataset into a 4D DataArray
        X_xr = X_ds.to_array()
        # We convert it into np array, put the t axis first
        X_array_t_first = np.swapaxes(X_xr.values, 0, 1)
        shape = X_array_t_first.shape
        # We reshape it to create one vector per time step, and slice the
        # valid range
        X_array = X_array_t_first.reshape(
            shape[0], shape[1] * shape[2] * shape[3])[valid_range]
        return X_array
