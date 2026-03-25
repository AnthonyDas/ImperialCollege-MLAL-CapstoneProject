import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer
from typing import Final

from random_sample_helper import DeterministicStdNormalSampler


class BaseTransformer:
    """
    Abstract base class for target and feature transformers.

    Enforces a consistent API for fitting, transforming, and inverse-transforming 
    data. Employs Monte Carlo to inverse transform (mean and variance) from
    transformed space back to the original space.

    Attributes:
        normal_sampler (DeterministicStdNormalSampler): Provided a fixed set of standard 
            normal samples used for repeatable Monte Carlo calculations.
    """
    
    def __init__(self, n_samples=10000):
        # Ensures we have the same Normal samples for all Monte Carlo transform inversions
        self.normal_sampler = DeterministicStdNormalSampler(n_samples=n_samples)
    
    def fit(self, y):
        raise NotImplementedError
    
    def transform(self, y):
        raise NotImplementedError

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)   

    def _inverse_transform(self, z):
        raise NotImplementedError

    def inverse_transform(self, z): 
        # Ensure 2D for the internal method '_inverse_transform', then return as 1D.
        z_arr = np.asarray(z)
        return self._inverse_transform(z_arr.reshape(-1, 1)).ravel()
    
    def inverse_transform_dist(self, mean_t, std_t):
        # Employ Monte Carlo to compute inverse y-transform
        
        n_points = len(mean_t)
        
        # Generate samples in transformed space
        z_scores = self.normal_sampler.z # Shape: (n_samples,)
        
        # Generate all Monte Carlo samples in transformed space upfront
        samples_t = mean_t[:, np.newaxis] + std_t[:, np.newaxis] * z_scores # Shape: (n_points, n_samples)

        # Inverse transform
        #samples_t_flat = samples_t.reshape(-1, 1)
        #samples_original_flat = self._inverse_transform(samples_t_flat)
        #samples_original = samples_original_flat.reshape(n_points, -1)
        samples_original = self._inverse_transform(samples_t.reshape(-1, 1)).reshape(n_points, -1)

        # Calculate the mean and std in y-original space
        # Use nanmean/nanvar to handle extreme tail errors - TODO: log/error when Nans detected
        mean_y = np.nanmean(samples_original, axis=1)
        std_y = np.nanstd(samples_original, axis=1)
        
        return mean_y, std_y

        
class IdentityTransformer(BaseTransformer):
    """Pass-through transformer that applies no data modifications."""
    
    def fit(self, y):
        return self
    
    def transform(self, y):
        return np.asarray(y) # No change
    
    def _inverse_transform(self, z):
        return np.asarray(z) # No change

    def inverse_transform_dist(self, mean_t, std_t):
        # For Identity, the distribution parameters don't change.
        return mean_t, std_t


class SklearnWrapper(BaseTransformer):
    """
    Wraps standard Scikit-Learn transformers to adhere to the BaseTransformer API.

    Args:
        transformer: Scikit-Learn transformer instance (e.g. StandardScaler).
    """
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def fit(self, y):
        self.transformer.fit(y)
        return self

    def transform(self, y):
        return self.transformer.transform(y)

    def _inverse_transform(self, z):
        # Ensure 2D for sklearn
        return self.transformer.inverse_transform(z)

        
class LogShiftTransformer(BaseTransformer):
    """
    Applies a Log(y + shift) transform where the shift ensures all values > 0.

    Range-based shift is used and calculated as `-min(y) + alpha * range(y)`.
    More robust than a simple log for datasets containing zero or negative values.
    """
    
    def __init__(self, alpha = 0.1):
        super().__init__()
        self.alpha = alpha
        self.shift_ = None
        self.y_min_ = None
        self.y_max_ = None
    
    def fit(self, y):
        y = np.asarray(y)
        
        self.y_min_ = np.min(y)
        self.y_max_ = np.max(y)
        
        y_range = self.y_max_ - self.y_min_
        
        # Handle constant target case
        if y_range == 0:
            y_range = 1.0
        
        self.shift_ = -self.y_min_ + (self.alpha * y_range)
        
        return self
    
    def transform(self, y):
        y = np.asarray(y)
        return np.log(y + self.shift_)

    def _inverse_transform(self, z):
        z = np.asarray(z)

        MAX_LOG = 700.0  # safe for float64
        z_safe = np.minimum(z, MAX_LOG)

        return np.exp(z_safe) - self.shift_

        
class LogShiftScaledTransformer(BaseTransformer):
    """
    This transformer applies LogShift followed by StandardScaler.
    
    This ensures the target is log-normal distributed and then centered/scaled 
    to have zero mean and unit variance.
    """
    
    def __init__(self):
        super().__init__()
        self.ls_ = LogShiftTransformer()
        self.scaler_ = StandardScaler()
    
    def fit(self, y):
        y = np.asarray(y).reshape(-1, 1)
        
        y_ls = self.ls_.fit_transform(y)
        self.scaler_.fit(y_ls)
        
        return self
    
    def transform(self, y):
        y = np.asarray(y).reshape(-1, 1)
        y_ls = self.ls_.transform(y)
        y_scaled = self.scaler_.transform(y_ls)
        return y_scaled.ravel()
    
    def _inverse_transform(self, z):
        z = np.asarray(z).reshape(-1, 1)
        
        y_unscaled = self.scaler_.inverse_transform(z)
        y_original = self.ls_._inverse_transform(y_unscaled)
        
        return y_original.ravel() 


class SymmetricLogTransformer(BaseTransformer):
    """
    Applies a symmetric log transform: sign(y) * log(1 + |y|/scale).

    Useful for data spanning orders of magnitude with both positive and negative values.
    """
    
    def __init__(self, alpha=0.5, beta=1.0, upper_q=0.95, lower_q=0.75, epsilon=1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.upper_q = upper_q # Upper quantile
        self.lower_q = lower_q # Lower quantile
        self.epsilon = epsilon
        
        self.scale_ = None
    
    def fit(self, y):
        y = np.asarray(y)
        
        q_high = np.quantile(y, self.upper_q)
        q_low  = np.quantile(y, self.lower_q)
        
        delta = q_high - q_low # Interquantile range
        
        s1 = self.alpha * abs(np.max(y))
        s2 = self.beta * delta
        
        self.scale_ = max(s1, s2, self.epsilon)
        
        return self
    
    def transform(self, y):
        y = np.asarray(y)
        return np.sign(y) * np.log1p(np.abs(y) / self.scale_) # log1p(x) = log(1 + x)

    def _inverse_transform(self, z):
        z = np.asarray(z)
    
        MAX_LOG = 700.0
        abs_z = np.minimum(np.abs(z), MAX_LOG)
    
        return np.sign(z) * self.scale_ * np.expm1(abs_z) # expm1(x) = exp(x - 1)

        
class SymmetricLogScaledTransformer(BaseTransformer):
    """
    This transformer applies SymmetricLog, followed by StandardScaler.
    
    Designed for datasets spanning multiple orders of magnitude that contain 
    both positive and negative values.
    """
    
    def __init__(self):
        super().__init__()
        self.st_ = SymmetricLogTransformer()
        self.scaler_ = StandardScaler()
    
    def fit(self, y):
        y = np.asarray(y).reshape(-1, 1)
        
        y_st = self.st_.fit_transform(y)
        self.scaler_.fit(y_st)
        
        return self
    
    def transform(self, y):
        y = np.asarray(y).reshape(-1, 1)
        y_st = self.st_.transform(y)
        y_scaled = self.scaler_.transform(y_st)
        return y_scaled.ravel()
    
    def _inverse_transform(self, z):
        z = np.asarray(z).reshape(-1, 1)
        
        y_unscaled = self.scaler_.inverse_transform(z)
        y_original = self.st_._inverse_transform(y_unscaled)
        
        return y_original.ravel()


# Extend PowerTransformer to make inverse_transform safe, i.e. prevent overflow
class SafePowerTransformer(PowerTransformer):
    """
    Extension of sklearn PowerTransformer applying clipping to prevent overflow.

    The intention is to prevent `inf` or `nan` results during inverse transformation
    by clipping input values to a safe numerical range for float64.
    """

    def _inverse_transform(self, z):
        z = np.asarray(z)
        z_clipped = np.clip(z, -1e6, 1e6)
        
        y = super().inverse_transform(z_clipped)

        return y
        

def get_y_transformers(inc_symlog = False):
    """
    Generates a dictionary of available y transformers.

    Args:
        inc_symlog (bool): Whether to include Symmetric Log variants.

    Returns:
        dict: Mapping of transformer names (str) to corresponding transformers.
    """
    
    transforms = {
        "none": IdentityTransformer(),
        "scaled": SklearnWrapper(StandardScaler()),
        "power": SklearnWrapper(SafePowerTransformer(method="yeo-johnson", standardize=False)),
        "pow-scaled": SklearnWrapper(SafePowerTransformer(method="yeo-johnson", standardize=True)),
        "logshift": LogShiftTransformer(),
        "logshift-scaled": LogShiftScaledTransformer(),
    }

    if inc_symlog:
        transforms["symlog"] = SymmetricLogTransformer()
        transforms["symlog-scaled"] = SymmetricLogScaledTransformer()
    
    return transforms

    
def get_x_transformers():
    """
    Generates a dictionary of available x transformers.

    Returns:
        dict: Mapping of transformer names (str) to corresponding transformers.
    """
    
    return {
        "none": IdentityTransformer(),
        "scaled": SklearnWrapper(StandardScaler()),
    }

    