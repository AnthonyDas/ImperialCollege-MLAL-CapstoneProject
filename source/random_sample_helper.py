import numpy as np

'''
By reusing the exact same standard Normal samples within Monte Carlo, we ensure that differing samples
won't be a source of randomness between different Monte Carlo runs, e.g. when estimating EI at different
grid points. Also, by setting the (same) seed, the samples will be deterministic, and results will be repeatable.
'''
class DeterministicStdNormalSampler:
    def __init__(self, n_samples=1000, seed=42):
        self.z = None # The actual standard Normal samples 
        self.n_samples = None # The number of samples
        
        if n_samples % 2 != 0:
            n_samples += 1  # Must be even for antithetic samples 
        
        rng = np.random.default_rng(seed)   
        
        n_half = n_samples // 2 # Floor div for integer 
        z_half = rng.standard_normal(n_half)
        
        # Antithetic pairing
        self.z = np.concatenate([z_half, -z_half])
        self.n_samples = len(self.z)
        self.seed = seed
        