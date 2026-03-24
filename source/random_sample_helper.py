import numpy as np


class DeterministicStdNormalSampler:
    """
    Generates and stores deterministic standard Normal samples for Monte Carlo usage.

    Employs antithetic samples to ensure the sample mean is exactly zero. By reusing the
    same Normal samples across different Monte Carlo calculations, we ensure that differing
    samples won't be a source of randomness impacting results.

    Attributes:
        z (np.ndarray): The array of standard Normal samples, N(0, 1).
        n_samples (int): The total number of samples (will always be even).
        seed (int): The random seed used for reproducibility.
    """
    
    def __init__(self, n_samples=1000, seed=42):
        """
        Initialises the sampler and generates the deterministic sample set.

        Args:
            n_samples (int): The target number of samples.
            seed (int): The random number generator seed.
        """
        
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
        