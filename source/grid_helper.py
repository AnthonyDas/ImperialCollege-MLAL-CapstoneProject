import pandas as pd
import numpy as np
import math


def hypercube_grid(steps_per_dim, x_col_names):
    """
    Generates a grid of points within a unit hypercube [0, 1]^d to be
    used as an exhaustive search grid for the global optimisation of
    acquisition functions. It generates linear spacings for each dimension
    based on the provided 'steps_per_dim' resolutions.

    Args:
        steps_per_dim (list of int): The number of intervals for each dimension. 
            For a 2D grid, [10, 10] would create an 11x11 point grid (including 
            endpoints).
        x_col_names (list of str): Names of the input features. The length of 
            this list defines the dimensionality of the hypercube.

    Returns:
        np.ndarray: A 2D array of shape (N, d), where N is the total number 
            of grid points and d is the number of dimensions.

    Example:
        >>> hypercube_grid([2, 2], ['x1', 'x2'])
        len(x_grid): 9
        array([[0. , 0. ], [0. , 0.5], [0. , 1. ],
               [0.5, 0. ], [0.5, 0.5], [0.5, 1. ],
               [1. , 0. ], [1. , 0.5], [1. , 1. ]])
    """
    
    # Create 1D linear spaces for each dimension from 0 to 1
    xi_vals = [np.linspace(0, 1, steps + 1) for steps in steps_per_dim] # +1 include endpoint
    
    xi_grid = np.meshgrid(*xi_vals, indexing='ij') # * to pass inner arrays, rather than the array of arrays

    #x_grid = [[xi, yi] for xi, yi in zip(x1_grid.ravel(), x2_grid.ravel())]
    #x_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    x_grid = np.stack(xi_grid, axis=-1).reshape(-1, len(steps_per_dim)) # type = np.array
    
    #x_grid = pd.DataFrame(x_grid, columns = x_col_names)
    
    print(f'len(x_grid): {len(x_grid)}')
    
    return x_grid
    