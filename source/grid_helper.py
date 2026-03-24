import pandas as pd
import numpy as np
import math

# TODO
def initial_grid_settings(x, step_size):
    # Obtain dimensions from first x point
    dimensions = len(x[0])
    
    bounds_per_dim = [ [0,1] for _ in range(dimensions)] # Always start spanning the unit hypercube
    step_size_per_dim = [ step_size for _ in range(dimensions) ]
    
    return bounds_per_dim, step_size_per_dim

# TODO
def next_grid_settings(next_pt, step_size_per_dim, steps_per_dim):
    next_bounds_per_dim = []
    next_step_size_per_dim = []

    ndigits = 6 # Round to 6 d.p. since that's how granular BBO submissions are
    
    for x, step_size, steps in list(zip(next_pt, step_size_per_dim, steps_per_dim)):  
        half_width = max(1e-6, round((0.5 * step_size), ndigits=ndigits)) # Can't be smaller than 1e-6
        lower_bound = max(0, round(x - half_width, ndigits=ndigits)) # Don't go lower than 0
        upper_bound = min(1, round(x + half_width, ndigits=ndigits)) # Don't go higher than 1
        next_bounds_per_dim.append([lower_bound, upper_bound])

        next_step_size = max(1e-6, round((step_size/steps), ndigits=ndigits)) # Can't be smaller than 1e-6
        next_step_size_per_dim.append(next_step_size)
        
    return next_bounds_per_dim, next_step_size_per_dim
    
# TODO
def create_grid(bounds_per_dim, step_size_per_dim):
    assert len(bounds_per_dim) > 0
    assert len(step_size_per_dim) > 0
    
    if len(bounds_per_dim) != 1 and len(step_size_per_dim) != 1:
        assert len(bounds_per_dim) == len(step_size_per_dim)

    dimensions = max(len(bounds_per_dim), len(step_size_per_dim))

    # TODO expand bounds_per_dim and step_size_per_dim if necessary

    steps_per_dim = []
    
    grid = [[]]
    grid_next = []
    
    for i in range(dimensions):
        bounds = bounds_per_dim[i]
        step_size = step_size_per_dim[i]
        
        steps = math.ceil((bounds[1] * 1e06 - bounds[0] * 1e06)/ (step_size * 1e06))
        steps_per_dim.append(steps)

        stop = round(bounds[0] + steps * step_size, ndigits=6)
        coords = np.linspace(bounds[0], stop, steps + 1, endpoint=True) # +1 to include both ends

        # Round to 6 d.p. since that's how granular BBO submissions are
        coords = [ float(round(coord, ndigits=6)) for coord in coords]

        # print(f'DEBUG: coords:\n{coords}')
        
        for pt in grid:
            for coord in coords:
                grid_next.append(pt + [coord])
        grid = grid_next
        grid_next = []
        # print(f'DEBUG: grid after dim {i}:\n{grid}')
        
    return np.array(grid), steps_per_dim


def smallest_possible_step_sizes(step_size_per_dim):
    is_smallest = [(step_size == 1e-6) for step_size in step_size_per_dim ]
    return sum(is_smallest) == len(step_size_per_dim)



def hypercube_grid(steps_per_dim, x_col_names):

    # Create 1D linear spaces for each dimension from 0 to 1
    xi_vals = [np.linspace(0, 1, steps + 1) for steps in steps_per_dim] # +1 include endpoint
    
    xi_grid = np.meshgrid(*xi_vals, indexing='ij') # * to pass inner arrays, rather than the array of arrays

    #x_grid = [[xi, yi] for xi, yi in zip(x1_grid.ravel(), x2_grid.ravel())]
    #x_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    x_grid = np.stack(xi_grid, axis=-1).reshape(-1, len(steps_per_dim)) # type = np.array
    
    #x_grid = pd.DataFrame(x_grid, columns = x_col_names)
    
    print(f'len(x_grid): {len(x_grid)}')
    
    return x_grid