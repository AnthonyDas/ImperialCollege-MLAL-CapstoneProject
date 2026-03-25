import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import minimize

import warnings

import distance_helper
import print_helper
from random_sample_helper import DeterministicStdNormalSampler


class BaseAcquisitionFn:
    """Base acquisition function to enforce a consistent API."""
    
    def __call__(self, mean_t, std_t):
        """Evaluate the acquisition function.

        Args:
            mean_t (np.ndarray): Predicted means in the transformed space.
            std_t (np.ndarray): Predicted standard deviations in the transformed space.

        Returns:
            np.ndarray: Acquisition values for each input point.
        """
        raise NotImplementedError


class UpperConfidenceBoundTransSpace(BaseAcquisitionFn):
    """Upper Confidence Bounds (UCB) acquisition function in y-transformed space."""
    
    def __init__(self, beta):
        """
        Args:
            beta (float): Exploration-exploitation trade-off parameter. Higher values encourage exploration.
        """
        self.beta = beta
    
    def __call__(self, mean_t, std_t):
        return mean_t + (self.beta * std_t)


class ExpectedImprovementTransSpace(BaseAcquisitionFn):
    """Expected Improvement (EI) acquisition function in y-transformed space."""
    
    def __init__(self, y_t_best, xi):
        """
        Args:
            y_t_best (float): The best observed value in y-transformed space.
            xi (float): Exploration-exploitation trade-off parameter. Higher values encourage exploration.
        """
        self.y_t_best = y_t_best
        self.xi = xi
    
    def __call__(self, mean_t, std_t):
        # Avoid division by zero
        std_t = np.clip(std_t, a_min=1e-12, a_max=None)
        
        improve = mean_t - self.y_t_best - self.xi
        z_score = improve/std_t # Division element wise
    
        ei = (improve * norm.cdf(z_score) + std_t * norm.pdf(z_score))
        return ei


class ExpectedImprovementOrigSpace(BaseAcquisitionFn):
    """
    Expected Improvement (EI) acquisition function in y-original space.
    
    Calculated by performing Monte Carlo sampling in the original y-space to
    accommodate non-linear transformations.
    """
    
    def __init__(self, y_best, xi, y_transform, normal_sampler):
        """
        Args:
            y_best (float): The best observed value in y-original space.
            xi (float): Exploration-exploitation trade-off parameter.
            y_transform: Transformer object with inverse_transform methods.
            normal_sampler: Sampler providing (deterministic) standard normal samples.
        """
        self.y_best = y_best
        self.xi = xi
        self.y_transform = y_transform
        self.normal_sampler = normal_sampler

    def __call__(self, mean_t, std_t): 
        # Introduced batching to prevent out-of-memory issues after vectorising processing 
        return self._batched(mean_t, std_t, batch_size=1000)

    def _batched(self, mean_t, std_t, batch_size):        
        # Safety
        mean_t = np.asarray(mean_t).ravel()
        std_t = np.asarray(std_t).ravel()

        # Should always have some variance due to noise
        std_t = np.maximum(std_t, 1e-12)
        
        N = len(mean_t)

        z = self.normal_sampler.z
        
        M = len(z)
        z = z[None, :] # z shape: (M) -> z shape: (1, M)

        ei_values = np.empty(N)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            means = mean_t[start:end][:, None] # mean_t shape: (N) -> (batch) -> (batch, 1)
            stds = std_t[start:end][:, None]   # std_t shape: (N)  -> (batch) -> (batch, 1)

            # Samples in transformed space
            # shape: (batch, 1) + (batch, 1)*(1, M) = (batch, M)
            samples_t = means + stds * z

            # We check samples_original after inverse_transform anyway, and any transformer issues should be fixed directly within the transformer
            # Clip to avoid extreme inverse-transform issues
            # samples_t = np.clip(samples_t, means - 6 * stds, means + 6 * stds)

            # Inverse transform
            samples_original = self.y_transform.inverse_transform(samples_t.ravel()).reshape(end - start, M) # Flatten -> inverse -> reshape back (batch, M)

            if not np.all(np.isfinite(samples_original)):
                indices = np.where(~np.isfinite(samples_original)) # Tuple of arrays, 1 array per dim to generate coordinates
                raise ValueError(f"Non-finite samples_original values [{len(indices[0])}] produced by y_transform.inverse_transform.\nsamples_original: {samples_original[indices]}.\nsamples_t: {samples_t[indices]}")

            improve = samples_original - self.y_best - self.xi
            improve = np.maximum(0.0, improve)

            ei_values[start:end] = improve.mean(axis=1) # (batch, M) -> (batch)

        return ei_values


def get_acq_fns(df, y_transform, ei_xis = [], ucb_betas = []):
    """
    Returns a dictionary of acquisition functions based on the EI xi and UCB beta values passed in.

    y_transform: Transformer object used to transform y-original outputs.
    
    """
    
    y = df['y'].values
    y_best = np.max(y) 

    # Recalculate y_t_best rather than using np.max(df["y_t"]) in case y_transform has been manually overridden
    y_t = y_transform.transform(y.reshape(-1, 1)).ravel()
    y_t_best = np.max(y_t) 
    
    # Ensure we have the same Normal samples for all EI evaluations
    normal_sampler = DeterministicStdNormalSampler(n_samples=10000)

    acq_fns = {}

    for xi in ei_xis:
        ei_name = f'EI trans space (xi={xi})'
        acq_fns[ei_name] = ExpectedImprovementTransSpace(y_t_best = y_t_best, xi = xi)

    for xi in ei_xis:
        ei_name = f'EI orig space (xi={xi})'
        acq_fns[ei_name] = ExpectedImprovementOrigSpace(y_best = y_best, xi = xi, y_transform = y_transform, normal_sampler = normal_sampler)
    
    for beta in ucb_betas:
        ucb_name = f'UCB trans space (beta={beta})'
        acq_fns[ucb_name] = UpperConfidenceBoundTransSpace(beta = beta)
    
    return acq_fns


def determine_next_eval_points(df, model, x_grid, x_col_names, x_transform, y_transform, x_dim, acq_fns, bounds = None, opt_trials=100):
    """
    Determines the next evaluation point for each acquisition function within acq_fns dictionary.

    For each acquisition function, processing occurs in two stages as a compromise between thoroughness and speed:
    Stage 1 - Evaluate the acquisition function over x_grid.
    Stage 2 - Select the best outputs from Stage 1 and use their corresponding x points as the starting points for local optimisation.

    Args:
        df (pd.DataFrame): Current BBO function data.
        model: Trained Gaussian Process surrogate model.
        x_grid (np.ndarray): Grid of points for the initial search in Stage 1.
        x_col_names (np.ndarray): x column names, e.g. [ 'x1', 'x2', ...].
        x_transform (object): Transformer used to transform x-original inputs to the model's training space. (Could be identity transformer).
        y_transform (object): Transformer used to transform y-original outputs. (Could be identity transformer).
        x_dim (int): number of x input features/dimensions.
        acq_fns (dict): dictionary of acquisition function names to corresponding object.
        bounds: (list of tuple, optional): List of (min, max) tuples for each 
            feature/dimension. Defaults to [0, 1) for all dimensions if None.
        opt_trials (int): Number of top points from Stage 1's grid search to use as optimisation seeds in Stage 2.

    Returns:
        pd.DataFrame: A summary of the best points found for each acquisition function.
        
    Raises:
        ValueError: If the model predicts non-finite values (NaN/Inf).
    """

    # Use x_grid transformed, since model trained in x-transformed space. (NB. x_transform could be identity transformer.)
    x_grid_t = x_transform.transform(x_grid) 

    # Output will always be in y-transformed space. (NB. y_transform could be identity transformer.)
    y_grid_mean, y_grid_std = model.predict(x_grid_t, return_std=True) 

    # np.isfinite() catches NaN and ±inf
    if not np.isfinite(y_grid_mean).all():
        raise ValueError("y_grid_mean contains non-finite values")

    if not np.isfinite(y_grid_std).all():
        raise ValueError("y_grid_std contains non-finite values")

    if bounds is None:
        bounds = [(0.0, 1.0 - 1e-6)] * x_dim
    
    df_results_rows = []
    
    for acq_fn_name, acq_fn in acq_fns.items():
        print(f'{acq_fn_name}...')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=r'.*Predicted variances smaller than 0. Setting those variances to 0.*')

            '''UserWarning: Some values in column 0 of the inverse-transformed data are NaN. This may be caused by numerical issues in the transformation process,
            e.g. extremely skewed data. Consider inspecting the input data or preprocessing it before applying the transformation.'''
            # We manually remove them
            warnings.filterwarnings("ignore", category=UserWarning, message=r'.*Some values in column 0 of the inverse-transformed data are NaN.*')
        
            try:
                if len(x_grid) >= opt_trials:
                   # Need to select the best opt_trials outputs
                    
                    acq_values = acq_fn(y_grid_mean, y_grid_std)
                    
                    #print(f'acq_values[:100]: {acq_values[:100]}')
            
                    # We don't require top_indices to be sorted; any ordering is fine
                    # top_indices = np.argsort(acq_values)[-opt_trials:]           
                    top_indices = np.argpartition(acq_values, -opt_trials)[-opt_trials:]
                    
                    starting_points = x_grid[top_indices]
                else:
                    # Can't run opt_trials optimisations because x_grid has fewer points
                    print(f'opt_trials ({opt_trials}) is larger than len(x_grid). Reducing to len(x_grid) = {len(x_grid)}.')
                    starting_points = x_grid
                
                x_best = None
                distance = None
                y_t_mean_best = None
                y_mean_best = None
                acq_best = -np.inf  
        
                # Precompute for performance since used many times within minimize(minimize_objective_fn)
                transform = x_transform.transform
                
                # Create a dict to store minimize's intermediate results
                intermediate_results = {}
                
                # Return the negative acq value since we optimise using minimize()
                def minimize_objective_fn(x):
                    x_t = transform(x.reshape(1, x_dim)) # pass in x_t, what the model was trained with
            
                    y_pred_mean, y_pred_std = model.predict(np.atleast_2d(x_t), return_std=True) # May or may not be y-transformed space
            
                    acq = acq_fn(y_pred_mean, y_pred_std)
        
                    if len(acq) != 1:
                        raise RuntimeError(f'acq_fn result did not have len = 1. acq: {acq}')
    
                    result = -acq[0] # Single entry 

                    # Store the result into intermediate_results
                    dict_key = tuple(x) # dict key cannot be a mutable array
                    if dict_key not in intermediate_results:
                        intermediate_results[dict_key] = result
                    
                    return result 
        
                # Minimize callback function: 'x_k' represents the intermediate result after iteration k
                # Doesn't seem to work so directy add to intermediate_results within minimize_objective_fn() above
                # def store_intermediate_result(x_k):
                #    intermediate_results.append(np.copy(x_k))
                
                # Local optimisation loop
                for index, x0 in enumerate(starting_points):
                    
                    res = minimize(minimize_objective_fn, x0=x0, bounds=bounds, method="L-BFGS-B" ) #, callback=store_intermediate_result)
    
                    if res.success and -res.fun > acq_best:
                        acq_best = -res.fun
                        x_best = res.x
                        
                        x_t_best = x_transform.transform(x_best.reshape(1, x_dim))
            
                        y_t_mean_best, y_t_std_best = model.predict(np.atleast_2d(x_t_best), return_std=True)
                        y_mean_best, y_std_best = y_transform.inverse_transform_dist(y_t_mean_best, y_t_std_best)
        
                        # Too verbose
                        # print(f"{index}: x0: {x0}, Acq Fn({x_best}) = {acq_best}, y_t_best: {y_t_mean_best}, y_best: {y_mean_best}")
                
                # Check x_best isn't an x point that's already been evaluated
                if not x_best is None:
                    already_evaled = np.isclose(df[x_col_names].values, x_best, atol=1e-7, rtol=0).all(axis=1).any()
                    if already_evaled:
                        print(f'x_best {x_best} is already evaluated. Checking minimize() intermediate results...')
    
                        x_best = None 
                        
                        #print(intermediate_results)
        
                        # Get sorted keys from intermediate_results dict ordered by dict values ('-acq', ascending)
                        sorted_keys = sorted(intermediate_results, key=intermediate_results.get)
        
                        for key in sorted_keys:
                            x_temp = np.array(key) # tuple -> array
                            distance = distance_helper.distance_to_nearest_point(df[x_col_names].values, x_temp) 
                            
                            already_evaled = np.isclose(df[x_col_names].values, x_temp, atol=1e-7, rtol=0).all(axis=1).any()
                            if already_evaled:
                                continue;
                                
                            acq_best = -intermediate_results[key]
                            x_best = x_temp
        
                            x_t_best = x_transform.transform(x_best.reshape(1, x_dim))
                
                            y_t_mean_best, y_t_std_best = model.predict(np.atleast_2d(x_t_best), return_std=True)
                            y_mean_best, y_std_best = y_transform.inverse_transform_dist(y_t_mean_best, y_t_std_best)
                            
                            print(f'x_best {x_best} and acq_best {acq_best} set from intermediate results')
                            break # We are done
    
                if x_best is None:
                     print('x_best could not be set using minimize() intermediate results. Unable to find a new acquisition function peak')
                
                else:    
                    distance = distance_helper.distance_to_nearest_point(df[x_col_names].values, x_best)     
                    print(f"Acq Fn({x_best}) = {acq_best}, distance: {distance}, y_t_best: {y_t_mean_best}, y_best: {y_mean_best}")
                    
                    # Save the result
                    df_results_rows.append(
                        {"acq_fn": acq_fn_name,
                         f'{x_col_names}': print_helper.format_point(x_best),
                         "distance": distance,
                         "y_t_mean": y_t_mean_best.item(), # Single element array        
                         "y_mean": y_mean_best.item(), # Single element array
                        })
    
            except Exception as e:
                print(f"{e}")
    
    df_results = pd.DataFrame(df_results_rows)
    
    return df_results
    