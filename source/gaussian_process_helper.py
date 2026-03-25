from collections.abc import Iterable

import copy
import math
import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.stats import beta

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, ConstantKernel, RBF, RationalQuadratic, Matern, WhiteKernel, DotProduct, ExpSineSquared
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

import time
from typing import Final
import warnings

import acquisition_fns_helper
import distance_helper
from random_sample_helper import DeterministicStdNormalSampler

WHITE_KERNEL: Final = "WhiteK"
LOWER_BOUND: Final = 1e-7
UPPER_BOUND: Final = 10

MODEL: Final = "Model"
Z_SMSE: Final = "Z_SMSE"
Z_MSE: Final = "Z_MSE"
Y_SMSE: Final = "Y_SMSE"           
Y_MSE: Final = "Y_MSE"
Y_SMSE_TOP: Final = "Y_SMSE_TOP"
Y_MSE_TOP: Final = "Y_MSE_TOP"  
LML: Final = "LML"
KERNEL_PARAMS: Final = "Kernel_Params"

NOISE: Final = "Noise (\u03B1)" # Unicode \u03B1 = alpha char

LOOCV: Final = "LOOCV"

X_TRANSFORM: Final = "x transform"
Y_TRANSFORM: Final = "y transform"


def format_sig_figs(value, sig_figs=6):
    """
    Formats a numeric value or collection of values to the specified precision 
    to standardise their visual representation.
    
    Format specifier 'g' is used to balance scientific and fixed-point notation
        dynamically.

    Args:
        value (Any): Input to format. Can be float, int, string, None, or an
            iterable of these types.
        sig_figs (int): The number of significant figures to show. 

    Returns:
        str or list: A formatted string representation of 'value', or a list of 
            formatted strings if the input was an Iterable.
    """
    if value is None:
        return ""
    
    if isinstance(value, str):
        return value

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value): # "NaN", "Inf", "-Inf"
            return str(value)

        # Use the 'g' format specifier which is designed for significant digits
        return f'{value:.{sig_figs}g}'
        
    if isinstance(value, Iterable):
        return [format_sig_figs(v, sig_figs) for v in value]
    
    return str(value) # Catch all
    

def get_kernels(x_dim):
    """
    Initialises various Gaussian Process kernels for model selection.

    Constructs a variety of kernels including Matern, RBF, Rational Quadratic
    and Periodic, both with and without an additive WhiteKernel component.
    
    Anisotropy (Automatic Relevance Determination) kernels are used by assigning
    independent length scales to each input dimension, except for within RationalQuadratic
    whose implementation currently only supports isotropic length scales. 

    Args:
        x_dim (int): The number of input dimensions.

    Returns:
        dict: Mapping from kernel names (str) to sklearn.gaussian_process.kernels objects.
    """
    
    # Specific bounds to prevent "hitting the wall"
    bounds = (LOWER_BOUND, UPPER_BOUND)
    
    # ConstantKernel scales the output variance
    CK = ConstantKernel(1.0)
    
    # WhiteKernel to represent noise (alpha)
    WK = WhiteKernel(noise_level=0.1, noise_level_bounds=bounds)

    # Ridge Kernel (kernel param sigma_0 is the intercept term)
    RK = DotProduct(sigma_0=1.0, sigma_0_bounds=bounds)
    
    # Anisotropy: using length_scale=[1.0, 1.0, ...] to allow for x_1, x_2,... to have independent length scales (i.e. sensitivity).
    length_scale = [1.0] * x_dim

    MAT25 = CK * Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=bounds)
    MAT15 = CK * Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=bounds)
    MAT05 = CK * Matern(length_scale=length_scale, nu=0.5, length_scale_bounds=bounds)
    
    RBFK   = CK * RBF(length_scale=length_scale, length_scale_bounds=bounds)

    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html
    # "Only the isotropic variant where length_scale is a scalar is supported"
    RQ    = CK * RationalQuadratic(length_scale=1.0, length_scale_bounds=bounds)
    
    local_length_scale = [0.1] * x_dim # Tighter focus
    LocalSens = ConstantKernel(0.1) * Matern(length_scale=local_length_scale, nu=1.5, length_scale_bounds=bounds)

    PER = CK * ExpSineSquared(length_scale_bounds=bounds, periodicity_bounds=bounds)

    LocPer = CK * RBF(length_scale=length_scale) * ExpSineSquared()

    LinPlusRBF = RK + RBFK
    
    kernels = {
        # Without WhiteKernel
        "Matern 2.5": MAT25,
        "Matern 1.5": MAT15,
        "Matern 0.5": MAT05,
        "RBF":        RBFK,
        "Rational Quad": RQ,
        "Periodic": PER,
        "Locally Periodic": LocPer,
        "Linear + RBF": LinPlusRBF,

        # With WhiteKernel
        "Matern 2.5 + " + WHITE_KERNEL: MAT25 + WK,
        "Matern 1.5 + " + WHITE_KERNEL: MAT15 + WK,
        "Matern 0.5 + " + WHITE_KERNEL: MAT05 + WK,
        "RBF + " + WHITE_KERNEL:        RBFK + WK,
        "Rational Quad + " + WHITE_KERNEL: RQ + WK,
        "Periodic + " + WHITE_KERNEL: PER + WK,
        "Locally Periodic + " + WHITE_KERNEL: LocPer + WK,
        "Linear + RBF + " + WHITE_KERNEL: LinPlusRBF + WK,
        
        # Local Sensitivity Kernel - This combines a global smooth trend with a high-frequency local component
        "Local Sens + Mat 2.5 + " + WHITE_KERNEL: LocalSens + MAT25 + WK, 

        # Ridge Kernel + Rational Quadratic
        "Ridge + RQ + " + WHITE_KERNEL: RK + RQ + WK,

        # Global Trend + Local Peaks + Ridge
        "Local Sens + Ridge + Mat 2.5 + " + WHITE_KERNEL: LocalSens + RK + MAT25 + WK,
    }

    return kernels


def extract_optimised_kernel_params_str(model):
    """
    Extracts the model's optimised kernel parameters.

    Example output: "0.977**2 * RBF(len_scale=[0.0116, 0.00739]) + WhiteK(noise=1e-07)"

    Args:
        model (GaussianProcessRegressor): A fitted GPR model.

    Returns:
        str: Optimised kernel parameters.
    """
    params = model.kernel_.get_params()
    params_strs = []

    for kernelID, value in params.items():
        
        # Check if a main kernelID (e.g. 'k1', 'k2', ... but not composite 'k1__k2',... )
        # AND value is a kernel object
        if '__' in kernelID or not isinstance(value, Kernel):
            continue

        # Shorten text
        value = str(value)
        value = value.replace("length_scale", "len_scale")
        value = value.replace("noise_level", "noise")
        value = value.replace("WhiteKernel", "WhiteK")
        value = value.replace("alpha", "\u03B1") # \u03B1 = alpha char
        
        params_strs.append(value)   
        
    return " + ".join(params_strs) # main kernels appear to be separated by addition


def get_model_noise(model):
    """
    Calculates the total observation noise captured by a GPR model.

    In Gaussian Processes, noise can be represented either by the `alpha` 
    parameter in the Regressor or an explicit `WhiteKernel` in the kernel 
    composition. This function sums both to find the total noise.

    Args:
        model (GaussianProcessRegressor): A fitted GPR model.

    Returns:
        float: The combined noise level (variance).
    """

    # Start with the GPR's alpha
    total_noise = model.alpha
    
    # Add WhiteKernel noise (if it exists)
    params = model.kernel_.get_params()
    for key, value in params.items():
        if "noise_level" in key and not key.endswith("_bounds"):
            total_noise += value
            
    return total_noise

    
def evaluate_model(model_name, model, df, x_col_names, x_transform, y_transform, df_tuning_results_rows, model_dict, n_samples=10000):
    """
    Performs analytical Leave-One-Out Cross-Validation (LOOCV) for GPR 'model'.

    This function fits the model in transformed space, then uses the Cholesky 
    decomposition to analytically derive Leave-One-Out predictions. It then 
    maps these predictions back to the original space using Monte Carlo (MC)
    inversion to calculate accuracy metrics.

    Args:
        model_name (str): Label for the specific kernel/x-transform/y-transform combination.
        model (GaussianProcessRegressor): The GP model template.
        df (pd.DataFrame): The training dataset.
        x_col_names (list): List of feature column names, e.g. ['x1', 'x2', ...].
        x_transform (BaseTransformer): Transformer for input features.
        y_transform (BaseTransformer): Transformer for target values.
        df_tuning_results_rows (list): List to append result metrics to.
        model_dict (dict): Dictionary to store the fitted model and its metadata.
        n_samples (int): MC samples for inverse distribution mapping.
    """
    
    # --- Data Prep ---  
    x = df[x_col_names].values
    y = df['y'].values 

    x_t = x_transform.fit_transform(x)
    y_t = y_transform.fit_transform(y.reshape(-1, 1)).ravel() # Keep 2D for transformers

    # Train the model using the full data in transformed space
    model.fit(x_t, y_t) 

    # ---  Analytical leave-one-out cross-validation (LOOCV) in y-transformed space --- 
    # 1. Extract the Cholesky decomposition L
    L = model.L_

    # 2. Compute the diagonal of the inverse: (K^-1)_ii
    # Step: solve(L, I) to get L_inv
    n = L.shape[0]
    L_inv = solve_triangular(L, np.eye(n), lower=True)

    # The diagonal of K^-1 is the row-wise sum of squares of L_inv
    # diag(K^-1) = diag(L_inv.T @ L_inv)
    diag_inv = np.sum(L_inv**2, axis=0)

    # 3. Get the 'alpha' weights (K^-1 @ y)
    # GPR stores this as gpr.alpha_
    alpha = model.alpha_

    # 4. LOOCV variance in y-transformed space
    mean_t = y_t - (alpha / diag_inv)

    # TODO: Unclear whether noise should be subtracted
    #noise = get_model_noise(model)
    #var_t = (1.0 / diag_inv) - noise

    var_t = 1.0 / diag_inv
    
    std_t = np.sqrt(np.maximum(var_t, 1e-12))

    # --- Computer inverse y-transform ---
    mean_y, std_y = y_transform.inverse_transform_dist(mean_t, std_t)

    # Calculate MSE using the predicted mean
    mse_i = (y.ravel() - mean_y)**2

    # --- Calculate Metrics ---

    # y-transformed Standardised Mean Squared Error (SMSE) and MSE
    y_t_mse = np.mean((y_t - mean_t)**2)
    y_t_smse = y_t_mse / np.var(y_t)

    # y-original Standardised Mean Squared Error (SMSE) and MSE
    y_mse = np.mean(mse_i)
    y_smse = y_mse / np.var(y)

    # y-original SMSE (Top 25%)
    n_top = max(1, n // 4)
    sorted_indices = np.argsort(y) # Ascending ordering
    
    mse_top = mse_i[sorted_indices][-n_top:]
    y_mse_top = np.mean(mse_top)

    # Calculate y_smse_top with the var(y_top) as denominator
    #y_top = y[sorted_indices][-n_top:]
    #y_smse_top = y_mse_top / np.var(y_top)

    # Calculate y_smse_top with the var(y) as denominator
    # Denominator var(y) is preferable to var(y_top) because then y_smse and y_smse_top
    # will have the same denominator and be directly comparable
    y_smse_top = y_mse_top / np.var(y)

    # Log marginal likelihood
    lml = model.log_marginal_likelihood()

    loocv_results = pd.DataFrame(data = {
        "y_t": y_t,
        "mean_t": mean_t,
        "resid_t": (mean_t - y_t),
        "std_t": std_t,
        "y_pred": mean_y,
    }, index = df.index)
    print(loocv_results) 
    
    print() # Blank line
    params = model.kernel_.get_params() # Must have underscore 'kernel_' to get kernel after .fit()
    for param_key, param_value in params.items():
        print(f"{param_key}: {param_value}")

    params_str = extract_optimised_kernel_params_str(model)

    # TODO: Could potentially drop Z_MSE, Y_MSE and Y_MSE_TOP as we have SMSE variants
    df_tuning_results_rows.append({
        MODEL: model_name,
        Z_SMSE: y_t_smse,           
        Z_MSE: y_t_mse,
        Y_SMSE: y_smse,           
        Y_MSE: y_mse,
        Y_SMSE_TOP: y_smse_top,
        Y_MSE_TOP: y_mse_top,    
        LML: lml,
        KERNEL_PARAMS: params_str,
    })

    model_dict[model_name] = {
        MODEL: model,
        LOOCV: loocv_results,
        X_TRANSFORM: x_transform, # After fit_transform()
        Y_TRANSFORM: y_transform, # After fit_transform()
    }

def tune_gaussian_process_surrogate(df, x_col_names, kernels, y_transforms, x_transforms):
    """
    Executes grid search over 'kernels', 'y_transforms' and 'x_transforms' to find the
    best combination for a GPR surrogate model.

    This is the main driver for model tuning. It suppresses common convergence 
    warnings that occur during hyperparameter optimisation (L-BFGS-B) and 
    systematically evaluates every combination of provided kernels and transforms.

    Args:
        df (pd.DataFrame): The training dataset.
        x_col_names (list): List of feature column names, e.g. ['x1', 'x2', ...].
        kernels (dict): Dictionary of kernel templates from `get_kernels`.
        y_transforms (dict): Dictionary of target transformers.
        x_transforms (dict): Dictionary of feature transformers.

    Returns:
        tuple: (df_tuning_results, model_dict)
            - df_tuning_results: DataFrame containing SMSE, MSE, and LML for all combinations.
            - model_dict: Dictionary containing the fitted model objects and transformers.
    """
    
    df_tuning_results_rows = []
    model_dict = {}
    
    with warnings.catch_warnings():
        '''ConvergenceWarning: lbfgs failed to converge after 15 iteration(s) (status=2):'''
        # n_restarts_optimizer=100, it is almost guaranteed that some of those 100 starts will fail to converge
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message=r'.*lbfgs failed to converge after.*')
        
        '''ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified upper bound 10.0.
        Increasing the bound and calling fit again may find a better value.''' 
        # This is just a symptom of the model trying to ignore some features by setting their lengthscale to the upper bound
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message=fr'.*is close to the specified upper bound {UPPER_BOUND}.*')

        '''The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound'''
        # We test composite kernel with additive WhiteKernel as well as without WhiteKernel. 
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message=r'.*parameter k2__noise_level is close to the specified lower bound.*')

        '''UserWarning: Some values in column 0 of the inverse-transformed data are NaN. This may be caused by numerical issues in the transformation process,
        e.g. extremely skewed data. Consider inspecting the input data or preprocessing it before applying the transformation.'''
        # We manually remove them
        warnings.filterwarnings("ignore", category=UserWarning, message=r'.*Some values in column 0 of the inverse-transformed data are NaN.*')
        
        for kernel_name, kernel_template in kernels.items():
            for x_transform_name, x_trans_template in x_transforms.items():
                for y_transform_name, y_trans_template in y_transforms.items():

                    # sklearn.base.clone() creates a 'fresh' composite kernel, so state doesn't leak across models 
                    kernel = clone(kernel_template)
    
                    # Copy transformers so each model has a 'fresh' version.
                    # Can't use sklearn.base.clone() because using our own custom transformer class
                    x_transform = copy.deepcopy(x_trans_template)
                    y_transform = copy.deepcopy(y_trans_template)
                    
                    model_name = f'{kernel_name}, x:{x_transform_name}, y:{y_transform_name}'
                    print( f'\n{model_name}...')  
        
                    # Set GP_alpha=0 because WhiteKernel will handle noise explicitly
                    GP_alpha = (0.0 if WHITE_KERNEL in kernel_name else 1e-8)
            
                    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=GP_alpha, random_state=42, normalize_y=False)
        
                    evaluate_model(
                        model_name = model_name,
                        model = model,
                        df = df,
                        x_col_names = x_col_names,
                        x_transform = x_transform, # Pass clean copy
                        y_transform = y_transform, # Pass clean copy
                        df_tuning_results_rows = df_tuning_results_rows,
                        model_dict = model_dict,
                    )
                    
    df_tuning_results = pd.DataFrame(df_tuning_results_rows)
    
    return df_tuning_results, model_dict

