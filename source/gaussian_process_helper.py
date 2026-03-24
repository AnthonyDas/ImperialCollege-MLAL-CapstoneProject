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
    Renders a number to a specific number of significant figures.
    Returns a string to preserve formatting.
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


# Leave-One-Out Cross-Validation (LOOCV) - Replaces with more performant analytical/virtual LOOCV
'''
#An SMSE of 1.0 means your model is performing no better than simply guessing the mean of the training data.
#An SMSE close to 0 indicates a high-performing model. If it's significantly above 1.0,
#your GP is actively misleading you (likely due to severe overfitting).
def LOOCV(model, x_t, y_t):
    df_results_rows = []
    
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(x):
        x_t_train, x_t_test = x_t[train_index], x_t[test_index]
        y_t_train, y_t_test = y_t[train_index], y_t[test_index]
        
        model.fit(x_t_train, y_t_train)
        y_t_pred = model.predict(x_t_test, return_std=False)

        df_results_rows.append({
            "y_t_true": y_t_test[0],
            "loocv_y_t_pred": y_t_pred[0],           
            "loocv_y_t_residual": (y_t_pred[0] - y_t_test[0]),           
        })

    return pd.DataFrame(df_results_rows)
'''
'''    
def warp_inputs(x, params):
        """Warps [0,1] inputs using Beta CDF per dimension."""
        x_warped = np.zeros_like(x)
        x_warped[:, 0] = beta.cdf(np.clip(x[:, 0], 1e-5, 1-1e-5), params[0], params[1])
        x_warped[:, 1] = beta.cdf(np.clip(x[:, 1], 1e-5, 1-1e-5), params[2], params[3])
        return x_warped
    
def LML_optimised_warped_inputs(model, x, z):
    
    def negative_LML(warping_params, x, z):
        x_warped = warp_inputs(x, warping_params)
        model.fit(x_warped, z)
        return -model.log_marginal_likelihood()

    model.fit(x, z)
    
    # Multi-start optimization for warping parameters
    best_neg_lml = -model.log_marginal_likelihood()
    best_params = [1.0, 1.0, 1.0, 1.0]

    seeds = [1.0, 1.0, 1.0, 1.0] # Identity
    seeds = np.vstack([seeds, np.random.uniform(0.1, 5.0, size=(4, 4))]) # Plus 4 random starts

    for index, seed in enumerate(seeds): 
        res = minimize(negative_LML, x0 = seed, args=(x, z), bounds=[(0.1, 10.0)] * 4, method='L-BFGS-B')
        
        if res.success:
            if res.fun < best_neg_lml:
                best_neg_lml = res.fun
                best_params = res.x

            print(f"{index + 1}/{len(seeds)} {seed} result_params: {res.x}, lml: {-res.fun}, best_params: {best_params}, best_lml: {-best_neg_lml}")

    return warp_inputs(x, best_params), best_params
'''
'''
DOESN'T WORK FOR COMPLEX KERNELS BECAUSE THERE COULD BE MULTIPLE MATERS, HENCE DICT KEY NAME CLASH
def extract_kernel_params(model):
    """
    Extracts optimised hyperparameters from a (potentially complex) additive/multiplicative kernel.
    Also renames nested kernel ids by mapping keys (e.g k1, k2) to their respective Kernel Class name.
    """
    params = model.kernel_.get_params()

    extracted_params = {}
    
    # Build a dict of kernel IDs (k1, k2,...) to kernel
    # e.g.'k1__k2' -> e.g. RationalQuadratic(alpha=1e+05, length_scale=0.375)
    id_to_kernel = {}
    for key, value in params.items():
        # Check if value is a kernel object
        if isinstance(value, Kernel):
            id_to_kernel[key] = value

    # Extract parameter values and rename key
    for key, value in params.items():

        if isinstance(value, Kernel):
            continue

        # Ignore parameter "bounds"
        if key.endswith("_bounds"):
            continue

       # Identify the id (everything before the last parameter name)
        # e.g., 'k1__k2__length_scale' -> id = 'k1__k2', param = 'length_scale'
        key_parts = key.rsplit("__", maxsplit=1) # Rightmost Split
        kernel_id = key_parts[0]
        param_name = key_parts[1]

        # Find class name using id_to_kernel dict to retrieve kernel if available, otherwise keep key as is
        new_key = key
        kernel = id_to_kernel.get(kernel_id, kernel_id)
        if isinstance(kernel, Kernel):
            new_key = f"{kernel.__class__.__name__}__{param_name}"
        
        extracted_params[new_key] = format_sig_figs(value, sig_figs=4)
            
    return extracted_params, id_to_kernel
'''

def extract_optimised_kernel_params_str(model):

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
    Extracts the noise level from a fitted GaussianProcessRegressor.
    Includes both the kernel's WhiteKernel and the GPR's alpha parameter.
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

    # 4. LOOCV variance in y-transformed spac
    mean_t = y_t - (alpha / diag_inv)

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
    
    #y_top = y[sorted_indices][-n_top:]
    #y_smse_top = y_mse_top / np.var(y_top)

    y_smse_top = y_mse_top / np.var(y) # NB. Still using the var of all y points

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

                    '''
                    if include_warping:
                        model_name = f'{kernel_name} warp'
                        x_warped, warp_params = LML_optimised_warped_inputs(model, x = df_x.values, z = df_z.values)
                        loocv_df = evaluate_model(
                            model_name = model_name,
                            model = model,
                            x_col_names = x_col_names,
                            x = x_warped,
                            y = df_y.values,
                            z = df_z.values,
                            inverse_transform = inverse_transform,
                            df_results_rows = df_results_rows
                        )
                        model_dict[model_name] = { MODEL : model, LOOCV: loocv_df, "Warp params" : warp_params }
                    '''
                    
    df_tuning_results = pd.DataFrame(df_tuning_results_rows)
    
    return df_tuning_results, model_dict
