import casadi as cs
import numpy as np


def rbf_kernel(X1, X2, lengthscale, outputscale):
    # Scaled RBF kernel formula:
    # outputscale * exp(-1/2 * ||x_di - x_dj||^2 / l_d^2))
    n1, n2 = X1.shape[0], X2.shape[0]
    lengthscale_sq = lengthscale**2
    
    # If we're working with CasADi matrices
    if type(X1) is cs.SX or type(X2) is cs.SX or type(X1) is cs.MX or type(X2) is cs.MX:
        # Vectorized squared distance computation
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
        X1_sq = cs.sum2(X1**2)
        X2_sq = cs.sum2(X2**2)
        cross_term = cs.mtimes(X1, X2.T)
        sq_dist = X1_sq + X2_sq.T - 2 * cross_term
        kernel = outputscale * cs.exp(-0.5 * sq_dist / lengthscale_sq)
        return kernel
    
    # Otherwise we can use numpy
    kernel = np.zeros((n1, n2))
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True)
    cross_term = X1 @ X2.T
    
    sq_dist = X1_sq + X2_sq.T - 2*cross_term
    kernel = outputscale * np.exp(-0.5 * sq_dist / lengthscale_sq)
    return kernel


def get_xdot(gp_model, x_train, y_train, control_dim, return_variance=False):
    state_dim = x_train.shape[1] - control_dim
    x = cs.MX.sym("x", 1, state_dim)
    u = cs.MX.sym("u", 1, control_dim)
    
    # Normalize training data if it is not already normalized
    already_normalized = (x_train.min() >= -0.01) and (x_train.max() <= 1.01)
    x_train_norm = x_train.copy()
    if not already_normalized:
        for i in range(x_train_norm.shape[1]):
            x_train_norm[:, i] = gp_model.min_max_norm(x_train_norm[:, i], gp_model.X_min_max[i, 0], gp_model.X_min_max[i, 1])
    
    # Get values for normalization
    X_min_max = gp_model.X_min_max
    state_min = cs.DM(X_min_max[:state_dim, 0]).T
    state_max = cs.DM(X_min_max[:state_dim, 1]).T
    control_min = cs.DM(X_min_max[state_dim:, 0]).T
    control_max = cs.DM(X_min_max[state_dim:, 1]).T

    # Normalize input data
    x_norm = (x - state_min) / (state_max - state_min)
    u_norm = (u - control_min) / (control_max - control_min)
    
    lengthscale = gp_model.model.covar_module.base_kernel.lengthscale.item()
    outputscale = gp_model.model.covar_module.outputscale.item()
    noise = gp_model.model.likelihood.noise.item()
    mean_const = gp_model.model.mean_module.constant.item()
    
    # Calculate mean using Cholesky decomposition
    K = rbf_kernel(x_train_norm, x_train_norm, lengthscale, outputscale)
    L = np.linalg.cholesky(K + noise * np.eye(len(x_train_norm)))
    y_centered = y_train - mean_const
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    k_star = rbf_kernel(cs.horzcat(x_norm, u_norm), x_train_norm, lengthscale, outputscale)
    mean = cs.mtimes(k_star, alpha) + mean_const
    
    if return_variance:
        v = cs.solve(L, k_star.T)
        variance = outputscale - cs.mtimes(v.T, v)
        # This is only used for comparison, when using the variance in ACADOS we don't want the noise, it's just a constant
        # variance = outputscale - cs.mtimes(v.T, v) + noise
        return cs.Function("x_dot", [x, u], [mean, variance], ["x", "u"], ["mean", "variance"])
    
    return cs.Function("x_dot", [x, u], [mean], ["x", "u"], ["x_dot"])
    

def split_inputs(f_cat: cs.Function, x_size: int, u_size: int):
    """
    Converts a CasADi function f_cat(X) with concatenated input X=[x; u] 
    into a function f(x, u) with separate inputs.
    
    Parameters
    ----------
    f_cat : cs.Function
        Original function with single input X = [x; u].
    x_size : int
        Dimension of x.
    u_size : int
        Dimension of u.
        
    Returns
    -------
    f_split : cs.Function
        New function with two inputs [x, u].
    """
    # Create separate symbolic inputs
    x = cs.MX.sym("x", x_size)
    u = cs.MX.sym("u", u_size)
    
    # Concatenate inputs
    X = cs.vertcat(x, u)
    
    # Evaluate original concatenated function
    y = f_cat(X)
    
    # Create new function with separate inputs
    f_split = cs.Function(f_cat.name() + "_split", [x, u], [y])
    
    return f_split


def concatenate_inputs(func: cs.Function, x_size, u_size):
    """
    Converts a CasADi function f(x, u) into f_cat(X) where
    X = [x; u] concatenated vector.
    
    Parameters
    ----------
    func : cs.Function
        Original function with two inputs [x, u].
    x_size : int
        Dimension of x.
    u_size : int
        Dimension of u.
        
    Returns
    -------
    f_cat : cs.Function
        New function with single input X = [x; u].
    """
    # Single input variable
    X = cs.MX.sym("X", x_size + u_size)
    
    # Split into original inputs - ensure correct order
    x = X[:x_size]  # First x_size elements are x
    u = X[x_size:x_size + u_size]  # Next u_size elements are u
    
    # Evaluate original function
    y = func(x, u)
    
    # Create new function
    f_cat = cs.Function(func.name() + "_cat", [X], [y])
    
    return f_cat


def get_function_estimator(f_casadi, bounds, n_points_per_dim=20, method='bspline'):
    """
    Create a CasADi interpolant for a function f(x).
    
    Parameters:
    - f_casadi: CasADi function, e.g., f = ca.Function('f', [x], [y])
    - bounds: list of tuples [(x1_min, x1_max), (x2_min, x2_max), ...]
    - n_points_per_dim: number of points in each dimension for interpolation
    - method: 'linear', 'bspline', or 'nearest' (CasADi supported)
    
    Returns:
    - interp: CasADi interpolant function
    """
    n_vars = len(bounds)
    
    # Create grids for each dimension as Python lists
    grids = [list(np.linspace(low, high, n_points_per_dim)) for (low, high) in bounds]
    
    # Create meshgrid with 'ij' indexing
    mesh = np.meshgrid(*grids, indexing='ij')
    
    # Flatten meshgrid points for function evaluation
    points = np.stack([m.flatten(order='C') for m in mesh], axis=-1)
    
    # Evaluate CasADi function
    y_vals = np.array([f_casadi(pt).full().item() for pt in points])
    
    # Reshape to grid shape
    grid_shape = [n_points_per_dim] * n_vars
    y_vals_grid = y_vals.reshape(grid_shape)
    
    # Reverse axes to fix variable ordering (so x0, x1, x2 -> x2, x1, x0)
    y_vals_grid_fixed = np.transpose(y_vals_grid, axes=range(n_vars-1, -1, -1))
    
    # Flatten back to 1D Python list
    y_vals_fixed = y_vals_grid_fixed.flatten().tolist()
    
    # Reverse grids as well, to match new axis ordering
    grids_fixed = grids[::-1]
    
    # Create CasADi interpolant
    interp = cs.interpolant('interp', method, grids_fixed, y_vals_fixed)
    
    return interp
