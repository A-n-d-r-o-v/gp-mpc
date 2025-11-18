import casadi as cs
import numpy as np


def get_xdot(gp_model, x_train, y_train, control_dim):
    state_dim = x_train.shape[1] - control_dim
    x = cs.reshape(cs.MX.sym("x", state_dim), 1, -1) # State variables
    u = cs.reshape(cs.MX.sym("u", control_dim), 1, -1) # Control variables
    
    def rbf_kernel(X1, X2, lengthscale, outputscale):
        # Scaled RBF kernel formula:
        # outputscale * exp(-1/2 * ||x_di - x_dj||^2 / l_d^2))
        n1, n2 = X1.shape[0], X2.shape[0]
        lengthscale_sq = lengthscale**2
        
        # If we're working with CasADi matrices
        if type(X1) is cs.MX or type(X2) is cs.MX:
            kernel = cs.MX.zeros(n1, n2)
            for i in range(n1):
                x_i = cs.reshape(X1[i,:], 1, X1.shape[1])
                for j in range(n2):
                    x_j = cs.reshape(X2[j,:], 1, X2.shape[1])
                    dif = x_i - x_j
                    kernel[i,j] = outputscale * cs.exp(-0.5 * cs.mtimes(dif, dif.T) / lengthscale_sq)
            return kernel
        
        # Otherwise we can use numpy
        kernel = np.zeros((n1, n2))
        for i in range(n1):
            x_i = X1[[i],:]
            for j in range(n2):
                x_j = X2[[j],:]
                dif = x_i - x_j
                kernel[i,j] = outputscale * np.exp(-0.5 * np.linalg.norm(dif)**2 / lengthscale_sq)
        
        return kernel
    
    # Normalize training data
    for i in range(x_train.shape[1]):
        x_train[:, i] = gp_model.min_max_norm(x_train[:, i], gp_model.X_min_max[i, 0], gp_model.X_min_max[i, 1])
    
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
    
    # mean formula: K(X_*, X)(K(X, X) + sigma_n^2 * I)^(-1) * y
    # sigma_n^2 = noise
    k = rbf_kernel(x_train, x_train, lengthscale, outputscale)
    y_centered = y_train - mean_const
    tmp = np.linalg.solve(k + noise * np.eye(len(x_train)), y_centered)
    k_star = rbf_kernel(cs.horzcat(x_norm, u_norm), x_train, lengthscale, outputscale)

    mean = cs.mtimes(k_star, tmp) + mean_const

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


def get_mean_and_variance(gp_model, x_train, y_train, control_dim):
    x = cs.reshape(cs.MX.sym("x", x_train.shape[1] - control_dim), 1, -1) # Input variables
    u = cs.reshape(cs.MX.sym("u", control_dim), 1, -1) # Control variables
    
    # Need to change this function if we ever change the kernel, but based on scaled RBF kernel for now
    def rbf_kernel(X1, X2, lengthscale, outputscale):
        # Scaled RBF kernel formula:
        # outputscale * exp(-1/2 * ||x_di - x_dj||^2 / l_d^2))
        n1, n2 = X1.shape[0], X2.shape[0]
        lengthscale_sq = lengthscale**2
        
        # If we're working with CasADi matrices
        if type(X1) is cs.MX or type(X2) is cs.MX:
            kernel = cs.MX.zeros(n1, n2)
            for i in range(n1):
                x_i = cs.reshape(X1[i,:], 1, X1.shape[1])
                for j in range(n2):
                    x_j = cs.reshape(X2[j,:], 1, X2.shape[1])
                    dif = x_i - x_j
                    kernel[i,j] = outputscale * cs.exp(-0.5 * cs.mtimes(dif, dif.T) / lengthscale_sq)
            return kernel
        
        # Otherwise we can use numpy
        kernel = np.zeros((n1, n2))
        for i in range(n1):
            x_i = X1[[i],:]
            for j in range(n2):
                x_j = X2[[j],:]
                dif = x_i - x_j
                kernel[i,j] = outputscale * np.exp(-0.5 * np.linalg.norm(dif)**2 / lengthscale_sq)
        
        return kernel
    
    # TODO: Add normalization here if this function is ever to be used
    
    lengthscale = gp_model.covar_module.base_kernel.lengthscale.item()
    outputscale = gp_model.covar_module.outputscale.item()
    noise = gp_model.likelihood.noise.item()
    mean_const = gp_model.mean_module.constant.item()
    
    # mean formula: K(X_*, X)(K(X, X) + sigma_n^2 * I)^(-1) * y
    # sigma_n^2 = noise
    k = rbf_kernel(x_train, x_train, lengthscale, outputscale)
    y_centered = y_train - mean_const
    k_with_noise = k + noise * np.eye(len(x_train))
    k_with_noise_inv = np.linalg.inv(k_with_noise)
    
    alpha = np.linalg.solve(k_with_noise, y_centered)
    k_star = rbf_kernel(cs.horzcat(x, u), x_train, lengthscale, outputscale)

    mean = cs.mtimes(k_star, alpha) + mean_const
    
    # variance formula: k(X_*, X_*) + sigma_n^2 - k(X_*, X)^T * (k(X, X) + sigma_n^2 * I)^(-1) * k(X_*, X)
    # k(X_*, X_*) = outputscale
    
    variance = outputscale + noise - cs.mtimes(cs.mtimes(k_star, k_with_noise_inv), k_star.T)
    
    return cs.Function("x_dot", [x, u], [mean, variance], ["x", "u"], ["mean", "variance"])
