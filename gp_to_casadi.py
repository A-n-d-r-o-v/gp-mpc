import casadi as cs
import numpy as np


def get_xdot(gp_model, x_train, y_train, control_dim):
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
    
    # Normalize before or here?
    # x_min_max = gp_model.X_min_max
    # periodic_inputs = gp_model.periodic_inputs
    
    lengthscale = gp_model.model.covar_module.base_kernel.lengthscale.item()
    outputscale = gp_model.model.covar_module.outputscale.item()
    noise = gp_model.model.likelihood.noise.item()
    mean_const = gp_model.model.mean_module.constant.item()
    
    # mean formula: K(X_*, X)(K(X, X) + sigma_n^2 * I)^(-1) * y
    # sigma_n^2 = noise
    k = rbf_kernel(x_train, x_train, lengthscale, outputscale)
    y_centered = y_train - mean_const
    tmp = np.linalg.solve(k + noise * np.eye(len(x_train)), y_centered)
    k_star = rbf_kernel(cs.horzcat(x, u), x_train, lengthscale, outputscale)

    mean = cs.mtimes(k_star, tmp) + mean_const

    # means = cs.vcat(means)
    return cs.Function("x_dot", [x, u], [mean], ["x", "u"], ["x_dot"])


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
    
    # means = cs.vcat(means)
    # variances = cs.vcat(variances)
    
    return cs.Function("x_dot", [x, u], [mean, variance], ["x", "u"], ["mean", "variance"])
