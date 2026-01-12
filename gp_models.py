from __future__ import annotations

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from gp_to_casadi import get_xdot

from typing import TYPE_CHECKING

from gp import GPSetting
from gp_data import DataGenerator

if TYPE_CHECKING:
    from ODEModel import ODEModel
    from gp import GPSetting


class TrainableGP:

    @classmethod
    def from_gp_setting(cls, model: "ODEModel", setting: "GPSetting"):
        """
        Create a single GP model and generate its data from one GPSetting.
        """
        # Generate data
        X, Y = setting.data_generator.generate_from_vars(
            model, setting.input_vars, setting.output_var_dots
        )
        train_ratio = setting.data_generator.train_ratio
        setting.data_generator.set_data(
            *DataGenerator.split_train_test(X, Y, train_ratio=train_ratio)
        )

        kwargs = setting.get_kwargs()

        # Create GP
        if setting.use_sparse:
            inducing_pts = setting.data_generator.generate_inducing_points(
                setting.sparse_n_inducing
            )
            gp = SingleTaskSparseGP(inducing_points=inducing_pts, **kwargs)
        else:
            if len(setting.output_var_dots) == 1:
                gp = SingleTaskExactGP(**kwargs)
            else:
                num_tasks=len(setting.output_var_dots)
                gp = MultiTaskExactGP(num_tasks=num_tasks, **kwargs)

        # Attach the setting for later reference
        gp.setting = setting
        return gp

    def train_model(self, train_x, train_y, iters=50, lr=0.1):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # single-output: squeeze train_y to (N,)
        if train_y.ndim == 2 and train_y.shape[1] == 1:
            train_y = train_y.squeeze(-1)

        if isinstance(self, gpytorch.models.ExactGP):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        else:
            mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self, num_data=train_y.shape[0]
            )

        for _ in range(iters):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(X))
            return preds.mean.cpu().numpy(), preds.variance.cpu().numpy()
        

    # ------------ Evaluation ------------ #

    @torch.no_grad()
    def eval_RMSE(self, X_test, Y_test, per_output: bool = False):
        """Root Mean Squared Error"""
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(X_test))
            error = pred.mean - Y_test

        if per_output and error.ndim > 1:
            rmse = torch.sqrt(torch.mean(error**2, dim=0))
        else:
            rmse = torch.sqrt(torch.mean(error**2))
        return rmse.item() if isinstance(rmse, torch.Tensor) and rmse.ndim == 0 else rmse

    @torch.no_grad()
    def eval_MAE(self, X_test, Y_test, per_output: bool = False):
        """Mean Absolute Error"""
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(X_test))
            error = torch.abs(pred.mean - Y_test)

        if per_output and error.ndim > 1:
            mae = torch.mean(error, dim=0)
        else:
            mae = torch.mean(error)
        return mae.item() if isinstance(mae, torch.Tensor) and mae.ndim == 0 else mae

    @torch.no_grad()
    def eval_NLPD(self, X_test, Y_test):
        """Negative Log Predictive Density"""
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(X_test))
            nlpd = gpytorch.metrics.negative_log_predictive_density(pred, Y_test)
        return nlpd.item()

    @torch.no_grad()
    def eval_MSLL(self, X_train, Y_train, X_test, Y_test):
        """Mean Standardized Log Loss"""
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(X_test))
            msll = gpytorch.metrics.mean_standardized_log_loss(pred, Y_test, Y_train)
        return msll.item()
        

    def evaluate_all(
        self,
        X_test,
        Y_test,
        X_train=None,
        Y_train=None,
        per_output: bool = False
    ) -> dict:
        """
        Evaluate multiple metrics for the GP on test (and optionally train) data.

        :param X_test: Test inputs (torch.Tensor or convertible to Tensor)
        :param Y_test: Test outputs (torch.Tensor or convertible to Tensor)
        :param X_train: Optional train inputs (needed for MSLL)
        :param Y_train: Optional train outputs (needed for MSLL)
        :param per_output: If True, RMSE and MAE are returned per output variable
        :return: Dictionary with keys: 'RMSE', 'MAE', 'NLPD', 'MSLL'
        """
        metrics = {}

        # Ensure tensors
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float64)
        if not isinstance(Y_test, torch.Tensor):
            Y_test = torch.tensor(Y_test, dtype=torch.float64)
        if X_train is not None and not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float64)
        if Y_train is not None and not isinstance(Y_train, torch.Tensor):
            Y_train = torch.tensor(Y_train, dtype=torch.float64)

        # RMSE
        metrics['RMSE'] = self.eval_RMSE(X_test, Y_test, per_output=per_output)

        # MAE
        metrics['MAE'] = self.eval_MAE(X_test, Y_test, per_output=per_output)

        # NLPD
        metrics['NLPD'] = self.eval_NLPD(X_test, Y_test)

        # MSLL (requires training data)
        if X_train is not None and Y_train is not None:
            metrics['MSLL'] = self.eval_MSLL(X_train, Y_train, X_test, Y_test)
        else:
            metrics['MSLL'] = None

        return metrics
        

class SingleTaskExactGP(TrainableGP, gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, mean_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        self.double()
        self.likelihood.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SingleTaskSparseGP(TrainableGP, gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, likelihood, kernel, mean_module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.likelihood = likelihood

        self.double()
        self.likelihood.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiTaskExactGP(TrainableGP, gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, mean_module, num_tasks):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            mean_module, num_tasks=num_tasks
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(kernel),
            num_tasks=num_tasks,
            rank=min(2, num_tasks),
        )

        self.double()
        self.likelihood.double()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)






## -- old stuff below -- ##

class SingleOutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class xDifferentialGP:
    def __init__(self, input_dim, periodic_inputs, device="cpu"):
        self.device = device
        self.model = None 
        self.likelihood = None
        self.periodic_inputs = periodic_inputs
        self.X = []
        self.Y = []

        self.X_min_max = []

        self.X_test = []
        self.Y_test = []

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray, normalize=True, test=False):

        # Init min max array
        self.X_min_max = np.zeros((X_new.shape[1], 2))
        
        # Perform normalization 
        for i in range(X_new.shape[1]):

            # periodic inputs are assumed to repeat with frequency of 2 pi
            if self.periodic_inputs[i]:
                X_new[:, i] = X_new[:, i] % (2*pi)

            self.X_min_max[i, 0] = np.min(X_new[:,i])
            self.X_min_max[i, 1] = np.max(X_new[:,i]) 
            if normalize:
                X_new[:,i] = self.min_max_norm(X_new[:,i], self.X_min_max[i, 0], self.X_min_max[i, 1])

        if test:
            self.X_test.append(torch.tensor(X_new, dtype=torch.float64, device=self.device))
            self.Y_test.append(torch.tensor(Y_new, dtype=torch.float64, device=self.device))
        else:
            self.X.append(torch.tensor(X_new, dtype=torch.float64, device=self.device))
            self.Y.append(torch.tensor(Y_new, dtype=torch.float64, device=self.device))
    
    def min_max_norm(self, data, min, max):
        return (data - min)/(max - min) 
    
    def clear_data(self):
        self.X = []
        self.Y = []
        self.X_test = []
        self.Y_test = []
        self.X_min_max = []
        self.model = None
        self.likelihood = None
    
    def print_hyperparameters(self):
        if self.model is None:
            print("Model not trained yet.")
            return

        print(f"GP Hyperparameters for model on device {self.device}:")
        # Assuming SingleOutputGP uses a standard RBF kernel with outputscale
        kernel = self.model.covar_module
        if hasattr(kernel, "base_kernel"):
            # For ScaleKernel(RBFKernel())
            lengthscale = kernel.base_kernel.lengthscale.detach().cpu().numpy()
            outputscale = kernel.outputscale.detach().cpu().numpy()
        else:
            lengthscale = kernel.lengthscale.detach().cpu().numpy()
            outputscale = getattr(kernel, "outputscale", None)
            if outputscale is not None:
                outputscale = outputscale.detach().cpu().numpy()

        # Noise from likelihood
        noise = self.likelihood.noise.detach().cpu().numpy()

        print(f"  Lengthscale: {lengthscale}")
        print(f"  Outputscale: {outputscale}")
        print(f"  Noise: {noise}")

    def train(self, training_iter=50, lr=0.1):
        train_x = torch.cat(self.X, dim=0)
        train_y = torch.cat(self.Y, dim=0)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = SingleOutputGP(train_x, train_y, self.likelihood).to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, X_new: np.ndarray):
        test_x = torch.tensor(X_new, dtype=torch.float64, device=self.device)

        for i in range(test_x.shape[1]):
            # periodic inputs are assumed to repeat with frequency of 2 pi
            if self.periodic_inputs[i]:
                test_x[:, i] = test_x[:, i] % (2*pi)
            test_x[:,i] = self.min_max_norm(test_x[:,i], self.X_min_max[i, 0], self.X_min_max[i, 1])

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))
            mu = preds.mean.unsqueeze(-1).cpu().numpy()
            var = preds.variance.unsqueeze(-1).cpu().numpy()

        return mu, var


"""
TODO: Merge code from MultiGP branch.
"""
class MultiGP:
    def __init__(self, input_dim, output_dim, device="cpu"):
        self.input_dim = input_dim # Not really used
        self.output_dim = output_dim
        self.device = device

        self.X = []
        self.Y = []

        self.models = []
        self.likelihoods = []

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray):
        self.X.append(torch.tensor(X_new, dtype=torch.float64, device=self.device))
        self.Y.append(torch.tensor(Y_new, dtype=torch.float64, device=self.device))

    def train(self, training_iter=50, lr=0.1):
        train_x = torch.cat(self.X, dim=0)
        train_y = torch.cat(self.Y, dim=0)

        self.models = []
        self.likelihoods = []

        for d in range(self.output_dim):
            y_col = train_y[:, d]

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = SingleOutputGP(train_x, y_col, likelihood).to(self.device)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(training_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, y_col)
                loss.backward()
                optimizer.step()

            self.models.append(model)
            self.likelihoods.append(likelihood)

    def predict(self, Xnew: np.ndarray):
        test_x = torch.tensor(Xnew, dtype=torch.float64, device=self.device)

        mus, vars_ = [], []
        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = likelihood(model(test_x))
                mus.append(preds.mean.unsqueeze(-1).cpu().numpy())
                vars_.append(preds.variance.unsqueeze(-1).cpu().numpy())

        mu = np.hstack(mus)  # (N, output_dim)
        var = np.hstack(vars_)  # (N, output_dim)
        return mu, var

def create_gps(gp_settings, device, X_train, Y_train, training_iter, lr, variances=False):
    """Create and train GPs for all given GP settings and translate to CasADi, return GP and CasADi representations in dictionaries"""
    gps = {}
    casadi_gps = {}

    for idx, setting in gp_settings.items():
        gp = xDifferentialGP(input_dim=setting.input_dim, device=device, periodic_inputs=setting.periodic_inputs)
        input_data = setting.get_input(X_train)
        output_data = setting.get_output(Y_train)
        print(f"Training gp for {setting.name}")
        gp.add_data(input_data, output_data)
        gp.train(training_iter, lr)
        gp.model.double() # This prevents a small error between the GP and CasADi variance model
        print("Converting to casadi")
        casadi_gp = get_xdot(gp, input_data, output_data, setting.control_dim, variances)
        gps[idx] = gp
        casadi_gps[idx] = casadi_gp

    return gps, casadi_gps
