import torch
import gpytorch
import numpy as np
from math import pi

# When you predic you want to set a known min max values such that the 
# value to predic is scaled in accordance with the training data
def min_max_norm(data, set_min=None, set_max=None):
    if set_min and set_max:
        min = set_min
        max = set_max
    else:
        min = np.min(data)
        max = np.max(data)
    
    return (data - min)/(max - min) 

class SingleOutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DifferentialGP:
    def __init__(self, input_dim, periodic_inputs, device="cpu"):
        self.device = device
        self.model = None 
        self.likelihood = None
        self.periodic_inputs = periodic_inputs
        self.X = []
        self.Y = []

        self.X_min_max = []

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray):

        # Init min max array
        self.X_min_max = np.zeros((X_new.shape[1], 2))
        
        # Perform normalization 
        for i in range(X_new.shape[1]):

            # periodic inputs are assumed to repeat with frequency of 2 pi
            if self.periodic_inputs[i]:
                X_new[:, i] = X_new[:, i] % (2*pi)

            self.X_min_max[i, 0] = np.min(X_new[:,i])
            self.X_min_max[i, 1] = np.max(X_new[:,i]) 
            X_new[:,i] = self.min_max_norm(X_new[:,i], self.X_min_max[i, 0], self.X_min_max[i, 1])

        self.X.append(torch.tensor(X_new, dtype=torch.float32, device=self.device))
        self.Y.append(torch.tensor(Y_new, dtype=torch.float32, device=self.device))
    
    def min_max_norm(self, data, min, max):
        return (data - min)/(max - min) 

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
        test_x = torch.tensor(X_new, dtype=torch.float32, device=self.device)

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
        self.X.append(torch.tensor(X_new, dtype=torch.float32, device=self.device))
        self.Y.append(torch.tensor(Y_new, dtype=torch.float32, device=self.device))

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
        test_x = torch.tensor(Xnew, dtype=torch.float32, device=self.device)

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
