import gpytorch

from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.models.model_list import IndependentModelList

import torch
import numpy as np
import matplotlib.pyplot as plt

from gp_data import DataGenerator, SamplingStrategy

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ODEModel import ODEModel # just need it for type hints


@dataclass
class GPSetting:

    # Target GP var(s) to predict
    output_var_dots: frozenset[str] 

    # Input GP var(s)
    input_vars: frozenset[str] 
    
    name: str 

    kernel: Kernel = gpytorch.kernels.RBFKernel()
    mean_module: Mean = gpytorch.means.ConstantMean()
    likelihood: Likelihood = None

    # Optional subset of input_vars that are periodic 
    periodic_vars: frozenset[str] = frozenset() 

    # Optional custom data generation settings
    data_generator: DataGenerator = field(
        default_factory=lambda: DataGenerator(sampling_strategy=SamplingStrategy.LHS)
    )
    #data: GPData | None = None

    training_iters=50
    lr=0.1

    # Use sparse GP
    use_sparse: bool = False

    sparse_n_inducing = 200


    def __post_init__(self):
        # Frozen sets are hashable and immutable
        self.output_var_dots = frozenset(self.output_var_dots)
        self.input_vars = frozenset(self.input_vars)

        if self.periodic_vars is not None:
            self.periodic_vars = frozenset(self.periodic_vars)
        else:
            self.periodic_vars = frozenset()

        num_tasks = len(self.output_var_dots)
        if self.likelihood is None:
            if num_tasks > 1:
                self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            else:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def validate(self):
        # output_var_dots, input_vars should not be empty sets.
        if not self.output_var_dots:
            raise ValueError("output_var_dots cannot be empty")
        if not self.input_vars:
            raise ValueError("input_vars cannot be empty")
        
        # All periodic vars need to exist in input_vars
        if not self.periodic_vars <= self.input_vars:
            raise ValueError("All periodic_vars must exist in input_vars")
        
        # Sparse GP is only supported for SingleTask 
        if self.use_sparse and len(self.output_var_dots) != 1:
            raise NotImplementedError("SparseGP is not implemented for MultiTask GPs and requires exactly 1 output var")
        
    def get_kwargs(self) -> dict[str, Any]:
        """
        Return keyword arguments for initializing a GP model.
        """
        # Ensure data exists
        if not hasattr(self, "data_generator") or self.data_generator is None:
            raise ValueError(f"GPSetting {self.name} has no data_generator assigned")
        
        train_x, train_y = self.data_generator.get_training_data()
        
        if len(self.output_var_dots) == 1:
            train_y = train_y.squeeze(-1)  # make shape (N,)

        return {
            "train_x": train_x,
            "train_y": train_y,
            "likelihood": self.likelihood,
            "kernel": self.kernel,
            "mean_module": self.mean_module,
        }

class GPManager:
    """
    Mostly for configuration: validating and parsing GP settings to instantiate GPs.
    """
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def create_ModelList(cls, from_model: "ODEModel") -> IndependentModelList:
        """
        Create a ModelList GP from model settings.
        """
        from gp_models import TrainableGP
        settings = from_model.gp_spec()
        models = []
        for setting in settings:
            models.append(TrainableGP.from_gp_setting(from_model, setting)) # build GP and data

        return IndependentModelList(*models)
    
    @classmethod
    def train_models(cls, model_list: IndependentModelList):
        print("Training models...")
        for gp in model_list.models:
            # Extract training data from GP's data generator
            train_X, train_Y = gp.setting.data_generator.X_train, gp.setting.data_generator.Y_train
            if train_X is None or train_Y is None:
                raise ValueError(f"GP {gp.setting.name} has no training data.")
            
            iters = gp.setting.training_iters
            lr = gp.setting.lr

            gp.train_model(train_X, train_Y, iters=iters, lr=lr)

    @classmethod
    def make_predictions(
        cls, 
        model_list: IndependentModelList, 
        input_dict: Dict[str, float]  # keys: variable names, values: scalars
    ) -> Dict[str, float]:
        """
        Predict for each GP in the model list using the provided input_dict.

        :param model_list: IndependentModelList of trained GPs
        :param input_dict: Dict containing all possible inputs (state, input, params)
        :return: Dict mapping output variable names to predicted scalar values
        """
        predictions: Dict[str, float] = {}

        for gp in model_list.models:
            # Determine which input_vars this GP uses
            input_vars = gp.setting.input_vars

            # Gather input values in the correct order
            x_list = [torch.tensor([input_dict[k]], dtype=torch.float64) for k in input_vars]

            if not x_list:
                raise ValueError(f"No input values found for GP {gp.setting.name}")

            # Shape (1, num_inputs)
            X_gp = torch.cat(x_list, dim=-1).unsqueeze(0)

            # Predict
            mu, _ = gp.predict(X_gp)  # mu shape: (1, num_outputs) or (1,)

            # Assign to corresponding output variables
            for i, out_var in enumerate(gp.setting.output_var_dots):
                predictions[out_var] = mu[0, i].item() if mu.ndim > 1 else mu.item()

        return predictions
    
    @classmethod
    def plot_GP_learning_curve(
        cls,
        ode_model: "ODEModel",
        model_list: IndependentModelList,
        training_sizes: Iterable[int],
        lr: float = 0.5,
        training_iter: int = 50,
    ):
        """
        Plot learning curves for all GPs in an IndependentModelList.
        Automatically generates data per GP using its GPSetting.
        """

        eval_data = {}

        for N in training_sizes:
            print(f"\n=== Training size N = {N} ===")
            eval_data[N] = {}

            for gp in model_list.models:
                setting = gp.setting
                print(f"GP: {setting.name}")

                X, Y = setting.data_generator.generate_from_vars(
                    ode_model, setting.input_vars, setting.output_var_dots
                )

                min_rows = min(X.shape[0], Y.shape[0])
                if min_rows < N:
                    raise ValueError(f"Not enough samples ({min_rows}) for requested N={N}")
                X = X[:min_rows, :]
                Y = Y[:min_rows, :]

                X = X[:N, :]
                Y = Y[:N, :]

                X_train, X_test, Y_train, Y_test = DataGenerator.split_train_test(
                    X, Y, train_ratio=0.8
                )

                print(f"Shapes after split: X_train={X_train.shape}, Y_train={Y_train.shape}, "
                    f"X_test={X_test.shape}, Y_test={Y_test.shape}")

                X_train = torch.tensor(X_train, dtype=torch.float64)
                X_test = torch.tensor(X_test, dtype=torch.float64)
                Y_train = torch.tensor(Y_train, dtype=torch.float64)
                Y_test = torch.tensor(Y_test, dtype=torch.float64)

                if isinstance(gp, gpytorch.models.ExactGP):
                    gp.set_train_data(inputs=X_train, targets=Y_train, strict=False)

                gp.train_model(X_train, Y_train, iters=training_iter, lr=lr)

                metrics = {
                    "RMSE": gp.eval_RMSE(X_test, Y_test),
                    "MAE": gp.eval_MAE(X_test, Y_test),
                    "NLPD": gp.eval_NLPD(X_test, Y_test),
                    "MSLL": gp.eval_MSLL(X_train, Y_train, X_test, Y_test)
                }
                eval_data[N][setting.name] = metrics

        # ---------- Plot ----------
        metrics_errors = ["RMSE", "MAE"]
        metrics_likelihood = ["NLPD", "MSLL"]
        colors_error = {"RMSE": "tab:blue", "MAE": "deepskyblue"}
        colors_likelihood = {"NLPD": "orange", "MSLL": "red"}

        training_sizes_ls = sorted(eval_data.keys())

        for gp in model_list.models:
            name = gp.setting.name
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: RMSE / MAE
            for metric in metrics_errors:
                values = [eval_data[N][name][metric] for N in training_sizes_ls]
                axes[0].plot(training_sizes_ls, values, marker='o', label=metric, color=colors_error[metric])
            axes[0].set_xlabel("Number of training samples (N)")
            axes[0].set_ylabel("RMSE / MAE")
            axes[0].set_title("Error metrics")
            axes[0].grid(True)
            axes[0].legend()

            # Right: NLPD / MSLL
            for metric in metrics_likelihood:
                values = [eval_data[N][name][metric] for N in training_sizes_ls]
                axes[1].plot(training_sizes_ls, values, marker='x', label=metric, color=colors_likelihood[metric])
            axes[1].set_xlabel("Number of training samples (N)")
            axes[1].set_ylabel("NLPD / MSLL")
            axes[1].set_title("Likelihood metrics")
            axes[1].grid(True)
            axes[1].legend()

            plt.suptitle(f"Learning curves for {name}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        return eval_data
