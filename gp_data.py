import torch
import numpy as np

from pyDOE3 import lhs

#from data import StateIdx, InputIdx
from typing import List, TYPE_CHECKING, Tuple
from dataclasses import dataclass
from enum import Enum

import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import dcor  # distance correlation library
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from ODEModel import ODEModel # just need it for type hints
    #from gp import GPSetting


class SamplingStrategy(Enum):
    LHS = "lhs"
    TRAJECTORY = "trajectory"

@dataclass
class DataGenerator:
    
    sampling_strategy: SamplingStrategy
    
    train_ratio = 0.8
    
    # LHS
    n_samples = 500


    # ...

    # Trajectory
    # ...
    
    X_train: list[torch.Tensor] | None = None
    Y_train: list[torch.Tensor] | None = None
    
    X_test: list[torch.Tensor] | None = None
    Y_test: list[torch.Tensor] | None = None

    def set_data(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def get_training_data(self):
        if self.X_train is None or self.Y_train is None:
            raise ValueError("Training data is unassigned.")
        return self.X_train, self.Y_train

    def get_test_data(self):
        if self.X_test is None or self.Y_test is None:
            raise ValueError("Test data is unassigned.")
        return self.X_test, self.Y_test
    
    @classmethod
    def plot_dcor_heatmap(cls, model: "ODEModel", n_samples: int = 20000):
        """
        Generate LHS data and plot a distance correlation heatmap of the state differences ΔS.
        Automatically filters out variables with zero variance.
        """
        # Generate data
        var_names = list(model.State.__dataclass_fields__.keys())
        X, Y = cls.generate_synthetic_lhs(model, n_samples=n_samples, output_var_dots=var_names)

        # Filter out constant columns
        filtered_names = []
        filtered_cols = []
        for i, name in enumerate(var_names):
            col = Y[:, i]
            if not np.allclose(col, col[0]):  # skip columns with zero variance
                filtered_names.append(name)
                filtered_cols.append(col)

        if len(filtered_cols) == 0:
            raise ValueError("All variables are constant; cannot compute distance correlations.")

        Y_filtered = np.stack(filtered_cols, axis=1)
        N_vars = len(filtered_names)

        # Compute dcor matrix
        dcor_matrix = np.zeros((N_vars, N_vars))
        for i in range(N_vars):
            for j in range(N_vars):
                val = dcor.distance_correlation(Y_filtered[:, i], Y_filtered[:, j])
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                dcor_matrix[i, j] = np.clip(val, 0, 1)  # ensure valid [0,1] range

        # Plot heatmap
        sns.set(style="white")
        plt.figure(figsize=(10, 8))
        sns.heatmap(dcor_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=filtered_names, yticklabels=filtered_names)
        plt.title("Distance Correlation of State Differences")
        plt.tight_layout()
        plt.show()

    def normalize(self):
        raise NotImplementedError

    def generate_from_vars(self, model: "ODEModel", input_vars, output_var_dots):
        
        if self.sampling_strategy == SamplingStrategy.LHS:
            gp_vars = set(input_vars) # | set(output_var_dots)
            ranges = model.get_LHS_data_ranges_safe()
            ranges = model.filtered_ranges(ranges, gp_vars)

            return __class__.generate_synthetic_lhs(
                model=model, 
                n_samples=self.n_samples, 
                ranges=ranges,
                output_var_dots=output_var_dots
            )

        elif self.sampling_strategy == SamplingStrategy.TRAJECTORY:
            return __class__.generate_from_trajectories(model=model)
        else:
            raise NotImplementedError
                
    @classmethod
    def split_train_test(
        cls, 
        X: np.ndarray, 
        Y: np.ndarray, 
        train_ratio: float = 0.8, 
        random_state: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        :param X: Input data, shape (N, D)
        :param Y: Output data, shape (N, O)
        :param train_ratio: Fraction of samples to use for training
        :param random_state: Optional seed for reproducibility
        :return: X_train, Y_train, X_test, Y_test
        """
        if random_state is not None:
            np.random.seed(random_state)

        N = X.shape[0]
        indices = np.random.permutation(N)
        split_idx = int(N * train_ratio)

        train_idx = indices[:split_idx]
        test_idx  = indices[split_idx:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test,  Y_test  = X[test_idx],  Y[test_idx]

        return X_train, Y_train, X_test, Y_test

    @classmethod
    def generate_synthetic_lhs(
            cls, 
            model: "ODEModel", 
            n_samples: int, 
            output_var_dots,
            ranges: dict[str, tuple[float, float]] | None = None,
            criterion: str = "center", 
            random_state: int | None = None, 
            as_tensor: bool = True
        ) -> Tuple[np.ndarray, np.ndarray] | Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data by sampling the state space using Latin Hypercube Sampling (LHS).
        For each LHS sample, a sampled input is applied to the state S, to land in state S'.
        The resulting training sample is the state difference S - S'.
        
        :param self: Description
        :param model: The ODE model to sample from.
        :type model: ODEModel
        """

        # TODO: optional second model B of same type as model A
        #       model B is a perturbed version of model A.
        #       if this is provided (not None), generate data
        #       as residual 

        if ranges is None:
            ranges = model.get_LHS_data_ranges_safe()
        keys = list(ranges.keys())
        dim = len(keys)

        if dim == 0:
            raise ValueError("Dim is 0")

        state_keys, input_keys, _ = model._classify_var_names(set(keys))

        # Generate LHS in [0,1] and then scale to ranges
        lhs_samples = lhs(dim, samples=n_samples, criterion=criterion, random_state=random_state)
        scaled_samples = cls._scale_lhs(lhs_samples, ranges)

        # Convert to State/Input objects
        states, inputs = [], []
        for row in scaled_samples:
            d = dict(zip(keys, row))
            state_dict = {k: d[k] for k in state_keys if k in d}
            input_dict = {k: d[k] for k in input_keys if k in d}

            #print(state_dict)
            #print(input_dict)
            s = model.State.from_dict(state_dict, fill_missing=True)
            u = model.Input.from_dict(input_dict, fill_missing=True)
            states.append(s)
            inputs.append(u)

        dot_dicts = []
        for s, u in zip(states, inputs):
            stepped = model.step(s, u, model.params, dt=1)
            dot_dict = (stepped - s).as_dict()  # equivalent to state_dot over dt
            dot_dicts.append(dot_dict)

        # Build X with only selected state and input variables
        X = np.array([
            np.concatenate([
                s.as_numpy()[[i for i, k in enumerate(model.State.__dataclass_fields__) if k in state_keys]],
                u.as_numpy()[[i for i, k in enumerate(model.Input.__dataclass_fields__) if k in input_keys]],
            ])
            for s, u in zip(states, inputs)
        ])

        Y = np.array([
            [dot[k] for k in output_var_dots] 
            for dot in dot_dicts
        ])

        if as_tensor:
            X = torch.tensor(X, dtype=torch.float64)
            Y = torch.tensor(Y, dtype=torch.float64)

        return X, Y
    
    @classmethod
    def generate_inducing_points(
        cls, 
        model: "ODEModel", 
        n_samples: int, 
        ranges: dict[str, tuple[float, float]] | None = None,
        **kwargs):
        X, _ = cls.generate_synthetic_lhs(model, n_samples, ranges=ranges, **kwargs)
        return X

    def generate_from_trajectories(self, model: "ODEModel"):
        """
        Generate training data by sampling different control (input) sequences.
        
        :param self: Description
        :param model: The ODE model to sample from.
        :type model: ODEModel
        """
        raise NotImplementedError
        # Needs to be generic! 
        # like, automatically sample different input sequences and check distance to existing data.

    @classmethod
    def _scale_lhs(
        cls,
        lhs_samples: np.ndarray, 
        ranges: dict
        ) -> np.ndarray:
        """
        Scale LHS samples in [0,1] to the given variable ranges.
        """
        lows = np.array([low for low, _ in ranges.values()])
        highs = np.array([high for _, high in ranges.values()])
        return lows + (highs - lows) * lhs_samples
    
