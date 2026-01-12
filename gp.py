import gpytorch

from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.models.model_list import IndependentModelList

import torch
import numpy as np

from gp_data import DataGenerator, SamplingStrategy

#from data import StateIdx, InputIdx
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, TYPE_CHECKING

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
        Create a ModelList GP from settings.
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
    def create_BatchIndependentMultitaskGPModel(cls, settings: Optional[List[GPSetting]] = None):
        """
        TODO: Could be interesting to look into.
        
        :param cls: 
        :param settings: Optionally build from this list of GP settings instead of self.gp_settings.
        :type settings: Optional[List[GPSetting]]
        """
        # each setting: 
        #   - Exactly one output var
        #   - Not sparse
        # All settings must use the same input_vars 
        num_tasks = len(settings)
        raise NotImplementedError




'''
@dataclass
class GPSettings:
    """Settings for a single GP model"""
    name: str
    state_indices: List[int]  # Which state variables are inputs
    control_indices: List[int]  # Which control variables are inputs
    output_index: int  # Which state derivative this GP predicts
    periodic_inputs: List[bool]  # Which inputs are periodic
    
    @property
    def input_dim(self):
        """Number of inputs in total"""
        return len(self.state_indices) + len(self.control_indices)
    
    @property
    def control_dim(self):
        """Number of control inputs"""
        return len(self.control_indices)
    
    @property
    def state_dim(self):
        """Number of state inputs"""
        return len(self.state_indices)
    
    def get_input_indices(self):
        """Get the indices for all inputs, state first then control"""
        return self.state_indices + self.control_indices
    
    def get_input(self, X: np.ndarray):
        """Extract the relevant input"""
        return X[:, self.get_input_indices()]
    
    def get_output(self, Y: np.ndarray):
        """Get the relevant output"""
        return Y[:, self.output_index]
'''

'''
def create_gp_settings(gps):
    """Create settings for all requested GPs and return them as a dictionary"""
    gp_settings = {}
    
    if StateIdx.V1 in gps:
        gp_settings[StateIdx.V1] = GPSettings(
            name="v1_dot",
            state_indices=[StateIdx.ROLL, StateIdx.PITCH, StateIdx.YAW],
            control_indices=[StateIdx.THRUST],
            output_index=StateIdx.V1,
            periodic_inputs=[False, False, False, False])

    if StateIdx.V2 in gps:
        gp_settings[StateIdx.V2] = GPSettings(
            name="v2_dot",
            state_indices=[StateIdx.ROLL, StateIdx.PITCH, StateIdx.YAW],
            control_indices=[StateIdx.THRUST],
            output_index=StateIdx.V2,
            periodic_inputs=[False, False, False, False])

    if StateIdx.V3 in gps:
        gp_settings[StateIdx.V3] = GPSettings(
            name="v3_dot",
            state_indices=[StateIdx.ROLL, StateIdx.PITCH],
            control_indices=[StateIdx.THRUST],
            output_index=StateIdx.V3,
            periodic_inputs=[False, False, False])

    if StateIdx.ROLL in gps:
        gp_settings[StateIdx.ROLL] = GPSettings(
            name="roll_dot",
            state_indices=[StateIdx.ROLL],
            control_indices=[StateIdx.ROLL_REF],
            output_index=StateIdx.ROLL,
            periodic_inputs=[False, False])

    if StateIdx.PITCH in gps:
        gp_settings[StateIdx.PITCH] = GPSettings(
            name="pitch_dot",
            state_indices=[StateIdx.PITCH],
            control_indices=[StateIdx.PITCH_REF],
            output_index=StateIdx.PITCH,
            periodic_inputs=[False, False])
    
    return gp_settings
'''



"""
# When you predic you want to set a known min max values such that the 
# value to predic is scaled in accordance with the training data
def min_max_norm(data, set_min=None, set_max=None):
    if set_min and set_max:
        min = set_min
        max = set_max
    else:
        min = np.min(data)
        max = np.max(data)
    
    return (data - min)/(max - min) if min != max else data
"""
