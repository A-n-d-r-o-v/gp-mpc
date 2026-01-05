import torch
import gpytorch
import numpy as np

from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.models.model_list import IndependentModelList

from math import pi
#from data import StateIdx, InputIdx
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Optional
from gp_to_casadi import get_xdot

if TYPE_CHECKING:
    from ODEModel import ODEModel # just need it for type hints


@dataclass
class GPSetting:

    # Target GP var(s) to predict
    output_vars: frozenset[str] 

    # Input GP var(s)
    input_vars: frozenset[str] 
    
    name: str 

    kernel: Kernel = gpytorch.kernels.RBFKernel()
    likelihood: Likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Optional subset of input_vars that are periodic 
    periodic_vars: frozenset[str] = frozenset() 

    # Use sparse GP
    use_sparse: bool = False

    def __post_init__(self):
        # Frozen sets are hashable and immutable
        self.output_vars = frozenset(self.output_vars)
        self.input_vars = frozenset(self.input_vars)

        if self.periodic_vars is not None:
            self.periodic_vars = frozenset(self.periodic_vars)
        else:
            self.periodic_vars = frozenset()

    def validate(self):
        # output_vars, input_vars should not be empty sets.
        if not self.output_vars:
            raise ValueError("output_vars cannot be empty")
        if not self.input_vars:
            raise ValueError("input_vars cannot be empty")
        
        # All periodic vars need to exist in input_vars
        
        if not self.periodic_vars <= self.input_vars:
            raise ValueError("All periodic_vars must exist in input_vars")
        
        # Sparse GP is only supported for SingleTask 
        if self.use_sparse and len(self.output_vars) != 1:
            raise NotImplementedError("SparseGP is not implemented for MultiTask GPs and requires exactly 1 output var")

class GPManager:
    """
    Mostly for configuration: validating and parsing GP settings to instantiate GPs.
    """
    def __init__(self, name: str, gp_settings: List[GPSetting]):
        self.name = name

        if __class__._validate_GPSettings(gp_settings):
            self.gp_settings = gp_settings

    @classmethod
    def _validate_GPSettings(self, settings: List[GPSetting]):
        # Each var in output_vars may only exist in one instance - they must be unique.
        # (NOTE: ^ I think this should hold true , intuitively)
        # An output variable cannot also be an input variable and vice versa.
        seen_in_vars, seen_out_vars = set(), set()
        
        for setting in settings:
            setting.validate()
            for in_var in setting.input_vars:
                seen_in_vars.add(in_var)
            for out_var in setting.output_vars:
                if out_var in seen_out_vars:
                    raise ValueError(f"output_var '{out_var}' appears in multiple GPSettings")
                seen_out_vars.add(out_var)       

        in_out_overlap = set.intersection(seen_in_vars, seen_out_vars)
        if len(in_out_overlap) != 0:
            raise ValueError(f"Overlapping input and output variables: {in_out_overlap}")
        
        return True

    @classmethod    
    def _build_gp(cls, setting: GPSetting):
        """
        Create a single GP model from one GPSetting.
        """
        if setting.use_sparse:
            # must be single-output
            return GPBase(None, None, setting.likelihood) #gpytorch.models.SparseSingleOutputGP(setting) # TODO: use our code
        else:
            if len(setting.output_vars) == 1:
                return GPBase(None, None, setting.likelihood)#gpytorch.models.ExactSingleOutputGP(setting) # TODO
            else:
                return GPBase(None, None, setting.likelihood) # gpytorch.models.ExactMultiOutputGP(setting) # TODO

    def create_ModelList(self, settings: Optional[List[GPSetting]] = None) -> IndependentModelList:
        """
        Create a ModelList GP from settings.
        
        :param self:
        :param settings: Optionally build from this list of GP settings instead of self.gp_settings.
        :type settings: Optional[List[GPSetting]]
        """
        if settings is None:
            settings = self.gp_settings
        models = []
        for setting in settings:
            models.append(__class__._build_gp(setting))

        return IndependentModelList(*models)

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


class GPData:

    def __init__(self):
        self.trainX: List[torch.Tensor] = []
        self.trainY: List[torch.Tensor] = []

        self.testX: List[torch.Tensor] = []
        self.testY: List[torch.Tensor] = []

    
    def generate_synthetic_lhs(self, model: "ODEModel"):
        """
        Generate training data by sampling the state space using Latin Hypercube Sampling (LHS).
        For each LHS sample, a sampled input is applied to the state S, to land in state S'.
        The resulting training sample is the state difference S - S'.
        
        :param self: Description
        :param model: The ODE model to sample from.
        :type model: ODEModel
        """
        pass

    def generate_from_trajectories(self, model: "ODEModel"):
        """
        Generate training data by sampling different control (input) sequences.
        
        :param self: Description
        :param model: The ODE model to sample from.
        :type model: ODEModel
        """
        pass
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



class GPBase(gpytorch.models.ExactGP):
     def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

    # Predict
    # AddData (or handle this in DataLoader)
    # ...


# Make SingleOutputGP and MultiOutputGP inherit from GPBase and move logic there.
# What do we do with DifferentialGP ? Maybe split into DataLoader and pick the appropriate GP? Or maybe this is just our new GPBase?

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
        gp = DifferentialGP(input_dim=setting.input_dim, device=device, periodic_inputs=setting.periodic_inputs)
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
