from __future__ import annotations

import numpy as np

from typing import List, Type, Optional, TYPE_CHECKING
from dataclasses import dataclass

from gpytorch.models import IndependentModelList
#from gp import min_max_norm
#from math import pi 
from gp import GPSetting, GPManager

if TYPE_CHECKING: # Type hints only
    from data import ODEData
    from gp_data import DataGenerator

from abc import ABC, abstractmethod

#Perturbation = Callable[[State, Input, Params, State], State]


class ODEModel(ABC):
    """
    Generic base class for an ODE model that is described both by a set of states, inputs, and parameters,
    and a set of ordinary differential equations that describe how these variables interact.
    This provides functionality for defining GP and ACADOS specifications in a centralized place
    for use in the rest of the pipeline.
    Can not be instantiated; must be inherited from.
    """

    # An ODE model needs the following classes that all inherit from ODEData.
    State: Type[ODEData] | None     = None
    Input: Type[ODEData] | None     = None
    Params: Type[ODEData] | None    = None # This contains constants.

    def __init__(
            self, 
            name: str = None, 
            params: Type[ODEData] = None):
        self.name = name
        self.params = params
        self._validate_unique_fields()
        self._validate_gp_spec()

    @classmethod
    def all_field_names(cls) -> set[str]:
        return (
            set(cls.State.__dataclass_fields__) |
            set(cls.Input.__dataclass_fields__) |
            set(cls.Params.__dataclass_fields__)
        )

    def _validate_unique_fields(self):
        """
        Enforce global uniqueness of the field names in State, Input, and Params.
        
        We do not e.g. want State and Input to share a field with the same name,
        since names hold conceptual meaning. "thrust" cannot be both a state and
        an input, for example.
        
        :param self: 
        """
        # Get the sets of fields
        states = set(self.State.__dataclass_fields__)
        inputs = set(self.Input.__dataclass_fields__)
        params = set(self.Params.__dataclass_fields__)

        # intersections between each set
        intersect = {
            ("State", "Input" ): states & inputs,
            ("State", "Params"): states & params,
            ("Input", "Params"): inputs & params,
        }

        # check overlap
        for (a, b), overlap in intersect.items():
            if overlap:
                raise ValueError(
                    f"Field name(s) {overlap} appear in both {a} and {b}. "
                    "Field names must be globally unique."
                )    
                    
        return True
    
    def _classify_var_names(self, deps: set[str]):
        """
        Classify a flat set of var name strings into appropriate dataclasses based on their respective 
        field names.
        
        :param self: Description
        :param deps: Description
        :type deps: set[str]
        """
        state_set = set(self.State.__dataclass_fields__)
        input_set = set(self.Input.__dataclass_fields__)
        param_set = set(self.Params.__dataclass_fields__)

        # Get the intersection
        state = deps & state_set
        input = deps & input_set
        params = deps & param_set

        unknowns = deps - state - input - params
        if unknowns:
            raise KeyError(f"Unknown dependency fields: {unknowns}")

        return (
            tuple(sorted(state)),
            tuple(sorted(input)),
            tuple(sorted(params)),
        )
    
    @abstractmethod
    def gp_spec(self) -> List[GPSetting]:
        raise NotImplementedError
    
    def _validate_gp_spec(self):
        # Each var in output_var_dots may only exist in one instance - they must be unique.
        # seen_in_vars, seen_out_vars = set(), set()
        
        settings = self.gp_spec()
        
        seen_out_vars = set()

        for setting in settings:
            setting.validate()
            # for in_var in setting.input_vars:
            #     seen_in_vars.add(in_var)
            for out_var in setting.output_var_dots:
                if out_var in seen_out_vars:
                    raise ValueError(f"output_var '{out_var}' appears in multiple GPSettings")
                seen_out_vars.add(out_var)       

        #in_out_overlap = set.intersection(seen_in_vars, seen_out_vars)
        #if len(in_out_overlap) != 0:
        #    raise ValueError(f"Overlapping input and output variables: {in_out_overlap}")
        
        return True

    @abstractmethod
    def _gps_to_model(self) -> Optional[{str}]:
        """
        Specify the set of GP outputs to use for prediction.
        If None, all outputs are used.
        
        :param self: Description
        :return: A set of strings containing the names of the GP output variables to model.
        :rtype: Any | None
        """
        return None

    @abstractmethod
    def _LHS_data_ranges(self, params) -> dict[str, tuple[float, float]]:
        """
        Specify the ranges of values to generate with Latin Hypercube Sampling (LHS).
        This should return a dictionary mapping {"var": (min, max), ...} and must contain
        ranges for all fields in State and Input.

        :param self: Description
        :return: Description
        :rtype: dict[str, tuple[float, float]]
        """
        raise NotImplementedError

    def get_LHS_data_ranges_safe(self, params=None) -> dict[str, tuple[float, float]]:
        """
        Get the LHS_data_ranges dict and validate that the dict structure is correctly implemented.
        Return the dict of ranges if no errors are present.
        
        TODO:   Instead of requiring all fields to be there, implement the option to
                default missing fields to (0,0)
        
        :param self: 
        :return: The ranges, if no errors are present.
        :rtype: dict[str, tuple[float, float]]
        """
        if not params:
            params = self.params

        ranges = self._LHS_data_ranges(params)
        if ranges is None:
            raise ValueError("Ranges is None. Ensure that it is implemented.")
        
        # Check that all state and input fields are represented by the dict 
        self.State.validate_covered_by_dict(ranges, allow_missing=False, allow_extra=True)
        self.Input.validate_covered_by_dict(ranges, allow_missing=False, allow_extra=True)

        # Validate range tuple
        for k, v in ranges.items():
            if (
                not isinstance(v, tuple)
                or len(v) != 2
                or not all(isinstance(x, (int, float)) for x in v)
                or v[0] > v[1]
            ):
                raise ValueError(f"Invalid range for '{k}': {v}")

        return ranges
    
    def filtered_ranges(
            self, 
            ranges: dict[str, tuple[float, float]],
            keys: set[str] | list[str] | frozenset[str]
        ) -> dict[str, tuple[float, float]]:
        """
        Return a filtered dict of LHS ranges, keeping only the specified keys.
        """
        return {k: v for k, v in ranges.items() if k in keys}
    
    @abstractmethod
    def default_data_generator(self) -> DataGenerator:
        pass

    @abstractmethod
    def _analytical_dot(self, state, input, params) -> dict[str, float]:
        raise NotImplementedError

    def differentiate(
            self,
            state, 
            input, 
            params = None,
            gp_model_list: IndependentModelList | None = None
            #perturbations: List[Perturbation] = None # TODO !!
        ) -> ODEData:
        """
        Docstring for differentiate TODO

        NOTE: We will want an option to *add* the GP to the ODE rather than replace it.
                Maybe bake this into the GP object: is it a residual or "absolute" from-scratch GP?
                Set this when we generate data.
                Data: Is Residual? --> GP --> Here
        
        :param self: 
        :param state: The state.
        :param input: The input.
        :param params: The parameters.
        :param diff_gps: TODO
        :type diff_gps: Optional[List] TODO
        :return: The resulting State.
        :rtype: ODEData
        """
        if params is None:
            params = self.params

        analytical_dot = self._analytical_dot(state, input, params)

        # check that everything is there...
        self.State.validate_covered_by_dict(analytical_dot, allow_missing=False, allow_extra=False) 

        # If a state variable is associated with a trained GP, replace the analytical_dot entry with the GP prediction.
        if gp_model_list:
            input_dict = {
                **{k: getattr(state, k) for k in state.__dataclass_fields__},
                **{k: getattr(input, k) for k in input.__dataclass_fields__},
                **{k: getattr(params, k) for k in params.__dataclass_fields__},
            }

            # Get GP predictions for all relevant variables
            gp_predictions = GPManager.make_predictions(gp_model_list, input_dict)
            gps_to_model = self._gps_to_model()

            # Update analytical_dot with GP outputs
            for k, v in gp_predictions.items():
                if gps_to_model:
                    if not k in gps_to_model:
                        continue
                #print(f"Replacing state_var {k} with gp prediction {v}")

                analytical_dot[k] = v  # or += v if using residual GPs

        # ------------------ #
        # -- Result state -- #
        # ------------------ #
        res_state = self.State.from_dict(analytical_dot)

        # TODO: Apply optional perturbations

        return res_state
    
    def step(self, state, input, params = None, gp_model_list: IndependentModelList | None = None, dt=1):
        if params is None:
            params = self.params
        step_state = self.differentiate(state, input, params=params, gp_model_list=gp_model_list)
        return state + step_state*dt
    
    '''
    @abstractmethod
    def position(self, state):
        return None
    
    @abstractmethod
    def orientation(self, state):
        return None
    '''

    # TODO: add ACADOS stuff, like cost, ...


