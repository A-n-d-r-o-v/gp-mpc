import numpy as np

from typing import List, Type, Optional, TYPE_CHECKING
from dataclasses import dataclass
#from gp import min_max_norm
#from math import pi 

if TYPE_CHECKING: # Type hints only
    from gp import GPSetting
    from data import ODEData

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
    State: Type[ODEData] = None
    Input: Type[ODEData] = None
    Params: Type[ODEData] = None

    def __init__(self, name: str = None):
        self.name = name
        self._validate_unique_fields()

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

        return {
            "state": tuple(sorted(state)),
            "input": tuple(sorted(input)),
            "params": tuple(sorted(params)),
        }
    
    @abstractmethod
    def gp_spec(self) -> List[GPSetting]:
        pass

    @abstractmethod
    def _analytical_dot(self, state, input, params) -> dict[str, float]:
        pass


    def differentiate(
            self,
            state, 
            input, 
            params,
            diff_gps: Optional[List] = None
            #perturbations: List[Perturbation] = None # TODO !!
        ) -> ODEData:
        """
        Docstring for differentiate TODO
        
        :param self: 
        :param state: The state.
        :param input: The input.
        :param params: The parameters.
        :param diff_gps: TODO
        :type diff_gps: Optional[List] TODO
        :return: The resulting State.
        :rtype: ODEData
        """
                #return super().differentiate(state, input, params, diff_gps)
       
        analytical_dot = self._analytical_dot(state, input, params)

        self.State._validate_field_dict(analytical_dot) # check that everything is there...

        # If a state variable is associated with a trained GP, replace the analytical_dot entry with the GP prediction.
        if diff_gps:
            for var_id, gp in diff_gps.items():
                if not var_id in analytical_dot:
                    raise KeyError(f"Unknown GP key '{var_id}' was not found in analytical_dot.")
                analytical_dot[var_id] = diff_gps[var_id].predict() # TODO. Predict on what? gotta grab those values

                # TODO: look over. Is this structure reasonable?
                deps = self._classify_var_names(gp.input_vars)
                gp_inputs = {name: getattr(state, name) for name in deps["state"]}
                gp_inputs.update({name: getattr(input, name) for name in deps["input"]})

                analytical_dot[var_id] = gp.predict(gp_inputs)


        # ------------------ #
        # -- Result state -- #
        # ------------------ #
        res_state = self.State.from_dict(analytical_dot)

        # TODO: Apply optional perturbations

        return res_state
    
    def step(self, state, input, params):
        pass
    
    # @abstractmethod
    # def create_GPs(self, state, input, params):
    #     # return dict of 
    #     pass
    
    @abstractmethod
    def position(self, state):
        pass
    
    @abstractmethod
    def orientation(self, state):
        pass
    
    # TODO: add ACADOS stuff, like cost, ...


