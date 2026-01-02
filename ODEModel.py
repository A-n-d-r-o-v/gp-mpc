import numpy as np

from typing import List, Type, Optional, TYPE_CHECKING
from dataclasses import dataclass
#from gp import min_max_norm
#from math import pi 

if TYPE_CHECKING: # Type hints only
    from gp import GPSetting
    from data import ODEData

from abc import abstractmethod

#Perturbation = Callable[[State, Input, Params, State], State]


class ODEModel:
    """
    Generic base class for an ODE model that is described both by a set of states, inputs, and parameters,
    and a set of ordinary differential equations that describe how these variables interact.
    This provides functionality for defining GP and ACADOS specifications in a centralized place
    for use in the rest of the pipeline.
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
        raise NotImplementedError("Not callable from superclass; must be overridden")

    @abstractmethod
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
        raise NotImplementedError("Not callable from superclass; must be overridden")
    
    # @abstractmethod
    # def create_GPs(self, state, input, params):
    #     # return dict of 
    #     pass
    
    @abstractmethod
    def position(self, state):
        raise NotImplementedError("Not callable from superclass; must be overridden")
    
    @abstractmethod
    def orientation(self, state):
        raise NotImplementedError("Not callable from superclass; must be overridden")
    
    # TODO: add ACADOS stuff, like cost, ...
    
    #def position(self, state):
    #    raise NotImplementedError("Not callable from superclass; must be overridden")

    '''
    def dynamics_analytical(
            self,
            state, 
            input, 
            params 
            #diff_gps: List[Perturbation] = None # TODO !!
        ) -> dict[str, float]:
        raise NotImplementedError("Not callable from superclass; must be overridden")
    '''


# {State.V1: {"idx": 4, "gp": gp_obj_hash} }


