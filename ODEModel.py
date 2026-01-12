from __future__ import annotations

import numpy as np

from typing import List, Type, Optional, TYPE_CHECKING
from dataclasses import dataclass
#from gp import min_max_norm
#from math import pi 
if TYPE_CHECKING: # Type hints only
    from gp import GPSetting
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
        Specify the set of GPs to use for prediction.
        TODO: Should maybe make use of the output fields instead of the GPSetting names. 
        
        :param self: Description
        :return: A set of strings containing the names of the GPSettings to use.
        :rtype: Any | None
        """
        return None

    def _filter_gp_spec(self, gp_spec: List["GPSetting"], names: {str}) -> List["GPSetting"]:
        """
        Docstring for filter_gp_spec
        TODO: Should maybe make use of the output fields instead of the GPSetting names. 
        
        :param self: Description
        :param gp_spec: The list of GP settings
        :type gp_spec: List["GPSetting"]
        :param names: Description
        :type names: List[str]
        :return: Description
        :rtype: List[GPSetting]
        """
        if not names:
            raise ValueError("Names cannot be an empty set.")
        
        seen_names = set()
        specs = []
        for spec in gp_spec:
            # Check for duplicates
            if spec.name in seen_names:
                raise ValueError(f"Duplicate GPSetting name: {spec.name}. Ensure that all GPSetting names are unique.")
            # Find matching specs
            if spec.name in names:
                specs.append(spec)
            seen_names.add(spec.names)

        if not specs:
            raise ValueError("No matching names found ")
        
        return specs

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
            keys: set[str] | list[str]
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
            diff_gps: Optional[List] = None
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

        # TODO: Maybe all this stuff should be moved to GPManager! Or maybe not..
        # We don't build GPs here...
        # But maybe we can still filter which GPs to predict with.
        gp_spec = self.gp_spec()
        gps_to_model = self._gps_to_model()
        if gps_to_model:
            gp_spec = self._filter_gp_spec(gp_spec, gps_to_model)

        # If a state variable is associated with a trained GP, replace the analytical_dot entry with the GP prediction.
        if diff_gps:
            for var_id, gp in diff_gps.items():
                if not var_id in analytical_dot:
                    raise KeyError(f"Unknown GP key '{var_id}' was not found in analytical_dot.")
                #analytical_dot[var_id] = diff_gps[var_id].predict() # TODO. Predict on what? gotta grab those values
                # --> {"v1_dot": prediction} (single gp)
                # --> {"v2_dot": prediction, "pitch": prediction} (multi gp)

                # Get state, input, param vars that match the GP input vars
                state_in, input_in, param_in = self._classify_var_names(gp.input_vars)
                gp_inputs = {name: getattr(state, name) for name in state_in}
                gp_inputs.update({name: getattr(input, name) for name in input_in})
                gp_inputs.update({name: getattr(params, name) for name in param_in})

                analytical_dot[var_id] = gp.predict(gp_inputs) # or +=

        # ------------------ #
        # -- Result state -- #
        # ------------------ #
        res_state = self.State.from_dict(analytical_dot)

        # TODO: Apply optional perturbations

        return res_state
    
    def step(self, state, input, params = None, dt=1): # TODO: optionally add GPs
        if params is None:
            params = self.params
        step_state = self.differentiate(state, input, params)
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


