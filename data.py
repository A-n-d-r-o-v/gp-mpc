import numpy as np
import torch
from dataclasses import asdict, fields
#from enum import IntEnum
from dataclasses import asdict, fields
from typing import ClassVar, TypeVar, Type


class ODEData:
    """
    This is a generic base for dataclasses like State, Input, etc, to inherit from.
    It provides a number of methods to handle things like type conversions,
    mathematical operations, etc.
    """

    T = TypeVar("T", bound="ODEData")
    DIM: ClassVar[int] = 0 # Number of class variables. Computed in post_init

    def __init__(self):
        for f in fields(self):
            setattr(self, f.name, 0.0)

    def __post_init__(self):
        cls = self.__class__
        if cls.DIM == 0:
            cls.DIM = len(fields(cls))    
        
    def as_numpy(self) -> np.ndarray:
        """Convert to numpy."""
        values = [getattr(self, f.name) for f in fields(self) if f.name != "DIM"]
        return np.array(values, dtype=float)

    @classmethod
    def from_numpy(cls, vec: np.ndarray):
        """Convert back to class type from numpy."""
        return cls(*vec.tolist())

    def as_tensor(self, dtype=torch.float64, device=None):
        """Convert to torch tensor."""
        values = [
            getattr(self, f.name)
            for f in fields(self)
            if f.name != "DIM"
        ]
        return torch.tensor(values, dtype=dtype, device=device)
    
    def as_dict(self) -> dict:
        """Convert instance to dictionary."""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def distance_to(self, other) -> float:
        """Euclidean distance to another instance of same type."""
        return np.linalg.norm(self.as_numpy() - other.as_numpy())

    def difference(self, other):
        """Return a new instance representing self - other."""
        diff_values = self.as_numpy() - other.as_numpy()
        return self.__class__.from_numpy(diff_values)

    def difference_dict(self, other) -> dict:
        """Return element-wise differences as a dict."""
        return self.difference(other).as_dict()
    
    @classmethod
    def _validate_field_dict(cls, d: dict, raise_missing_field_error = True) -> bool:
        """
        Check that a dictionary d that maps {"var_name" -> value} contains
        all the corresponding fields defined in the dataclass.
        Optionally avoid raising an error for a missing field. 
        
        :param cls:
        :param d: The dictionary mapping {"state_var_name" -> value}.
        :type d: dict
        :param raise_missing_field_error: Raise an error if a field is missing from the dict.
        :return: Whether the set of keys in the provided dict fully matches the set of dataclass fields.
        :rtype: bool
        """
        fields_set = set(cls.__dataclass_fields__.keys())
        keys_set = set(d.keys())

        missing = fields_set - keys_set
        extra = keys_set - fields_set

        if missing:
            if raise_missing_field_error:
                raise ValueError(f"Missing dict key(s) for: {missing}")
            else: 
                print(f"Warning: Missing dict key(s) for: {missing}")
        if extra:
            raise ValueError(f"Unknown dict key(s): {extra}")
        
        return not missing and not extra

    @classmethod
    def from_dict(cls: type["ODEData"], d: dict[str, float], fill_missing: bool = False) -> "ODEData":
        """
        Convert a dictionary to an instance of this dataclass.
        """
        if not fill_missing:
            cls._validate_state_dict(d, raise_missing_field_error=True)
        else:
            # Fill missing keys with 0
            fields_set = set(cls.__dataclass_fields__.keys())
            for key in fields_set:
                d.setdefault(key, 0.0)

            # Check for extra keys
            extra_keys = set(d.keys()) - fields_set
            if extra_keys:
                raise ValueError(f"Unknown dict key(s): {extra_keys}")

        return cls(**d)


    @classmethod
    def zero(cls: Type[T]) -> T: # Use the type of whatever inherits this class
        """
        Return a zero-init state. 
        NOTE: Sets each cls field to 0.0 which currently is fine as
        we assume everything is float... This needs to be looked over 
        if this assumption changes in the future.
        """
        return cls(**{f.name: 0.0 for f in fields(cls)})
    


# The code below is the old version for reference

"""
@dataclass
class State(ODEData):
    v1: float
    v2: float # 1
    v3: float # 2
    roll: float # 3
    pitch: float
    yaw: float # 5
    p1: float
    p2: float # 7
    p3: float # 8
    roll_ref: float # 9
    pitch_ref: float 
    thrust: float # 11
    roll_ref_dot: float # 12
    pitch_ref_dot: float
    thrust_dot: float
       

State.DIM = 15
# State.DIM = 7

@dataclass
class Input(ODEData):
    roll_ref_ddot: float
    pitch_ref_ddot: float
    thrust_ddot: float
    
Input.DIM = 3
    

@dataclass
class Params(ODEData):
    g: float
    roll_gain: float
    roll_tau: float
    pitch_gain: float
    pitch_tau: float
    dragacc1: float
    dragacc2: float

Params.DIM = 7


class StateIdx(IntEnum):
    V1 = 0
    V2 = 1
    V3 = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    P1 = 6
    P2 = 7
    P3 = 8
    ROLL_REF = 9
    PITCH_REF = 10
    THRUST = 11
    ROLL_REF_DOT = 12
    PITCH_REF_DOT = 13
    THRUST_DOT = 14


class InputIdx(IntEnum):
    ROLL_REF_DDOT = 0
    PITCH_REF_DDOT = 1
    THRUST_DDOT = 2
        :param d: Dictionary mapping {"var_name" -> value}
        :param fill_missing: If True, missing keys are initialized to 0 instead of raising an error
        :return: Instance of cls

"""
