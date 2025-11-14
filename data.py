import numpy as np
import torch
from dataclasses import dataclass, asdict, fields
from enum import IntEnum


class ODEData:

    # DIM: ClassVar[int] = 0 # Number of class variables. Computed in post_init

    def __init__(self):
        for f in fields(self):
            setattr(self, f.name, 0.0)
        # self.DIM = len(fields(self))
        # DIM = self.DIM
        
    def as_numpy(self) -> np.ndarray:
        """Convert to numpy."""
        values = [getattr(self, f.name) for f in fields(self) if f.name != "DIM"]
        return np.array(values, dtype=float)

    @classmethod
    def from_numpy(cls, vec: np.ndarray):
        """Convert back to class type from numpy."""
        return cls(*vec.tolist())

    def as_tensor(self, dtype=torch.float32, device=None):
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
    def zero(cls) -> "State":
        """Return a zero-init state."""
        # return cls()
        return cls.from_numpy(np.zeros(len(fields(cls))))


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
