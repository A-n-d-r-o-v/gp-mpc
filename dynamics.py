#from data import * # oof..
from typing import List, Type, Optional
from dataclasses import dataclass
#from gp import min_max_norm
#from math import pi 
from data import ODEData

import numpy as np
#import torch

from abc import abstractmethod

#Perturbation = Callable[[State, Input, Params, State], State]


class ODEModel:
    """
    Generic base class for an ODE model that is described by ..
    """

    # An ODE model needs the following classes that all inherit from ODEData.
    State: Type[ODEData] = None
    Input: Type[ODEData] = None
    Params: Type[ODEData] = None

    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def differentiate(
            self,
            state, 
            input, 
            params,
            diff_gps: Optional[List] = None
            #perturbations: List[Perturbation] = None # TODO !!
        ) -> ODEData:
        raise NotImplementedError("Not callable from superclass; must be overridden")
    
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


class DJIMatrice100(ODEModel):
    
    @dataclass 
    class MatriceState(ODEData):
        v1: float
        v2: float
        v3: float
        roll: float
        pitch: float
        yaw: float
        p1: float
        p2: float
        p3: float
        roll_ref: float
        pitch_ref: float 
        thrust: float 
        roll_ref_dot: float
        pitch_ref_dot: float
        thrust_dot: float

    @dataclass 
    class MatriceInput(ODEData):
        roll_ref_ddot: float
        pitch_ref_ddot: float
        thrust_ddot: float
        
    @dataclass(frozen=True)
    class MatriceParams(ODEData):
        g: float
        roll_gain: float
        roll_tau: float
        pitch_gain: float
        pitch_tau: float
        dragacc1: float
        dragacc2: float

    State = MatriceState
    Input = MatriceInput
    Params = MatriceParams

    def differentiate(self, state, input, params, diff_gps = None):
        #return super().differentiate(state, input, params, diff_gps)
       
        # ----------- #
        # -- State -- #
        # ----------- #
        # Velocity
        v1, v2, v3 = state.v1, state.v2, state.v3 

        # Orientation
        roll, pitch, yaw = state.roll, state.pitch, state.yaw 

        # Position
        p1, p2, p3 = state.p1, state.p2, state.p3 

        # Ref
        roll_ref, pitch_ref = state.roll_ref, state.pitch_ref

        # Ref dot
        roll_ref_dot, pitch_ref_dot = state.roll_ref_dot, state.pitch_ref_dot
        
        # Thrust
        thrust, thrust_dot = state.thrust, state.thrust_dot 

        # ----------- #
        # -- Input -- #
        # ----------- #
        roll_ref_ddot = input.roll_ref_ddot
        pitch_ref_ddot = input.pitch_ref_ddot
        thrust_ddot = input.thrust_ddot

        # ------------ #
        # -- Params -- #
        # ------------ #
        g = params.g
        roll_gain = params.roll_gain
        roll_tau = params.roll_tau
        pitch_gain = params.pitch_gain
        pitch_tau = params.pitch_tau
        dragacc1 = params.dragacc1
        dragacc2 = params.dragacc2

        # -------------- #
        # -- Dynamics -- #
        # -------------- #
        analytical_dot = {
            "v1": (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1 * v1, # * v1 ? 
            "v2": (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2 * v2, # * v2 ?
            "v3": -g + np.cos(pitch) * np.cos(roll) * thrust,
            "roll": (roll_gain * roll_ref - roll) / roll_tau,
            "pitch": (pitch_gain * pitch_ref - pitch) / pitch_tau,
            "yaw": 0.0,
            "p1": v1, "p2": v2, "p3": v3,

            # Reference dynamics
            "roll_ref": roll_ref_dot,
            "pitch_ref": pitch_ref,
            "thrust": thrust_dot,

            # Second derivatives = input
            "roll_ref_dot": roll_ref_ddot,
            "pitch_ref_dot": pitch_ref_ddot,
            "thrust_dot": thrust_ddot
        }

        self.State._validate_field_dict(analytical_dot) # check that everything is there...

        # If a state variable is associated with a trained GP, replace the analytical_dot entry with the GP prediction.
        if diff_gps:
            for var_id in diff_gps:
                if not var_id in analytical_dot:
                    raise KeyError(f"Unknown GP key '{var_id}' was not found in analytical_dot.")
                analytical_dot[var_id] = diff_gps[var_id].predict() # TODO. Predict on what? gotta grab those values

        # ------------------ #
        # -- Result state -- #
        # ------------------ #
        res_state = self.State.from_dict(analytical_dot)

        # TODO: Apply optional perturbations

        return res_state

        

    #def __init__(self, name: str = None):
    #    super().__init__(name=name)



'''

def differentiate(state: State, 
                  input: Input, 
                  params: Params,
                  perturbations: List[Perturbation] = None) -> State:
    """
    Compute time-derivative of the state vector.
    """
    # unpack state
    v1, v2, v3 = state.v1, state.v2, state.v3
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    p1, p2, p3 = state.p1, state.p2, state.p3
    roll_ref, pitch_ref, thrust = state.roll_ref, state.pitch_ref, state.thrust
    roll_ref_dot, pitch_ref_dot, thrust_dot = (
        state.roll_ref_dot, state.pitch_ref_dot, state.thrust_dot
    )

    # unpack input
    roll_ref_ddot = input.roll_ref_ddot
    pitch_ref_ddot = input.pitch_ref_ddot
    thrust_ddot = input.thrust_ddot

    # unpack params
    g = params.g
    roll_gain = params.roll_gain
    roll_tau = params.roll_tau
    pitch_gain = params.pitch_gain
    pitch_tau = params.pitch_tau
    dragacc1 = params.dragacc1
    dragacc2 = params.dragacc2

    # translational dynamics
    v1_dot = (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1 * v1
    v2_dot = (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2 * v2
    v3_dot = -g + np.cos(pitch) * np.cos(roll) * thrust
    #print(f"v3_act: {v3_dot}, pitch: {pitch}, roll: {roll},  thrust: {thrust}")

    # rotational dynamics
    roll_dot = (roll_gain * roll_ref - roll) / roll_tau
    pitch_dot = (pitch_gain * pitch_ref - pitch) / pitch_tau
    # print(f"roll_dot_roll: {roll_dot}, roll: {roll},  roll_ref: {roll_ref}")
    yaw_dot = 0.0

    # position dynamics
    p1_dot, p2_dot, p3_dot = v1, v2, v3

    # reference dynamics
    roll_ref_dot_val = roll_ref_dot
    pitch_ref_dot_val = pitch_ref_dot
    thrust_dot_val = thrust_dot

    # second derivatives = input
    roll_ref_dot_dot = roll_ref_ddot
    pitch_ref_dot_dot = pitch_ref_ddot
    thrust_dot_dot = thrust_ddot

    res_state = State(
        v1_dot, v2_dot, v3_dot,
        roll_dot, pitch_dot, yaw_dot,
        p1_dot, p2_dot, p3_dot,
        roll_ref_dot_val, pitch_ref_dot_val, thrust_dot_val,
        roll_ref_dot_dot, pitch_ref_dot_dot, thrust_dot_dot
    )

    if perturbations:
        for p_fn in perturbations:
            residual = p_fn(state, input, params, res_state)
            res_state = State.from_numpy(res_state.as_numpy() + residual.as_numpy())
    
    return res_state

def differentiate_features(
        X: np.ndarray, params: Params) -> np.ndarray:
    """
    Compute time-derivative given feature vectors.
    NOTE: Has not been touched for a while.
    """
    states = X[:, :State.DIM]
    input = X[:, State.DIM:]

    # unpack states
    v1, v2, v3 = states[:, 0], states[:, 1], states[:, 2]
    roll, pitch, yaw = states[:, 3], states[:, 4], states[:, 5]
    p1, p2, p3 = states[:, 6], states[:, 7], states[:, 8]
    roll_ref, pitch_ref, thrust = states[:, 9], states[:, 10], states[:, 11]
    roll_ref_dot, pitch_ref_dot, thrust_dot = states[:, 12], states[:, 13], states[:, 14]

    # unpack input
    roll_ref_ddot, pitch_ref_ddot, thrust_ddot = input[:, 0], input[:, 1], input[:, 2]

    # unpack params
    g = params.g
    roll_gain, roll_tau = params.roll_gain, params.roll_tau
    pitch_gain, pitch_tau = params.pitch_gain, params.pitch_tau
    dragacc1, dragacc2 = params.dragacc1, params.dragacc2

    # translational dynamics
    v1_dot = (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1 * v1
    v2_dot = (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2 * v2
    v3_dot = -g + np.cos(pitch) * np.cos(roll) * thrust

    # rotational dynamics
    roll_dot = (roll_gain * roll_ref - roll) / roll_tau
    pitch_dot = (pitch_gain * pitch_ref - pitch) / pitch_tau
    yaw_dot = np.zeros_like(yaw)

    # position dynamics
    p1_dot, p2_dot, p3_dot = v1, v2, v3

    # reference dynamics
    roll_ref_dot_val, pitch_ref_dot_val, thrust_dot_val = roll_ref_dot, pitch_ref_dot, thrust_dot

    # second derivatives
    roll_ref_dot_dot, pitch_ref_dot_dot, thrust_dot_dot = roll_ref_ddot, pitch_ref_ddot, thrust_ddot

    # stack back
    state_dot = np.vstack([
        v1_dot, v2_dot, v3_dot,
        roll_dot, pitch_dot, yaw_dot,
        p1_dot, p2_dot, p3_dot,
        roll_ref_dot_val, pitch_ref_dot_val, thrust_dot_val,
        roll_ref_dot_dot, pitch_ref_dot_dot, thrust_dot_dot
    ]).T

    return state_dot


def differentiate_gp(state: State, 
                  input: Input, 
                  params: Params,
                  gps: Dict[StateIdx, DifferentialGP]
                ) -> State:
    """
    Compute time-derivative of the state vector.
    """
    # unpack state
    v1, v2, v3 = state.v1, state.v2, state.v3
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    p1, p2, p3 = state.p1, state.p2, state.p3
    roll_ref, pitch_ref, thrust = state.roll_ref, state.pitch_ref, state.thrust
    roll_ref_dot, pitch_ref_dot, thrust_dot = (
        state.roll_ref_dot, state.pitch_ref_dot, state.thrust_dot
    )

    # unpack input
    roll_ref_ddot = input.roll_ref_ddot
    pitch_ref_ddot = input.pitch_ref_ddot
    thrust_ddot = input.thrust_ddot

    # unpack params
    g = params.g
    roll_gain = params.roll_gain
    roll_tau = params.roll_tau
    pitch_gain = params.pitch_gain
    pitch_tau = params.pitch_tau
    dragacc1 = params.dragacc1
    dragacc2 = params.dragacc2

    # translational dynamics
    #v1_dot = (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1 * v1 # GP:able
    #v2_dot = (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2 * v2 # GP:able
    # v3_dot = -g + np.cos(pitch) * np.cos(roll) * thrust
    v1_dot = gps[StateIdx.V1].predict([[roll, pitch, yaw, thrust]])[0][0, 0]
    v2_dot = gps[StateIdx.V2].predict([[roll, pitch, yaw, thrust]])[0][0, 0]
    v3_dot = gps[StateIdx.V3].predict([[roll, pitch, thrust]])[0][0, 0]

    # rotational dynamics
    # roll_dot = (roll_gain * roll_ref - roll) / roll_tau # GP:able
    # pitch_dot = (pitch_gain * pitch_ref - pitch) / pitch_tau #GP:able
    roll_dot = gps[StateIdx.ROLL].predict([[roll, roll_ref]])[0][0, 0]
    pitch_dot = gps[StateIdx.PITCH].predict([[pitch, pitch_ref]])[0][0, 0]

    yaw_dot = 0.0

    # position dynamics
    p1_dot, p2_dot, p3_dot = v1, v2, v3

    # reference dynamics
    roll_ref_dot_val = roll_ref_dot
    pitch_ref_dot_val = pitch_ref_dot
    thrust_dot_val = thrust_dot

    # second derivatives = input
    roll_ref_dot_dot = roll_ref_ddot
    pitch_ref_dot_dot = pitch_ref_ddot
    thrust_dot_dot = thrust_ddot

    res_state = State(
        v1_dot, v2_dot, v3_dot,
        roll_dot, pitch_dot, yaw_dot,
        p1_dot, p2_dot, p3_dot,
        roll_ref_dot_val, pitch_ref_dot_val, thrust_dot_val,
        roll_ref_dot_dot, pitch_ref_dot_dot, thrust_dot_dot
    )

    return res_state


def differentiate_gp_casadi(state: State, 
                  input: Input, 
                  params: Params,
                  gps: Dict[StateIdx, Any]
                ) -> State:
    """
    Compute time-derivative of the state vector.
    """
    # unpack state
    v1, v2, v3 = state.v1, state.v2, state.v3
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    p1, p2, p3 = state.p1, state.p2, state.p3
    roll_ref, pitch_ref, thrust = state.roll_ref, state.pitch_ref, state.thrust
    roll_ref_dot, pitch_ref_dot, thrust_dot = (
        state.roll_ref_dot, state.pitch_ref_dot, state.thrust_dot
    )

    # unpack input
    roll_ref_ddot = input.roll_ref_ddot
    pitch_ref_ddot = input.pitch_ref_ddot
    thrust_ddot = input.thrust_ddot

    # unpack params
    g = params.g
    roll_gain = params.roll_gain
    roll_tau = params.roll_tau
    pitch_gain = params.pitch_gain
    pitch_tau = params.pitch_tau
    dragacc1 = params.dragacc1
    dragacc2 = params.dragacc2

    # translational dynamics
    #v1_dot = (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1 * v1 # GP:able
    #v2_dot = (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2 * v1 # GP:able
    # v3_dot = -g + np.cos(pitch) * np.cos(roll) * thrust
    v1_dot = float(gps[StateIdx.V1]([roll, pitch, yaw], [thrust]))
    v2_dot = float(gps[StateIdx.V2]([roll, pitch, yaw], [thrust]))
    v3_dot = float(gps[StateIdx.V3]([roll, pitch], [thrust]))

    roll_dot = float(gps[StateIdx.ROLL]([roll], [roll_ref]))
    pitch_dot = float(gps[StateIdx.PITCH]([pitch], [pitch_ref]))

    yaw_dot = 0.0

    # position dynamics
    p1_dot, p2_dot, p3_dot = v1, v2, v3

    # reference dynamics
    roll_ref_dot_val = roll_ref_dot
    pitch_ref_dot_val = pitch_ref_dot
    thrust_dot_val = thrust_dot

    # second derivatives = input
    roll_ref_dot_dot = roll_ref_ddot
    pitch_ref_dot_dot = pitch_ref_ddot
    thrust_dot_dot = thrust_ddot

    res_state = State(
        v1_dot, v2_dot, v3_dot,
        roll_dot, pitch_dot, yaw_dot,
        p1_dot, p2_dot, p3_dot,
        roll_ref_dot_val, pitch_ref_dot_val, thrust_dot_val,
        roll_ref_dot_dot, pitch_ref_dot_dot, thrust_dot_dot
    )

    
    return res_state


def step(state: State, input: Input, params: Params, dt: float, perturbations=None) -> State:
    """
    Euler integration: x_{t+dt} = x_t + f(x_t)*dt
    """
    state_dot = differentiate(state, input, params, perturbations)
    return State.from_numpy(state.as_numpy() + state_dot.as_numpy() * dt)

def step_gp(state: State, input: Input, params: Params, dt: float, gps) -> State:
    """
    Euler integration: x_{t+dt} = x_t + f(x_t)*dt
    But now we more GP:s!
    """
    state_dot = differentiate_gp(state, input, params, gps)
    return State.from_numpy(state.as_numpy() + state_dot.as_numpy() * dt)


def step_gp_casadi(state: State, input: Input, params: Params, dt: float, gps) -> State:
    """
    Euler integration: x_{t+dt} = x_t + f(x_t)*dt
    But now we more GP:s!
    """
    state_dot = differentiate_gp_casadi(state, input, params, gps)
    return State.from_numpy(state.as_numpy() + state_dot.as_numpy() * dt)


def step_features(X: np.ndarray, params: Params, dt: float) -> np.ndarray:
    """
    '' but for many states and input together in a feature vector.
    NOTE: Has not been touched for a while.
    """
    states = X[:, :State.DIM]
    state_dots = differentiate_features(X, params)
    return states + state_dots * dt

    
    '''