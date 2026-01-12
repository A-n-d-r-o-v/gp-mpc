
import numpy as np

from dataclasses import dataclass
from data import ODEData
from ODEModel import ODEModel
from typing import List
import data_gen_cfg

from gp import GPSetting
from gp_data import DataGenerator, SamplingStrategy


class DJIMatrice100(ODEModel):

    # ===============================
    # State, Input, Param setup
    # ===============================
    
    @dataclass(frozen=True)
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

    @dataclass(frozen=True)
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

    def __init__(self,
                 params: MatriceParams | None = None,
                 name = None
        ):
        #self.input = input

        # Get from CFG unless specified in constructor 
        if params is None:
            print("Getting params from cfg")
            params = self.MatriceParams.from_dict(data_gen_cfg.PARAMS)
        
        self.params = params
        print(self.params, "\n----------")

        super().__init__(name=name, params=params)

    # ===============================

    def gp_spec_v2(self) -> List["GPSetting"]:
        """
        GP spec based on distance correlation matrix data.
        
        :param self: Description
        :return: Description
        :rtype: List[GPSetting]
        """
        return [
            GPSetting(
                output_var_dots = {"v1", "roll"},
                input_vars  = {"roll", "roll_ref", "pitch", "yaw", "thrust", "v1"}, 
                periodic_vars = None,
                name = "v1_roll",
            ),
            GPSetting(
                output_var_dots = {"v2", "pitch"}, 
                input_vars  = {"roll", "pitch", "pitch_ref", "yaw", "thrust", "v2"}, 
                periodic_vars = None,
                name = "v2_pitch"
            ),
            GPSetting(
                output_var_dots = {"v3"}, 
                input_vars  = {"roll", "pitch", "thrust"}, 
                periodic_vars = None,
                name = "v3",
            )
        ]
    
    def gp_spec(self) -> List["GPSetting"]:
        return [
            GPSetting(
                output_var_dots = {"v1"}, 
                input_vars  = {"roll", "pitch", "yaw", "thrust", "v1"}, 
                periodic_vars = None,
                name = "v1"
                # DataGenerator.. (automatically merge with default)
            ),
            GPSetting(
                output_var_dots = {"v2"}, 
                input_vars  = {"roll", "pitch", "yaw", "thrust", "v2"}, 
                periodic_vars = None,
                name = "v2"
            ),
            GPSetting(
                output_var_dots = {"v3"}, 
                input_vars  = {"roll", "pitch", "thrust"}, 
                periodic_vars = None,
                name = "v3"
            ),
            GPSetting(
                output_var_dots = {"roll"}, 
                input_vars  = {"roll", "roll_ref"}, 
                periodic_vars = None,
                name = "roll"
            ),
            GPSetting(
                output_var_dots = {"pitch"}, 
                input_vars  = {"pitch", "pitch_ref"}, 
                periodic_vars = None,
                name = "pitch"
            )
        ]
    
    def _gps_to_model(self):
        pass #return {"v1", "v2", "v3"}

    def _LHS_data_ranges(self, params: "Params"):
        return {
            # VELOCITY
            "v1":               (-5, 5),
            "v2":               (-5, 5),
            "v3":               (-5, 5),

            # ORIENTATION
            "roll":             (-3.1415, 3.1415),
            "pitch":             (-3.1415, 3.1415),
            "yaw":               (0, 0), # unused in the model 
            #"pitch":            (0, 0),

            # POSITION, the position of the drone has no effect on the derivatives
            "p1":               (0, 0),
            "p2":               (0, 0),
            "p3":               (0, 0),

            # TARGET/REFERENCE STATES? 
            "roll_ref":         (-10, 10),
            "pitch_ref":        (-10, 10),
            "thrust":           (-5, 150),
            # "thrust":           (2 ,2),

            # FIRST DERIVATIVES 
            "roll_ref_dot":     (-2, 2),
            "pitch_ref_dot":    (-2, 2),
            "thrust_dot":       (-2, 2),

            # INPUT (SECOND DERIVATIVES)
            "roll_ref_ddot":    (0, 0),
            "pitch_ref_ddot":   (0, 0),
            "thrust_ddot":      (-3*params.g, 3*params.g),
        }
    
    def default_data_generator(self):
        return DataGenerator(
            sampling_strategy=SamplingStrategy.LHS, 
            n_samples=500
        )

    def _analytical_dot(self, state: "State", input: "Input", params: "Params"):
        # ----------- #
        # -- State -- #
        # ----------- #
        # Position, velocity, orientation
        p1, p2, p3 = state.p1, state.p2, state.p3 
        v1, v2, v3 = state.v1, state.v2, state.v3 
        roll, pitch, yaw = state.roll, state.pitch, state.yaw 

        # Ref, ref dot
        roll_ref, pitch_ref = state.roll_ref, state.pitch_ref
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
        return {
            "v1": (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust - dragacc1, # * v1 ? 
            "v2": (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust - dragacc2, # * v2 ?
            "v3": -g + np.cos(pitch) * np.cos(roll) * thrust,
            "roll": (roll_gain * roll_ref - roll) / roll_tau,
            "pitch": (pitch_gain * pitch_ref - pitch) / pitch_tau,

            "yaw": 0.0,
            "p1": v1, "p2": v2, "p3": v3,

            # Reference dynamics
            "roll_ref": roll_ref_dot,
            "pitch_ref": pitch_ref_dot,
            "thrust": thrust_dot,

            # Second derivatives = input
            "roll_ref_dot": roll_ref_ddot,
            "pitch_ref_dot": pitch_ref_ddot,
            "thrust_dot": thrust_ddot
        }


'''

def differentiate(state: State, 
                  input: Input, 
                  params: Params,
                  perturbations: List[Perturbation] = None) -> State:
    """
    Compute time-derivative of the state vector.
    """
    
    # [unpack...]

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