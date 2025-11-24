import torch
import gpytorch
import casadi as cs

from data import *
from itertools import product

from dynamics import step, step_gp, step_gp_casadi
from gp import MultiGP, create_gp_settings, create_gps

from data_gen_cfg import PARAMS, RANGES, CFG1_PERTURBATIONS, CFG2_PERTURBATIONS
from generate_data import generate_data
from gp_to_casadi import get_function_estimator, concatenate_inputs, split_inputs

import matplotlib
matplotlib.use("Agg") # Had to do this because of using WSL
import matplotlib.pyplot as plt


def make_feature(state, inputs) -> np.ndarray:
    """
    Concatenate State and Input into one feature vector.
    """
    return np.hstack([state.as_numpy(), inputs.as_numpy()])


def generate_discrete_inputs(roll_vals, pitch_vals, thrust_vals):
    """
    Cartesian product of 3 lists of candidate values.
    """
    return [Input(r, p, t) for r, p, t in product(roll_vals, pitch_vals, thrust_vals)]


def most_uncertain_input(state: State, candidates: list[Input], gp: MultiGP):
    """
    Return the candidate input with the highest GP predictive uncertainty.
    """
    X = np.vstack([make_feature(state, u) for u in candidates])
    mu, var = gp.predict(X)
    total_uncertainty = var.sum(axis=1)
    idx = int(np.argmax(total_uncertainty))
    return candidates[idx]


def walk_two_drones(gp: MultiGP, params: Params, dt: float, input_sequence: list[Input]):
    # initial states
    drone_true = State.zero()
    drone_gp = State.zero()

    true_positions = []
    gp_positions = []

    for u in input_sequence:
        # true step
        drone_true = step(drone_true, u, params, dt)

        # GP step: predict derivative (state_dot_hat)
        x_feat = make_feature(drone_gp, u).reshape(1, -1)
        mu, _ = gp.predict(x_feat)  # (1, state_dim)
        state_dot_hat = mu[0]
        next_state_gp = State.from_numpy(drone_gp.as_numpy() + state_dot_hat * dt)
        drone_gp = next_state_gp

        # record positions
        true_positions.append([drone_true.p1, drone_true.p2, drone_true.p3])
        gp_positions.append([drone_gp.p1, drone_gp.p2, drone_gp.p3])

    return np.array(true_positions), np.array(gp_positions)


# This is a walking drone sim locked to one axis 
def walk_drones_vert(diff_gps, params: Params, dt: float, input_sequence: list[Input], casadi_gps=None):
    # initial states
    drone_true = State.zero()
    drone_gp = State.zero()
    drone_casadi = State.zero()

    true_positions = []
    gp_positions = []
    casadi_positions = []

    for u in input_sequence:
        # Base line ODE prediction
        drone_true = step(drone_true, u, params, dt)

        # GP step: predict derivative
        drone_gp = step_gp(drone_gp, u, params, dt, diff_gps)

        # record positions
        true_positions.append([drone_true.p1, drone_true.p2, drone_true.p3])
        gp_positions.append([drone_gp.p1, drone_gp.p2, drone_gp.p3])
        
        if casadi_gps:
            drone_casadi = step_gp_casadi(drone_casadi, u, params, dt, casadi_gps)
            casadi_positions.append([drone_casadi.p1, drone_casadi.p2, drone_gp.p3])

    if casadi_gps:
        return np.array(true_positions), np.array(gp_positions), np.array(casadi_positions)

    return np.array(true_positions), np.array(gp_positions)


def construct_xdot():
    params = Params(
        g=9.8066,
        roll_gain=1.101, roll_tau=0.253,
        pitch_gain=1.097, pitch_tau=0.267,
        dragacc1=0.01, dragacc2=0.01
    )
    training_iter = 300  #100
    lr = 0.01            #0.5
    N = 1000              #100
    X, Y = generate_data(N, RANGES, lhs_random_state=1234)
    
    gpytorch.settings.eval_cg_tolerance._set_value(0)
    gpytorch.settings.max_cg_iterations._set_value(100000)
    
    test_size = 0.2
    test_count = int(N * test_size)
    train_idx = np.arange(N)[test_count:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("training on device ", device)

    # gps_to_create = [StateIdx.V1, StateIdx.V2, StateIdx.V3, StateIdx.ROLL, StateIdx.PITCH]
    #gps_to_create = [StateIdx.PITCH, StateIdx.ROLL]
    gps_to_create = [StateIdx.V3]
    
    gp_settings = create_gp_settings(gps_to_create)
    print("training")
    gps, casadi_gps = create_gps(gp_settings, device, X_train, Y_train, training_iter, lr)
    print("done training")

    state = cs.MX.sym("state", State.DIM)
    input_vec = cs.MX.sym("input", Input.DIM)
    state_dot_list = [None] * 15
    
    roll = state[StateIdx.ROLL]
    pitch = state[StateIdx.PITCH]
    yaw = state[StateIdx.YAW]
    known_dynamics = {
        StateIdx.V1: (cs.cos(roll) * np.cos(yaw) * cs.sin(pitch) + cs.sin(roll) * cs.sin(yaw)) * state[StateIdx.THRUST] - state[StateIdx.V1] * params.dragacc1,
        StateIdx.V2: (cs.cos(roll) * np.sin(pitch) * cs.sin(yaw) - cs.cos(yaw) * cs.sin(roll)) * state[StateIdx.THRUST] - state[StateIdx.V2] * params.dragacc2,
        StateIdx.V3: -params.g + cs.cos(pitch) * cs.cos(roll) * state[StateIdx.THRUST],
        StateIdx.ROLL: (params.roll_gain * state[StateIdx.ROLL_REF] - roll) / params.roll_tau,
        StateIdx.PITCH: (params.pitch_gain * state[StateIdx.PITCH_REF] - pitch) / params.pitch_tau,
        StateIdx.YAW: cs.MX.zeros(1),
        StateIdx.P1: state[StateIdx.V1],
        StateIdx.P2: state[StateIdx.V2],
        StateIdx.P3: state[StateIdx.V3],
        StateIdx.ROLL_REF: state[StateIdx.ROLL_REF_DOT],
        StateIdx.PITCH_REF: state[StateIdx.PITCH_REF_DOT],
        StateIdx.THRUST: state[StateIdx.THRUST_DOT],
        StateIdx.ROLL_REF_DOT: input_vec[InputIdx.ROLL_REF_DDOT],
        StateIdx.PITCH_REF_DOT: input_vec[InputIdx.PITCH_REF_DDOT],
        StateIdx.THRUST_DOT: input_vec[InputIdx.THRUST_DDOT],
    }

    for idx, gp in casadi_gps.items():
        states = [state[i] for i in gp_settings[idx].state_indices]
        controls = [state[i] for i in gp_settings[idx].control_indices]
        # gp is a f(x, u) function. We need a f(X) to work with (simpler)
        gp_func = gp
        num_variables = len(states) + len(controls)
        """
        gp_concat = concatenate_inputs(gp_func, len(states), len(controls))
        gp_est = get_function_estimator(gp_concat, [(-4, 15) for i in range(num_variables)])
        gp_split = split_inputs(gp_est, len(states), len(controls))
        state_dot_list[idx] = gp_split(cs.vertcat(*states), cs.vertcat(*controls))
        """
        state_dot_list[idx] = gp_func(cs.vertcat(*states), cs.vertcat(*controls))

    for idx in set(StateIdx) - casadi_gps.keys():
        state_dot_list[idx] = known_dynamics[idx]
    
    state_dot = cs.vertcat(*state_dot_list)
    
    #return cs.Function("x_dot", [state, input_vec], [state_dot], ["x", "u"], ["x_dot"])
    casFunc = cs.Function("x_dot", [state, input_vec], [state_dot], ["x", "u"], ["x_dot"])

    return casFunc

def test_gps():
    params = Params(
        g=9.8066,
        roll_gain=1.101, roll_tau=0.253,
        pitch_gain=1.097, pitch_tau=0.267,
        dragacc1=0.01, dragacc2=0.01
    )

    training_iter = 100
    lr = 0.5
    dt = 0.01  # 100hz
    N = 1000
    X, Y = generate_data(N, RANGES)
    
    gpytorch.settings.eval_cg_tolerance._set_value(0)
    gpytorch.settings.max_cg_iterations._set_value(100000)
    
    test_size = 0.2
    test_count = int(N * test_size)
    test_idx = np.arange(N)[:test_count]
    train_idx = np.arange(N)[test_count:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gps_to_create = [StateIdx.V1, StateIdx.V2, StateIdx.V3, StateIdx.ROLL, StateIdx.PITCH]
    gp_settings = create_gp_settings(gps_to_create)
    print("training")
    gps, casadi_gps = create_gps(gp_settings, device, X_train, Y_train, training_iter, lr)
    
    print("simulating")
    true_path, gp_path, casadi_path = walk_drones_vert(gps, params, dt, [Input(0.1, 0.0, 5)] * 500, casadi_gps)

    for i in range(true_path.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.plot(true_path[:, i], "b-", label="True Dynamics")
        axs.plot(gp_path[:, i], "r-", label="GP Dynamics")
        axs.plot(casadi_path[:, i], "g--", label="CasADi Dynamics")
        fig.legend()
        plt.savefig(f"gpg_dim{i}.png")
    
    for i, gp in gps.items():
        casadi_pred = np.array([casadi_gps[i](x[gp_settings[i].state_indices], x[gp_settings[i].control_indices]).full().flatten()[0] for x in X_test])
        mu_pred, _ = gp.predict(gp_settings[i].get_input(X_test))
        plt.figure()
        plt.plot(mu_pred, "o-", label="GP mean")
        plt.plot(casadi_pred, "x-", markersize=7, label="CasADi mean")
        plt.title(f"Output dimension {i}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gp_mean_comparison_dim{i}.png")


if __name__ == "__main__":
    test_gps()
    