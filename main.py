import torch
import gpytorch

from data import *
from itertools import product

from dynamics import step, differentiate_features, step_gp, step_gp_casadi
from gp import MultiGP, min_max_norm, DifferentialGP
from gp_to_casadi import get_xdot

from data_gen_cfg import PARAMS, RANGES, CFG1_PERTURBATIONS, CFG2_PERTURBATIONS
from generate_data import generate_data

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
        # pick input

        # true step
        # print(f"before :{drone_true}")
        drone_true = step(drone_true, u, params, dt)
        # print(f"after:{drone_true}")
        # print()

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
def walk_two_drones_vert(diff_gps, params: Params, dt: float, input_sequence: list[Input]):
    # initial states
    drone_true = State.zero()
    drone_gp = State.zero()

    true_positions = []
    gp_positions = []

    for u in input_sequence:
        # Base line ODE prediction
        drone_true = step(drone_true, u, params, dt)

        # GP step: predict derivative
        drone_gp = step_gp(drone_gp, u, params, dt, diff_gps)

        # record positions
        true_positions.append([drone_true.p1, drone_true.p2, drone_true.p3])
        gp_positions.append([drone_gp.p1, drone_gp.p2, drone_gp.p3])

    return np.array(true_positions), np.array(gp_positions)

# This is a walking drone sim locked to one axis 
def walk_three_drones_vert(diff_gps, diff_gps_casadi, params: Params, dt: float, input_sequence: list[Input]):
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
        
        drone_casadi = step_gp_casadi(drone_casadi, u, params, dt, diff_gps_casadi)

        # record positions
        true_positions.append([drone_true.p1, drone_true.p2, drone_true.p3])
        gp_positions.append([drone_gp.p1, drone_gp.p2, drone_gp.p3])
        casadi_positions.append([drone_casadi.p1, drone_casadi.p2, drone_gp.p3])

    return np.array(true_positions), np.array(gp_positions), np.array(casadi_positions)


if __name__ == "__main__":
    drone = State.zero()
    params = Params(
        g=9.8066,
        roll_gain=1.101, roll_tau=0.253,
        pitch_gain=1.097, pitch_tau=0.267,
        dragacc1=0.01, dragacc2=0.01
    )

    training_iter = 100
    lr = 0.5
    dt = 0.01  # 100hz
    N = 100
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
    
    v1_dot_gp = DifferentialGP(input_dim=4, device=device, periodic_inputs = [0, 0, 0, 0])
    v2_dot_gp = DifferentialGP(input_dim=4, device=device, periodic_inputs = [0, 0, 0, 0])
    v3_dot_gp = DifferentialGP(input_dim=3, device=device, periodic_inputs = [0, 0, 0])

    roll_dot_gp = DifferentialGP(input_dim=2, device=device, periodic_inputs = [0, 0]) # At least I _believe_ that roll_ref is not periodic
    pitch_dot_gp = DifferentialGP(input_dim=2, device=device, periodic_inputs = [0, 0])
    
    v1_dot_xtrain = X_train[:, [3, 4, 5, 11]] # roll pitch yaw thrust
    v2_dot_xtrain = X_train[:, [3, 4, 5, 11]] # roll pitch yaw thrust
    v3_dot_xtrain = X_train[:, [3, 4, 11]] # roll pitch thrust
    roll_dot_xtrain = X_train[:, [3, 9]] # roll roll_ref
    pitch_dot_xtrain = X_train[:, [4, 10]] # pitch pitch_ref
    
    v1_dot_gp.add_data(v1_dot_xtrain, Y_train[:, 0]) 
    v2_dot_gp.add_data(v2_dot_xtrain, Y_train[:, 1]) 
    v3_dot_gp.add_data(v3_dot_xtrain, Y_train[:, 2]) 
    roll_dot_gp.add_data(roll_dot_xtrain, Y_train[:, 3])
    pitch_dot_gp.add_data(pitch_dot_xtrain, Y_train[:, 4]) 

    print("training")
    v1_dot_gp.train(training_iter=training_iter, lr=lr) 
    v2_dot_gp.train(training_iter=training_iter, lr=lr)
    v3_dot_gp.train(training_iter=training_iter, lr=lr)

    roll_dot_gp.train(training_iter=training_iter, lr=lr)
    pitch_dot_gp.train(training_iter=training_iter, lr=lr)
    diff_gps = {"v1_dot": v1_dot_gp, "v2_dot": v2_dot_gp, "v3_dot": v3_dot_gp, "roll_dot": roll_dot_gp, "pitch_dot": pitch_dot_gp}
    
    v1_dot_gp_casadi = get_xdot(v1_dot_gp, v1_dot_xtrain, Y_train[:, 0], control_dim=1)
    v2_dot_gp_casadi = get_xdot(v2_dot_gp, v2_dot_xtrain, Y_train[:, 1], control_dim=1)
    v3_dot_gp_casadi = get_xdot(v3_dot_gp, v3_dot_xtrain, Y_train[:, 2], control_dim=1)
    roll_dot_gp_casadi = get_xdot(roll_dot_gp, roll_dot_xtrain, Y_train[:, 3], control_dim=1)
    pitch_dot_gp_casadi = get_xdot(pitch_dot_gp, pitch_dot_xtrain, Y_train[:, 4], control_dim=1)
    
    diff_gps_casadi = {"v1_dot": v1_dot_gp_casadi, "v2_dot": v2_dot_gp_casadi, 
                       "v3_dot": v3_dot_gp_casadi, "roll_dot": roll_dot_gp_casadi, 
                       "pitch_dot": pitch_dot_gp_casadi}
    
    true_path, gp_path = walk_two_drones_vert(diff_gps, params, dt, [Input(0.1, 0.0, 5)] * 500)
    true_path, gp_path, casadi_path = walk_three_drones_vert(diff_gps, diff_gps_casadi, params, dt, [Input(0.1, 0.0, 5)] * 500)

    for i in range(true_path.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.plot(true_path[:, i], "b-", label="True Dynamics")
        axs.plot(gp_path[:, i], "r-", label="GP Dynamics")
        axs.plot(casadi_path[:, i], "g--", label="CasADi Dynamics")
        fig.legend()
        plt.savefig(f"gpg_dim{i}.png")
    