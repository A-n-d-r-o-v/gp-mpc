MAIN.PY


import torch

torch.set_default_dtype(torch.float64)

from data import *
#from itertools import product
from typing import Iterable

from dynamics import step, step_gp
from gp import MultiGP, DifferentialGP

import gpytorch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import numpy as np

from data_gen_cfg import PARAMS, RANGES, CFG1_PERTURBATIONS, CFG2_PERTURBATIONS
from generate_data import generate_data



def make_feature(state, inputs) -> np.ndarray:
    """
    Concatenate State and Input into one feature vector.
    """
    return np.hstack([state.as_numpy(), inputs.as_numpy()])


'''
def generate_discrete_inputs(roll_vals, pitch_vals, thrust_vals):
    """
    Cartesian product of 3 lists of candidate values.
    """
    return [Input(r, p, t) for r, p, t in product(roll_vals, pitch_vals, thrust_vals)]
'''

'''
def most_uncertain_input(state: State, candidates: list[Input], gp: MultiGP):
    """
    Return the candidate input with the highest GP predictive uncertainty.
    """
    X = np.vstack([make_feature(state, u) for u in candidates])
    mu, var = gp.predict(X)
    total_uncertainty = var.sum(axis=1)
    idx = int(np.argmax(total_uncertainty))
    return candidates[idx]
'''

""" 
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
 """
""" # This is a walking drone sim locked to one axis 
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

 """
def eval_GP(GP):
    assert GP.model is not None, "GP.model is None"
    assert GP.likelihood is not None, "GP.likelihood is None"

    X_train, Y_train = GP.X[0], GP.Y[0]
    X_test, Y_test = GP.X_test[0], GP.Y_test[0]

    device = next(GP.model.parameters()).device
    def to_tensor(x, device):
        """
        Convert input to a float64 tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.float().to(device)

        if isinstance(x, np.ndarray):
            # Need a numeric dtype
            if x.dtype.kind not in {'f', 'i'}:
                raise ValueError(f"Unsupported numpy dtype: {x.dtype}")
            return torch.from_numpy(x).float().to(device)

        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.float64, device=device)

        if isinstance(x, (int, float)):
            return torch.tensor([x], dtype=torch.float64, device=device)

        raise TypeError(f"Unsupported type for to_tensor: {type(x)}")
    
    X_train = to_tensor(X_train, device)
    Y_train = to_tensor(Y_train, device)
    X_test = to_tensor(X_test, device)
    Y_test = to_tensor(Y_test, device)

    '''
    print("\nEvaluating GP with the following:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape, "\n")
    '''

    # with torch.no_grad():
    #     K = roll_dot_gp.model.covar_module(X_train).evaluate()  # full covariance
    #     print("Cov matrix shape:", K.shape)
    #     print("Eigenvalues:", torch.linalg.eigvalsh(K))

    #raise NotImplementedError()
    rmse = eval_RMSE(GP, X_test=X_test, Y_test=Y_test)
    mae = eval_MAE(GP, X_test=X_test, Y_test=Y_test)
    nlpd = eval_NLPD(GP, X_test=X_test, Y_test=Y_test)
    msll = eval_MSLL(GP, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("NLPD: ", nlpd)
    print("MSLL: ", msll)

    return {"RMSE": rmse, "MAE": mae, "NLPD": nlpd, "MSLL": msll}

def eval_GP_learning_curve(gp_data, training_sizes: Iterable[int], lr=0.5, normalize=True):
    eval_data = {}

    for N in training_sizes:
        print(f"N: {N}")
        X, Y = generate_data(N, RANGES)

        if normalize:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X = x_scaler.fit_transform(X)
            Y = y_scaler.fit_transform(Y)

        test_size = 0.2
        test_count = int(N * test_size)
        test_idx = np.arange(N)[:test_count]
        train_idx = np.arange(N)[test_count:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        eval_data[N] = {}

        # Train each GP
        for name, info in gp_data.items():
            print(f"GP: {name}")
            info["gp"].clear_data()
            info["gp"].add_data(X_train[:, info["x_idxs"]], Y_train[:, info["y_idx"]], normalize=normalize)
            info["gp"].add_data(X_test[:, info["x_idxs"]], Y_test[:, info["y_idx"]], normalize=normalize, test=True)
            info["gp"].train(training_iter=training_iter, lr=lr)

            info["gp"].print_hyperparameters()

            gp_eval = eval_GP(info["gp"])
            eval_data[N][name] = gp_eval

     # ---------- Plotting ----------
    metrics_errors = ["RMSE", "MAE"]
    metrics_likelihood = ["NLPD", "MSLL"]

    # Assign specific colors for each metric
    colors_error = {"RMSE": "tab:blue", "MAE": "deepskyblue"}
    colors_likelihood = {"NLPD": "orange", "MSLL": "red"}

    training_sizes_ls = sorted(eval_data.keys())

    for name in gp_data.keys():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left subplot: RMSE / MAE
        for metric in metrics_errors:
            values = [eval_data[N][name][metric] for N in training_sizes_ls]
            axes[0].plot(training_sizes_ls, values, marker='o', label=metric, color=colors_error[metric])
        axes[0].set_xlabel("Number of training samples (N)")
        axes[0].set_ylabel("RMSE / MAE")
        axes[0].set_title("Error metrics")
        axes[0].grid(True)
        axes[0].legend()

        # Right subplot: NLPD / MSLL
        for metric in metrics_likelihood:
            values = [eval_data[N][name][metric] for N in training_sizes_ls]
            axes[1].plot(training_sizes_ls, values, marker='x', label=metric, color=colors_likelihood[metric])
        axes[1].set_xlabel("Number of training samples (N)")
        axes[1].set_ylabel("NLPD / MSLL")
        axes[1].set_title("Likelihood metrics")
        axes[1].grid(True)
        axes[1].legend()

        plt.suptitle(f"Learning curves for {name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def eval_GP_learning_curve2(gp_data, training_sizes: Iterable[int], lr=0.5):
    eval_data = {}

    for N in training_sizes:
        print(f"N: {N}")
        # Data setup for N samples
        X, Y = generate_data(N, RANGES)

        # scale
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)

        test_size = 0.2
        test_count = int(N * test_size)
        test_idx = np.arange(N)[:test_count]
        train_idx = np.arange(N)[test_count:]
        
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        eval_data[N] = {}

        # Train each GP
        for name, info in gp_data.items():
            print(f"GP: {name}")
            info["gp"].clear_data() # reset
            info["gp"].add_data(X_train[:, info["x_idxs"]], Y_train[:, info["y_idx"]])
            info["gp"].add_data(X_test[:, info["x_idxs"]], Y_test[:, info["y_idx"]], test=True)
            info["gp"].train(training_iter=training_iter, lr=lr)

            info["gp"].print_hyperparameters()

            gp_eval = eval_GP(info["gp"])
            eval_data[N][name] = gp_eval

    metrics_list = ["RMSE", "MAE", "NLPD", "MSLL"]  # metrics in gp_eval
    training_sizes_ls = sorted(eval_data.keys())      # all N values

    for name in gp_data.keys():  # iterate over GP names
        plt.figure(figsize=(8, 5))
        
        for metric in metrics_list:
            # collect metric values for all training sizes
            values = [eval_data[N][name][metric] for N in training_sizes_ls]
            plt.plot(training_sizes_ls, values, marker='o', label=metric)
        
        plt.title(f"Learning curve for {name}")
        plt.xlabel("Number of training samples (N)")
        plt.ylabel("Metric value")
        plt.legend()
        plt.grid(True)
        plt.show()


def eval_RMSE(GP, X_test, Y_test):
    GP.model.eval()
    GP.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = GP.likelihood(GP.model(X_test))
        rmse = torch.sqrt(torch.mean((pred.mean - Y_test) ** 2))
    return rmse.item()

def eval_MAE(GP, X_test, Y_test):
    GP.model.eval()
    GP.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = GP.likelihood(GP.model(X_test))
        mae = torch.mean(torch.abs(pred.mean - Y_test))
    return mae.item()

def eval_NLPD(GP, X_test, Y_test):
    GP.model.eval()
    GP.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = GP.likelihood(GP.model(X_test))
        nlpd = gpytorch.metrics.negative_log_predictive_density(pred, Y_test)
        #nlpd = -pred.log_prob(Y_test).mean()
    return nlpd.item()

def eval_MSLL(GP, X_train, Y_train, X_test, Y_test):
    GP.model.eval()
    GP.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = GP.likelihood(GP.model(X_test))
        
        #ll_model = pred.log_prob(Y_test)

        #y_mean = Y_train.mean()
        #y_var = Y_train.var()
        #baseline_ll = -0.5 * (torch.log(2 * torch.pi * y_var) + ((Y_test - y_mean) ** 2) / y_var)

        #msll = -(ll_model.mean() - baseline_ll.mean())
        msll = gpytorch.metrics.mean_standardized_log_loss(pred, Y_test, Y_train)
        #msll = gpytorch.metrics.quantile_coverage_error()

    return msll.item()


if __name__ == "__main__":
    drone = State.zero()
    params = Params(
        g=9.8066,
        roll_gain=1.101, roll_tau=0.253,
        pitch_gain=1.097, pitch_tau=0.267,
        dragacc1=0.01, dragacc2=0.01
    )

    dt = 0.01
    N = 1000

    training_iter = 100
    #training_iter = 50 / This will make things worse 
    lr = 0.5
    # X = np.random.randn(N, State.DIM + Input.DIM) * 2
    
    X, Y = generate_data(N, RANGES)
    # Y = differentiate_features(X, params)

    #gpytorch.settings.eval_cg_tolerance._set_value(1e-10)
    gpytorch.settings.eval_cg_tolerance._set_value(0)
    #gpytorch.settings.max_cg_iterations._set_value(10000)
    gpytorch.settings.max_cg_iterations._set_value(100000)
    #gpytorch.settings.cholesky_jitter._set_value(float_value=2e-5, double_value=2e-7, half_value=2e-5)


    test_size = 0.2
    test_count = int(N * test_size)
    test_idx = np.arange(N)[:test_count]
    train_idx = np.arange(N)[test_count:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: You are about to train on cpu!") 

    #if True: 
    gp_data = {
        "v1_dot": {
            "gp": DifferentialGP(input_dim=4, device=device, periodic_inputs=[0, 0, 0, 0]),
            "x_idxs": [3, 4, 5, 11], # r,y,p,y
            "y_idx": 0
        },
        "v2_dot": {
            "gp": DifferentialGP(input_dim=4, device=device, periodic_inputs=[0, 0, 0, 0]),
            "x_idxs": [3, 4, 5, 11], # r,y,p,y
            "y_idx": 1
        },
        "v3_dot": {
            "gp": DifferentialGP(input_dim=3, device=device, periodic_inputs=[0, 0, 0]),
            "x_idxs": [3, 4, 11], # r,p,y
            "y_idx": 2
        },
        "roll_dot": {
            "gp": DifferentialGP(input_dim=2, device=device, periodic_inputs=[0, 0]),
            "x_idxs": [3, 9], # r, r_r
            "y_idx": 3
        },
        "pitch_dot": {
            "gp": DifferentialGP(input_dim=2, device=device, periodic_inputs=[0, 0]),
            "x_idxs": [4, 10], # p, p_r
            "y_idx": 4
        }
    }

    eval_GP_learning_curve(gp_data, training_sizes=range(10, 1550, 150), normalize=False)

    diff_gps = {name: info["gp"] for name, info in gp_data.items()}


    """
        v1_dot_gp.add_data(X_train[:, [3, 4, 5, 11]], Y_train[:, 0]) # roll pitch yaw thurst
        v2_dot_gp.add_data(X_train[:, [3, 4, 5, 11]], Y_train[:, 1]) # roll pitch yaw thurst
        v3_dot_gp.add_data(X_train[:, [3, 4, 11]], Y_train[:, 2]) # roll pitch thurst
        roll_dot_gp.add_data(X_train[:, [3, 9 ]], Y_train[:, 3]) # roll roll_ref
        pitch_dot_gp.add_data(X_train[:, [4, 10]], Y_train[:, 4]) # pitch pitch_ref 
    
        print("training")
        v1_dot_gp.train(training_iter=training_iter, lr=lr) 
        v2_dot_gp.train(training_iter=training_iter, lr=lr)
        v3_dot_gp.train(training_iter=training_iter, lr=lr)
        roll_dot_gp.train(training_iter=training_iter, lr=lr)
        pitch_dot_gp.train(training_iter=training_iter, lr=lr)

        # Lazy copy paste
        v1_dot_gp.add_data(X_test[:, [3, 4, 5, 11]], Y_test[:, 0], test=True) # roll pitch yaw thurst
        v2_dot_gp.add_data(X_test[:, [3, 4, 5, 11]], Y_test[:, 1], test=True) # roll pitch yaw thurst
        v3_dot_gp.add_data(X_test[:, [3, 4, 11]], Y_test[:, 2], test=True) # roll pitch thurst
        roll_dot_gp.add_data(X_test[:, [3, 9 ]], Y_test[:, 3], test=True) # roll roll_ref
        pitch_dot_gp.add_data(X_test[:, [4, 10]], Y_test[:, 4], test=True) # pitch pitch_ref 

        diff_gps = {"v1_dot": v1_dot_gp, "v2_dot": v2_dot_gp, "v3_dot": v3_dot_gp, "roll_dot": roll_dot_gp, "pitch_dot": pitch_dot_gp}
    
        for GP_id in diff_gps.keys():
            GP = diff_gps[GP_id]
            
            print(f"Evaluating {GP_id}...")
            eval_GP(GP)
    """
         
    print("predicting")

    true_path, gp_path = walk_two_drones_vert(diff_gps, params, dt, [Input(0.1, 0.0, 5)] * 500)

    fig, axs = plt.subplots(1, 1)
    axs.plot(true_path[:, 2], label="True Dynamics")
    axs.plot(gp_path[:, 2],  label="GP Dynamics")
    
    plt.savefig("gpg.pdf")
    plt.show()
