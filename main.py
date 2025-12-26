# Read files (Lattice groups)

# describe ODE
# optimize stuff?

# Module for saving primitives to file

from state_info import State
from grid_point import GridPoint
from ACADOSDJIMatrice import ACADODjiMatrice
from old_main import construct_xdot
import math
import time

import casadi as cs
from gp_to_casadi import get_function_estimator, concatenate_inputs, split_inputs

SAVE_SUCCESS_PATH = "primitives/"
SAVE_FAILED_PATH = "primitives_failed/"
GOLD_PATH = "primitives_gold/" 
def save_primitive_to_file(group, id, name, traj):
    file_name = "primitive_group_" + str(group) + "_id_" + str(id) + ".txt"
    
    if traj.valid:
        path = SAVE_SUCCESS_PATH
    else:
        path = SAVE_FAILED_PATH

<<<<<<< Updated upstream
    # ensure file does not exists
    try:
        f = open(path+file_name, "x")
    except:
        f = open(path+file_name, "w")
        f.truncate()
    
    # write stuff
    f.write(
        "Generated trajectory:" + "\n" + 
        "Group: " + str(group) + "\n" + 
        "ID: " + str(id) + "\n" + 
        "Name: " + str(name)
    )
    f.write(str(traj))
    f.close()
=======
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
>>>>>>> Stashed changes


def main():
    """
    x = cs.SX.sym("x", 2)
    u = cs.SX.sym("u", 1)
    y = cs.sin(x[0]) + cs.cos(x[1] - u)
    f = cs.Function("f", [x, u], [y])
    val = f([0.9, 0.5], [0.1])
    print("standard", val.full())
    
    print("new stuff")

    f_cat = concatenate_inputs(f, 2, 1)
    val = f_cat([0.9, 0.5, 0.1])
    print("concat", val.full())

    f_est = get_function_estimator(f_cat, [(-1, 1) for i in range(3)])
    val = f_est([0.1, 0.5, 0.9])
    print("reverse order", val.full())
    val = f_est([0.9, 0.5, 0.1])
    print("correct order", val.full())

    f_est_split = split_inputs(f_est, 2, 1)
    val = f_est_split([0.9, 0.5], [0.1])
    print("split f_est", val.full())
    return
    #"""

    number_of_groups = 14
    sampling_time = 0.1 # 50 Hz
    x_dot = construct_xdot()
    shouldGenerate = True

    for group in range(0, number_of_groups+1):
        lattice_file = "config/lattice_group" + str(group) + ".txt"
        f = open(lattice_file)

        numberOfPrimitives = int(f.readline())

        for i in range(numberOfPrimitives):
            inS = State()
            fiS = State()

            f.readline() # read the "\n"

            id = int(f.readline())
            name = str(f.readline())
            finalTime = float(f.readline())

            inS.x, inS.y, inS.z, inS.vx, inS.vy, inS.vz = map(float, f.readline().split())
            
            fiS.x, fiS.y, fiS.z, fiS.vx, fiS.vy, fiS.vz = map(float, f.readline().split())
            inS.time = 0.0
            fiS.time = finalTime

            gridPoint = GridPoint(inS, fiS, name, id, group)
            nSegments = math.floor(finalTime / sampling_time)+1
            start = time.time()
            
            traj = ACADODjiMatrice(inS, fiS, nSegments, finalTime, x_dot, shouldGenerate)
            shouldGenerate = False

            end = time.time()
            print("Time to create trajectory:", end - start)

            save_primitive_to_file(group, id, name, traj)

        print(numberOfPrimitives)

        pass
    pass

if __name__ == "__main__":
    main()


<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes
