import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange

import seaborn as sns
import dcor
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram


from pyDOE3 import lhs

from data import State, Input
from data_gen_cfg import PARAMS, RANGES, CFG1_PERTURBATIONS, CFG2_PERTURBATIONS
from ODEModel import step

from ODEModel import step, differentiate_features
# ==========================
# Parameters and initialization
# ==========================


# function to compute rotation matrix from roll/pitch/yaw
def body_axes(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def _scale_lhs(
        lhs_samples: np.ndarray, 
        ranges: dict
        ) -> np.ndarray:
    """Scale LHS samples in [0,1] to the given variable ranges."""
    lows = np.array([low for low, _ in ranges.values()])
    highs = np.array([high for _, high in ranges.values()])
    return lows + (highs - lows) * lhs_samples

def generate_states_actions(
        n_samples: int,
        ranges: dict,
        criterion: str = "center", # maximin: more realistic for real data (not 100% coverage). center: more centered around mean. None=randomize
        random_state: int = None #          https://pydoe3.readthedocs.io/en/latest/randomized.html
    ):
    """
    Generate states and actions using Latin Hypercube Sampling. 
    The state space is sampled and then scaled to fit the intervals defined by 'ranges'.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    ranges : dict
        Dictionary mapping variable names to (min, max) tuples specifying the
        sampling interval for each dimension.
    criterion : str, optional
        LHS sampling criterion, controls the distribution of points:
            - "center" : place points at the center of each interval (clustered around mean)
            - "maximin": maximize minimum distance between points (more uniform coverage)
            - None     : uniform random LHS
            - more: https://pydoe3.readthedocs.io/en/latest/randomized.html
    random_state : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    states : list of State
        List of sampled State objects.
    inputs : list of Input
        List of sampled Input objects
    """
    dim = len(ranges)

    # Latin Hypercube Sampling
    lhs_samples = lhs(dim, samples=n_samples, criterion=criterion, random_state=random_state)

    data = _scale_lhs(lhs_samples, ranges)

    states, inputs = [], []
    keys = list(ranges.keys())

    for row in data:
        d = dict(zip(keys, row))
        states.append(State(
            v1=d["v1"], v2=d["v2"], v3=d["v3"],
            roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"],
            p1=d["p1"], p2=d["p2"], p3=d["p3"],
            roll_ref=d["roll_ref"], pitch_ref=d["pitch_ref"],
            thrust=d["thrust"],
            roll_ref_dot=d["roll_ref_dot"], pitch_ref_dot=d["pitch_ref_dot"],
            thrust_dot=d["thrust_dot"],
        ))
        inputs.append(Input(
            roll_ref_ddot=d["roll_ref_ddot"],
            pitch_ref_ddot=d["pitch_ref_ddot"],
            thrust_ddot=d["thrust_ddot"],
        ))
    return states, inputs

def generate_data(
        n_samples: int,
        ranges: dict,
        # dt: float,
        lhs_criterion: str = "center",
        lhs_random_state: int = None,
        as_tensor: bool = True,
        return_nominal: bool = False
        ):
    """
    Generate training data for GP: X = [state,input], Y = state differences or nominal states.

    Args:
        n_samples: Number of training samples.
        ranges: Dict specifying state/input sampling ranges.
        dt: Time step for simulation.
        lhs_criterion: Criterion for Latin Hypercube Sampling.
        lhs_random_state: RNG seed.
        as_tensor: Return torch.Tensors if True; else lists.
        return_nominal: If True, trainY = nominal stepped states; else differences.

    Returns:
        trainX, trainY
    
    NOTE: Can probably be parallelized with dynamics.step_features if needed
    """
    states, inputs = generate_states_actions(n_samples, ranges, 
                                             criterion=lhs_criterion, 
                                             random_state=lhs_random_state)
    trainY_list = []

    state_input_list = [np.concatenate((s.as_numpy(), i.as_numpy())) for (s, i) in zip(states, inputs)]
    starting_states = np.dstack(state_input_list).squeeze().T

    #moded_starting_states = starting_states.copy()
    #from math import pi
    #moded_starting_states[:, [3, 4]] = moded_starting_states[:, [3, 4]] % (2*2*pi) 
    #moded_state_dot = differentiate_features(moded_starting_states, PARAMS)
    
    state_dot = differentiate_features(starting_states, PARAMS)

    #print(state_dot[:, 2])
    #print(moded_state_dot[:,2])
    #print(np.sum((state_dot[:, 2] - moded_state_dot[:,2])**2))
    #1/0
    

    return starting_states, state_dot
    

    # print(starting_states[0])
    # print(state_dot[0])
    # 1/0
    
    # for sample_idx in trange(n_samples):
    #     state = states[sample_idx]
    #     action = inputs[sample_idx]
        
    #     # step dynamics
    #     complete_state_action_sample = np.concatenate((state.as_numpy(), action.as_numpy()))
        
    #     state_step_nom = differentiate_features(complete_state_action_sample, PARAMS)
    #     # state_step_nom = step(state, action, PARAMS, dt)
    #     # state_step_pert = step(state, action, PARAMS, dt, perturbations=CFG2_PERTURBATIONS)

    #     # if return_nominal:         
    #     #     trainY_list.append(state_step_nom.difference(state_step_pert)) 
    #     # else:
    #     trainY_list.append(state_step_nom)

    # if as_tensor:
    #     # Combine {state, input} as one tensor (trainX)
    #     trainX = torch.stack([torch.cat([s.as_tensor(), u.as_tensor()]) for s, u in zip(states, inputs)])
    #     trainY = torch.stack([sd.as_tensor() for sd in trainY_list])
    # else:
    #     trainX = list(zip(states, inputs))
    #     trainY = trainY_list

    # return trainX, trainY

def plot_covariance(data):
    pass

def plot_distribution(data):
    pass


if __name__ == '__main__':

    trainX, trainY = generate_data(1000, RANGES, dt=1)
    print(f"{trainX.shape=}")
    print(f"{trainY.shape=}")


    '''
    draw = False

    scale = 0.2  # body axes length
    dt = 1
    """ state = State.zero()
    pos = np.array([state.p1, state.p2, state.p3])
    R = body_axes(state.roll, state.pitch, state.yaw)
    vel = np.array([state.v1, state.v2, state.v3]) """

    positions = []

    states_diffs = []

    if draw:
        plt.ion()  # interactive mode
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        trajectory_line, = ax.plot([], [], [], 'gray', linestyle='--', alpha=0.5)

        """ q_x = ax.quiver(*pos, *(R[:,0]*scale), color='r')
        q_y = ax.quiver(*pos, *(R[:,1]*scale), color='g')
        q_z = ax.quiver(*pos, *(R[:,2]*scale), color='b')
        q_vel = ax.quiver(*pos, *(vel*0.1), color='k', linestyle='dashed') """


    num_samples = 5000 # number of unique start states
    num_steps = 1 # simulation steps per start state

    # TODO: Need #num_samples action sequences of length #num_steps
    #       Sample spline control points with LHS?

    # TODO: We can make it faster by calling differentiate_features instead (vectorize !!!)
    #       if we feel the need to...
    #       it may already be fast enough to generate more than enough data.
    states, inputs = generate_states_actions(num_samples, RANGES)

    print("Generating training data...")
    for sample_idx in trange(num_samples):
        state = states[sample_idx]
        action = inputs[sample_idx]

        # Sim loop
        for step_idx in range(num_steps): # Unused for now (num_steps = 1)
            # record position
            pos = np.array([state.p1, state.p2, state.p3])
            positions.append(pos)
            roll, pitch, yaw = state.roll, state.pitch, state.yaw
            vel = np.array([state.v1, state.v2, state.v3])
            R = body_axes(roll, pitch, yaw)

            if draw:
                # update trajectory
                positions_arr = np.array(positions)
                trajectory_line.set_data(positions_arr[:,0], positions_arr[:,1])
                trajectory_line.set_3d_properties(positions_arr[:,2])

                # remove old quivers
                for q in [q_x, q_y, q_z, q_vel]:
                    q.remove()

                # redraw quivers at new position
                q_x = ax.quiver(*pos, *(R[:,0]*scale), color='r')
                q_y = ax.quiver(*pos, *(R[:,1]*scale), color='g')
                q_z = ax.quiver(*pos, *(R[:,2]*scale), color='b')
                q_vel = ax.quiver(*pos, *(vel*0.1), color='k', linestyle='dashed')

                plt.draw()
                plt.pause(dt)

            # step dynamics
            state_step_nom = step(state, action, PARAMS, dt)
            state_step_pert = step(state, action, PARAMS, dt, perturbations=CFG2_PERTURBATIONS)

            #states_diffs.append(state_step_nom.difference(state_step_pert)) # State ( Nominal - Perturbed )
            states_diffs.append(state_step_nom.difference(state)) 
        
        #states_diffs = states # .append() 

    #print(state.as_dict())

    print(states_diffs[0])
    print(states_diffs[-1])

    delta_array = np.array([s.as_numpy() for s in states_diffs])  # shape (N_samples, N_vars)
    N_vars = delta_array.shape[1]

    print("GO!")

    # get automatically from State or Input
    var_names = [
        "v1", "v2", "v3",
        "roll", "pitch", "yaw",
        "p1", "p2", "p3",
        "roll_ref", "pitch_ref", "thrust",
        "roll_ref_dot", "pitch_ref_dot", "thrust_dot"
    ]

    # Compute distance correlation matrix
    dcor_matrix = np.zeros((N_vars, N_vars))
    for i in range(N_vars):
        for j in range(N_vars):
            dcor_matrix[i, j] = dcor.distance_correlation(delta_array[:, i], delta_array[:, j])

    # Optional: cluster variables based on dcor
    Z = linkage(1 - dcor_matrix, method='average')
    sns.set(style="white")
    plt.figure(figsize=(10,8))
    dendro = dendrogram(Z, labels=var_names, leaf_rotation=90)
    order = dendro['leaves']

    # Reorder matrix to match dendrogram
    dcor_ordered = dcor_matrix[order, :][:, order]
    ordered_names = [var_names[i] for i in order]

    # Plot
    sns.heatmap(dcor_ordered, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=ordered_names, yticklabels=ordered_names)
    plt.title("Distance Correlation of State Differences")
    plt.show()


    # Plot distributions
    df = pd.DataFrame(delta_array, columns=var_names)

    # summary stats
    summary = df.describe().T[["min", "max", "mean", "std"]]
    summary["var"] = df.var().values

    print(summary)

    # Plot histograms / KDEs
    fig, axes = plt.subplots(
        nrows=(len(var_names) // 3) + 1, ncols=3, figsize=(15, 4 * ((len(var_names) // 3) + 1))
    )
    axes = axes.flatten()

    for i, col in enumerate(var_names):
        sns.histplot(df[col], bins=50, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(
            f"{col}\nmin={df[col].min():.2f}, max={df[col].max():.2f}, var={df[col].var():.2e}"
        )
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    '''
