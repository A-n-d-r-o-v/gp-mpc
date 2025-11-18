# Read files (Lattice groups)

# describe ODE
# optimize stuff?

# Module for saving primitives to file

from state_info import State
from grid_point import GridPoint
from ACADOSDJIMatrice import ACADODjiMatrice
from old_main import construct_xdot
import math

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
            traj = ACADODjiMatrice(inS, fiS, nSegments, finalTime, x_dot)

            save_primitive_to_file(group, id, name, traj)

        print(numberOfPrimitives)

        pass
    pass

if __name__ == "__main__":
    main()


