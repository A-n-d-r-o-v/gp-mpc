import numpy as np
import matplotlib.pyplot as plt

#GOLD_PATH = "primitives_gold/"
GOLD_PATH = "primitives_acados_standard/"
COMPARE_PATH = "primitives_all_gp/"

def read_primitive_into_matrix(path : str)->np.array:
    arr = []
    try:
        with open(path) as f:
            for l in f.readlines():
                spl = l.split()
                if len(spl) > 1:
                    for s in spl:
                        try:
                            v = float(s)
                            arr.append(v)
                        except:
                            pass
                pass
            pass
    except:
        return np.array([-1])
    return np.array(arr)
    

def compare_primitives(group, id):
    try:
        file_name = "primitive_group_" + str(group) + "_id_" + str(id) + ".txt"
        v1 = read_primitive_into_matrix(GOLD_PATH + file_name)
        v2 = read_primitive_into_matrix(COMPARE_PATH + file_name)

        # path error
        path1 = v1[16:]
        path1 = path1.reshape((int(path1.size / 11), 11))
        path2 = v2[16:]
        path2 = path2.reshape((int(path2.size / 11), 11))

        if (path1.shape != path2.shape):
            print(group, id, "failed! Non-matching path-shapes")
            return (-1, -1, -1)

        row_diff_mean = np.mean(np.abs(path1 - path2), axis=1)
        #print(row_diff_mean)

        path_error = np.mean(row_diff_mean)

        # Cost error (relative error)
        cost_error = (v2[3] - v1[3]) / abs(v1[3])

        # Time error (relative error)
        time_error = (v2[2] - v1[2]) / abs(v1[2])

        return (path_error, cost_error, time_error)
    except:
        print("unknown error at ", group, id)
        return (-1, -1, -1)

def findGroupAndIDFromIndex(groups, index):
    for g, i in enumerate(groups):
        if index <= i:
            return (g, index)
        else:
            index -= i
    return (-1, -1)


if __name__ == "__main__":
    groups = [
        0, # disregard first one
        10,
        16,
        30,
        48,
        32,
        16,
        16,
        20,
        30,
        30,
        40,
        32,
        16,
        16
    ]

    all_path_error = []
    all_cost_error = []
    all_time_error = []

    num_skipped = 0
    
    for group, numberOfPrimitives in enumerate(groups):
        for i in range(numberOfPrimitives):
            p, c, t = compare_primitives(group, i)

            if True:#abs(c) < 1.0:
                all_path_error.append(p)
                all_cost_error.append(c)
                all_time_error.append(t)
            else:
                num_skipped += 1

    print("Total number of primitives: ", sum(groups[1:]))
    print("Number of primitives skipped: ", num_skipped)
    print("Percent of primitives skipped: ", (num_skipped / (sum(groups[1:]))) * 100)
    
    #print largest 5 cost error
    amax = np.argmax(all_cost_error)
    amax = findGroupAndIDFromIndex(groups, amax)
    print(amax)

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(all_path_error)
    ax1.set_title("absolute cumulative path error")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(all_cost_error)
    ax2.set_title("relative cost error")


    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(all_time_error)
    ax3.set_title("relative time error")
    plt.show()
