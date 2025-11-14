from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
from state_info import State

class Trajectory:
    def __init__(self, solver:AcadosOcpSolver, start : State, end : State, valid : bool):
        segments = solver.N
        self.path_list = []
        self.final_time = solver.get(0, "x")[-1]
        self.cost = solver.get_cost()
        self.valid = valid

        self.toFromLabel = "x y z vx vy vz"
        self.pathFrom = start.getXYZVXVYVZStr()
        self.pathTo = end.getXYZVXVYVZStr()
        t = np.linspace(0.0, self.final_time, segments+1)

        for i in range(len(t)):
            r = solver.get(i, "x")
            s_pos = r[6:9]
            s_velocity = r[0:3]
            s_rotation = r[3:6]
            s_thrust = r[11]
            s_t = t[i]

            self.path_list.append(
                [s_t, 
                s_pos[0], s_pos[1], s_pos[2], 
                s_velocity[0], s_velocity[1], s_velocity[2],
                s_rotation[0], s_rotation[1], s_rotation[2],
                s_thrust]
            )

            #print(f"Pos: {s_pos[0]:.6f} {s_pos[1]:.6f} {s_pos[2]:.6f} Vel: {s_velocity[0]:.6f} {s_velocity[1]:.6f} {s_velocity[2]:.6f} Rot: {s_rotation[0]:.6f} {s_rotation[1]:.6f} {s_rotation[2]:.6f} {s_thrust:.6f}")


    def __str__(self):
        return str(
            f"finalTime: {self.final_time:.6f}\n"+
            f"Cost: {self.cost:.6f}\n"+
            f"ToFromLabel: {self.toFromLabel}\n"+
            f"From: {self.pathFrom}\n"+
            f"To: {self.pathTo}\n"+
            f"Trajectory: \n"+
            "t        x        y         z        vx       vy       vz       roll     pitch     yaw     thrust \n"+
            "".join(["".join(f"{j:.6f} " for j in i) + "\n" for i in self.path_list])
            )
        