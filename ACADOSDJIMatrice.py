
from state_info import State
from grid_point import GridPoint
from trajectory import Trajectory

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, MX, vertcat, sin, cos, Function
import numpy as np


def dummy_calc_cost(ocp : AcadosOcp, acados_ocp_solver : AcadosOcpSolver):
    N = ocp.dims.N   # horizon length
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # --- extract results ---
    x_traj = np.zeros((N+1, nx))
    u_traj = np.zeros((N, nu))

    for k in range(N+1):
        x_traj[k, :] = acados_ocp_solver.get(k, "x")
        if k < N:
            u_traj[k, :] = acados_ocp_solver.get(k, "u")

    tf = ocp.solver_options.tf  # final horizon (could be T in free-time problems)
    N = ocp.dims.N

    stage_cost_fun = Function('stage_cost', [ocp.model.x, ocp.model.u], [ocp.model.cost_expr_ext_cost])
    term_cost_fun  = Function('term_cost',  [ocp.model.x], [ocp.model.cost_expr_ext_cost_e])
    
    T = float(term_cost_fun(x_traj[-1, :]))

    h = T / N
    
    total_cost = 0.0
    for k in range(N):
        total_cost += float(stage_cost_fun(x_traj[k, :], u_traj[k, :])) * h
    total_cost += T

    return total_cost


def ACADODjiMatrice(start : State, end : State, nsegments:int, finalTime: float, x_dot) -> Trajectory:
    PI = 3.1415
    nsegments = nsegments - 1
    # Differential States
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    
    roll = MX.sym("roll")
    pitch = MX.sym("pitch")
    yaw = MX.sym("yaw")

    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    
    roll_ref = MX.sym("roll_ref")
    pitch_ref = MX.sym("pitch_ref")
    thrust = MX.sym("thrust")
    
    roll_ref_dot = MX.sym("roll_ref_dot")
    pitch_ref_dot = MX.sym("pitch_ref_dot")
    thrust_dot = MX.sym("thrust_dot")

    T = MX.sym("T")

    x_state = vertcat(vx, vy, vz,
                     roll, pitch, yaw,
                     px, py, pz,
                     roll_ref, pitch_ref, thrust,
                     roll_ref_dot, pitch_ref_dot, thrust_dot,
                     T
                    )

    # The GPs don't use T
    x_state_for_gp = x_state[:-1]

    # Control
    roll_ref_ddot   = MX.sym("roll_ref_ddot")
    pitch_ref_ddot  = MX.sym("pitch_ref_ddot")
    thrust_ddot     = MX.sym("thrust_ddot")
    
    u_control = vertcat(roll_ref_ddot, pitch_ref_ddot, thrust_ddot)

    # Constants
    roll_tau   = 0.253
    roll_gain  = 1.101
    pitch_tau  = 0.267
    pitch_gain = 1.097
    linear_drag_coefficient1 = 0.01
    linear_drag_coefficient2 = 0.01
    g = 9.8066
    
    f_expl_expr = x_dot(x_state_for_gp, u_control)
    # f_expl_expr = vertcat(
    #     (cos(roll)*cos(yaw)*sin(pitch) + sin(roll)*sin(yaw))*thrust - vx * linear_drag_coefficient1,
    #     (cos(roll)*sin(pitch)*sin(yaw) - cos(yaw)*sin(roll))*thrust - vy * linear_drag_coefficient2,
    #     (-g + cos(pitch)*cos(roll)*thrust),
    #     (roll_gain*roll_ref - roll)/roll_tau,
    #     (pitch_gain*pitch_ref - pitch)/pitch_tau,
    #     0,
    #     vx,
    #     vy,
    #     vz,
    #     roll_ref_dot,
    #     pitch_ref_dot,
    #     thrust_dot,
    #     roll_ref_ddot,
    #     pitch_ref_ddot,
    #     thrust_ddot
    # )

    # Scale it (maybe correct)
    f_expl_expr_scaled = vertcat(T * f_expl_expr, 0)

    # Create model
    model = AcadosModel()
    model.f_expl_expr = f_expl_expr_scaled
    model.x = x_state
    model.u = u_control
    model.name = "Quadcopter"

    # Begin to define ocp!
    ocp = AcadosOcp()
    ocp.model = model

    #ocp.dims.N = nsegments
    ocp.solver_options.N_horizon = nsegments
    ocp.solver_options.tf = 1.0

    # constraints
    minTime = finalTime*0.5
    maxTime = finalTime*1.5

    ocp.constraints.lbx = np.array([ # USE LBU AND UBU for control stuff
        -0.8*28*PI/180, # roll_ref
        -0.8*28*PI/180, # pitch_ref
        1.2*g/2, # thrust
        -5, # vx
        -5, # vy
        -5, # vz
        minTime 
    ])

    ocp.constraints.ubx = np.array([
        0.8*28*PI/180, # Roll_ref
        0.8*28*PI/180, # Pitch_ref
        20*g*0.8, # Thrust
        5, # vx
        5, # vy
        5, # vz
        maxTime
    ])
    
    # indeces for the constraints
    ocp.constraints.idxbx = np.array([
        9, 10, 11, 0, 1, 2, 15
    ])

    # Setup initial conditions for (most) states BUT leave T free at initial node.
    # Use idxbx_0 / lbx_0 / ubx_0 to constrain only the indices we want at the initial shooting node.
    # We'll constrain all original 15 states at start (indices 0..14), but not the T_state (index 15). (we ignore this right now)
    idxbx_0 = np.arange(16)
    ocp.constraints.idxbx_0 = idxbx_0

    x0_vals_lower = np.array([
        start.vx, start.vy, start.vz,
        start.roll, start.pitch, start.yaw,
        start.x, start.y, start.z,
        start.roll_ref, start.pitch_ref, start.thrust,
        0.0, 0.0, 0.0,
        minTime
    ])
    x0_vals_upper = np.array([
        start.vx, start.vy, start.vz,
        start.roll, start.pitch, start.yaw,
        start.x, start.y, start.z,
        start.roll_ref, start.pitch_ref, start.thrust,
        0.0, 0.0, 0.0,
        maxTime
    ])

    ocp.constraints.lbx_0 = x0_vals_lower
    ocp.constraints.ubx_0 = x0_vals_upper

    # end constraints
    end_values_l = np.array([
        end.vx, end.vy, end.vz, 
        end.roll, end.pitch, end.yaw,
        end.x, end.y, end.z,
        end.roll_ref, end.pitch_ref, end.thrust,
        0.0, 0.0, 0.0,
        minTime
    ])
    end_values_u = np.array([
        end.vx, end.vy, end.vz, 
        end.roll, end.pitch, end.yaw,
        end.x, end.y, end.z,
        end.roll_ref, end.pitch_ref, end.thrust,
        0.0, 0.0, 0.0,
        maxTime
    ])
    ocp.constraints.lbx_e = end_values_l
    ocp.constraints.ubx_e = end_values_u
    ocp.constraints.idxbx_e = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ])

    # OPTIMAL CONTROL PROBLEM
    cost_rpp       = roll_ref**2 + pitch_ref**2
    cost_rppt_dot  = roll_ref_dot**2 + pitch_ref_dot**2 + thrust_dot**2
    cost_rppt_ddot = roll_ref_ddot**2 + pitch_ref_ddot**2 + thrust_ddot**2

    time_smooth_ratio = 4
    lagrange_expr = (1.0 / nsegments) * (1/time_smooth_ratio) * (1*cost_rpp + 1*cost_rppt_dot + 0.5*cost_rppt_ddot)

    # Lagrange cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = lagrange_expr

    # Mayer (terminal) cost
    ocp.model.cost_expr_ext_cost_e = T# * T #SX(1.0)  # equivalent to minimizeMayerTerm(T)

    # Solve things!
    """
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"    # ACADO uses "FULL_CONDENSING_QPOASES" (?)
    ocp.solver_options.hessian_approx = "EXACT"                  # same idea
    ocp.solver_options.integrator_type = "ERK"                   # explicit Runge–Kutta
    ocp.solver_options.nlp_solver_type = "SQP"                   # full SQP
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.qp_solver_iter_max = 500
    """

    #"""
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'#'FULL_CONDENSING_QPOASES'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 5000
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.sim_method_num_steps = 30
    ocp.solver_options.qp_solver_iter_max = 450
    # new thing
    ocp.solver_options.num_stages = 4
    ocp.solver_options.num_steps = 6
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    #"""


    # DONT PRINT
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_verbosity = 1

    solver = AcadosOcpSolver(ocp, verbose=False)

    # warm-start
    # Linear interpolation between start and end states
    if True:
        x0_start = np.array([
            start.vx, start.vy, start.vz,
            start.roll, start.pitch, start.yaw,
            start.x, start.y, start.z,
            start.roll_ref, start.pitch_ref, start.thrust,
            0.0, 0.0, 0.0,
            finalTime
        ])
        xn_end = np.array([
            end.vx, end.vy, end.vz, 
            end.roll, end.pitch, end.yaw,
            end.x, end.y, end.z,
            end.roll_ref, end.pitch_ref, end.thrust,
            0.0, 0.0, 0.0,
            finalTime
        ])

        for k in range(nsegments+1):
            alpha = k / nsegments
            x_guess = (1 - alpha) * x0_start + alpha * xn_end
            solver.set(k, "x", x_guess)

    status = solver.solve()
    solved_successfully = status == 0

    print("dummy dum dum", dummy_calc_cost(ocp, solver), "actual cost", solver.get_cost())

    traj = Trajectory(solver, start, end, solved_successfully)
    #print(traj)
    return traj