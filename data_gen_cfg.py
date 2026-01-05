#from data import State, Params

# ============== #
#     PARAMS     #
# ============== #

PARAMS = {
    "g": 9.8066,
    "roll_gain": 1.101, "roll_tau": 0.253,
    "pitch_gain": 1.097, "pitch_tau": 0.267,
    "dragacc1": 0.01, "dragacc2": 0.01
}


# ============== #
#     RANGES     #
# ============== #

# Possible {State, Input} ranges for LHS.

# ===================== #
#     PERTURBATIONS     #
# ===================== #
'''
quad_vel_drag = lambda s, i, p, sd: State(
        -0.1*s.v1**2, -0.1*s.v2**2, -0.1*s.v3**2,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0
    )

weaker_thrust = lambda s, i, p, sd: State(
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,-0.3*s.thrust,
        0,0,0
    )

pitch_sensor_bias = lambda s, i, p, sd: State(
        0, 0, 0,
        0.0, +0.1, 0.0,  # bias in pitch measurement
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
)

cubic_roll_input = lambda s, i, p, sd: State(
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        4 * i.roll_ref_ddot**3, 0, 0
)

poly_pitch_input = lambda s, i, p, sd: State(
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 2 * i.pitch_ref_ddot**2 + 1 * i.pitch_ref_ddot**3, 0
)

CFG1_PERTURBATIONS = [weaker_thrust, pitch_sensor_bias, cubic_roll_input, poly_pitch_input]
CFG2_PERTURBATIONS = [quad_vel_drag, weaker_thrust, pitch_sensor_bias]
'''
