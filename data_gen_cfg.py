from data import State, Params



# ============== #
#     PARAMS     #
# ============== #

PARAMS = Params(
    g=9.8066,
    roll_gain=1.101, roll_tau=0.253,
    pitch_gain=1.097, pitch_tau=0.267,
    dragacc1=0.01, dragacc2=0.01
)


# ============== #
#     RANGES     #
# ============== #

# Possible {State, Input} ranges for LHS.
RANGES = {
    # VELOCITY
    "v1":               (0, 0),
    "v2":               (0, 0),
    "v3":               (0, 0),

    # ORIENTATION
    # "roll":              (10,20),
    # "pitch":             (10,20),
    # "yaw":               (0,0),
    
    "roll":             (-3.1415, 3.1415),
    "pitch":             (-3.1415, 3.1415),
    "yaw":               (-3.1415, 3.1415), 
    #"pitch":            (0, 0),

    # POSITION, the position of the drone has _no_ effect on the derivatives
    "p1":               (0, 0),
    "p2":               (0, 0),
    "p3":               (0, 0),

    # TARGET/REFERENCE STATES, what is this? 
    "roll_ref":         (-10, 10),
    "pitch_ref":        (-10, 10),
    "thrust":           (-5, 150),
    # "thrust":           (2 ,2),


    # FIRST DERIVATIVES 
    "roll_ref_dot":     (0, 0),
    "pitch_ref_dot":    (0, 0),
    "thrust_dot":       (0, 0),

    # INPUT (SECOND DERIVATIVES)
    "roll_ref_ddot":    (0, 0),
    "pitch_ref_ddot":   (0, 0),
    "thrust_ddot":      (-PARAMS.g*2, PARAMS.g*7),
}


# ===================== #
#     PERTURBATIONS     #
# ===================== #

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

