'''parameters for calculation'''
import numpy as np


case_N200_k400 = {
    # numerical parameter
    'num_points'    : 200,              # number of points along two axes
    'kpoints_max'   : 400,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 0.1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'mode_list'     : [[0,-1]],          # np.array([p_m, q_m])
    'gamma'         : 0.03,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#3.2,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 10000000000,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 2000000,          # plot state every this number of steps
    'dump_interval'   : 2e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
}

case_N400_k200 = {
    # numerical parameter
    'num_points'    : 400,              # number of points along two axes
    'kpoints_max'   : 200,              # maximum k in truncated summation
    'plot_lim'      : 2.5,              # ceiling of stress value
    # material property
    'burgers'       : 0.1,             # reduced magnitude of unit Burgers vector
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 0.5,                # weight of dislocation core energy
    # external condition
    'temperature'   : 0.0012,            # reduced temperature
    'tau_ext'       : 18.0,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 1000000000,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 200000,          # plot state every this number of steps
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'path_state'    : './state',        # directory for saving transient states
}

case = case_N200_k400
