'''parameters for calculation'''
import numpy as np
simulations = ['mmc', 'gmc', 'kmc']

case_N5_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 5,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#0.50,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 2e8,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 10000000,          # plot state every this number of steps
    'dump_interval'   : 1e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 1000,
    'read_restart' : False,
    'start_points' : [100000000],
    'initial_dim'  : (5,5),
}

case_N150_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 150,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#0.50,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 2e8,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 10000000,          # plot state every this number of steps
    'dump_interval'   : 1e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 1000,
    'read_restart' : False,
    'start_points' : [100000000],
    'initial_dim'  : (5,5),
}

case_N200_k14000 = {
    # select simulation type
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 200,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0.01,#3.2,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 10000000000,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 50000000,          # plot state every this number of steps
    'plot_lim'     :2.5,
    'dump_interval'   : 5e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    # read restart
    'read_restart' : False,
    'start_points' : [500000],
    'initial_dim'  : (200,200),
}

case_N200_k8000 = {
    # select simulation type
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 200,              # number of points along two axes
    'kpoints_max'   : 8000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0.01,#3.2,                # externally applied reduced    stress
    # simulation parameters
    'maxiter'       : 10000000000,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this  number of steps
    'plot_state_step' : 50000000,          # plot state every this number of    steps
    'plot_lim'     :2.5,
    'dump_interval'   : 5e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving     stress kernel file
    # read restart
    'read_restart' : False,
    'start_points' : [500000],
    'initial_dim'  : (200,200),
}

case_N400_k14000 = {
    # numerical parameter
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    'num_points'    : 400,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#3.2,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 1e12,         # max number of iteration steps
    'recalc_stress_step' : 100000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 100000000,          # plot state every this number of steps
    'dump_interval'   : 5e9,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    'read_restart' : False,
    'initial_dim'  : (400,400),
    'start_points' : [1000000],
}

case_N100_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 100,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,         # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1, 
    'Q'             : 1,
    'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    'gamma'         : 1,#0.10,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0.0,#0.04, new               # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 1e14,         # max number of iteration steps
    'recalc_stress_step' : 100000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 100000000,          # plot state every this number of steps
    'dump_interval'   : 1e8,#1e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 0.1,
    'read_restart' : False,
    'initial_dim'  : (100,100),
    'start_points'  : [500000],
}

case_N50_k14000 = {
    # numerical parameter
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    'num_points'    : 50,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    #'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    #'mode_list'     : [[0,2.5],[1,-1]],   #
    'mode_list'     : [[1,-1]],
    #'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#3.2,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 4e14,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 200000000,#10000000,          # plot state every this number of steps
    'dump_interval'   : 1e9,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 2500,
    'read_restart' : False,
    'initial_dim'  : (50,50),
    'start_points'  : [3000000000],
}

'''
case_N50_k400 = {
    # numerical parameter
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][0],['screen','non-screen'][0]],
    'num_points'    : 50,              # number of points along two axes
    'kpoints_max'   : 400,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 0.5,
    'v'             : 1,
    'Q'             : 1,
    #'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    #'mode_list'     : [[0,2.5],[1,-1]],   #
    'mode_list'     : [[1,-2]],
    #'mode_list'     : [[1,-1]],
    'gamma'         : 1,#(1-0.3)*np.pi*0.46e-10/26e-11/(4.05/np.sqrt(34))*2, 
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,#3.2,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 4e14,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 200000000,#10000000,          # plot state every this number of steps
    'dump_interval'   : 1e9,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 2500,
    'read_restart' : False,
    'initial_dim'  : (50,50),
    'start_points'  : [3000000000],
}
'''


case_N10_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 10,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    #'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    #'mode_list'     : [[0,2.5],[1,-1]],   #
    'mode_list'     : [[1,-1]],
    #'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,                # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 2e9,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 10000000,          # plot state every this number of steps
    'dump_interval'   : 1e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 1000,
    'read_restart' : False,
    'initial_dim'  : (10.1),
    'start_points'  : [100000000],
}

case_N20_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 20,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    #'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    #'mode_list'     : [[0,2.5],[1,-1]],   #
    'mode_list'     : [[1,-1]],
    #'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,               # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 2e9,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 10000000,          # plot state every this number of steps
    'dump_interval'   : 5e7,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 1000,
    'read_restart' : False,
    'initial_dim'  : (20,20),
    'start_points'  : [100000000],
}

case_N800_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter              
    'num_points'    : 800,              # number of points along two axes
    'kpoints_max'   : 5000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 1,                # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    #'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    #'mode_list'     : [[0,2.5],[1,-1]],   #
    'mode_list'     : [[1,-1]],
    #'mode_list'     : [[1,-1]],
    'gamma'         : 1,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0,               # externally applied reduced stress
    # simulation parameters        
    'maxiter'       : 2e9,         # max number of iteration steps
    'recalc_stress_step' : 10000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 10000000,          # plot state every this number of steps
    'dump_interval'   : 5e7,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 1000,
    'read_restart' : False,
    'initial_dim'  : (20,20),
    'start_points'  : [100000000],
} 

case_N100_k14000 = {
    'simulation_type': simulations[0],
    'disl_dipole'   : [[True,False][1],['screen','non-screen'][0]],
    # numerical parameter
    'num_points'    : 100,              # number of points along two axes
    'kpoints_max'   : 14000,              # maximum k in truncated summation
    # material property
    'nu'            : 0.3,              # Poisson's ratio
    'zeta'          : 4,         # weight of dislocation core energy
    'a_dsc'         : 1,
    'v'             : 1,
    'Q'             : 1,
    'mode_list'     : [[1,-1]],          # np.array([p_m, q_m])  T = 0.05
    'gamma'         : 1,#0.10,
    # external condition
    'temperature'   : 24,#0.0006,            # reduced temperature
    'tau_ext'       : 0.0,#0.04, new               # externally applied reduced stress
    # simulation parameters
    'maxiter'       : 1e14,         # max number of iteration steps
    'recalc_stress_step' : 100000000,     # recalculate stress field every this number of steps
    'plot_state_step' : 100000000,          # plot state every this number of steps
    'dump_interval'   : 1e8,#1e8,
    'path_stress_kernel' : './stress_kernel_origin', # directory for saving stress kernel file
    'plot_lim'     :2.5,
    #'write_step'   : 0.1,
    'read_restart' : False,
    'initial_dim'  : (100,100),
    'start_points'  : [500000],
}


case = case_N100_k14000
