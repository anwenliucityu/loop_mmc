import numpy as np
from stress_kernel_shift import stress_kernel_shift
from file_handling import output_path 
import sys
from param import case

temperature = float(sys.argv[1])
simulation_type = case['simulation_type']
#tau_ext = case['tau_ext']
tau_ext = float(sys.argv[2])
psi_ext = float(sys.argv[3])
nu = case['nu']
zeta = case['zeta']
a_dsc = case['a_dsc']
gamma = case['gamma']
mode_list = case['mode_list']
num_points = case['num_points']
kpoints_max = case['kpoints_max']
latt_dim = (num_points, num_points)
maxiter = case['maxiter']
recalc_stress_step = case['recalc_stress_step']
plot_state_step = case['plot_state_step']
path_stress_kernel = case['path_stress_kernel']
dump_interval = case['dump_interval']
read_restart =  case['read_restart']

# create a directory for saving the states generated during simulation
path_state = output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, \
                mode_list, temperature, tau_ext, psi_ext, simulation_type, mkdir=True)

# build up neighborlist
import neighborlist
nblist_mat, nblist_arr = neighborlist.generate_neighbor_index_mat(latt_dim)

# initialize lattice state and lattice height
import initialization
if read_restart == False:
    latt_state, latt_height = initialization.one_state_init(latt_dim)
# read restart file
else:
    start_points = case['start_points']
    initial_dim  = case['initial_dim']
    initial_config_path = output_path(initial_dim[0], kpoints_max, nu,zeta,a_dsc, gamma, mode_list, temperature, tau_ext)
    latt_state, latt_height = initialization.grow_config_size(start_points, initial_config_path, latt_dim, initial_dim)
    boundary_index = initialization.get_boundary_region_index(initial_dim, repeat=int(latt_dim[0]/initial_dim[0]),relax_wdith=8)

# read in stress kernel Xi
from file_handling import read_stress_kernel_from_txt
stress_kernel = read_stress_kernel_from_txt(num_points, kpoints_max, path_stress_kernel)

# compute stress field
from compute_fields import compute_stress_field
stress_kernel_center = stress_kernel_shift(stress_kernel, (int(np.ceil(latt_dim[0]/2-1)), int(np.ceil(latt_dim[1]/2-1)))) 
latt_stress = compute_stress_field(latt_state, stress_kernel_center, a_dsc)

# compute initial energy
E_total, E_core, E_elas, E_step = \
        initialization.compute_energy(latt_state,latt_height,latt_stress,nblist_mat,a_dsc,nu,zeta,gamma)


# monte carlo
if simulation_type == 'mmc' or simulation_type == 'gmc':
    from monte_carlo_simulator import mc
    mc(latt_state, latt_stress, latt_height, stress_kernel, a_dsc, gamma, nblist_mat, nblist_arr, temperature, tau_ext, maxiter, nu, zeta, recalc_stress_step, plot_state_step, mode_list, E_total, E_core, E_elas, E_step, dump_interval, simulation_type, path_state)
elif simulation_type == 'kmc':
    from monte_carlo_simulator import kmc
    Q = case['Q']
    kmc(latt_state,latt_stress,latt_height,stress_kernel,a_dsc,gamma,nblist_mat,nblist_arr, temperature,tau_ext,psi_ext,maxiter,nu,zeta,recalc_stress_step,plot_state_step,mode_list,E_total,E_core,E_elas,E_step,dump_interval,Q,path_state)
