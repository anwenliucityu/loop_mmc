import numpy as np
from stress_kernel_shift import stress_kernel_shift
from compute_quantities import compute_step_energy, compute_core_energy, compute_elas_energy
from file_handling import output_path 
import sys
from param import case

temperature = float(sys.argv[1])
tau_ext = case['tau_ext']
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

# create a directory for saving the states generated during simulation
path_state = output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, temperature, tau_ext)

# build up neighborlist
import neighborlist
nblist_mat, nblist_arr = neighborlist.generate_neighbor_index_mat(latt_dim)

# initialize lattice state
import initialization
latt_state = initialization.one_state_init(latt_dim)

# initialize lattice height
latt_height = initialization.one_state_init(latt_dim)

# read in stress kernel Xi
from file_handling import read_stress_kernel_from_txt
stress_kernel = read_stress_kernel_from_txt(num_points, kpoints_max, path_stress_kernel)

# compute stress field
from compute_fields import compute_stress_field
stress_kernel_center = stress_kernel_shift(stress_kernel, (int(np.ceil(latt_dim[0]/2-1)), int(np.ceil(latt_dim[1]/2-1)))) 
latt_stress = compute_stress_field(latt_state, tau_ext, stress_kernel_center, a_dsc)

# compute initial total energy E
E_core = compute_core_energy(latt_state, nblist_mat, a_dsc, nu, zeta)
E_step = compute_step_energy(gamma, a_dsc, latt_height, nblist_mat)
E_elas = compute_elas_energy(a_dsc, latt_state, latt_stress)
E_total = E_core + E_step + E_elas

# metropolis monte carlo
from monte_carlo_simulator import mmc
latt_state, latt_stress = mmc(latt_state, latt_stress, latt_height, stress_kernel, a_dsc, gamma, nblist_mat, nblist_arr, temperature, tau_ext, maxiter, nu, zeta, recalc_stress_step, plot_state_step, mode_list, E_total, E_core, E_elas, E_step, dump_interval, path_state)