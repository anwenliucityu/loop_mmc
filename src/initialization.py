'''initialize lattice state'''
import numpy as np
import itertools
from compute_quantities import compute_step_energy, compute_core_energy, compute_elas_energy
import random

state_list = [-1, 0, 1]

def rand_init(latt_dim):
    '''randomly assign the state in list to each lattice point'''
    return np.random.choice(state_list, latt_dim)

def one_state_init(latt_dim):
    '''set states of all lattice points to 0'''
    return state_list[1] * np.ones(shape=latt_dim), \
           state_list[1] * np.ones(shape=latt_dim)

def circle_init(latt_dim):
    '''set states in a circle to 1'''
    radius = np.min(latt_dim) / 8 # in unit of cells
    center = np.floor(np.array(latt_dim) / 2) # center index of circle
    latt_state = state_list[1] * np.ones(shape=latt_dim)
    for ind, _ in np.ndenumerate(latt_state):
        distance = np.linalg.norm(ind - center)
        if distance <= radius:
            latt_state[ind] = state_list[2]
    return latt_state

def read_state(start_point, path_state):
    restart_path = path_state + '/snapshot/'
    latt_state_file = restart_path + f's_{start_point}.txt'
    latt_height_file = restart_path + f'z_{start_point}.txt'
    latt_state = np.loadtxt(latt_state_file)
    latt_height = np.loadtxt(latt_height_file)
    state_size = latt_state.shape
    return latt_state, latt_height, state_size

def grow_config_size(start_points, path_state, latt_dim):
    '''repeat a 200*200 config into a 800*800 config by random rotation'''
    if len(start_points) != 16:
        start_points_new = (start_points*int(np.ceil(16/len(start_points))))[:16]
    random.shuffle(start_points_new)
    start_points_new = np.array(start_points_new).reshape(4,4)
    latt_state_list  = []
    latt_height_list = []
    grow_latt_state  = np.zeros(shape = latt_dim)
    grow_latt_height = np.zeros(shape = latt_dim)
    for (i,j), start_point in np.ndenumerate(start_points_new):
        latt_state, latt_height, size = read_state(start_point, path_state)
        grow_latt_state[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] = latt_state
        grow_latt_height[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] = latt_height
    return grow_latt_state, grow_latt_height, size

def get_boundary_region_index(initial_dim, repeat=4, relax_wdith=2):
    relax_region = (relax_wdith/2, initial_dim[0]-relax_wdith/2)
    boundary_cell = []
    for (i,j) in itertools.product(range(initial_dim[0]), range(initial_dim[1])):
        if i<relax_region[0] or i>=relax_region[1] or j<relax_region[0] or j>=relax_region[1]:
            boundary_cell.append([i,j])
    boundary_cell = np.array(boundary_cell)

    boundary_index = np.empty(shape=(boundary_cell.shape[0]*repeat**2,2), dtype=int)
    boundary_cell_num = boundary_cell.shape[0]
    replicate=0
    for (i,j) in itertools.product(range(repeat), range(repeat)):
        shift = np.array((i*initial_dim[0], j*initial_dim[1]))
        boundary_index[replicate:replicate+boundary_cell_num] = boundary_cell + shift
        replicate+=boundary_cell_num
    return boundary_index

def compute_energy(latt_state, latt_height, latt_stress, nblist_mat, a_dsc, nu, zeta, gamma):
    E_core = compute_core_energy(latt_state, nblist_mat, a_dsc, nu, zeta)
    E_step = compute_step_energy(gamma, a_dsc, latt_height, nblist_mat)
    E_elas = compute_elas_energy(a_dsc, latt_state, latt_stress)
    E_total = E_core + E_step + E_elas
    return E_total, E_core, E_elas, E_step


if __name__ == '__main__':
    latt_dim = (3,3)
    a = one_state_init(latt_dim)
    print(a)
