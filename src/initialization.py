'''initialize lattice state'''
import numpy as np

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
    restart_path = path_state + '/snapshoti/'
    latt_state_file = restart_path + f'{s_{start_point}.txt}'
    latt_height_file = restart_path + f'{z_{start_point}.txt}'
    latt_state = np.loadtxt(latt_state_file)
    latt_height = np.loadtxt(latt_height_file)
    state_size = latt_state.shape
    return latt_state, latt_height, latt_size

def grow_config_size(start_points, path_state, latt_dim):
    latt_state_list  = []
    latt_height_list = []
    grow_latt_state  = np.zeros(shape = latt_dim)
    grow_latt_height = np.zeros(shape = latt_dim)
    for i, start_point in enumerate(start_points):




if __name__ == '__main__':
    latt_dim = (3,3)
    a = one_state_init(latt_dim)
    print(a)
