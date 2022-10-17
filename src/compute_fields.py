'''Compute fields of a state'''
import numpy as np
import scipy
from stress_kernel_shift import stress_kernel_shift
'''
def compute_stress_field(latt_state, stress_kernel, a_dsc):
    N = latt_state.shape[0]
    sum = np.zeros(shape=latt_state.shape)
    for ind, state in np.ndenumerate(latt_state):
        sum += state * stress_kernel_shift(stress_kernel, ind)
    return 2 * a_dsc/N * sum 
'''
#'''
def compute_stress_field(latt_state, stress_kernel_center, a_dsc):
    N = latt_state.shape[0]
    return 2 * a_dsc/N * scipy.signal.convolve2d(latt_state, stress_kernel_center, mode='same', boundary='wrap')
#'''
