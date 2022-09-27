import numpy as np
import scipy

def calc_correlation(quantity, write_step, delta_t):
    N = quantity.shape[0]
    delta_step_index_array = np.arange(0,10,1)
    correlation = []
    for index in delta_step_index_array:
        count = 0
        ave = 0
        for i in range(N-1):
            step_0 = i
            step_delta_step = i+index
            if step_delta_step>N-2:
                break
            v_0 = quantity[step_0]
            v_delta_step = quantity[step_delta_step]
            ave += v_0 * v_delta_step
            count +=1
        ave /= count
        correlation.append(ave)
    return delta_step_index_array*write_step*delta_t, correlation

def calc_inverse_viscosity(tau_array, correlation, T, num_points):
    integrate = 2*scipy.integrate.trapezoid(correlation, x=tau_array)*num_points**2/T
    return integrate

