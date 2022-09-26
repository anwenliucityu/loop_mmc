import numpy as np
import scipy

def calc_correlation(quantity, T, num_points):
    dump_interval = num_points**2
    N = quantity.shape[0]
    tau_array = np.arange(0,10,1)
    correlation = []
    for tau in tau_array:
        count = 0
        integral = 0
        for i in range(N-1):
            t0 = i
            t_tau = i+tau
            if t_tau>N-2:
                break
            v0 = quantity[t0]
            v_tau = quantity[t_tau]
            integral += v0 * v_tau 
            count +=1
        #integral = integral/count/T * 2 * num_points**2
        correlation.append(integral)
    return tau_array*dump_interval, correlation

def calc_inverse_viscosity(tau_array, correlation, T, num_points):
    integrate = scipy.integrate.trapezoid(correlation, x=tau_array)*num_points**2/T
    return integrate

