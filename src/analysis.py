import numpy as np
import scipy

def calc_correlation(quantity, write_step, delta_t, a_dsc):
    N = quantity.shape[0]
    delta_step_index_array = np.arange(0,6,1)
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
        ave *= a_dsc**2/(count*(2*write_step*delta_t)**2)
        correlation.append(ave)
    return delta_step_index_array*write_step*delta_t, correlation

def calc_inverse_viscosity(tau_array, correlation, T, num_points):
    integrate = 2*scipy.integrate.trapezoid(correlation, x=tau_array)*num_points**2/T
    return integrate

def fit_mob(time, mob):
    return mob*time

def get_quantity_at_time_t(t, quantity, t_list):
    index_l = np.where(t_list<t)[0][-1]
    index_r = index_l + 1
    quantity_t = quantity[index_l]+(t-t_list[index_l])/(t_list[index_r]-t_list[index_l])*(quantity[index_r]-quantity[index_l])
    return quantity_t

def calc_msd(quantity, t_list, tau_array):
    tau_num = tau_array.shape[0]
    t_max = t_list[-2]
    N = t_list.shape[0]
    msd_list = []
    for j in range(tau_num):
        tau = tau_array[j]
        M = 0
        square_sum = 0
        for i in range(2,N):
            t_i = t_list[i] + tau
            if t_i > t_max:
                break
            M+=1
            quantity1 = get_quantity_at_time_t(t_list[i], quantity, t_list)
            quantity2 = get_quantity_at_time_t(t_i, quantity, t_list)
            diff = quantity2 - quantity1
            square_sum += diff**2
        msd = square_sum/M
        msd_list.append(msd)
    return msd_list

