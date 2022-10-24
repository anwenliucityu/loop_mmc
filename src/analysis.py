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

def calc_vvcorrelation_displacement(displacement, time, delta_t):
    N = displacement.shape[0]
    n=20
    delta_t_array = np.arange(0,delta_t*n,delta_t)
    tmax = time[-2]

    v_correlation = []
    for index in range(n):
        count = 0
        ave = 0
        tau = delta_t_array[index]
        for i in range(1,N-1):
            time0 = time[i]
            time1 = time[i] + tau
            if time1 > tmax:
                break
            v0 = calc_v0(i, displacement, time)
            if tau == 0:
                v1 = v0
            else:
                v1 = calc_v1(time1, displacement, time)
            ave += v0*v1
            count +=1
        ave /= count
        v_correlation.append(ave)
    return delta_t_array, v_correlation
        

def calc_v0(i, displacement, time):
    ti = time[i]
    tm = time[i-1]
    tp = time[i+1]
    ui = displacement[i]
    um = displacement[i-1]
    up = displacement[i+1]
    ui = ((ti-tm)*up + (tp-ti)*um)/(tp-tm)
    vi = ((ti-tm)/(tp-ti)*(up-ui)+(tp-ti)/(ti-tm)*(ui-um))/(tp-tm)
    return vi

def calc_v1(time1, displacement, time):
    i_m = np.where(time1>time)[0][-1]
    if (time1-time[i_m])*2>time[i_m+1]-time[i_m]:
        i_p = i_m+2
    else:
        i_m=i_m-1
        i_p = i_m+2
    tm = time[i_m]
    tp = time[i_p]
    um = displacement[i_m]
    up = displacement[i_p]
    ui = ((time1-tm)*up + (tp-time1)*um)/(tp-tm)
    vi = ((time1-tm)/(tp-time1)*(up-ui)+(tp-time1)/(time1-tm)*(ui-um))/(tp-tm)
    return vi


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

