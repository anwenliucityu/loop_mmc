import numpy as np
import random
from stress_kernel_shift import stress_kernel_shift
from compute_quantities import compute_core_energy_change, compute_elastic_energy_change, compute_step_energy_change, \
        compute_step_energy, compute_core_energy, compute_elas_energy
from compute_fields import compute_stress_field
from plot_state import plot_state_stress
from file_handling import write_total_energy_wu_to_txt, write_s_z_average_to_txt
from pathlib import Path
import os

def mc(latt_state_init, latt_stress_init, latt_height_init, stress_kernel, a_dsc, gamma, nblist_mat, nblist_arr, temperature, tau_ext, maxiter, nu, zeta, recalc_stress_step, plot_state_step, mode_list,E_total, E_core,E_elas,E_step, dump_interval, simulation_type, path_state=None, phi_ext =0):
    '''metropolis and glauber monte carlo'''
    latt_state = latt_state_init
    latt_stress = latt_stress_init
    latt_height = latt_height_init
    N = latt_state.shape[0]
    write_energy = N**2
    choice_num = len(mode_list)
    stress_kernel_center = stress_kernel_shift(stress_kernel, (int(np.ceil(N/2-1)), int(np.ceil(N/2-1))))
    
    # create a txt file for writing E_total and w_u
    os.system(f'rm {path_state}/quantities.txt')
    os.system(f'rm {path_state}/s_z.txt')
    Path(f'{path_state}/quantities.txt').touch()
    Path(f'{path_state}/s_z.txt').touch()
    
    W_stress = 0
    W_free_energy = 0
    for i in range(maxiter+1):
        # plot state every several steps
        if np.mod(i, plot_state_step) == 0:
            plot_state_stress(latt_height, latt_state, latt_stress, tau_ext, i, path_state)

        # write E_total
        if np.mod(i, write_energy) == 0:
            stress_mean = np.mean(latt_stress)
            write_total_energy_wu_to_txt(i, E_total,E_core,E_elas,E_step, W_stress, W_free_energy, stress_mean, path_state)
            s_mean = np.mean(latt_state)
            s_square_mean = np.mean(latt_state**2)
            h_mean = np.mean(latt_height)
            h_square_mean = np.mean(latt_height**2)
            write_s_z_average_to_txt(i, s_mean, s_square_mean, h_mean, h_square_mean, path_state)

        # write state
        if np.mod(i, dump_interval) == 0 and i != 0:
            np.savetxt(path_state+f'/s_{i}.txt', latt_state, fmt='%.1f')
            np.savetxt(path_state+f'/z_{i}.txt', latt_height, fmt='%.1f')

        # recalculate stress field to eliminate accumulated error
        if np.mod(i, recalc_stress_step) == 0 and i > 0:
            print(f"Recalculate stress field at Step {i}!")
            latt_stress = compute_stress_field(latt_state, stress_kernel_center, a_dsc)
            E_core = compute_core_energy(latt_state, nblist_mat, a_dsc, nu, zeta)
            E_step = compute_step_energy(gamma, a_dsc, latt_height, nblist_mat)
            E_elas = compute_elas_energy(a_dsc, latt_state, latt_stress)
            E_total = E_core + E_step + E_elas

        # nblist_mat, nblist_arr
        rand_site = (np.random.randint(N), np.random.randint(N))

        rand_site_neighbor = ((nblist_arr[0][rand_site[0]], rand_site[1]), # I-1, J
                              (nblist_arr[1][rand_site[0]], rand_site[1]), # I+1. J
                              (rand_site[0], nblist_arr[2][rand_site[1]]), # I, J-1
                              (rand_site[0], nblist_arr[3][rand_site[1]])) # I, J+1

        state_change = np.random.choice([-1, 1])*np.array(random.choice(mode_list))

        # compute change in energy
        core_energy_change = compute_core_energy_change(latt_state, rand_site, state_change[0], rand_site_neighbor, a_dsc, nu, zeta)
        elas_energy_change = compute_elastic_energy_change(latt_stress, rand_site, state_change[0], stress_kernel, a_dsc)
        step_energy_change = compute_step_energy_change(latt_height, state_change[1],rand_site, rand_site_neighbor, a_dsc, gamma)
        ext_stress_work  = -a_dsc * tau_ext * state_change[0]
        free_energy_work = -a_dsc * phi_ext * state_change[1]
        enthalpy_change = core_energy_change + elas_energy_change + step_energy_change + ext_stress_work + free_energy_work

        # accept or reject
        if simulation_type == 'mmc':
            if enthalpy_change >= 0:
                if np.random.rand() >= np.exp(-enthalpy_change / temperature):
                    continue
        elif simulation_type == 'gmc':
            if np.random.rand() >= 0.5-0.5*np.tanh(enthalpy_change/ (2*temperature)):
                continue
        # accept
        latt_state[rand_site] += state_change[0]
        latt_stress += 2 * a_dsc/N  * state_change[0] * stress_kernel_shift(stress_kernel, rand_site)
        latt_height[rand_site] += state_change[1]

        # calc energy and w_u
        E_total += core_energy_change + elas_energy_change + step_energy_change
        E_core += core_energy_change
        E_elas += elas_energy_change
        E_step += step_energy_change
        W_stress += ext_stress_work
        W_free_energy += free_energy_work

def kmc(latt_state_init, latt_stress_init, latt_height_init, stress_kernel, a_dsc, gamma, nblist_mat, nblist_arr, temperature, tau_ext, maxiter, nu, zeta, recalc_stress_step, plot_state_step, mode_list,E_total, E_core,E_elas,E_step, dump_interval, v,Q, path_state=None, phi_ext=0):
    '''kinetic monte carlo'''
    latt_state = latt_state_init
    latt_stress = latt_stress_init
    latt_height = latt_height_init
    N = latt_state.shape[0]
    mode_num = len(mode_list)
    stress_kernel_center = stress_kernel_shift(stress_kernel, (int(np.ceil(N/2-1)), int(np.ceil(N/2-1))))

    event_num = mode_num * 2 * N * N
    event_index_1d = np.arange(0, event_num, 1, dtype=int)
    event_index_4d = event_index_1d.reshape(mode_num, 2, N ,N)
    pre_exp_factor = np.exp(-Q/temperature)

    # create a txt file for writing E_total and w_u
    os.system(f'rm {path_state}/quantities.txt')
    os.system(f'rm {path_state}/s_z.txt')
    Path(f'{path_state}/quantities.txt').touch()
    Path(f'{path_state}/s_z.txt').touch()

    # nblist_mat, nblist_arr
    site_index_x = np.linspace(0, N-1, N, endpoint=True, dtype=int)
    site_index_y = np.linspace(0, N-1, N, endpoint=True, dtype=int)
    X,Y = np.meshgrid(site_index_x, site_index_y, indexing='xy')
    X = X.T
    Y = Y.T

    site_neighbor = ((nblist_arr[0][X], Y), # I-1, J
                     (nblist_arr[1][X], Y), # I+1. J
                     (X, nblist_arr[2][Y]), # I, J-1
                     (X, nblist_arr[3][Y])) # I, J+1
    time = 0
    W_stress = 0
    W_free_energy = 0
    enthalpy_4darray = np.zeros(shape=(mode_num, 2, N, N))
    core_eng_4darray = np.zeros(shape=(mode_num, 2, N, N))
    elas_eng_4darray = np.zeros(shape=(mode_num, 2, N, N))
    step_eng_4darray = np.zeros(shape=(mode_num, 2, N, N))
    for i in range(maxiter):
        # plot state every several steps
        if np.mod(i, plot_state_step) == 0:
            plot_state_stress(latt_height, latt_state, latt_stress, tau_ext, i, path_state)

        # write E_total
        if np.mod(i, 1) == 0:
            stress_mean = np.mean(latt_stress)
            write_total_energy_wu_to_txt(time, E_total,E_core,E_elas,E_step,W_stress, W_free_energy, stress_mean, path_state)
            s_mean = np.mean(latt_state)
            s_square_mean = np.mean(latt_state**2)
            h_mean = np.mean(latt_height)
            h_square_mean = np.mean(latt_height**2)
            write_s_z_average_to_txt(time, s_mean, s_square_mean, h_mean, h_square_mean, path_state)

        # write state
        if np.mod(i, dump_interval) == 0 and i != 0:
            np.savetxt(path_state+f'/s_{i}.txt', latt_state, fmt='%.1f')
            np.savetxt(path_state+f'/z_{i}.txt', latt_height, fmt='%.1f')

        # recalculate stress field to eliminate accumulated error
        if np.mod(i, recalc_stress_step) == 0 and i > 0:
            print(f"Recalculate stress field at Step {i}!")
            latt_stress = compute_stress_field(latt_state, stress_kernel_center, a_dsc)
            E_core = compute_core_energy(latt_state, nblist_mat, a_dsc, nu, zeta)
            E_step = compute_step_energy(gamma, a_dsc, latt_height, nblist_mat)
            E_elas = compute_elas_energy(a_dsc, latt_state, latt_stress)
            E_total = E_core + E_step + E_elas

        # compute change in energy
        for j in range(mode_num):
            mode = mode_list[j]
            b = mode[0]
            h = mode[1]
            for (k,), sign in np.ndenumerate([1,-1]):
                if i == 0:
                    # at first step, calculate core and step energy change for all events
                    core_eng_4darray[j,k,:,:] = compute_core_energy_change(latt_state, (X,Y), sign*b, site_neighbor, a_dsc, nu, zeta)
                    step_eng_4darray[j,k,:,:] = compute_step_energy_change(latt_height, sign*h, (X,Y), site_neighbor, a_dsc, gamma)
                else:
                    # after first step, update the energy change at selected site and its neighbours
                    update_site = ((select_i, select_j),    # find this 5 sites
                                   (nblist_arr[0][select_i], select_j), # I-1, J
                                   (nblist_arr[1][select_i], select_j), # I+1. J
                                   (select_i, nblist_arr[2][select_j]), # I, J-1
                                   (select_i, nblist_arr[3][select_j])) # I, J+1
                    for (index_i,index_j) in update_site:   # find their neighbours and calculated the energy change one by one
                        site_neighbor_ij = ((nblist_arr[0][index_i],index_j), # I-1, J
                                            (nblist_arr[1][index_i], index_j), # I+1. J
                                            (index_i, nblist_arr[2][index_j]), # I, J-1
                                            (index_i, nblist_arr[3][index_j])) # I, J+1
                        core_eng_4darray[j,k,index_i,index_j] = compute_core_energy_change(latt_state, (index_i, index_j), sign*b, site_neighbor_ij, a_dsc, nu, zeta)
                        step_eng_4darray[j,k,index_i,index_j] = compute_step_energy_change(latt_height, sign*h, (index_i, index_j), site_neighbor_ij, a_dsc, gamma)
                # elastic energy is global
                elas_eng_4darray[j,k,:,:] = compute_elastic_energy_change(latt_stress, (X,Y), sign*b, stress_kernel, a_dsc)
                ext_stress_work  = -a_dsc * tau_ext * sign*b
                free_energy_work = -a_dsc * phi_ext * sign*h
                enthalpy_4darray[j,k,:,:] = core_eng_4darray[j,k,:,:] + elas_eng_4darray[j,k,:,:] + \
                                            step_eng_4darray[j,k,:,:] + ext_stress_work + free_energy_work

        
        enthalpy_1darray = enthalpy_4darray.reshape(-1,)
        frequency_1darray = pre_exp_factor * np.exp(-enthalpy_1darray/(2*temperature)) 
        frequency_sequence = np.sort(frequency_1darray)[::-1]
        frequency_index = np.argsort(frequency_1darray)[::-1]
        sum = np.sum(frequency_1darray)
        event_prob = frequency_sequence/sum
        time += -1/sum * np.log(random.random())

        # generate a random number in [0,1)
        random_num = random.random()
        prob = 0
        for l in range(event_num):
            prob += event_prob[l]
            if random_num < prob:
                #select_1d_index = l
                break
        select_1d_index = frequency_index[l]
        select_4d_index = tuple(np.argwhere(select_1d_index==event_index_4d)[0])
        
        mode_index, mode_direction_index, select_i, select_j = select_4d_index
        mode_direction = [1,-1][mode_direction_index]
        delta_s = mode_direction*mode_list[mode_index][0]
        delta_z = mode_direction*mode_list[mode_index][1]
        # accept
        latt_state[(select_i, select_j)] += delta_s
        latt_stress += 2 * a_dsc/N  * delta_s * stress_kernel_shift(stress_kernel, (select_i, select_j))
        latt_height[(select_i, select_j)] += delta_z

        # calc energy and w_u
        E_total += enthalpy_4darray[select_4d_index]
        E_core += core_eng_4darray[select_4d_index]
        E_elas += elas_eng_4darray[select_4d_index]
        E_step += step_eng_4darray[select_4d_index]
        W_stress      += -a_dsc * tau_ext * delta_s
        W_free_energy += -a_dsc * phi_ext * delta_z


