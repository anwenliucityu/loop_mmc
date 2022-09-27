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

def mmc(latt_state_init, latt_stress_init, latt_height_init, stress_kernel, a_dsc, gamma, nblist_mat, nblist_arr, temperature, tau_ext, maxiter, nu, zeta, recalc_stress_step, plot_state_step, mode_list,E_total, E_core,E_elas,E_step, dump_interval, write_step, path_state=None):
    '''metropolis monte carlo'''
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
    
    for i in range(maxiter+1):
        # plot state every several steps
        if np.mod(i, plot_state_step) == 0:
            plot_state_stress(latt_height, latt_state, latt_stress, tau_ext, i, path_state)

        # write E_total
        if np.mod(i, write_energy) == 0:
            # calc wz and wu
            #w_u = np.sqrt(s_square_mean - s_mean**2)
            #w_z = np.sqrt(h_square_mean - h_mean**2)
            stress_mean = np.mean(latt_stress)
            write_total_energy_wu_to_txt(i, E_total,E_core,E_elas,E_step, stress_mean, path_state)

        # write s and z average and square average
        if np.mod(i, write_step) == 0:
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
            latt_stress = compute_stress_field(latt_state, tau_ext, stress_kernel_center, a_dsc)
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
        energy_change = core_energy_change + elas_energy_change + step_energy_change

        # accept or reject
        if energy_change >= 0:
            if np.random.rand() >= np.exp(-energy_change / temperature):
                continue
        # accept
        latt_state[rand_site] += state_change[0]
        latt_stress += 2 * a_dsc/N  * state_change[0] * stress_kernel_shift(stress_kernel, rand_site)
        latt_height[rand_site] += state_change[1]

        # calc energy and w_u
        E_total += energy_change
        E_core += core_energy_change
        E_elas += elas_energy_change
        E_step += step_energy_change

    return (latt_state, latt_stress)
