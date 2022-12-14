'''write or read a file'''
import numpy as np
from pathlib import Path
import os

def write_stress_to_txt(stress, component_ind, proj_plane):
    '''write a stress numpy array to a txt file'''

    mat = np.matrix(stress)
    with open(f"stress_{component_ind}_{proj_plane}.txt", 'w') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%.20f')

def read_stress_from_txt(component_ind, proj_plane):
    '''read a stress numpy array from a txt file'''
    with open(f"stress_{component_ind}_{proj_plane}.txt", 'r') as file:
        return np.array([[float(num) for num in line.split(' ')] for line in file])

def write_stress_kernel_to_txt(stress_kernel, num_points, kpoints_max, nu, path_stress_kernel=None):
    '''write stress kernel Xi to a txt file'''
    if not path_stress_kernel:
        path_stress_kernel = "./stress_kernel_origin"
    # create a directory if it does not exist
    Path(path_stress_kernel).mkdir(parents=False, exist_ok=True)
    mat = np.matrix(stress_kernel)
    with open(path_stress_kernel + \
    f"/stress_kernel_origin_N{num_points}_k{kpoints_max}_nu{nu}.txt", 'w') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%.20f')

def read_stress_kernel_from_txt(num_points, kpoints_max, nu, path_stress_kernel):
    '''read a stress kernel Xi numpy array from a txt file'''
    with open(path_stress_kernel + \
    f"/stress_kernel_origin_N{num_points}_k{kpoints_max}_nu{nu}.txt", 'r') as file:
        return np.array([[float(num) for num in line.split(' ')] for line in file])

def write_total_energy_wu_to_txt(i, E_total,E_core, E_elas,E_step,W_stress, W_free_energy, stress_mean, path_state):
    with open(f'{path_state}/quantities.txt', 'a') as f:
        if type(i) == int:
            f.write(f"{i} {E_total} {E_core} {E_elas} {E_step} {W_stress:>6.4e} {W_free_energy:>6.4e} {stress_mean:>6.4e}\n")
        else:
            f.write(f"{i:>6.4e} {E_total} {E_core} {E_elas} {E_step} {W_stress:>6.4e} {W_free_energy:>6.4e} {stress_mean:>6.4e}\n")


def write_s_z_average_to_txt(i, s_mean, s_square_mean, h_mean, h_square_mean, w_s, w_z, path_state):
    with open(f'{path_state}/s_z.txt', 'a') as f:
        if type(i)==int:
            f.write(f"{i} {s_mean:>6.4e} {s_square_mean:>6.4e} {h_mean:>6.4e} {h_square_mean:>6.4e} {w_s} {w_z}\n")
        else:
            f.write(f"{i:>6.4e} {s_mean:>6.4e} {s_square_mean:>6.4e} {h_mean:>6.4e} {h_square_mean:>6.4e} {w_s} {w_z}\n")

def output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, temperature, tau_ext, psi_ext, simulation_type, mkdir=False, disl_dipole=False, delta_over_N=0,screen='screen'):
    mode_name = ''
    for i in range(len(mode_list)):
        for j, string in enumerate(['b','h']):
            mode_name += string
            mode_name += str(mode_list[i][j])
        if i!=len(mode_list)-1:
            mode_name+='_'
    scratch_path = f'/gauss12/home/cityu/anwenliu/scratch/loop/{simulation_type}/'
    if disl_dipole==True:
        scratch_path += f'disl_dipole/{screen}/'
    elif tau_ext!=0:
        scratch_path += 'stress_driven/'
    elif psi_ext!=0:
        scratch_path += 'free_energy_driven/'
    else:
        scratch_path += f'thermo_equilibrium/'
    path_state = scratch_path + f'N{num_points}_k{kpoints_max}/nu{nu}_zt{zeta}_adsc{a_dsc}_gm{gamma}/{mode_name}/stress{tau_ext:.3f}_psi{psi_ext:.3f}/T{temperature:.3f}'
    if disl_dipole==True:
        path_state += f'/delta_{delta_over_N}/'
    if os.path.exists(path_state)==False and mkdir==True:
        os.makedirs(path_state)
    #if os.path.exists(path_state)==False and mkdir==False:
        #path_state = scratch_path + f'N{num_points}_k{kpoints_max}/nu{nu}_zt{zeta}_adsc{a_dsc}_gm{gamma}/{mode_name}/stress{tau_ext:.2f}_phi{phi_ext:.2f}/T{temperature:.2f}'
    if mkdir==True:
        print(f'working dir = {path_state}')
    return path_state



