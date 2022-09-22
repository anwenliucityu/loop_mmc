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

def write_stress_kernel_to_txt(stress_kernel, num_points, kpoints_max, path_stress_kernel=None):
    '''write stress kernel Xi to a txt file'''
    if not path_stress_kernel:
        path_stress_kernel = "./stress_kernel_origin"
    # create a directory if it does not exist
    Path(path_stress_kernel).mkdir(parents=False, exist_ok=True)
    mat = np.matrix(stress_kernel)
    with open(path_stress_kernel + \
    f"/stress_kernel_origin_N{num_points}_k{kpoints_max}.txt", 'w') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%.20f')

def read_stress_kernel_from_txt(num_points, kpoints_max, path_stress_kernel):
    '''read a stress kernel Xi numpy array from a txt file'''
    with open(path_stress_kernel + \
    f"/stress_kernel_origin_N{num_points}_k{kpoints_max}.txt", 'r') as file:
        return np.array([[float(num) for num in line.split(' ')] for line in file])

def write_total_energy_wu_to_txt(i, E_total,E_core, E_elas,E_step, wu, wz, path_state):
    with open(f'{path_state}/E_total_wu.txt', 'a') as f:
        f.write(f"{i} {E_total:>10.8f} {E_core:>10.8f} {E_elas:>10.8f} {E_step:>10.8f} {wu:>10.8f} {wz:>10.8f}\n")

def output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, temperature, tau_ext):
    mode_name = ''
    for i in range(len(mode_list)):
        for j in range(2):
            if j==0:
                mode_name+='s'
            if j==1:
                mode_name+='z'
            mode_name += str(mode_list[i][j])
            if i!=len(mode_list)-1:
                mode_name+='_'
    scratch_path = '/gauss12/home/cityu/anwenliu/scratch/loop_mc/'
    path_state = scratch_path + f'N{num_points}_k{kpoints_max}/nu{nu}_zt{zeta}_adsc{a_dsc}_gm{gamma}/{mode_name}/T{temperature:.2f}_tau{tau_ext}'
    if os.path.exists(path_state)==False:
        os.makedirs(path_state)
    return path_state



