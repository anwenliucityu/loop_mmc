import os
import numpy as np
from param import case
from file_handling import output_path
#tau_ext = case['tau_ext']
nu = case['nu']
zeta = case['zeta']
a_dsc = case['a_dsc']
gamma = case['gamma']
mode_list = case['mode_list']
num_points = case['num_points']
kpoints_max = case['kpoints_max']
simulation_type = case['simulation_type']
latt_dim = (num_points, num_points)

def write_file(T, path, simulation_type, tau_ext, phi_ext):
    with open(f'{path}/job.sh', 'w') as f:
        f.write(f'#!/usr/bin/env bash \n'
                f'#SBATCH --job-name="{simulation_type}_{T:.2f}"\n'
                f'#SBATCH --partition=xlong\n'
                  f'#SBATCH --ntasks-per-core=1\n'
                  f'#SBATCH --ntasks=1\n'
                  f'#SBATCH --mem=2G \n'
                  f'#SBATCH --exclude=gauss[6,16],gauss\n'
                  f'#SBATCH --nodes=1\n'
                  f'#SBATCH --cpus-per-task=1\n\n'

                  f'cd {os.getcwd()} \n'
                  f'python main.py {T:.2f} {tau_ext:.2f} {phi_ext:.2f}\n')

def sbatch_job(T, tau_ext_all, phi_ext_all):
    T = np.array(T)
    tau_ext_all = np.array(tau_ext_all)
    phi_ext_all = np.array(phi_ext_all)
    for k in range(phi_ext_all.shape[0]):
        phi_ext = phi_ext_all[k]
        for j in range(tau_ext_all.shape[0]):
            tau_ext = tau_ext_all[j]
            for i in range(T.shape[0]):
                path_state = output_path(num_points, kpoints_max, nu, \
                        zeta, a_dsc, gamma, mode_list,T[i], tau_ext, phi_ext, simulation_type, mkdir=True)
                write_file(T[i], path_state, simulation_type, tau_ext, phi_ext)
                path = os.getcwd()
                os.chdir(path_state)
                os.system(f'sbatch job.sh')
                os.chdir(path)


if __name__ == '__main__':
    T = np.arange(0., 2.0, 0.2)[2:]
    tau_ext = np.array([0]) 
    phi_ext = np.array([0.1,0.2,0.3,0.4])

    T = [1.0]
    tau = [0]
    phi_ext = [0]
    #T = np.array([0.2,0.4,0.6])
    #T = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
    #T = np.arange(0, 0.08, 0.01)[1:]
    #T = np.array([0.01,0.03])
    #T = np.array([0.2])
    #T = np.arange(0., 0.2, 0.02)[1:]
    #T = np.arange(4, 8, 0.4)
    sbatch_job(T, tau_ext, phi_ext)
