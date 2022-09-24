import os
import numpy as np
from param import case
from file_handling import output_path
tau_ext = case['tau_ext']
nu = case['nu']
zeta = case['zeta']
a_dsc = case['a_dsc']
gamma = case['gamma']
mode_list = case['mode_list']
num_points = case['num_points']
kpoints_max = case['kpoints_max']
latt_dim = (num_points, num_points)

def write_file(T, path):
    with open(f'{path}/job.sh', 'w') as f:
        f.write(f'#!/usr/bin/env bash \n'
                f'#SBATCH --job-name="T{T:.2f}"\n'
                f'#SBATCH --partition=xlong\n'
                  f'#SBATCH --ntasks-per-core=1\n'
                  f'#SBATCH --ntasks=1\n'
                  f'#SBATCH --mem=2G \n'
                  f'#SBATCH --exclude=gauss[6,16],gauss\n'
                  f'#SBATCH --nodes=1\n'
                  f'#SBATCH --cpus-per-task=1\n\n'

                  f'cd {os.getcwd()} \n'
                  f'python main.py {T:.2f} \n')

def sbatch_job(T):
    for i in range(T.shape[0]):
        path_state = output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, T[i], tau_ext, mkdir=True)
        write_file(T[i], path_state)
        path = os.getcwd()
        os.chdir(path_state)
        os.system(f'sbatch job.sh')
        os.chdir(path)


if __name__ == '__main__':
    T = np.arange(0., 2, 0.2)[1:]
    T = np.array([0.01,0.03])
    T = np.array([0.2])
    #T = np.arange(0., 0.2, 0.02)[1:]
    #T = np.arange(4, 8, 0.4)
    sbatch_job(T)
