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
disl_dipole = case['disl_dipole'][0]
screen = case['disl_dipole'][1]
latt_dim = (num_points, num_points)

def write_file(T, path, simulation_type, tau_ext, psi_ext,disl_dipole=False,delta_over_N=0.1):
    with open(f'{path}/job.sh', 'w') as f:
        f.write(f'#!/usr/bin/env bash \n'
                f'#SBATCH --job-name="T{T:.3f}"\n'
                f'#SBATCH --partition=xlong\n'
                  f'#SBATCH --ntasks-per-core=1\n'
                  f'#SBATCH --ntasks=1\n'
                  f'#SBATCH --mem=2G \n'
                  f'#SBATCH --exclude=gauss[1,6,16],gauss\n' # 1 is slow
                  f'#SBATCH --nodes=1\n'
                  f'#SBATCH --cpus-per-task=1\n\n'

                  f'cd {os.getcwd()} \n')
        if disl_dipole==False:
            f.write(f'python main.py {T:.3f} {tau_ext:.3f} {psi_ext:.3f}\n')
        else:
            f.write(f'python main.py {T:.3f} {tau_ext:.3f} {psi_ext:.3f} {delta_over_N}\n')

def sbatch_job(T, tau_ext_all, psi_ext_all,disl_dipole=False, screen='screen',copy_restart=False,restart_num=0):
    T = np.array(T)
    #tau_ext_all = np.array(tau_ext_all)
    psi_ext_all = np.array(psi_ext_all)
    delta_over_N = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    #delta_over_N = [0.1,0.9]
    for i in range(T.shape[0]):
        for k in range(psi_ext_all.shape[0]):
            psi_ext = psi_ext_all[k]
            for j in range(tau_ext_all.shape[0]):
                tau_ext = tau_ext_all[j]
            #for m in range(len(tau_ext_all[i])):
                #tau_ext = tau_ext_all[i][m]
                if disl_dipole==False:
                    path_state = output_path(num_points, kpoints_max, nu, \
                        zeta, a_dsc, gamma, mode_list,T[i], tau_ext, psi_ext, simulation_type, mkdir=True)
                    write_file(T[i], path_state, simulation_type, tau_ext, psi_ext)
                    if copy_restart==True:
                        copy_path = output_path(num_points, kpoints_max, nu, \
                            zeta, a_dsc, gamma, mode_list,T[i], 0, 0, 'kmc',)
                        os.system(f'cp {copy_path}/s_{restart_num}.txt {path_state}')
                        os.system(f'cp {copy_path}/z_{restart_num}.txt {path_state}')
                    path = os.getcwd()
                    os.chdir(path_state)
                    os.system(f'sbatch job.sh')
                    os.chdir(path)
                else:
                    for l in range(len(delta_over_N)):
                        path_state = output_path(num_points, kpoints_max, nu, \
                          zeta, a_dsc, gamma, mode_list,T[i], tau_ext, psi_ext, simulation_type, mkdir=True,\
                          disl_dipole=True, delta_over_N = delta_over_N[l], screen=screen)
                        write_file(T[i], path_state, simulation_type, tau_ext, psi_ext,disl_dipole=True,\
                                delta_over_N=delta_over_N[l])
                        path = os.getcwd()
                        os.chdir(path_state)
                        os.system(f'sbatch job.sh')
                        os.chdir(path)

if __name__ == '__main__':
    copy_restart= case['read_restart']
    restart_num = int(case['start_points'][0])
    T = np.arange(0., 2.0, 0.2)[1:]

    T = np.array([0.05,0.10,0.15,0.20,0.25,0.27,0.28,0.29,0.30,0.31,0.32,0.35,0.40,0.45])
    T = np.array([0.26,0.278,0.279,0.281])
    T = np.arange(0.1,1.7,0.1)
    T = np.arange(0.5,5,0.25)
    T = np.array([0.125,0.175,0.225,0.285,0.31,0.33])
    #T = np.array([0.15,0.25,0.27,0.29,0.31,0.33,0.35])
    #T = np.arange(4,15,1)
    #T = np.array(8.5,9.5)
    #T = np.array([0.28,0.30])
    T = np.array([0.9,1.2])
    T = np.array([0.8,1.1])
    T = np.array([4,6.5])
    #T = np.arange(4,8,0.5)
    #T = np.array([22,28])
    T = np.arange(0.1,1.7,0.1)
    #tau_ext = np.arange(0.2,0.7,0.1)
    #tau_ext = np.array([0.1])
    T = np.array([0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.26,0.27,0.275,0.28,0.285,0.29,0.30,0.31,0.33,0.35,0.40])
    T = np.array([0.7,1.2])
    T = np.array([2,3])
    T = np.array([4])
    T = np.arange(0.25,4.25,0.25)
    #T = np.arange(1,10,1)
    T = np.array([2,4])
    T = np.arange(1,5,0.5)
    T = np.arange(0.2,2.2,0.2)
    T = np.arange(4,12,1)
    T = np.arange(1,8,0.5)
    T = np.arange(0.3,3.3,0.3)
    T = np.arange(0.5,4,0.5)
    T = np.arange(0.2,3.0,0.2)
    T = np.arange(0.1,1,0.1)
    T = np.arange(1,4.2,0.2)
    T = np.arange(1,8,0.25)
    T = np.arange(0.2,0.8,0.04)
    T = np.arange(0.2,2.2,0.05)
    #T = np.array([0.58])
    T = np.array([1,1.5,2,2.5,3,3.5,4,4.5])
    T = np.array([2.5])
    T = np.array([3.0])
    T = np.array([1,1.5,2,2.5,3,3.5,4,4.5])
    #T = np.array([5])
    T = np.array([5])
    T = np.arange(3,9,0.25)
    T = np.arange(0.4,1.9,0.1)
    T = np.arange(1,6,0.25)
    #T = np.arange(5,14,1)
    T = np.arange(1,9,0.3)
    T = np.arange(0.1,3,0.2)
    T = np.arange(3,5,0.2)
    T = np.arange(2,15,1)
    T = np.array([4,5,6,7,8,9])
    T = np.arange(0.2,3,0.2)
    T = np.arange(3,6.2,0.2)
    T = np.arange(1,5.2,0.2)
    T = np.arange(0.2,1,0.2)
    T = np.arange(0.1,1.6,0.1)
    T = np.array([1.5])
    T = np.array([0.1,0.2,0.3,0.4])
    T = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45])
    #T = np.arange(3,6,0.3)
    T = np.arange(0.1,2,0.1)
    #psi_ext = np.array([0.001,0.002,0.003,0.004,0.005])
    #psi_ext = np.array([0.01,0.02,0.03,0.04,0.05])
    #psi_ext = np.array([0.2,0.4,0.6,0.8])
    T = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35])
    T = np.array([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4])

    T = np.arange(0.075,0.5,0.05)
    T = np.arange(1,3,0.25)
    T = np.arange(0.2,3.2,0.2)
    T = np.arange(0.05,0.5,0.05)
    T = np.arange(0.4,1.9,0.1)
    T = np.arange(1,4.2,0.2)
    T = np.arange(1,8,0.25)
    T = np.array([2.4,2.8])
    T = np.arange(0.2,2.2,0.05)
    T = np.array([1.5,2.5])
    T = np.arange(1,6,0.3)
    T = np.arange(0.1,2,0.2)
    T = np.array([4.9,4.6])
    #T = np.arange(0.1,1.5,0.05)
    tau_ext = np.array([0]) 
    #psi_ext = np.array([0.02,0.04,0.06,0.08,0.10])

    #tau_ext = np.array([0.12,0.14,0.16,0.18])
    psi_ext = np.array([0])

    '''
    tau_ext = [[1.3,1.4,1.5,1.6,1.7],
       [1.2,1.3,1.4,1.5,1.6],
       [1.1,1.2,1.3,1.4,1.5],
       [1.0,1.1,1.2,1.3,1.4],
       [0.9,1.0,1.1,1.2,1.3],
       [0.8,0.9,1.0,1.1,1.2],
       [0.7,0.8,0.9,1.0],
       [0.7,0.8,0.9],
       [0.7]]  
    T = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    '''
    sbatch_job(T, tau_ext, psi_ext, disl_dipole=disl_dipole, screen=screen, copy_restart=copy_restart, restart_num=restart_num)
