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
disl_dipole = case['disl_dipole']
latt_dim = (num_points, num_points)

def write_file(T, path, simulation_type, tau_ext, phi_ext,disl_dipole=False,delta_over_N=0.1):
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
            f.write(f'python main.py {T:.3f} {tau_ext:.3f} {phi_ext:.3f}\n')
        else:
            f.write(f'python main.py {T:.3f} {tau_ext:.3f} {phi_ext:.3f} {delta_over_N}\n')

def sbatch_job(T, tau_ext_all, phi_ext_all,disl_dipole=False):
    T = np.array(T)
    tau_ext_all = np.array(tau_ext_all)
    phi_ext_all = np.array(phi_ext_all)
    delta_over_N = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for k in range(phi_ext_all.shape[0]):
        phi_ext = phi_ext_all[k]
        for j in range(tau_ext_all.shape[0]):
            tau_ext = tau_ext_all[j]
            for i in range(T.shape[0]):
                if disl_dipole==False:
                    path_state = output_path(num_points, kpoints_max, nu, \
                        zeta, a_dsc, gamma, mode_list,T[i], tau_ext, phi_ext, simulation_type, mkdir=True)
                    write_file(T[i], path_state, simulation_type, tau_ext, phi_ext)
                    path = os.getcwd()
                    os.chdir(path_state)
                    #os.system(f'sbatch job.sh')
                    os.chdir(path)
                else:
                    for l in range(len(delta_over_N)):
                        path_state = output_path(num_points, kpoints_max, nu, \
                          zeta, a_dsc, gamma, mode_list,T[i], tau_ext, phi_ext, simulation_type, mkdir=True,\
                          disl_dipole=True, delta_over_N = delta_over_N[l])
                        write_file(T[i], path_state, simulation_type, tau_ext, phi_ext,disl_dipole=True,\
                                delta_over_N=delta_over_N[l])
                        path = os.getcwd()
                        os.chdir(path_state)
                        os.system(f'sbatch job.sh')
                        os.chdir(path)

if __name__ == '__main__':
    T = np.arange(0., 2.0, 0.2)[1:]
    #T =  np.arange(0.01, 0.065, 0.005)
    #T = np.arange(0.2, 3.2, 0.2)
    #T = np.arange(0.1,0.8,0.1)
    T = np.arange(1,10,1) #[1,-1] gamma=1.4
    T = np.arange(4,7.1,0.1)
    T = np.arange(1,20,2)
    T = np.arange(1, 10, 0.25)
    #T = np.arange(2,22,2)
    #T = np.arange(0.2,1.2,0.2)
    #T = np.arange(1,10,1)
    #T = np.array([3.25,3.75])
    #T = np.arange(0,1.2,0.1)[1:]
    #T = np.arange(1,2.5,0.1)
    #T = np.arange(0.05,0.55,0.05)
    #T = np.arange(0.1,1.9,0.2)
    T = np.arange(0.2,2,0.2)
    tau_ext = np.array([0]) 
    #phi_ext = np.array([0.001,0.002,0.003,0.004])
    phi_ext = np.array([0])

    #T = [1.0]
    #tau = [0]
    #phi_ext = [0]
    #T = np.array([0.2,0.4,0.6])
    #T = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
    #T = np.arange(0, 0.08, 0.01)[1:]
    #T = np.array([0.01,0.03])
    #T = np.array([0.2])
    #T = np.arange(0., 0.2, 0.02)[1:]
    #T = np.arange(4, 8, 0.4)
    sbatch_job(T, tau_ext, phi_ext, disl_dipole=disl_dipole)
