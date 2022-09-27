'''generate stress kernel Xi and write it to a txt file'''
import numpy as np
import matplotlib.pyplot as plt

from param import case
nu = case['nu']
num_points = case['num_points']
kpoints_max = case['kpoints_max']
n_max = m_max = kpoints_max
path_stress_kernel = case['path_stress_kernel']
plot_lim = case['plot_lim']

ind_I_arr = np.arange(0, num_points)
ind_J_arr = np.arange(0, num_points)
ind_I_mat, ind_J_mat = np.meshgrid(ind_I_arr, ind_J_arr, indexing='ij')

# pre-calculation
n_list = np.arange(1, n_max+1, 1)
m_list = np.arange(1, m_max+1, 1)
sinc_n = np.sinc(n_list / (n_max + 1))
sinc_m = np.sinc(m_list / (m_max + 1))
sin_n = np.sin(np.pi * n_list / num_points)
sin_m = np.sin(np.pi * m_list / num_points)
nsq = n_list**2
msq = m_list**2
Nsq, Msq = np.meshgrid(nsq, msq)
sqrt_nsq_msq = np.sqrt(Nsq + Msq)

sum1 = 0.0
for i in range(n_max):
    n = n_list[i]
    lanczos_n = sinc_n[i]
    for j in range(m_max):
        m = m_list[j]
        lanczos_m = sinc_m[j]
        sum1 += lanczos_n * lanczos_m * \
        (1/m) * ((1-nu)*sqrt_nsq_msq[i,j]/n + nu*n/sqrt_nsq_msq[i,j]) * \
        sin_n[i] * sin_m[j] * \
        np.cos(2*np.pi*n*ind_I_mat/num_points) * \
        np.cos(2*np.pi*m*ind_J_mat/num_points)
sum1 *= 2
sum2 = 0.0
for j in range(m_max):
    m = m_list[j]
    lanczos_m = sinc_m[j]
    sum2 += lanczos_m * sin_m[j] * \
    np.cos(2*np.pi*m*ind_J_mat/num_points)
sum2 *= (1 - nu) * np.pi / num_points
sum3 = 0.0
for i in range(n_max):
    n = n_list[i]
    lanczos_n = sinc_n[i]
    sum3 += lanczos_n * sin_n[i] * \
    np.cos(2*np.pi*n*ind_I_mat/num_points)
sum3 *= np.pi / num_points
stress_kernel = - sum1 - sum2 - sum3

from file_handling import write_stress_kernel_to_txt
write_stress_kernel_to_txt(stress_kernel, num_points, kpoints_max, path_stress_kernel)

# plot and check
sizeOfFont = 16
fontProperties = {
'family' : 'serif',
'serif' : ['Computer Modern Serif'],
'weight' : 'normal',
'size' : sizeOfFont
}
plt.rc('text', usetex=True)
plt.rc('font', **fontProperties)

fig, ax = plt.subplots()
ax.pcolor(stress_kernel.T, vmin=-plot_lim, vmax=plot_lim, cmap='rainbow', shading='auto', edgecolor='k', linewidth=0.2)
ax.set_xlim(0, num_points)
ax.set_ylim(0, num_points)
ax.set_xticks(np.linspace(0, num_points, 11, endpoint=True))
ax.set_yticks(np.linspace(0, num_points, 11, endpoint=True))
ax.set_xlabel(r'Index $I$', fontsize=16)
ax.set_ylabel(r'Index $J$', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.set_title(rf"$\Xi_{{I,J;0,0}}$ for $N = {num_points}$, $k_{{\rm max}} = {kpoints_max}$", fontsize=16)
ax.set_aspect('equal', adjustable='box')
fig.set_size_inches(6,5)
plt.tight_layout()
plt.savefig(path_stress_kernel +'/plot.png')
plt.close()
#plt.show()
