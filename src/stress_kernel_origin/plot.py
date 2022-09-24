import numpy as np
import matplotlib.pyplot as plt

num_points = 800
kpoints_max = 400
stress_kernel_name = f'stress_kernel_origin_N{num_points}_k{kpoints_max}.txt'
stress_kernel = np.loadtxt(stress_kernel_name)
plot_lim=0.2
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
ax.pcolor(stress_kernel, vmin=-plot_lim, vmax=plot_lim, cmap='rainbow', shading='auto',    edgecolor='k', linewidth=0.2)
ax.set_xlim(0, num_points)
ax.set_ylim(0, num_points)
ax.set_xticks(np.linspace(0, num_points, 11, endpoint=True))
ax.set_yticks(np.linspace(0, num_points, 11, endpoint=True))
ax.set_xlabel(r'Index $I$', fontsize=16)
ax.set_ylabel(r'Index $J$', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.set_title(rf"$\Xi_{{I,J;0,0}}$ for $N = {num_points}$, $k_{{\rm max}} =                 {kpoints_max}$", fontsize=16)
ax.set_aspect('equal', adjustable='box')
fig.set_size_inches(6,5)
plt.tight_layout()
#plt.close()
#plt.savefig(f'N{num_points}_k{kpoints_max}.pdf')
plt.show()     
