import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as op
import matplotlib as mpl
import matplotlib
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
import matplotlib.pyplot as plt
from file_handling import output_path
from param import case
from analysis import calc_correlation, calc_inverse_viscosity

equlibrium_num = 400
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rc('axes', linewidth=1.)
def c():
    size=20
    #mpl.rc('mathtext', fontset='stixsans')
    mpl.rc('axes', linewidth=1.25)
    mpl.rc('xtick.major', pad=8)
    mpl.rc('font', size=size)          # controls default text sizes
    plt.rc('axes', titlesize=1.4*size)     # fontsize of the axes title
    plt.rc('axes', labelsize=size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels

    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=size)   # fontsize of the figure title
    #plt.grid(True, which='both')

    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
#c()

tau_ext = case['tau_ext']
nu = case['nu']
zeta = case['zeta']
a_dsc = case['a_dsc']
gamma = case['gamma']
mode_list = case['mode_list']
num_points = case['num_points']
kpoints_max = case['kpoints_max']
latt_dim = (num_points, num_points)
write_step = case['write_step']
delta_t = (2*len(mode_list)*num_points**2)**(-1)

T = np.arange(0,2,0.2)[1:]
T = np.array([0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8])
T = np.arange(0,2,0.2)[1:] #[[0,2.5],[1,-1]]
#T = np.arange(0,0.08,0.01)[1:]
#T = np.arange(0,0.2,0.02)[1:] #[0,-1]
#T = np.arange(0,4,0.4)[1:]  # [[0,2.5],[1,-1]]
#T = np.arange(0,4,0.2)[1:]  #[[1,0]]  and [[1,-1]]

C_list = []
C_core = []
C_elas = []
C_step = []
W_list = []
Z_list = []
labelsize=17
titlesize=18
#T = np.array([0.00015])
figsize_x, figsize_y = (2,7)
fig,ax = plt.subplots(figsize_x,figsize_y, figsize=(40,10))
H_list = []
S = 0
delta_T =  T[1]-T[0]
inverse_viscosity = []
mobility = []
for i in range(T.shape[0]):
    temperature = float(T[i])
    path_state = output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, T[i], tau_ext)
    dat = np.loadtxt(path_state+'/quantities.txt')[equlibrium_num:]
    #dat = np.loadtxt(path_state+'/E_total_wu.txt')[equlibrium_num:]
    step = dat[:,0]
    E = dat[:,1]
    E_core = dat[:,2]
    E_elas = dat[:,3]
    E_step = dat[:,4]
    u = dat[:,5]
    z = dat[:,6]
    s_mean = dat[:,7]
    h_mean = dat[:,9]
    s_square_mean = dat[:,8]
    h_square_mean = dat[:,10]

    C = (np.mean(E**2)-np.mean(E)**2)/(num_points**2*T[i]**2)
    if i==0:
        S += C/T[i]*delta_T/2 
    else:
        S += (C_list[-1]/T[i-1]+C/T[i])*num_points**2*delta_T/2
    H_list.append((np.mean(E)-T[i]*S)/num_points**2)
    C_list.append(C)
    core = (np.mean(E_core**2)-np.mean(E_core)**2)/(num_points**2*T[i]**2)
    C_core.append(core)
    elas = (np.mean(E_elas**2)-np.mean(E_elas)**2)/(num_points**2*T[i]**2)
    C_elas.append(elas)
    c_step = (np.mean(E_step**2)-np.mean(E_step)**2)/(num_points**2*T[i]**2)
    C_step.append(c_step)
    W = np.mean(u)
    W_list.append(W)
    Z = np.mean(z)
    Z_list.append(Z)
    ax[1][0].plot(step, E/num_points**2, '-', label=f'{round(T[i],2)}')
    ax[0][2].plot(step, u, '-', label=f'{round(T[i],2)}')
    ax[1][2].plot(step, z, '-', label=f'{round(T[i],2)}')
    ax[0][3].plot(step, s_mean, '-')
    ax[1][3].plot(step, h_mean, '-')
    ax[0][4].plot(step, s_square_mean, '-')
    ax[1][4].plot(step, h_square_mean, '-')

    # calculate v-v correlation
    v = a_dsc*(np.diff(s_mean[1:], n=1) + np.diff(s_mean[:-1], n=1))/(2*delta_t*write_step)
    h = a_dsc*(np.diff(h_mean[1:], n=1) + np.diff(h_mean[:-1], n=1))/(2*delta_t*write_step)
    tau_array_v , correlation_v = calc_correlation(v, write_step, delta_t)
    ax[0][5].plot(tau_array_v, correlation_v, 'o-')
    tau_array_h , correlation_h = calc_correlation(h, write_step, delta_t)
    ax[1][5].plot(tau_array_h, correlation_h, 'o-')

    integrate_vs = calc_inverse_viscosity(tau_array_v, correlation_v, T[i], num_points)
    integrate_vz = calc_inverse_viscosity(tau_array_h, correlation_h, T[i], num_points)
    inverse_viscosity.append(integrate_vs)
    mobility.append(integrate_vz)

ax[0,5].set_xlabel(r'$\Delta$step', fontsize=labelsize)
ax[0,5].set_ylabel(r'$\mathrm{cor}_v$', fontsize=labelsize)
ax[1,5].set_xlabel(r'$\Delta$step', fontsize=labelsize)
ax[1,5].set_ylabel(r'$\mathrm{cor}_h$', fontsize=labelsize)

ax[1][0].set_xlabel('step', fontsize=labelsize)
ax[1][0].set_ylabel(r'$E_\mathrm{tot}$ per pixel', fontsize=labelsize)
ax[0][2].set_xlabel('step', fontsize=labelsize)
ax[0][2].set_ylabel(r'$w_s$', fontsize=labelsize)
ax[1][2].set_xlabel('step', fontsize=labelsize)
ax[1][2].set_ylabel(r'$w_z$', fontsize=labelsize)

ax[0][3].set_xlabel('step', fontsize=labelsize)
ax[0][3].set_ylabel(r'$\overline{s}$', fontsize=labelsize)
ax[1][3].set_xlabel('step', fontsize=labelsize)
ax[1][3].set_ylabel(r'$\overline{h}$', fontsize=labelsize)
ax[0][4].set_xlabel('step', fontsize=labelsize)
ax[0][4].set_ylabel(r'$\overline{s^2}$', fontsize=labelsize)
ax[1][4].set_xlabel('step', fontsize=labelsize)
ax[1][4].set_ylabel(r'$\overline{h^2}$', fontsize=labelsize)

for i in range(figsize_x):
    for j in range(figsize_y):
        ax[i][j].yaxis.major.formatter._useMathText = True
        ax[i][j].xaxis.major.formatter._useMathText = True
        ax[i][j].ticklabel_format(style='sci', scilimits=(-1,2), axis='both')
        ax[i][j].tick_params(labelsize=labelsize-2)
        ax[i][j].xaxis.get_offset_text().set_fontsize(labelsize-1)
        ax[i][j].yaxis.get_offset_text().set_fontsize(labelsize-1)

ax[0][0].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][1].set_xlabel(r'$T$', fontsize=labelsize)
ax[1][1].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][0].set_ylabel(r'$C$', fontsize=labelsize)
ax[0][1].set_ylabel(r'<$w_s$>', fontsize=labelsize)
ax[1][1].set_ylabel(r'<$w_z$>', fontsize=labelsize)

ax[0][1].plot(T, np.array(W_list), 'o-',)
ax[0][0].plot(T, np.array(C_list), 'o-',label=r'$C_\mathrm{tot}$', markersize=7, markerfacecolor='none', mew=2)
ax[0][0].plot(T, np.array(C_core), '>-',label=r'$C_\mathrm{core}$')
ax[0][0].plot(T, np.array(C_elas), 's-',label=r'$C_\mathrm{elas}$')
ax[0][0].plot(T, np.array(C_step), '<-',label=r'$C_\mathrm{step}$')
ax[1][1].plot(T, np.array(Z_list), 'o-')
ax[0][0].legend(fancybox=False)
ax[0][2].legend(fancybox=False)


# H free energy
ax[0,6].plot(T, np.array(H_list), 'o-')
ax[0][6].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][6].set_ylabel(r'$F$ per pixel', fontsize=labelsize)

ax[1,6].plot(T, inverse_viscosity, 'o-', label = r'inverse_viscosity')
ax[1,6].plot(T, mobility, 'o-', label = r'mobility')
ax[1,6].set_ylabel(r'$d/\eta$', fontsize = labelsize)
ax[1,6].set_xlabel(r'$T$', fontsize = labelsize)
ax[1,6].legend(fancybox=False)

#ax[1,7].plot(1/T, np.log(inverse_viscosity), 'o-')
#ax[1,7].plot(1/T, np.log(mobility), 'o-')

plt.tight_layout()
plt.show()

