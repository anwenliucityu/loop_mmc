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
from param import case_N200_k400 as case

equlibrium_num = 0
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

T = np.arange(0,2,0.2)[1:]
T = np.array([0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8])
T = np.arange(0,0.2,0.02)[1:] #[[0,-1]]
T = np.arange(0,4,0.4)[1:]  # [[0,2.5],[1,-1]]
T = np.arange(0,4,0.2)[1:]  #[[1,0]]  and [[1,-1]]

C_list = []
C_core = []
C_elas = []
C_step = []
W_list = []
Z_list = []
labelsize=17
titlesize=18
#T = np.array([0.00015])
fig,ax = plt.subplots(2,3, figsize=(15,10))
for i in range(T.shape[0]):
    temperature = float(T[i])
    path_state = output_path(num_points, kpoints_max, nu, zeta, a_dsc, gamma, mode_list, T[i], tau_ext)
    dat = np.loadtxt(path_state+'/E_total_wu.txt')[equlibrium_num:]
    step = dat[:,0]
    E = dat[:,1]
    E_core = dat[:,2]
    E_elas = dat[:,3]
    E_step = dat[:,4]
    u = dat[:,5]
    z = dat[:,6]

    C = (np.mean(E**2)-np.mean(E)**2)/(num_points**2*T[i]**2)
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
    ax[1][0].plot(step, E, '-', label=f'{round(T[i],2)}')
    ax[1][1].plot(step, u, '-', label=f'{round(T[i],2)}')
    ax[1][2].plot(step, z, '-', label=f'{round(T[i],2)}')

ax[1][0].set_xlabel('step', fontsize=labelsize)
ax[1][0].set_ylabel(r'$E_\mathrm{tot}$', fontsize=labelsize)
ax[1][1].set_xlabel('step', fontsize=labelsize)
ax[1][1].set_ylabel(r'$w_u$', fontsize=labelsize)
ax[1][2].set_xlabel('step', fontsize=labelsize)
ax[1][2].set_ylabel(r'$w_z$', fontsize=labelsize)

for i in range(2):
    for j in range(3):
        ax[i][j].yaxis.major.formatter._useMathText = True
        ax[i][j].xaxis.major.formatter._useMathText = True
        ax[i][j].ticklabel_format(style='sci', scilimits=(-1,2), axis='both')
        ax[i][j].tick_params(labelsize=labelsize-2)
        ax[i][j].xaxis.get_offset_text().set_fontsize(labelsize-1)
        ax[i][j].yaxis.get_offset_text().set_fontsize(labelsize-1)

ax[0][0].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][1].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][2].set_xlabel(r'$T$', fontsize=labelsize)
ax[0][0].set_ylabel(r'$C$', fontsize=labelsize)
ax[0][1].set_ylabel(r'<$w_u$>', fontsize=labelsize)
ax[0][2].set_ylabel(r'<$w_z$>', fontsize=labelsize)

ax[0][1].plot(T, np.array(W_list), 'o-',)
ax[0][0].plot(T, np.array(C_list), 'o-',label=r'$C_\mathrm{tot}$', markersize=7, markerfacecolor='none', mew=2)
ax[0][0].plot(T, np.array(C_core), '>-',label=r'$C_\mathrm{core}$')
ax[0][0].plot(T, np.array(C_elas), 's-',label=r'$C_\mathrm{elas}$')
ax[0][0].plot(T, np.array(C_step), '<-',label=r'$C_\mathrm{step}$')
ax[0][2].plot(T, np.array(Z_list), 'o-')
ax[0][0].legend(fancybox=False)
ax[1][1].legend(fancybox=False)

plt.tight_layout()
plt.show()

