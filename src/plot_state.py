import numpy as np
import os
import matplotlib.pyplot as plt

def plot_state_stress(latt_height, latt_state, latt_stress, tau_ext, step, path_state=None):
    Nx, Ny = latt_state.shape
    x = np.linspace(0, 1, Nx+1, endpoint=True)
    y = np.linspace(0, 1, Ny+1, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='xy')

    sizeOfFont = 16
    fontProperties = {
    'family' : 'serif',
    'serif' : ['Computer Modern Serif'],
    'weight' : 'normal',
    'size' : sizeOfFont
    }
    plt.rc('text', usetex=True)
    plt.rc('font', **fontProperties)

    title_list = (r"$z$",r"$s$", r"$\sigma_{13}$")
    fig, ax = plt.subplots(1,3)
    ax[0].pcolormesh(X, Y, latt_height.T, vmax=4, vmin=-4, cmap='rainbow', linewidth=0)
    ax[1].pcolormesh(X, Y, latt_state.T, vmax=4, vmin=-4, cmap='rainbow', linewidth=0)
    ax[2].pcolormesh(X, Y, latt_stress.T - tau_ext, vmax=1.0, vmin=-1.0, cmap='rainbow', linewidth=0)
    for i in [0,1,2]:
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].set_xticks((0, 0.5, 1))
        ax[i].set_yticks((0, 0.5, 1))
        ax[i].set_xlabel(r'$x_1/L$')
        ax[i].set_ylabel(r'$x_2/L$')
        ax[i].tick_params(labelsize=14)
        ax[i].set_aspect('equal', adjustable='box')
        ax[i].set_title(title_list[i])
    fig.suptitle(f"Step = {step}")
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    if not path_state:
        plt.show(block=False)
        plt.pause(3)
    else:
        plot_path = f'{path_state}/snapshot'
        if os.path.exists(plot_path)==False:
            os.makedirs(plot_path)
        plt.savefig(plot_path + f"/state_{step}.jpg")
    plt.close()
