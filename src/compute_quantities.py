import numpy as np

def compute_core_energy_change(latt_state, rand_site, state_change, rand_site_neighbor, a_dsc, nu, zeta):
    s_0 = latt_state[rand_site]
    s_I_neg = latt_state[rand_site_neighbor[0]]
    s_I_pos = latt_state[rand_site_neighbor[1]]
    s_J_neg = latt_state[rand_site_neighbor[2]]
    s_J_pos = latt_state[rand_site_neighbor[3]]
    return zeta * a_dsc**2 * state_change \
    * ((2*s_0 - s_I_pos - s_I_neg + state_change) + (1-nu)*(2*s_0 - s_J_pos - s_J_neg + state_change))

def compute_elastic_energy_change(latt_stress, rand_site, state_change, stress_kernel, a_dsc):
    N = latt_stress.shape[0]
    return - a_dsc * state_change \
    * (latt_stress[rand_site] + a_dsc/N * state_change * stress_kernel[0,0])

def compute_step_energy_change(latt_height, state_change, rand_site, rand_site_neighbor, a_dsc, gamma):
    latt_height_new = latt_height + state_change
    z_0 = latt_height[rand_site]
    z_I_neg = latt_height[rand_site_neighbor[0]]
    z_I_pos = latt_height[rand_site_neighbor[1]]
    z_J_neg = latt_height[rand_site_neighbor[2]]
    z_J_pos = latt_height[rand_site_neighbor[3]]

    z_0_p = latt_height_new[rand_site]
    z_I_neg_p = latt_height_new[rand_site_neighbor[0]]
    z_I_pos_p = latt_height_new[rand_site_neighbor[1]]
    z_J_neg_p = latt_height_new[rand_site_neighbor[2]]
    z_J_pos_p = latt_height_new[rand_site_neighbor[3]]

    return gamma*a_dsc \
            *(abs(z_0_p-z_I_neg)+ abs(z_0_p-z_I_pos)+abs(z_0_p-z_J_pos)+abs(z_0_p-z_J_neg) \
               -abs(z_0-z_I_neg) -abs(z_0-z_I_pos)  -abs(z_0-z_J_pos)  -abs(z_0-z_J_neg))

def compute_step_energy(gamma, a_dsc, latt_height, nblist_mat):
    z = latt_height
    return  gamma * a_dsc * np.sum(abs(z-z[nblist_mat[2], nblist_mat[3]])+\
            abs(z-z[nblist_mat[6], nblist_mat[7]]))

def compute_core_energy(latt_state, nblist_mat, a_dsc, nu, zeta):
    s = latt_state
    return zeta * a_dsc**2/2 *np.sum( ((s - s[nblist_mat[2], nblist_mat[3]])**2 + \
            (1-nu)*(s - s[nblist_mat[6], nblist_mat[7]])**2) )

def compute_elas_energy(a_dsc, latt_state, latt_stress):
    return -a_dsc/2 * np.sum(latt_state * latt_stress) 

