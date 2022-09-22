'''shift original stress kernel to where a pixel loop is introduced'''
import numpy as np

def stress_kernel_shift(stress_kernel, loop_pos):
    num_points = stress_kernel.shape[0]
    ind_I_arr = np.arange(num_points)
    ind_J_arr = np.arange(num_points)

    # shift the loop position by shifting indices
    ind_I_arr_shift = ind_I_arr - loop_pos[0]
    ind_J_arr_shift = ind_J_arr - loop_pos[1]

    # periodic boundary condition
    condition_index = np.argwhere(ind_I_arr_shift < 0)
    ind_I_arr_shift[condition_index] += num_points
    condition_index = np.argwhere(ind_I_arr_shift >= num_points)
    ind_I_arr_shift[condition_index] -= num_points
    condition_index = np.argwhere(ind_J_arr_shift < 0)
    ind_J_arr_shift[condition_index] += num_points
    condition_index = np.argwhere(ind_J_arr_shift >= num_points)
    ind_J_arr_shift[condition_index] -= num_points

    ind_I_mat_shift, ind_J_mat_shift = np.meshgrid(ind_I_arr_shift, ind_J_arr_shift, indexing='ij')
    return stress_kernel[ind_I_mat_shift, ind_J_mat_shift]

def shift_field(field, pos_ind):
    num_points_x, num_points_y = field.shape
    ind_I_arr = np.arange(num_points_x)
    ind_J_arr = np.arange(num_points_y)

    # shift the loop position by shifting indices
    ind_I_arr_shift = ind_I_arr - pos_ind[0]
    ind_J_arr_shift = ind_J_arr - pos_ind[1]

    # periodic boundary condition
    condition_index = np.argwhere(ind_I_arr_shift < 0)
    ind_I_arr_shift[condition_index] += num_points_x
    condition_index = np.argwhere(ind_I_arr_shift >= num_points_x)
    ind_I_arr_shift[condition_index] -= num_points_x
    condition_index = np.argwhere(ind_J_arr_shift < 0)
    ind_J_arr_shift[condition_index] += num_points_y
    condition_index = np.argwhere(ind_J_arr_shift >= num_points_y)
    ind_J_arr_shift[condition_index] -= num_points_y

    ind_I_mat_shift, ind_J_mat_shift = np.meshgrid(ind_I_arr_shift, ind_J_arr_shift, indexing='ij')
    return field[ind_I_mat_shift, ind_J_mat_shift]

