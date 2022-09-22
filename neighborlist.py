'''build up neighborlist'''
import numpy as np

'''
def build_nblist(latt_dim):
    """build up neighborlist"""
    neighbour = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    size = (latt_dim[0], latt_dim[1], len(neighbour),2)  # 4 neighbour for a pixel loop
    latt_nblist = np.zeros(size, dtype=int)#latt_dim, dtype=list)
    for ind, _ in np.ndenumerate(latt_nblist):
        move = neighbour[ind[2]]
        nb_ind = [ind[0] + move[0], ind[1] + move[1]]
        # pbc
        for axis in [0, 1]:
            if nb_ind[axis] == -1:
                nb_ind[axis] = latt_dim[axis] - 1
            elif nb_ind[axis] == latt_dim[axis]:
                nb_ind[axis] = 0
        latt_nblist[ind[:2]] = nb_ind
    latt_nblist = tuple(map(tuple,latt_nblist))
    return latt_nblist
'''

def generate_neighbor_index_mat(num_points):
    '''
    return a tuple which contains 8 matrices:
    0:  i-index of left neighbor (cell in x- direction)
    1:  j-index of left neighbor (cell in x- direction)
    2:  i-index of right neighbor (cell in x+ direction)
    3:  j-index of right neighbor (cell in x+ direction)
    4:  i-index of down neighbor (cell in y- direction)
    5:  j-index of down neighbor (cell in y- direction)
    6:  i-index of upper neighbor (cell in y+ direction)
    7:  j-index of upper neighbor (cell in y+ direction)
    '''
    ind_I_arr = np.arange(num_points[0])
    ind_J_arr = np.arange(num_points[1])

    # 0:  i-index of left neighbor (cell in x- direction)
    # 1:  j-index of left neighbor (cell in x- direction)
    ind_I_arr_I_neg = ind_I_arr - 1
    ind_I_arr_I_neg[np.where(ind_I_arr_I_neg == -1)] = num_points[0] - 1
    ind_J_arr_I_neg = ind_J_arr
    ind_I_mat_I_neg, ind_J_mat_I_neg = np.meshgrid(ind_I_arr_I_neg, ind_J_arr_I_neg, indexing='ij')

    # 2:  i-index of right neighbor (cell in x+ direction)
    # 3:  j-index of right neighbor (cell in x+ direction)
    ind_I_arr_I_pos = ind_I_arr + 1
    ind_I_arr_I_pos[np.where(ind_I_arr_I_pos == num_points[0])] = 0
    ind_J_arr_I_pos = ind_J_arr
    ind_I_mat_I_pos, ind_J_mat_I_pos = np.meshgrid(ind_I_arr_I_pos, ind_J_arr_I_pos, indexing='ij')

    # 4:  i-index of down neighbor (cell in y- direction)
    # 5:  j-index of down neighbor (cell in y- direction)
    ind_I_arr_J_neg = ind_I_arr
    ind_J_arr_J_neg = ind_J_arr - 1
    ind_J_arr_J_neg[np.where(ind_J_arr_J_neg == -1)] = num_points[1] - 1
    ind_I_mat_J_neg, ind_J_mat_J_neg = np.meshgrid(ind_I_arr_J_neg, ind_J_arr_J_neg, indexing='ij')

    # 6:  i-index of upper neighbor (cell in y+ direction)
    # 7:  j-index of upper neighbor (cell in y+ direction)
    ind_I_arr_J_pos = ind_I_arr
    ind_J_arr_J_pos = ind_J_arr + 1
    ind_J_arr_J_pos[np.where(ind_J_arr_J_pos == num_points[1])] = 0
    ind_I_mat_J_pos, ind_J_mat_J_pos = np.meshgrid(ind_I_arr_J_pos, ind_J_arr_J_pos, indexing='ij')

    return (ind_I_mat_I_neg, ind_J_mat_I_neg, ind_I_mat_I_pos, ind_J_mat_I_pos, ind_I_mat_J_neg, ind_J_mat_J_neg, ind_I_mat_J_pos,     ind_J_mat_J_pos), \
           (ind_I_arr_I_neg, ind_I_arr_I_pos, ind_J_arr_J_neg, ind_J_arr_J_pos)


if __name__ == '__main__':
    a = np.arange(1,10,1).reshape(3,3)
    mat, arr = generate_neighbor_index_mat((3,3))
    print(a[mat[0], mat[1]])
    site = [0, 0]
    print(a[arr[0][site[0]], site[1]])

    
