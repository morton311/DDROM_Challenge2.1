import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time
import sys
import h5py
from tqdm import tqdm
from scipy.sparse.linalg import factorized

def random_patch_sampling(data, patch_size):
    num_patches = 10000
    num_images = 1
    data = data.squeeze()
    print(f"Data shape: {data.shape}, patch_size: {patch_size}, num_patches: {num_patches}")
    nx = data.shape[0]
    ny = data.shape[1]
    sz = patch_size
    BUFF = 0
    totalsamples = 0
    X = np.zeros((sz ** 2, num_patches))
    
    for i in range(num_images):
        this_image = data

        # Determine how many patches to take
        getsample = num_patches // num_images
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(getsample):
            d1 = BUFF + np.random.randint(0, nx - sz - 2 * BUFF)
            d2 = BUFF + np.random.randint(0, ny - sz - 2 * BUFF)
            
            totalsamples += 1
            temp = this_image[d1:d1 + sz, d2:d2 + sz].reshape(sz ** 2, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X

def modal_decomp_2D(data, patch_size):
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal

def FEM_shape_calculator_2D_ortho_gfemlr(x, y, xpt, ypt):
    sumxpt = np.sum(xpt) / 4
    sumypt = np.sum(ypt) / 4

    dxpt = (-xpt[0] + xpt[1] + xpt[2] - xpt[3]) / 2
    dypt = (ypt[0] + ypt[1] - ypt[2] - ypt[3]) / 2

    zeta_i = [-1, 1, 1, -1]
    eta_i = [1, 1, -1, -1]

    # Inverse transform for parallelogram elements, bilinear shape functions
    zeta = 2 * (x - sumxpt) / dxpt
    eta = 2 * (y - sumypt) / dypt

    N = np.zeros((4,1))
    # shape function values
    for i in range(4):
        N[i] = (1 / 4) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta)
    return N


def dls_Decomp_2D(data, x_grid, y_grid, patch_size, num_modes, modemat_u=None, modemat_v=None, mode_snaps=1):

    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2
    if data.ndim == 4:
        num_snaps, nx, ny, _ = data.shape
    elif data.ndim == 5:
        num_batches, num_snaps, nx, ny, _ = data.shape

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    if modemat_u is None or modemat_v is None:
        local_modes_u, eigVal_u = modal_decomp_2D(data[:mode_snaps, ..., 0], patch_size)
        local_modes_v, eigVal_v = modal_decomp_2D(data[:mode_snaps, ..., 1], patch_size)
        modemat_u = local_modes_u[:, :num_modes]
        modemat_v = local_modes_v[:, :num_modes]
    

        modes_grid_u = modemat_u.reshape((patch_size, patch_size, num_modes), order='F')
        modes_grid_v = modemat_v.reshape((patch_size, patch_size, num_modes), order='F')

        # Mode grid components for the four quadrants
        F1 = list(range(0, mid_pt))
        F2 = list(range(mid_pt-1, nskip_sample + 1))
        F3 = list(range(0, mid_pt))
        F4 = list(range(mid_pt-1, nskip_sample + 1))

        comp1_x, comp1_y = np.meshgrid(F1, F4)
        comp2_x, comp2_y = np.meshgrid(F2, F4)
        comp3_x, comp3_y = np.meshgrid(F2, F3)
        comp4_x, comp4_y = np.meshgrid(F1, F3)

        modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

        modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
        modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

        for i in range(num_modes):
            modes_grid_1_comp_u[:, :, i] = modes_grid_u[comp1_x, comp1_y, i]
            modes_grid_2_comp_u[:, :, i] = modes_grid_u[comp2_x, comp2_y, i]
            modes_grid_3_comp_u[:, :, i] = modes_grid_u[comp3_x, comp3_y, i]
            modes_grid_4_comp_u[:, :, i] = modes_grid_u[comp4_x, comp4_y, i]

            modes_grid_1_comp_v[:, :, i] = modes_grid_v[comp1_x, comp1_y, i]
            modes_grid_2_comp_v[:, :, i] = modes_grid_v[comp2_x, comp2_y, i]
            modes_grid_3_comp_v[:, :, i] = modes_grid_v[comp3_x, comp3_y, i]
            modes_grid_4_comp_v[:, :, i] = modes_grid_v[comp4_x, comp4_y, i]

        modes_vec_comp_1_u = modes_grid_1_comp_u.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_2_u = modes_grid_2_comp_u.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_3_u = modes_grid_3_comp_u.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_4_u = modes_grid_4_comp_u.reshape((mid_pt**2, num_modes), order='F')

        modes_vec_comp_1_v = modes_grid_1_comp_v.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_2_v = modes_grid_2_comp_v.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_3_v = modes_grid_3_comp_v.reshape((mid_pt**2, num_modes), order='F')
        modes_vec_comp_4_v = modes_grid_4_comp_v.reshape((mid_pt**2, num_modes), order='F')


        i, j = 0, 0

        x1 = x_grid[(i  )*nskip , (j+1)*nskip]
        x2 = x_grid[(i+1)*nskip , (j+1)*nskip]
        x3 = x_grid[(i+1)*nskip , (j  )*nskip] 
        x4 = x_grid[(i  )*nskip , (j  )*nskip]

        y1 = y_grid[(i  )*nskip , (j+1)*nskip]
        y2 = y_grid[(i+1)*nskip , (j+1)*nskip]
        y3 = y_grid[(i+1)*nskip , (j  )*nskip] 
        y4 = y_grid[(i  )*nskip , (j  )*nskip]

        xpt = [x1, x2, x3, x4]
        ypt = [y1, y2, y3, y4]

        N1 = np.zeros((nskip+1)**2)
        N2 = np.zeros((nskip+1)**2)
        N3 = np.zeros((nskip+1)**2)
        N4 = np.zeros((nskip+1)**2)

        for kx in range(nskip+1):
            indx = i*nskip + kx
            for ky in range(nskip+1):
                indy = j*nskip + ky
                x_val = x_grid[indx,indy]
                y_val = y_grid[indx,indy]

                # shape functions over the grid points

                iind = ky*(nskip+1) + kx

                N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

                N1[iind] = N[0][0]
                N2[iind] = N[1][0]
                N3[iind] = N[2][0]
                N4[iind] = N[3][0]

        modemat_local_u = np.hstack([
            N1[:, np.newaxis],
            N1[:, np.newaxis] * modes_vec_comp_1_u,
            N2[:, np.newaxis],
            N2[:, np.newaxis] * modes_vec_comp_4_u,
            N3[:, np.newaxis],
            N3[:, np.newaxis] * modes_vec_comp_3_u,
            N4[:, np.newaxis],
            N4[:, np.newaxis] * modes_vec_comp_2_u
        ])
        
        modemat_local_v = np.hstack([
            N1[:, np.newaxis],
            N1[:, np.newaxis] * modes_vec_comp_1_v,
            N2[:, np.newaxis],
            N2[:, np.newaxis] * modes_vec_comp_4_v,
            N3[:, np.newaxis],
            N3[:, np.newaxis] * modes_vec_comp_3_v,
            N4[:, np.newaxis],
            N4[:, np.newaxis] * modes_vec_comp_2_v
        ])

    else:
        modemat_local_u = modemat_u
        modemat_local_v = modemat_v

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global M GFEM matrix')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            M_u[np.ix_(lltogl, lltogl)] += M_local_u
            M_v[np.ix_(lltogl, lltogl)] += M_local_v
    
    M_u = M_u.tocsc()
    M_v = M_v.tocsc()

    #spy M_u
    # import matplotlib.pyplot as plt
    # plt.spy(M_u, markersize=1)
    # plt.title('Sparsity pattern of M_u')
    # plt.show()

    # Pre-factorize the matrices for efficiency
    solve_M_u = factorized(M_u)
    solve_M_v = factorized(M_v)
    print('Done prefactorizing M')

    

    # Handle both 4D and 5D data arrays
    if data.ndim == 4:
        num_batches = 1
        data_batches = [data]
    elif data.ndim == 5:
        num_batches = data.shape[0]
        data_batches = [data[b] for b in range(num_batches)]
    
    dof_u_all = []
    dof_v_all = []
    
    for batch_idx, data_batch in enumerate(data_batches):
        dof_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
        dof_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

        L_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
        L_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

        for i in range(nx_g-1):
            for j in range(ny_g-1):
                lltogl = np.zeros(dof_elem, dtype=int)
                for lindx in range(4):
                    indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                    indx_dof_end = indx_dof_start + dof_node

                    lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

                indx_cell = i*nskip
                indy_cell = j*nskip
                Q_local_u = data_batch[:, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, 0].transpose(1,2,0)
                Q_local_v = data_batch[:, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, 1].transpose(1,2,0)

                Q_local_u_vec = np.zeros(((nskip+1)**2, num_snaps))
                Q_local_v_vec = np.zeros(((nskip+1)**2, num_snaps))

                for kx in range(nskip+1):
                    for ky in range(nskip+1):
                        iind = ky*(nskip+1) + kx
                        
                        Q_local_u_vec[iind, :] = Q_local_u[kx, ky, :]
                        Q_local_v_vec[iind, :] = Q_local_v[kx, ky, :]

                L_local_u = modemat_local_u_wt.T @ Q_local_u_vec
                L_local_v = modemat_local_v_wt.T @ Q_local_v_vec

                L_u[lltogl,:] = L_u[lltogl, :] + L_local_u
                L_v[lltogl,:] = L_v[lltogl, :] + L_local_v

        dof_u = solve_M_u(L_u)
        dof_v = solve_M_v(L_v)

        dof_u = dof_u.T
        dof_v = dof_v.T
        
        dof_u_all.append(dof_u)
        dof_v_all.append(dof_v)
    
    # Stack batches or return single batch
    dof_u = np.stack(dof_u_all) if num_batches > 1 else dof_u_all[0]
    dof_v = np.stack(dof_v_all) if num_batches > 1 else dof_v_all[0]

    dls_config = {
        'patch_size': patch_size,
        'num_modes': num_modes,
        'nskip': nskip,
        'nx_g': nx_g,
        'ny_g': ny_g,
        'nx_t': nx_t,
        'ny_t': ny_t,
        'dof_node': dof_node,
        'num_gfem_nodes': num_gfem_nodes,
        'modemat_local_u': modemat_local_u,
        'modemat_local_v': modemat_local_v
    }

    return dof_u, dof_v, dls_config


def dls_Rec_2D(dof_u, dof_v, dls_config):

    nskip = dls_config['nskip']
    patch_size = dls_config['patch_size']
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    nx_g = dls_config['nx_g']
    ny_g = dls_config['ny_g']

    num_dof, num_snaps = dof_u.shape
    dof_node = dls_config['dof_node']
    num_gfem_nodes = dls_config['num_gfem_nodes']

    nx_t = dls_config['nx_t']
    ny_t = dls_config['ny_t']

    sample_x = range(0, nx_t, nskip)
    sample_y = range(0, ny_t, nskip)

    modemat_local_u = dls_config['modemat_local_u']
    modemat_local_v = dls_config['modemat_local_v']

    rec_u = np.zeros((num_snaps, nx_t, ny_t))
    rec_v = np.zeros((num_snaps, nx_t, ny_t))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            lltogl = np.zeros(dof_node*4, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx * dof_node: (lindx + 1) * dof_node] = np.arange(indx_dof_start, indx_dof_end)

            indx_cell = i*nskip
            indy_cell = j*nskip

            rec_local_u_vec = modemat_local_u @ dof_u[lltogl]
            rec_local_v_vec = modemat_local_v @ dof_v[lltogl]

            rec_local_u_vec_reshaped = rec_local_u_vec.reshape((nskip+1, nskip+1, num_snaps), order='F')
            rec_local_v_vec_reshaped = rec_local_v_vec.reshape((nskip+1, nskip+1, num_snaps), order='F')

            rec_u[:, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1] = rec_local_u_vec_reshaped.transpose(2,0,1)
            rec_v[:, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1] = rec_local_v_vec_reshaped.transpose(2,0,1)

        rec = np.zeros((2, num_snaps, nx_t, ny_t))
        rec[0] = rec_u
        rec[1] = rec_v
    return rec


def vis_modes(config, save_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.mplot3d import Axes3D

    paper_width = 470 # pt
    width = paper_width / 72.27 # inches
    height = width / 1.618 # inches
    # plt.rcParams['text.usetex'] = True
    # set default font size
    plt.rcParams['font.size'] = 8 # Change default font size to 12
    plt.rcParams['axes.titlesize'] = 10 # Change axes title font size
    plt.rcParams['axes.labelsize'] = 8 # Change axes labels font size
    plt.rcParams['xtick.labelsize'] = 8 # Change x-axis tick labels font size
    plt.rcParams['ytick.labelsize'] = 8 # Change y-axis tick labels font size
    plt.rcParams['legend.fontsize'] = 8 # Change legend font size
    plt.rcParams['figure.constrained_layout.use'] = True

    shape_funcs = config['modemat_local_u']
    dims = shape_funcs.shape
    nskip = ((config['patch_size']-1) // 2)
    patch_size = config['patch_size']
    num_modes = config['num_modes']
    shape_funcs = shape_funcs.reshape(int(np.sqrt(dims[0])), int(np.sqrt(dims[0])), -1)

    shape_func_reshaped = np.zeros((patch_size, patch_size, num_modes+1), dtype=shape_funcs.dtype)

    for i in range(num_modes+1):
        shape_func_reshaped[:nskip+1, nskip:, i] = shape_funcs[..., i]
        shape_func_reshaped[:nskip+1, :nskip+1, i] = shape_funcs[..., i+num_modes+1]
        shape_func_reshaped[ nskip:, :nskip+1, i] = shape_funcs[..., i+2*num_modes+2]
        shape_func_reshaped[ nskip:, nskip:, i] = shape_funcs[..., i+3*num_modes+3]

    N, M = 2, 3+num_modes//2
    size = 0.8
    fig, axes = plt.subplots(N, M, sharey=True, sharex=True, figsize=(size*width, 2/7*size*width), gridspec_kw={'hspace': 0.05, 'wspace': 0.05}) #

    ax = axes.flatten()
    for i in range(num_modes):
        j = i
        if i > num_modes//2 - 1:
            j += 3
        vmax = np.max(np.abs(shape_func_reshaped[...,i+1]))
        ax[3+j].pcolormesh(shape_func_reshaped[...,i+1], cmap='seismic', shading='Gouraud', vmin=-vmax, vmax=vmax)
        ax[3+j].set_xticks([])
        ax[3+j].set_yticks([])
        ax[3+j].set_aspect('equal')

    for row in range(2):
        for col in range(3):
            fig.delaxes(axes[row, col])

    big_ax = fig.add_axes([0.05, -0.05,
                           0.25, 0.95])
    vmax = np.max(np.abs(shape_func_reshaped[...,0]))
    im0 = big_ax.pcolormesh(shape_func_reshaped[...,0], cmap='seismic', shading='Gouraud', vmin=-vmax, vmax=vmax)
    big_ax.set_aspect('equal')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_title('2D Linear FEM Shape Function')

    # for row in range(2):
    #     for col in range(2, M):
    #         pos = axes[row, col].get_position()
    #         axes[row, col].set_position([
    #             pos.x0 + 0.001,
    #             pos.y0,
    #             pos.width,
    #             pos.height
    #         ])

    ax[5].set_title('Linear FEM enriched with local POD modes')
    plt.savefig(save_dir, dpi=600, bbox_inches='tight')
    plt.close()

    # fig = plt.figure(figsize=(width, height))
    # num_to_plot = num_modes+1
    # for i in range(num_to_plot):
    #     j = i
    #     if i > num_modes//2:
    #         j +=1
    #     ax3d = fig.add_subplot(2, num_modes//2, j+1, projection='3d')
    #     X, Y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    #     Z = shape_func_reshaped[..., i]
    #     vmax = np.max(np.abs(Z))
    #     surf = ax3d.plot_surface(X, Y, Z, cmap='seismic', alpha=0.95, vmin=-vmax, vmax=vmax)
    #     ax3d.set_xticklabels([])
    #     ax3d.set_yticklabels([])
    #     ax3d.set_zticklabels([])
    #     ax3d.set_title(f'Shape Function {i}')
    
    # plt.savefig('gfem_shape_functions_3d.png', dpi=600)
    # plt.close()


def gfem_2d(data, patch_size, num_modes):
    num_snaps, nx, ny, _ = data.shape
    grid_x = np.linspace(1, nx, nx)
    grid_y = np.linspace(1, ny, ny)
    [grid_x, grid_y] = np.meshgrid(grid_x, grid_y)
    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2
    mode_snaps = 1

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    Q_grid_u = data[:,:nx_t, :ny_t, 0]
    Q_grid_v = data[:,:nx_t, :ny_t, 1]

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element


    # Compute local modes
    local_modes_u, eigVal_u = modal_decomp_2D(data[:mode_snaps, ..., 0], patch_size)
    local_modes_v, eigVal_v = modal_decomp_2D(data[:mode_snaps, ..., 1], patch_size)
    # print(local_modes.shape)
    mode_grid_u = local_modes_u[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')
    mode_grid_v = local_modes_v[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')

    # Mode grid components for the four quadrants
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt-1, nskip_sample + 1))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt-1, nskip_sample + 1))

    modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

    modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

    for i in range(num_modes):
        modes_grid_1_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

        modes_grid_1_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

    modes_vec_1_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_u = np.zeros((mid_pt ** 2, num_modes))

    modes_vec_1_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_v = np.zeros((mid_pt ** 2, num_modes))

    for i in range(num_modes):
        modes_vec_1_comp_u[:, i] = modes_grid_1_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_u[:, i] = modes_grid_2_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_u[:, i] = modes_grid_3_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_u[:, i] = modes_grid_4_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')

        modes_vec_1_comp_v[:, i] = modes_grid_1_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_v[:, i] = modes_grid_2_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_v[:, i] = modes_grid_3_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_v[:, i] = modes_grid_4_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
    
    i = 1
    j = 1

    M_local_u = np.zeros((dof_elem, dof_elem))
    M_local_v = np.zeros((dof_elem, dof_elem))
    
    # x, y locations of the GFEM element nodes
    x1 = grid_x[i*nskip,       (j-1)*nskip]
    x2 = grid_x[i*nskip,       j*nskip]
    x3 = grid_x[(i-1)*nskip,   j*nskip]
    x4 = grid_x[(i-1)*nskip,   (j-1)*nskip]

    y1 = grid_y[i*nskip,       (j-1)*nskip]
    y2 = grid_y[i*nskip,       j*nskip]
    y3 = grid_y[(i-1)*nskip,   j*nskip]
    y4 = grid_y[(i-1)*nskip,   (j-1)*nskip]


    # Combining x, y nodal coordinates into vector form
    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((nskip+1)**2)
    N2 = np.zeros((nskip+1)**2)
    N3 = np.zeros((nskip+1)**2)
    N4 = np.zeros((nskip+1)**2)

    for kx in range(nskip+1):
        indx = (i-1)*nskip + kx
        for ky in range(nskip+1):
            indy = (j-1)*nskip + ky
            x_val = grid_x[indy,indx]
            y_val = grid_y[indy,indx]

            # shape functions over the grid points

            iind = ky*(nskip+1) + kx

            N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    modemat_local_u = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_u,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_u,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_u,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_u
    ])
    
    modemat_local_v = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_v,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_v,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_v,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_v
    ])

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    L_u = np.zeros((num_gfem_nodes * dof_node, num_snaps))
    L_v = np.zeros((num_gfem_nodes * dof_node, num_snaps))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global GFEM matrices')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            L_local_u = np.zeros((dof_elem, 1))
            L_local_v = np.zeros((dof_elem, 1))

            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            indx_cell = (i-1) * nskip
            indy_cell = (j-1) * nskip

            M_u[np.ix_(lltogl, lltogl)] = M_u[np.ix_(lltogl, lltogl)] + M_local_u
            M_v[np.ix_(lltogl, lltogl)] = M_v[np.ix_(lltogl, lltogl)] + M_local_v

            for id in range(num_snaps):
                indx_cell = i * nskip
                indy_cell = j * nskip

                Q_local_u = Q_grid_u[id, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1]
                Q_local_v = Q_grid_v[id, indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1]

                Q_local_u_vec = np.zeros((nskip+1)**2)
                Q_local_v_vec = np.zeros((nskip+1)**2)


                for kx in range(nskip+1):
                    for ky in range(nskip+1):
                        iind = ky*(nskip+1) + kx
                        
                        Q_local_u_vec[iind] = Q_local_u[kx, ky]
                        Q_local_v_vec[iind] = Q_local_v[kx, ky]

                L_local_u = modemat_local_u_wt.T @ Q_local_u_vec.T
                L_local_v = modemat_local_v_wt.T @ Q_local_v_vec.T

                L_u[lltogl, id] = L_u[lltogl, id] + L_local_u
                L_v[lltogl, id] = L_v[lltogl, id] + L_local_v

    
    print('Done constructing global matrices')
    # plt.spy(M, markersize=2)
    print(M_u.shape, L_u.shape)

    # Convert lilmatrix to csr matrix
    M_u = csr_matrix(M_u)
    M_v = csr_matrix(M_v)

    dof_u = spsolve(M_u, L_u)
    dof_v = spsolve(M_v, L_v)

    class Config:
        def __init__(self, data, patch_size, num_modes, modemat_local_u, modemat_local_v):
            self.nx = data.shape[1]
            self.ny = data.shape[2]
            self.num_snaps = data.shape[3]
            self.patch_size = patch_size
            self.num_modes = num_modes
            self.nskip = (patch_size - 1) // 2
            self.nskip_sample = patch_size - 1
            self.mid_pt = 1 + self.nskip_sample // 2
            self.sample_x = range(0, self.nx, self.nskip)
            self.sample_y = range(0, self.ny, self.nskip)
            self.nx_t = max(self.sample_x) + 1
            self.ny_t = max(self.sample_y) + 1
            self.nx_g = len(self.sample_x)
            self.ny_g = len(self.sample_y)
            self.num_gfem_nodes = self.nx_g * self.ny_g
            self.dof_node = num_modes + 1
            self.dof_elem = 4 * self.dof_node
            self.modemat_local_u = modemat_local_u
            self.modemat_local_v = modemat_local_v
            self.compression_ratio = data.shape[0]*num_snaps*self.nx*self.ny / (data.shape[0]*num_snaps*self.dof_node + data.shape[0] * num_modes * self.patch_size**2 )


    config = Config(data=data, patch_size=patch_size, num_modes=num_modes, modemat_local_u=modemat_local_u, modemat_local_v=modemat_local_v)

    return dof_u, dof_v, config

def gfem_recon(dof_u, dof_v, config):

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    nskip = config.nskip
    dof_node = config.dof_node # DOFs/node
    dof_elem = config.dof_elem # DOFs/element
    
    # if one dimensional, make 2d
    if len(dof_u.shape) == 1:
        dof_u = dof_u[:, np.newaxis]
        dof_v = dof_v[:, np.newaxis]

    Q_rec_u = np.zeros((config.nx_t, config.ny_t, dof_u.shape[-1]))
    Q_rec_v = np.zeros((config.nx_t, config.ny_t, dof_v.shape[-1]))

    for i in range(config.nx_g-1):
        for j in range(config.ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*config.ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)
            # print(lltogl)
            for id in range(dof_u.shape[-1]):
                Q_rec_local_u_vec = config.modemat_local_u @ dof_u[lltogl, id]
                Q_rec_local_v_vec = config.modemat_local_v @ dof_v[lltogl, id]
                
                Q_rec_local_u = Q_rec_local_u_vec.reshape((nskip+1, nskip+1), order='F')
                Q_rec_local_v = Q_rec_local_v_vec.reshape((nskip+1, nskip+1), order='F')

                Q_rec_u[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_u
                Q_rec_v[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_v
    Q_rec = np.zeros((config.nx_t, config.ny_t, dof_u.shape[-1], 2))
    Q_rec[:, :, :, 0] = Q_rec_u
    Q_rec[:, :, :, 1] = Q_rec_v
    return Q_rec