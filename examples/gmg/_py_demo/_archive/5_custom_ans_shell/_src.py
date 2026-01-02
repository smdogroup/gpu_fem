import numpy as np
import matplotlib.pyplot as plt

def lagrange_basis_1d(xi, vals, order:int=2):
    """1d order 2, basis interp (3 points)"""

    if order == 2:
        result = -0.5 * xi * (1.0 - xi) * vals[0]
        result += (1.0 - xi**2) * vals[1]
        result += 0.5 * xi * (1.0 + xi) * vals[2]
    elif order == 1:
        result = 0.5 * (1-xi) * vals[0]
        result += 0.5 * (1 + xi) * vals[1]
    return result

def lagrange_basis_1d_transpose(xi, out_bar, order:int=2):
    """1d order 2, basis interp (3 points)"""
    if order == 2:
        vals_bar = np.zeros(3)
        vals_bar[0] = -0.5 * xi * (1.0 - xi) * out_bar
        vals_bar[1] = (1.0 - xi**2) * out_bar
        vals_bar[2] = 0.5 * xi * (1.0 + xi) * out_bar
        return vals_bar
    elif order == 1:
        vals_bar = np.zeros(2)
        vals_bar[0] = 0.5 * (1.0 - xi) * out_bar
        vals_bar[1] = 0.5 * (1.0 + xi) * out_bar
        return vals_bar

def lagrange_basis_1d_grad(xi, vals, order:int=2):
    """get d/dxi derivs of the 1D lagrange basis"""
    if order == 2:
        result = (xi - 0.5) * vals[0]
        result += (-2.0 * xi) * vals[1]
        result += (xi + 0.5) * vals[2]
        return result
    elif order == 1:
        result = 0.5 * (-vals[0] + vals[1])
        return result

def lagrange_basis_1d_grad_transpose(xi, out_bar, order:int=2):
    """get d/dxi derivs of the 1D lagrange basis"""
    if order == 2:
        vals_bar = np.zeros(3)
        vals_bar[0] = (xi - 0.5) * out_bar
        vals_bar[1] = (-2.0 * xi) * out_bar
        vals_bar[2] = (xi + 0.5) * out_bar
        return vals_bar
    elif order == 1:
        vals_bar = np.zeros(2)
        vals_bar[0] = -0.5 * out_bar
        vals_bar[1] = 0.5 * out_bar
        return vals_bar

def lagrange_basis_1d_grad2(xi, vals, order:int=2):
    """get d^2/dxi^2 derivs of the 1D lagrange basis"""
    result = 1.0 * vals[0]
    result += -2.0 * vals[1]
    result += 1.0 * vals[2]
    return result

def lagrange_basis_1d_grad2_transpose(xi, out_bar, order:int=2):
    """get d/dxi derivs of the 1D lagrange basis"""
    vals_bar = np.zeros(3)
    vals_bar[0] = out_bar
    vals_bar[1] = -2.0 * out_bar
    vals_bar[2] = out_bar
    return vals_bar

def lagrange_basis_2d(xi, eta, vals, order:int=2):
    """2d order 2, basis interp (9 points)"""
    # sum-factor method, first interp over each eta and add in
    n = order + 1
    eta_vals = [lagrange_basis_1d(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
    return lagrange_basis_1d(eta, eta_vals, order)

def lagrange_basis_2d_transpose(xi, eta, out_bar, order:int=2):
    """2d order 2, basis interp (9 points)"""
    n = order + 1
    vals_bar = np.zeros(n*n)
    eta_vals_bar = lagrange_basis_1d_transpose(eta, out_bar, order)
    for i in range(n):
        vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_transpose(xi, eta_vals_bar[i], order)
    return vals_bar

def lagrange_basis_2d_grad(xi, eta, vals, deriv:int=0, order:int=2):
    """2d order 2, basis interp (9 points)"""
    n = order + 1
    if deriv == 0: # d/dxi deriv
        eta_vals = [lagrange_basis_1d_grad(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
        return lagrange_basis_1d(eta, eta_vals, order)

    else: # d/deta deriv
        eta_vals = [lagrange_basis_1d(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
        return lagrange_basis_1d_grad(eta, eta_vals, order)
    
def lagrange_basis_2d_grad_transpose(xi, eta, out_bar, deriv:int=0, order:int=2):
    """2d order 2, basis interp (9 points)"""
    n = order + 1
    vals_bar = np.zeros(n*n)
    if deriv == 0: # d/dxi deriv
        eta_vals_bar = lagrange_basis_1d_transpose(eta, out_bar, order)
        for i in range(n):
            vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_grad_transpose(xi, eta_vals_bar[i], order)

    else: # d/deta deriv
        eta_vals_bar = lagrange_basis_1d_grad_transpose(eta, out_bar, order)
        for i in range(n):
            vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_transpose(xi, eta_vals_bar[i], order)
    return vals_bar
    
def lagrange_basis_2d_grad2(xi, eta, vals, case:int=0, order:int=2):
    """case in 1,2,3 for _11, _12, _22 derivs"""
    # unfortunately the second derivs here become constant (and may lose some effectiveness of the DOF)
    # but only in that particular direction..
    n = order + 1
    if case == 0: # d^2/dxi^2
        eta_vals = [lagrange_basis_1d_grad2(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
        return lagrange_basis_1d(eta, eta_vals, order)
    elif case == 1: # d^2/dxi/deta (mixed deriv)
        eta_vals = [lagrange_basis_1d_grad(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
        return lagrange_basis_1d_grad(eta, eta_vals, order)
    elif case == 2: # d^2/deta^2
        eta_vals = [lagrange_basis_1d(xi,vals[n*i:(n*i+n)], order) for i in range(n)]
        return lagrange_basis_1d_grad2(eta, eta_vals, order)
        
    
def lagrange_basis_2d_grad2_transpose(xi, eta, out_bar, case:int=0, order:int=2):
    """case in 1,2,3 for _11, _12, _22 derivs"""
    n = order + 1
    vals_bar = np.zeros(n*n)
    if case == 0: # d^2/dxi^2
        eta_vals_bar = lagrange_basis_1d_transpose(eta, out_bar, order)
        for i in range(n):
            vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_grad2_transpose(xi, eta_vals_bar[i], order)
    elif case == 1: # d^2/dxi/deta
        eta_vals_bar = lagrange_basis_1d_grad_transpose(eta, out_bar, order)
        for i in range(n):
            vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_grad_transpose(xi, eta_vals_bar[i], order)
    elif case == 2: # d^2/dxi/deta
        eta_vals_bar = lagrange_basis_1d_grad2_transpose(eta, out_bar, order)
        for i in range(n):
            vals_bar[n*i:(n*i+n)] = lagrange_basis_1d_transpose(xi, eta_vals_bar[i], order)

    return vals_bar

def lagrange_basis_1d_vec(xi):
    return np.array([
        -0.5 * xi * (1.0 - xi),
        (1.0 - xi**2),
        0.5 * xi * (1.0 + xi),
    ])

def lagrange_basis_2d_vec(xi, eta):
    xi_vec = lagrange_basis_1d_vec(xi)
    eta_vec = lagrange_basis_1d_vec(eta)
    return np.array([xi_vec[i]*eta_vec[j] for j in range(3) for i in range(3)])

def get_xpts_basis(xi, eta, xpts):
    """get xpts a_xi, a_eta, a_zeta natural coords basis"""

    # get nodal vals here
    xyz = [xpts[i::3] for i in range(3)]

    # get param grads
    r_xi = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=0) for _vec in xyz])
    r_eta = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=1) for _vec in xyz])

    # magnitudes
    A_xi, A_eta = np.linalg.norm(r_xi), np.linalg.norm(r_eta)

    # unit vecs for basis
    a_xi, a_eta = r_xi / A_xi, r_eta / A_eta
    a_gam = np.cross(a_xi, a_eta)
    a_gam /= np.linalg.norm(a_gam)

    return a_xi, a_eta, a_gam, A_xi, A_eta

def get_dsurf_jac(xi, eta, xpts):
    """get |d(x,y)/d(xi,eta)| jacobian"""

    # get nodal vals here
    xyz = [xpts[i::3] for i in range(3)]

    # get param grads
    r_xi = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=0) for _vec in xyz])
    r_eta = np.array([lagrange_basis_2d_grad(xi, eta, _vec, deriv=1) for _vec in xyz])

    out = np.cross(r_xi, r_eta)
    return np.linalg.norm(out)

def get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts):
    """get Tinv matrix.."""

    a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

    # get shell normals here in local cartesian basis first
    e_x = shell_xi_axis.copy()
    e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
    e_y = np.cross(a_gam, e_x)
    e_z = a_gam.copy() # since e_z, a_gam assumed same direction

    # make transform from global disps to local cartesian basis
    T_LG = np.zeros((3,3))
    T_LG[0] = e_x
    T_LG[1] = e_y
    T_LG[2] = e_z
    # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

    # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
    cross_nrm = np.linalg.norm(np.cross(a_xi, a_eta))
    T = np.array([
        [np.dot(e_y, a_eta), -np.dot(e_y, a_xi), 0.0],
        [-np.dot(e_x, a_eta), np.dot(e_x, a_xi), 0.0],
        [0.0, 0.0, cross_nrm]
    ]) / cross_nrm
    return T

def get_cov_nodal_disps(shell_xi_axis, xpts, vars):
    """compute shell normals and covariant basis, then rotate disps and rotations to each local nodal cov basis (uvw-tildes, etc.)"""
    # you do need the displacements rotated in their natural coord basis
    # to account for curvature effects

    vars_cov = np.zeros_like(vars)

    for node in range(9):
        inode, jnode = node % 3, node // 3
        xi, eta = 1.0*(inode-1), 1.0*(jnode-1)

        # get cov basis at the current node
        a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

        # get shell normals here in local cartesian basis first
        e_x = shell_xi_axis.copy()
        e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
        e_y = np.cross(a_gam, e_x)
        e_z = a_gam.copy() # since e_z, a_gam assumed same direction

        # make transform from global disps to local cartesian basis
        T_LG = np.zeros((3,3))
        T_LG[0] = e_x
        T_LG[1] = e_y
        T_LG[2] = e_z
        # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

        # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
        Tinv = np.array([
            [np.dot(e_x, a_xi), np.dot(e_y, a_xi), 0],
            [np.dot(e_x, a_eta), np.dot(e_y, a_eta), 0],
            [0, 0, 1.0],
        ])
        Tinv_planar = Tinv[:2, :][:, :2]

        # rotate the current shell disps and shear strains
        uvw_loc_xyz = np.dot(T_LG, vars[5*node:(5*node+3)])
        gams = vars[5*node+3:(5*node+5)]

        vars_cov[5*node:(5*node+3)] = np.dot(Tinv, uvw_loc_xyz) # uv rotate
        vars_cov[(5*node+3):(5*node+5)] = np.dot(Tinv_planar, gams) # rotate trv shear strains

    return vars_cov

def get_cov_nodal_disps_transpose(shell_xi_axis, xpts, vars_cov_bar):
    """compute shell normals and covariant basis, then rotate disps and rotations to each local nodal cov basis (uvw-tildes, etc.)"""
    # you do need the displacements rotated in their natural coord basis
    # to account for curvature effects

    vars_bar = np.zeros_like(vars_cov_bar)

    for node in range(9):
        inode, jnode = node % 3, node // 3
        xi, eta = 1.0*(inode-1), 1.0*(jnode-1)

        # get cov basis at the current node
        a_xi, a_eta, a_gam, _, _ = get_xpts_basis(xi, eta, xpts)

        # get shell normals here in local cartesian basis first
        e_x = shell_xi_axis.copy()
        e_x -= np.dot(a_gam, e_x) # remove normal component of 1 axis
        e_y = np.cross(a_gam, e_x)
        e_z = a_gam.copy() # since e_z, a_gam assumed same direction

        # make transform from global disps to local cartesian basis
        T_LG = np.zeros((3,3))
        T_LG[0] = e_x
        T_LG[1] = e_y
        T_LG[2] = e_z
        # T_LG * global disp => local cartesian disp so that if [u,v,w] global in direction of e_x for instance => we get just u in local cartesian basis

        # now compute the cov basis transform matrix (inv form so that Tinv*e => a)
        Tinv = np.array([
            [np.dot(e_x, a_xi), np.dot(e_y, a_xi), 0],
            [np.dot(e_x, a_eta), np.dot(e_y, a_eta), 0],
            [0, 0, 1.0],
        ])
        Tinv_planar = Tinv[:2, :][:, :2]

        # now begin transpose steps
        uvw_loc_xyz_bar = np.dot(Tinv.T, vars_cov_bar[5*node:(5*node+3)])
        vars_bar[5*node:(5*node+3)] += np.dot(T_LG.T, uvw_loc_xyz_bar)

        # transpose rotate the trv shear strains
        vars_bar[5*node+3:(5*node+5)] = np.dot(Tinv_planar.T, vars_cov_bar[(5*node+3):(5*node+5)])

    return vars_bar

def get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts):
    """for rotating xi,eta strains => xyz strains (membrane and bending)"""

    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    T2 = np.array([
        [T[0,0]**2, T[0,1]**2, T[0,0] * T[0,1]],
        [T[1,0]**2, T[1,1]**2, T[1,0] * T[1,1]],
        [2 * T[0,1] * T[1,0], 2 * T[0,1] * T[1,1], T[0,0] * T[1,1] + T[0,1] * T[1,0]]
    ])
    return T2

def get_membrane_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    mem_strains = np.zeros(3)

    # get length scale conversions
    _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    # compute eps_11 = 1/A_xi * du/dxi
    u_nodal = vars_cov[0::5]
    mem_strains[0] = lagrange_basis_2d_grad(xi, eta, u_nodal, deriv=0) / A_xi

    # compute eps_22 = 1/A_eta * dv/deta
    v_nodal = vars_cov[1::5]
    mem_strains[1] = lagrange_basis_2d_grad(xi, eta, v_nodal, deriv=1) / A_eta

    # compute gam_12 = 1/A_xi * dv/dxi + 1/A_eta * du/eta (no 1/2 cause gam_12 not eps_12)
    dv_dx = lagrange_basis_2d_grad(xi, eta, v_nodal, deriv=0) / A_xi
    du_dy = lagrange_basis_2d_grad(xi, eta, u_nodal, deriv=1) / A_eta
    mem_strains[2] = dv_dx + du_dy

    # rotate membrane strains to local cartesian from nat cov basis
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    mem_strains_xyz = np.dot(T2, mem_strains)

    return mem_strains_xyz

def get_membrane_strains_transpose(shell_xi_axis, xi, eta, xpts, mem_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    mem_strains_bar = np.dot(T2.T, mem_strains_xyz_bar)

    _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    # vars_cov is already the local nodal cov rotated bases
    # mem_strains[0] = lagrange_basis_2d_grad(xi, eta, u_nodal, deriv=0) / A_xi
    vars_cov_bar[0::5] += lagrange_basis_2d_grad_transpose(xi, eta, mem_strains_bar[0] / A_xi, deriv=0)

    # now get the e_eta,eta strains
    vars_cov_bar[1::5] += lagrange_basis_2d_grad_transpose(xi, eta, mem_strains_bar[1] / A_eta, deriv=1)

    # now do the e_{xi,eta} in-plane 
    vars_cov_bar[1::5] += lagrange_basis_2d_grad_transpose(xi, eta, mem_strains_bar[2] / A_xi, deriv=0)
    vars_cov_bar[0::5] += lagrange_basis_2d_grad_transpose(xi, eta, mem_strains_bar[2] / A_eta, deriv=1)
    
    return vars_cov_bar

def get_bending_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    bend_strains = np.zeros(3)

    # get length scale conversions
    _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    # compute the rotations alpha, beta from w and trv shear strains (in local cov basis)
    gam_13 = vars_cov[3::5].copy()
    gam_23 = vars_cov[4::5].copy()
    w = vars_cov[2::5].copy()

    # can't construct alpha, beta first then interpolate again (cause then prolongation operator with single FEA interp)
    # not compatible with double FEA interp probably..

    # now use these interped rotations to compute bending strains
    # k11 = dalpha/dx = dgam13/dx - d^2w/dx^2
    bend_strains[0] = lagrange_basis_2d_grad(xi, eta, gam_13, deriv=0) / A_xi - lagrange_basis_2d_grad2(xi, eta, w, case=0) / A_xi**2
    # NOTE : does the 1/A_xi**2 in second deriv above result in worse conditioning of matrix though? maybe.. because it actually becomes something like 1/h^4 since B^T and B show up
    # maybe some other way around it, we'll see..
    # NOTE : maybe I can add some auxillary rotation-like DOF for the w that are just tied to w interp, not directly rotations?
    # like hermite cubic does..

    # k22 = dbeta/dy = dgam23/dy - d^2w/dy^2
    bend_strains[1] = lagrange_basis_2d_grad(xi, eta, gam_23, deriv=1) / A_eta - lagrange_basis_2d_grad2(xi, eta, w, case=2) / A_eta**2

    # gam12 = dbeta/dx + dalpha/dy = dgam23/dx + dgam13/dy - 2 * d^2w/dx/dy
    # TODO : how do the second derivs here perform on curved grids though? maybe worse.. will have to see about this later and improve the scheme maybe
    dbeta_dx = lagrange_basis_2d_grad(xi, eta, gam_23, deriv=0) / A_xi - lagrange_basis_2d_grad2(xi, eta, w, case=1) / A_xi / A_eta
    dalpha_dy = lagrange_basis_2d_grad(xi, eta, gam_13, deriv=1) / A_eta - lagrange_basis_2d_grad2(xi, eta, w, case=1) / A_xi / A_eta
    bend_strains[2] = dbeta_dx + dalpha_dy
    bend_strains[2] *= 2.0 # ??, this seemed to correct the disp.. TODO : double check this later

    # rotate bending strains to local cartesian from nat cov basis
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    bend_strains_xyz = np.dot(T2, bend_strains)
    return bend_strains_xyz

def get_bending_strains_transpose(shell_xi_axis, xi, eta, xpts, bend_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T2 = get_natural_to_local_strain_transform(shell_xi_axis, xi, eta, xpts)
    bend_strains_bar = np.dot(T2.T, bend_strains_xyz_bar)

    _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    bend_strains_bar[2] *= 2.0 # ? # ??, this seemed to correct the disp.. TODO : double check this later

    # vars_cov is already the local nodal cov rotated bases
    vars_cov_bar[3::5] += lagrange_basis_2d_grad_transpose(xi, eta, bend_strains_bar[0] / A_xi, deriv=0)
    vars_cov_bar[2::5] += lagrange_basis_2d_grad2_transpose(xi, eta, -bend_strains_bar[0] / A_xi**2, case=0)

    # bend strains bar 1
    vars_cov_bar[4::5] += lagrange_basis_2d_grad_transpose(xi, eta, bend_strains_bar[1] / A_eta, deriv=1)
    vars_cov_bar[2::5] += lagrange_basis_2d_grad2_transpose(xi, eta, -bend_strains_bar[1] / A_eta**2, case=2)

    # bend strains bar 2
    vars_cov_bar[3::5] += lagrange_basis_2d_grad_transpose(xi, eta, bend_strains_bar[2] / A_eta, deriv=1)
    vars_cov_bar[2::5] += lagrange_basis_2d_grad2_transpose(xi, eta, -2.0 * bend_strains_bar[2] / A_xi / A_eta, case=1)
    vars_cov_bar[4::5] += lagrange_basis_2d_grad_transpose(xi, eta, bend_strains_bar[2] / A_xi, deriv=0)
    
    return vars_cov_bar

def get_trv_shear_strains(shell_xi_axis, xi, eta, xpts, vars_cov):
    """compute first cov then global basis membrane strains (3 of them)"""
    trv_shear_strains = np.zeros(2)

    gam13_nodal = vars_cov[3::5]
    trv_shear_strains[0] = lagrange_basis_2d(xi, eta, gam13_nodal)

    gam23_nodal = vars_cov[4::5]
    trv_shear_strains[1] = lagrange_basis_2d(xi, eta, gam23_nodal)

    # rotate membrane strains to local cartesian from nat cov basis
    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    trv_shear_strains_xyz = np.dot(T[:2,:][:,:2], trv_shear_strains)

    return trv_shear_strains_xyz

def get_trv_shear_strains_transpose(shell_xi_axis, xi, eta, xpts, trv_shear_strains_xyz_bar):
    """compute first cov then global basis membrane strains (3 of them)"""

    vars_cov_bar = np.zeros(45)

    # undo the xi,eta => xyz strain transform (transpose)
    T = get_natural_rot_matrix(shell_xi_axis, xi, eta, xpts)
    T2 = T[:2,:][:,:2]
    trv_shear_strains_bar = np.dot(T2.T, trv_shear_strains_xyz_bar)

    # _, _, _, A_xi, A_eta = get_xpts_basis(xi, eta, xpts)

    vars_cov_bar[3::5] += lagrange_basis_2d_transpose(xi, eta, trv_shear_strains_bar[0])
    vars_cov_bar[4::5] += lagrange_basis_2d_transpose(xi, eta, trv_shear_strains_bar[1])

    return vars_cov_bar

def get_quadpt_strains(shell_xi_axis, xi, eta, xpts, vars):
    """get quadpt strains (important step in the kelem formulation)"""

    vars_cov = get_cov_nodal_disps(shell_xi_axis, xpts, vars)

    quadpt_strains = np.zeros(8)
    quadpt_strains[:3] = get_membrane_strains(shell_xi_axis, xi, eta, xpts, vars_cov)
    quadpt_strains[3:6] = get_bending_strains(shell_xi_axis, xi, eta, xpts, vars_cov)
    quadpt_strains[6:] = get_trv_shear_strains(shell_xi_axis, xi, eta, xpts, vars_cov)
    return quadpt_strains

def get_quadpt_stresses(shell_xi_axis, xi, eta, xpts, vars, E:float, nu:float, thick:float):
    """get quadpt stresses (important step in the kelem formulation)"""

    quadpt_strains = get_quadpt_strains(shell_xi_axis, xi, eta, xpts, vars)

    # now get A,D,As matrix (for metal)
    C = E / (1.0 - nu**2) * np.array([
        [1, nu, 0], [nu, 1, 0], [0, 0, 0.5 * (1 - nu)],
    ])
    A = C * thick
    D = C * thick**3 / 12.0
    ks = 5.0 / 6.0 # shear correction factor
    As = ks * A[-1, -1] * np.diag([1.0, 1.0])

    # now get the shell stress resultants
    N = np.dot(A, quadpt_strains[:3])
    M = np.dot(D, quadpt_strains[3:6])
    Q = np.dot(As, quadpt_strains[6:])

    quadpt_stresses = np.concatenate([N, M, Q], axis=0)

    return quadpt_stresses

def get_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain):
    """transpose operation of strain sens back to disps"""

    vars_cov_bar = get_membrane_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[:3])
    vars_cov_bar += get_bending_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[3:6])
    vars_cov_bar += get_trv_shear_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain[6:])

    vars_bar = get_cov_nodal_disps_transpose(shell_xi_axis, xpts, vars_cov_bar)
    return vars_bar

def get_quadpt_kelem(shell_xi_axis, xi, eta, xpts, E:float, nu:float, thick:float):
    """get linear kelem stiffness matrix"""

    N = 45
    Kelem = np.zeros((N,N))

    # TODO : need to loop over each of 45 input strains to do hess-vec prod method..
    for i in range(N):
        p_vars = np.zeros(N)
        p_vars[i] = 1.0
        quadpt_stresses = get_quadpt_stresses(shell_xi_axis, xi, eta, xpts, p_vars, E, nu, thick)

        vars_bar = get_strains_transpose(shell_xi_axis, xi, eta, xpts, dstrain=quadpt_stresses)
        Kelem[:,i] = vars_bar

    # plt.imshow(Kelem)
    # plt.show()

    return Kelem

def get_kelem(shell_xi_axis, xpts, E:float, nu:float, thick:float):
    """get full kelem by looping over quadpts and weights"""

    rt_35 = np.sqrt(3.0 / 5.0)
    pts = np.array([-rt_35, 0.0, rt_35])
    weights = np.array([5.0/9, 8.0/9, 5.0/9])

    full_Kelem = np.zeros((45,45))

    for ieta in range(3):
        eta, wt_eta = pts[ieta], weights[ieta]

        for ixi in range(3):
            xi, wt_xi = pts[ixi], weights[ixi]
            weight = wt_xi * wt_eta
            dxy_jac = get_dsurf_jac(xi, eta, xpts)

            _kelem = get_quadpt_kelem(shell_xi_axis, xi, eta, xpts, E, nu, thick)

            full_Kelem += _kelem * weight * dxy_jac

            iquad = 3 * ieta + ixi + 1
            print(f"full kelem done with {iquad=}/9")

    # # _temp = np.log(np.abs(full_Kelem))
    # # _temp[np.abs(full_Kelem) < 1e0] = np.nan
    # plt.imshow(full_Kelem)
    # plt.show()

    return full_Kelem

def get_plate_K_global(nxe, E:float, nu:float, thick:float):
    """get a global stiffness matrix for a plate geometry"""

    shell_xi_axis = np.array([1.0, 0.0, 0.0])
    nx = 2 * nxe + 1
    nnodes = nx**2
    ndof = 5 * nnodes
    h = 1.0 / (nx-1)
    K = np.zeros((ndof, ndof))

    # same elem_xpts shape (just shifts, so make it once)
    elem_xpts = np.zeros(27)
    for inode in range(3):
        x = h * inode
        for jnode in range(3):
            y = h * jnode
            z = 0.0
            node = 3 * jnode + inode
            elem_xpts[3*node:(3*node+3)] = np.array([x, y, z])[:]

    # get kelem once (since it's the same for each element here in this structured grid)
    Kelem = get_kelem(shell_xi_axis, elem_xpts, E, nu, thick)

    max_node = 0

    for iye in range(nxe):
        for ixe in range(nxe):
            node = 2 * iye * nx + 2 * ixe
            elem_nodes = np.array([node, node+1, node+2, node+nx, node+nx+1, node+nx+2, node+2*nx, node+2*nx+1, node+2*nx+2])
            elem_dof = np.array([5*_node+_idof for _node in elem_nodes for _idof in range(5)])
            
            c_max_node = np.max(elem_nodes)
            max_node = np.max([max_node, c_max_node + 1])

            arr_ind = np.ix_(elem_dof, elem_dof)
            K[arr_ind] += Kelem[:,:]

    # make sure we covered all the nodes..
    assert(max_node == nnodes)

    return K

def get_global_plate_loads(nxe, load_type:str="sine-sine", magnitude:float=1.0):

    nx = 2 * nxe + 1
    nnodes = nx**2
    ndof = 5 * nnodes
    F = np.zeros(ndof)
    h = 1.0 / (nx-1)

    if load_type == "sine-sine":
        # apply w bending plate loads...
        for inode in range(nnodes):
            ix, iy = inode % nx, inode // nx
            x, y = ix * h, iy * h

            # (2,1) mode so at least we can see which direction is which
            q = magnitude * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
            F[5 * inode + 2] = q

    else:
        raise AssertionError("No other current load types")
    
    return F

def get_bcs(nxe, uv_fix:bool=False):
    # now apply bcs to the stiffness matrix and forces
    nx = 2 * nxe + 1
    bcs = []
    for iy in range(nx):
        for ix in range(nx):
            inode = nx * iy + ix

            if ix in [0, nx-1] or iy in [0, nx-1]:
                bcs += [5 * inode + 2] # w constr
            
            # if ix in [0, nx-1]: bcs += [5 * inode + 3] # theta_x = 0 on x=const edge (y-varying)
            # if iy in [0, nx-1]: bcs += [5 * inode + 4] # theta y = 0 on y=const edge

            # bcs are flipped from rots case (these are the trv shear strains not alpha,beta which have rotation through director)
            if ix in [0, nx-1]: bcs += [5 * inode + 4] 
            if iy in [0, nx-1]: bcs += [5 * inode + 3] 

            if uv_fix:
                bcs += [5*inode] # u constr
                bcs += [5*inode+1] # v constr
            else:
                if ix == 0: bcs += [5*inode] # u constr
                if iy == 0: bcs += [5*inode+1] # v constr

    bcs = np.array(bcs)
    return bcs

def _apply_bcs_helper(bcs, K, F):
    K[bcs,:] = 0.0
    K[:,bcs] = 0.0
    for bc in bcs:
        K[bc,bc] = 1.0
    F[bcs] = 0.0
    return K, F

def apply_bcs(nxe, K, F):
    """apply bcs to the global stiffness matrix and forces"""
    bcs = get_bcs(nxe)
    return _apply_bcs_helper(bcs, K, F)

def remove_bcs(nxe, _K, _F):
    nx = 2 * nxe + 1
    ndof = 5 * nx**2
    bcs = get_bcs(nxe)
    free_dof = np.array([dof for dof in range(ndof) if not(dof in bcs)])
    K, F = _K[free_dof,:][:,free_dof], _F[free_dof]
    return K, 

def order1_gam_subspace(nxe):
    constrs = []
    nx = 2 * nxe + 1
    nxe_c = nxe // 2
    nx_c = 2 * nxe_c + 1
    # N_c = nx_c**2
    N = nx**2
    ndof = 5 * N

    # print(f"{N=} {ndof=}")

    # TODO : there's something to fix on the bndry here.. (it's not respecting gam_xz or gam_yz constraints on the bndry)

    import scipy as sp

    constrs = []

    def _get_dof(_ix_f, _iy_f, _idof):
        _node = _iy_f * nx + _ix_f
        return 5 * _node + _idof

    for ielem in range(nxe**2):
        ixe, iye = ielem % nxe, ielem // nxe
        ix_f, iy_f = 2 * ixe, 2 * iye

        # now apply list of constraints here..

        for _idof in [3,4]:
            # 5 constraints per each dof and elem (so rotations are actually linear subspace)

            # constraint 1
            _constr = np.zeros(ndof)
            n1, n2, n3 = _get_dof(ix_f, iy_f, _idof), _get_dof(ix_f + 1, iy_f, _idof), _get_dof(ix_f + 2, iy_f, _idof)
            _constr[n1] = 0.5
            _constr[n2] = -1.0
            _constr[n3] = 0.5
            constrs += [_constr]

            # constraint 2
            _constr = np.zeros(ndof)
            n1, n2, n3 = _get_dof(ix_f, iy_f, _idof), _get_dof(ix_f, iy_f + 1, _idof), _get_dof(ix_f, iy_f + 2, _idof)
            _constr[n1] = 0.5
            _constr[n2] = -1.0
            _constr[n3] = 0.5
            constrs += [_constr]

            # constraint 3
            _constr = np.zeros(ndof)
            n1, n2, n3 = _get_dof(ix_f, iy_f, _idof), _get_dof(ix_f + 2, iy_f, _idof), _get_dof(ix_f + 1, iy_f + 1, _idof)
            n4, n5 = _get_dof(ix_f, iy_f + 2, _idof), _get_dof(ix_f + 2, iy_f + 2, _idof)
            _constr[n1] = 0.25
            _constr[n2] = 0.25
            _constr[n3] = -1.0
            _constr[n4] = 0.25
            _constr[n5] = 0.25
            constrs += [_constr]

            # constraint 4
            if iye == nxe - 1: # only add if at end of y elems
                _constr = np.zeros(ndof)
                n1, n2, n3 = _get_dof(ix_f, iy_f + 2, _idof), _get_dof(ix_f + 1, iy_f + 2, _idof), _get_dof(ix_f + 2, iy_f + 2, _idof)
                _constr[n1] = 0.5
                _constr[n2] = -1.0
                _constr[n3] = 0.5
                constrs += [_constr]

            # constraint 5
            if ixe == nxe - 1: # only add if at end of x elems (otherwise redundant)
                _constr = np.zeros(ndof)
                n1, n2, n3 = _get_dof(ix_f + 2, iy_f, _idof), _get_dof(ix_f + 2, iy_f + 1, _idof), _get_dof(ix_f + 2, iy_f + 2, _idof)
                _constr[n1] = 0.5
                _constr[n2] = -1.0
                _constr[n3] = 0.5
                constrs += [_constr]

    constrs = np.array(constrs)
    # constrs = sp.sparse.csr_matrix(constrs)

    Z = sp.linalg.null_space(constrs)

    print(f"{constrs.shape=} {Z.shape=}")

    return Z

def gmg_plate_coarse_fine_matrices(nxe_fine, apply_bcs:bool=True):
    """prolongation or coarse-fine operator of the plate"""
    nxe_c, nxe_f = nxe_fine // 2, nxe_fine
    nelems_c, nelems_f = nxe_c**2, nxe_f**2
    nx_c, nx_f = 2 * nxe_c + 1, 2 * nxe_f + 1
    nnodes_c, nnodes_f = nx_c**2, nx_f**2
    ndof_c, ndof_f = 5 * nnodes_c, 5 * nnodes_f

    I_cf = np.zeros((ndof_f, ndof_c))

    fine_node_cts = np.zeros(nnodes_f).astype(np.int32)

    for ielem_c in range(nelems_c):
        ixe_c, iye_c = ielem_c % nxe_c, ielem_c // nxe_c
        node_c = 2 * iye_c * nx_c + 2 * ixe_c
        c_nodes = [node_c + nx_c * i + j for i in range(3) for j in range(3)]

        # 5x5 set of fine nodes inside each 3x3 coarse elem
        ix_f, iy_f = 4 * ixe_c, 4 * iye_c
        node_f = iy_f * nx_f + ix_f
        f_nodes = np.array([node_f + nx_f * i + j for i in range(5) for j in range(5)])

        # count how many times we added into this fine node.. so that we can average contributions
        fine_node_cts[f_nodes] += 1
        
        # same interp for each DOF.
        # NOTE : if thickness varies throughout elem, then there is an extra adjustment for rots in J. Fish paper
        # I'll have to add, rushing to get demo for meeting, so not doing that yet.. just basic interp

        for nodal_dof in range(5):
            # for single DOF like u or w for instance, we now interp among the nodes.. with 2d lagrange basis..
            c_dof = np.array([5*_node + nodal_dof for _node in c_nodes])
            f_dof = np.array([5 *_node + nodal_dof for _node in f_nodes])

            # loop over each fine node..
            for node, dof in enumerate(f_dof):
                ix_loc, iy_loc = node % 5, node // 5
                xi, eta = (ix_loc / 2.0) - 1.0, (iy_loc / 2.0) - 1.0

                N = lagrange_basis_2d_vec(xi, eta)
                # print(f"{node=} {xi=} {eta=} {N=} {np.sum(N)=}")
                # I_cf[f_dof[node], c_dof] = N
                I_cf[f_dof[node], c_dof] += N
                # not += N?

    I_fc = I_cf.copy().T

    # normalize coarse=>fine by row-sums (since sometimes multiple coarse nodes add to fine, like on edge boundaries)
    for ifine in range(I_cf.shape[0]):
        I_cf[ifine,:] /= np.sum(I_cf[ifine,:])

    # NOTE : should I row-normalize like this?

    if apply_bcs:
        # normalize coarse operator (since it's also basically an interp also)
        for icoarse in range(I_fc.shape[0]):
            I_fc[icoarse,:] /= np.sum(I_fc[icoarse,:])

        # plt.imshow(I_cf)

        # apply bcs (in place, dof not removed)
        fine_bcs = get_bcs(nxe_f)
        coarse_bcs = get_bcs(nxe_c)
        I_cf[fine_bcs,:] = 0.0
        I_cf[:, coarse_bcs] = 0.0

        # apply bcs on fc also
        I_fc[coarse_bcs,:] = 0.0
        I_fc[:,fine_bcs] = 0.0

    return I_cf, I_fc


def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, ndof:int=5):
    """
    Perform Block Gauss-Seidel smoothing for 3 DOF per node.
    A: csr_matrix of size (3*nnodes, 3*nnodes)
    b: RHS vector (3*nnodes,)
    x0: initial guess (3*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    n = A.shape[0] // ndof

    r = b - A @ x0
    init_norm = np.linalg.norm(r)

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                
    # show conv..
    rf = b - A @ x
    fin_norm = np.linalg.norm(rf)
    print(F"block-GS {num_iter=} {init_norm=:.2e} => {fin_norm=:.2e}")

    return x

def damped_jacobi(A:np.ndarray, b:np.ndarray, n_iter:int, x0:np.ndarray=None, omega:float=2.0 / 3.0):
    if x0 is None:
        x0 = np.zeros_like(b)
    x = x0.copy()
    Dinv = 1.0 / np.diag(A)

    for i in range(n_iter):
        r = b - A @ x
        # dx = omega * Dinv * r
        s = Dinv * r

        # energy norm opt update?
        energy_omega = np.dot(r, s) / np.dot(s, np.dot(A, s))
        dx = energy_omega * s
        # dx *= -1
        x += dx

        r_norm = np.linalg.norm(r)
        print(f"{i=} : {r_norm=:.3e}")

    return x