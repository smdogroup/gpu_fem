# we choose the hermite cubic polynomials as the basis functions of the element
import matplotlib.pyplot as plt
import numpy as np

def hermite_cubic_polynomials_1d(ibasis):
    # node 1 is xi = -1, node 2 is xi = 1
    if ibasis == 0: # w for node 1
        return [0.5, -0.75, 0.0, 0.25]
    elif ibasis == 1: # dw/dxi for node 1
        return [0.25, -0.25, -0.25, 0.25]
    elif ibasis == 2: # w for node 2
        return [0.5, 0.75, 0.0, -0.25]
    elif ibasis == 3: # dw/dxi for node 2
        return [-0.25, -0.25, 0.25, 0.25]
    
def eval_polynomial(poly_list, value):
    poly_list_arr = np.array(poly_list)
    var_list_arr = np.array([value**(ind) for ind in range(len(poly_list))])
    return np.dot(poly_list_arr, var_list_arr)

def hermite_cubic_1d(ibasis, xi):
    poly_list = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(poly_list, xi)

def plot_hermite_cubic():
    xi_vec = np.linspace(-1, 1, 100)
    for ibasis in range(4):
        poly = hermite_cubic_polynomials_1d(ibasis)
        h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
        plt.plot(xi_vec, h_vec, label=f"phi_{ibasis}")
    plt.legend()
    plt.show()

# and the following quadrature rule for 1D elements
def get_quadrature_rule(order:int=3):
    if order <= 3:
        rt35 = np.sqrt(3.0/5.0)
        return [(-rt35, 5.0/9.0), (0.0, 8.0/9.0), (rt35, 5.0/9.0)]
    elif order == 4:
        pts = [
            np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0)**0.5),
            np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0)**0.5)
        ]
        wts = [
            (18.0 + 30.0**0.5) / 36.0,
            (18.0 - 30.0**0.5) / 36.0,
        ]

        return [
            (-pts[0], wts[0]),
            (pts[0], wts[0]),
            (-pts[1], wts[1]),
            (pts[1], wts[1]),
        ]
    
    
# this is how to compute the element stiffness matrix:

def get_hermite_hess(ibasis, xi, xscale):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)

    dphi_xi2_poly = [2.0 * xi_poly[-2], 6.0 * xi_poly[-1]]
    dphi_xi2 = eval_polynomial(dphi_xi2_poly, xi)
    dphi_dx2 = 1.0/xscale**2 * dphi_xi2
    return dphi_dx2

def lagrange_polynomials_1d(ibasis):
    if ibasis == 0:
        return [0.5, -0.5]   # 0.5 - 0.5*xi
    elif ibasis == 1:
        return [0.5, 0.5]    # 0.5 + 0.5*xi

def lagrange_value(ibasis, xi):
    return eval_polynomial(lagrange_polynomials_1d(ibasis), xi)

def get_lagrange_grad(ibasis, xi, xscale):
    if ibasis == 0:
        return -0.5 / xscale
    else:
        return 0.5 / xscale

def get_basis_fcn(ibasis, xi):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(xi_poly, xi)

def get_hermite_grad(ibasis, xi, xscale=1.0):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)

    dphi_dxi_poly = [xi_poly[-3], 2.0 * xi_poly[-2], 3.0 * xi_poly[-1]]
    dphi_dxi = eval_polynomial(dphi_dxi_poly, xi)
    return dphi_dxi / xscale

def interp_hermite_disp(xi, elem_disp, fine_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    elem_disp2 = elem_disp.copy()
    elem_disp2[np.array([1, 3])] *= fine_xscale # dw/dx => dw/dxi
    # even though we take in coarse rotations
    # and interp on coarse element, we ought to use dw/dxi of fine xscale (makes it smaller rotations for fine size)
    # to interp to fine elements.. (this correction fixed convergence, a bit tricky for hermite here)

    w = 0.0
    for ibasis in range(4):
        w += get_basis_fcn(ibasis, xi) * elem_disp2[ibasis]
        # w_xi_coarse += get_hermite_grad(ibasis, xi) * elem_disp2[ibasis]
    return w

def interp_hermite_rotation(xi, elem_disp, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    elem_disp2 = elem_disp.copy()
    elem_disp2[np.array([1, 3])] *= coarse_xscale # dw/dx => dw/dxi
    w_xi_coarse = 0.0
    for ibasis in range(4):
        w_xi_coarse += get_hermite_grad(ibasis, xi) * elem_disp2[ibasis]
    th = w_xi_coarse / coarse_xscale
    return th

def interp_lagrange_rotation(xi, elem_disp):
    # for some reason the th are much lower when hermite interp (mins energy?) like this (missing high freq error)
    # so trying lagrange basis instead

    thetas = elem_disp[np.array([1,3])]
    th = 0.0
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        th += N_i * thetas[ibasis]
    return th

def interp_hermite_disp_transpose(xi, w_in, fine_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out = np.zeros(4)
    for ibasis in range(4):
        coarse_out[ibasis] += get_basis_fcn(ibasis, xi) * w_in
        # coarse_out[ibasis] += get_hermite_grad(ibasis, xi) * w_xi_in
    
    # even though it's interp from coarse, you should interp it like it's fine xscale (this correction stabilizes prolong..)
    # a bit tricky in hermite cubic

    # TODO : how to fix that?
    coarse_out[np.array([1,3])] *= fine_xscale
    return coarse_out

def interp_hermite_rotation_transpose(xi, th_in, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    w_xi_coarse_in = th_in / coarse_xscale
    coarse_out = np.zeros(4)
    for ibasis in range(4):
        coarse_out[ibasis] += get_hermite_grad(ibasis, xi) * w_xi_coarse_in
    coarse_out[np.array([1,3])] *= coarse_xscale #* 2.0
    return coarse_out

def interp_lagrange_rotation_transpose(xi, th_in):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out1 = np.zeros(2)
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        coarse_out1[ibasis] = N_i * th_in

    coarse_out = np.array([0.0, coarse_out1[0], 0.0, coarse_out1[1]])
    return coarse_out

def get_nl_kelem(xscale, EI, EA, elem_disp):
    """get element stiffness matrix"""
    nquad = 4
    nbasis = 6

    quad_pts = get_quadrature_rule(nquad)

    w_dof = elem_disp[np.array([1,2,4,5])].copy()
    # convert rot DOF from dw/dx to dw/dxi units
    w_dof[np.array([1,3])] *= xscale

    Kelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        xi, weight = quad_pts[iquad]

        # compute nonlinear parts of strain
        dwdx = np.dot(w_dof, np.array([get_hermite_grad(ibasis, xi, xscale) for ibasis in range(4)]) )
        dwdx_sq = dwdx * dwdx
        
        for i in range(2):
            for j in range(2):

                # uu term
                factor = weight * xscale
                Kelem[i,j] += factor * EA * get_lagrange_grad(i, xi, xscale) * get_lagrange_grad(j, xi, xscale)

            for k in range(4):
                # uw term and wu term
                factor = weight * xscale
                if k % 2 == 1: factor *= xscale # hermite rescaling
                Kelem[i, 2+k] += factor * EA * dwdx * get_lagrange_grad(i, xi, xscale) * get_hermite_grad(k, xi, xscale)
                Kelem[2+k, i] += factor * EA * dwdx * get_lagrange_grad(i, xi, xscale) * get_hermite_grad(k, xi, xscale)
                

        # hermite term here ww part
        for i in range(4):
            for j in range(4):
                factor = weight * xscale
                if i % 2 == 1: factor *= xscale
                if j % 2 == 1: factor *= xscale
                Kelem[2+i,2+j] += factor * EI * get_hermite_hess(i, xi, xscale) * get_hermite_hess(j, xi, xscale)
                Kelem[2+i,2+j] += factor * EA * dwdx_sq * get_hermite_grad(i, xi, xscale) * get_hermite_grad(j, xi, xscale)
    
    # now reorder it [u; w] block parts to nodal
    reorder = np.array([0,2,3,1,4,5])
    Kelem = Kelem[reorder,:][:,reorder]

    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # ax[0].imshow(Kelem)
    # ax[1].imshow(Kelem_exact)
    # plt.show()
    return Kelem

def get_felem(xscale):
    """get element load vector"""
    nquad = 4
    nbasis = 6

    quad_pts = get_quadrature_rule(nquad)

    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = quad_pts[iquad]
        for ibasis in range(4):
            scaling = 1.0
            if ibasis % 2 == 1: scaling *= xscale
            felem[2 + ibasis] += scaling * weight * xscale * get_basis_fcn(ibasis, xi)
    reorder = np.array([0,2,3,1,4,5])
    felem = felem[reorder]
    return felem