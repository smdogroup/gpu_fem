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
def get_quadrature_rule(iquad):
    # 3rd order
    rt35 = np.sqrt(3.0/5.0)
    if iquad == 0:
        return -rt35, 5.0/9.0
    elif iquad == 1:
        return 0.0, 8.0/9.0
    elif iquad == 2:
        return rt35, 5.0/9.0
    
# this is how to compute the element stiffness matrix:
def get_hess(ibasis, xi, xscale):
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

def get_basis_fcn(ibasis, xi):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(xi_poly, xi)

def get_hermite_grad(ibasis, xi):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)

    dphi_dxi_poly = [xi_poly[-3], 2.0 * xi_poly[-2], 3.0 * xi_poly[-1]]
    dphi_dxi = eval_polynomial(dphi_dxi_poly, xi)
    return dphi_dxi

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

def get_kelem(xscale):
    """get element stiffness matrix"""
    nquad = 3
    nbasis = 4
    Kelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule(iquad)
        for i in range(nbasis):
            for j in range(nbasis):
                factor = weight * xscale
                if i % 2 == 1: factor *= xscale
                if j % 2 == 1: factor *= xscale
                Kelem[i,j] += factor * get_hess(i, xi, xscale) * get_hess(j, xi, xscale)
    
    # compare Kelem to exact from Reddy FEA book (with EI = 1 at this step)
    # h = xscale
    # Kelem_exact = 2.0 / h**3 * np.array([
    #     [6, -3 * h, -6, -3 * h],
    #     [-3 * h, 2 * h**2, 3 * h, h**2],
    #     [-6, 3 * h, 6, 3 * h],
    #     [-3 * h, h**2, 3 * h, 2*h**2],
    # ])

    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # ax[0].imshow(Kelem)
    # ax[1].imshow(Kelem_exact)
    # plt.show()
    return Kelem

def get_felem(xscale):
    """get element load vector"""
    nquad = 3
    nbasis = 4
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule(iquad)
        for ibasis in range(nbasis):
            scaling = 1.0
            if ibasis % 2 == 1: scaling *= xscale
            felem[ibasis] += scaling * weight * xscale * get_basis_fcn(ibasis, xi)
    return felem