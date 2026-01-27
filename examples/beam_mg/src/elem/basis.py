import numpy as np

# =============================
# QUADRATURE
# =============================

def zero_order_quadrature():
    return [0.0], [1.0]

def first_order_quadrature():
    irt3 = np.sqrt(1.0/3.0)
    return [-irt3, irt3], [1.0, 1.0]

def second_order_quadrature():
    rt35 = np.sqrt(3.0 / 5.0)
    return [-rt35, 0.0, rt35], [5.0/9.0, 8.0/9.0, 5.0/9.0]

def third_order_quadrature():
    a =  0.8611363115940526
    b = 0.3399810435848563
    wa = 0.3478548451374539
    wb = 0.6521451548625461
    return [-a, -b, b, a], [wa, wb, wb, wa]

# =============================
# BASIS
# =============================

def hermite_cubic(ibasis, xi):
        if ibasis == 0: # w for node 1
            return 0.5 - 0.75 * xi + 0.25 * xi**3
        elif ibasis == 1: # dw/dxi for node 1
            return 0.25 - 0.25 * xi - 0.25 * xi**2 + 0.25 * xi**3
        elif ibasis == 2: # w for node 2
            return 0.5 + 0.75 * xi - 0.25 * xi**3
        elif ibasis == 3:
            return -0.25 - 0.25 * xi + 0.25 * xi**2 + 0.25 * xi**3

def hermite_cubic_grad(ibasis, xi):
    if ibasis == 0:
        return -0.75 + 0.75 * xi**2
    elif ibasis == 1:
        return -0.25 - 0.5 * xi + 0.75 * xi**2
    elif ibasis == 2:
        return 0.75 - 0.75 * xi**2
    elif ibasis == 3:
        return -0.25 - 0.5 * xi + 0.75 * xi**2
    
def hermite_cubic_hess(ibasis, xi):
    if ibasis == 0:
        return 1.5 * xi
    elif ibasis == 1:
        return -0.5 + 1.5 * xi
    elif ibasis == 2:
        return -1.5 * xi
    elif ibasis == 3:
        return 0.5 + 1.5 * xi
    
def lagrange(ibasis, xi):
    if ibasis == 0:
        return 0.5 - 0.5 * xi
    elif ibasis == 1:
        return 0.5 + 0.5 * xi
    
def lagrange_grad(ibasis, xi, J:float):
    if ibasis == 0:
        return -0.5 / J
    elif ibasis == 1:
        return 0.5 / J
    

# ====================================
# multigrid interpolations
# ====================================


def interp_hermite_disp(xi, elem_disp, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    elem_disp2 = elem_disp.copy()
    elem_disp2[np.array([1, 3])] *= coarse_xscale # dw/dx => dw/dxi
    w = 0.0
    for ibasis in range(4):
        # w += get_basis_fcn(ibasis, xi) * elem_disp2[ibasis]
        w += hermite_cubic(ibasis, xi) * elem_disp2[ibasis]
    return w

def interp_lagrange_rotation(xi, elem_disp):
    # for some reason the th are much lower when hermite interp (mins energy?) like this (missing high freq error)
    # so trying lagrange basis instead
    thetas = elem_disp[np.array([1,3])]
    th = 0.0
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        th += N_i * thetas[ibasis]
    return th

def interp_lagrange(xi, elem_disp):
    w, th = 0.0, 0.0
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        w += N_i * elem_disp[2 * ibasis]
        th += N_i * elem_disp[2 * ibasis + 1]
    return w, th

def interp_lagrange_transpose(xi, nodal_in):
    w_in, th_in = nodal_in[0], nodal_in[1]
    coarse_out = np.zeros(4)
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        coarse_out[2 * ibasis] += N_i * w_in
        coarse_out[2 * ibasis + 1] += N_i * th_in
    return coarse_out

def interp_hermite_disp_transpose(xi, w_in, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out = np.zeros(4)
    for ibasis in range(4):
        coarse_out[ibasis] += hermite_cubic(ibasis, xi) * w_in
    coarse_out[np.array([1,3])] *= coarse_xscale
    return coarse_out

def interp_lagrange_rotation_transpose(xi, th_in):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out1 = np.zeros(2)
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        coarse_out1[ibasis] = N_i * th_in

    coarse_out = np.array([0.0, coarse_out1[0], 0.0, coarse_out1[1]])
    return coarse_out


def interp_hermite_disp(xi, elem_disp, fine_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    hermite_disp = elem_disp[np.array([0, 1, 3, 4])]
    hermite_disp[np.array([1, 3])] *= fine_xscale # dw/dx => dw/dxi
    # hermite disp should interp with dw/dxi smaller (down to fine xscale) to give better conv
    # hard to explain (but in-element rotations typically are exagerrated too much if use coarse xscale)

    w = 0.0
    for ibasis in range(4):
        w += hermite_cubic(ibasis, xi) * hermite_disp[ibasis]
        # w_xi_coarse += get_hermite_grad(ibasis, xi) * elem_disp2[ibasis]
    return w

# def interp_hermite_rotation(xi, elem_disp, fine_xscale):
#     """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

#     # convert rotations back to dw/dxi for interpolation
#     hermite_disp = elem_disp[np.array([0, 1, 3, 4])]
#     hermite_disp[np.array([1, 3])] *= fine_xscale # dw/dx => dw/dxi
#     w_xi_coarse = 0.0
#     for ibasis in range(4):
#         w_xi_coarse += get_hermite_grad(ibasis, xi) * hermite_disp[ibasis]
#     th = w_xi_coarse / fine_xscale
#     return th

# =======================================
# HERMITE
# =======================================


def interp6_hermite_disp(xi, elem_disp, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    hermite_disp = elem_disp[np.array([0, 1, 3, 4])]
    hermite_disp[np.array([1, 3])] *= coarse_xscale # dw/dx => dw/dxi
    # hermite disp should interp with dw/dxi smaller (down to fine xscale) to give better conv
    # hard to explain (but in-element rotations typically are exagerrated too much if use coarse xscale)

    w = 0.0
    for ibasis in range(4):
        w += hermite_cubic(ibasis, xi) * hermite_disp[ibasis]
    return w

def interp6_lagrange_rotation(xi, elem_disp):
    # for some reason the th are much lower when hermite interp (mins energy?) like this (missing high freq error)
    # so trying lagrange basis instead

    thetas = elem_disp[np.array([1,4])]
    theta_shears = elem_disp[np.array([2,5])]
    th, th_s = 0.0, 0.0
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        th += N_i * thetas[ibasis]
        th_s += N_i * theta_shears[ibasis]
    return th, th_s

def interp6_hermite_disp_transpose(xi, w_in, coarse_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out = np.zeros(4)
    for ibasis in range(4):
        coarse_out[ibasis] += hermite_cubic(ibasis, xi) * w_in    
    coarse_out[np.array([1,3])] *= coarse_xscale #* 2.0
    coarse_out2 = np.array([coarse_out[0], coarse_out[1], 0.0, coarse_out[2], coarse_out[3], 0.0])
    return coarse_out2

def interp6_lagrange_rotation_transpose(xi, th_in, th_shear_in):
    theta_out = np.zeros(2)
    theta_shear_out = np.zeros(2)
    
    for ibasis in range(2):
        N_i = lagrange(ibasis, xi)
        theta_out[ibasis] += N_i * th_in
        theta_shear_out[ibasis] += N_i * th_shear_in
    coarse_out = np.array([0.0, theta_out[0], theta_shear_out[0], 0.0, theta_out[1], theta_shear_out[1]])
    return coarse_out