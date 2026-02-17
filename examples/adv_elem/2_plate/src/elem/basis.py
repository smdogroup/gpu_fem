import numpy as np

# =============================
# QUADRATURE
# =============================

def zero_order_quadrature():
    return [0.0], [2.0] # why 2.0 here? because 1/2 in integration?

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
        return -0.25 + 0.5 * xi + 0.75 * xi**2
    
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
    

def get_lagrange_basis_01(xi):
    N = np.array([1.0 - xi, xi])
    dN = np.array([-1.0, 1.0])
    return N, dN

def get_lagrange_basis(xi):
    N = 0.5 * np.array([1.0 - xi, 1.0 + xi])
    dN = 0.5 * np.array([-1.0, 1.0])
    return N, dN

def get_lagrange_basis_2d_all(xi, eta):
    N1, dN1 = get_lagrange_basis(xi)
    N2, dN2 = get_lagrange_basis(eta)
    N = np.zeros(4);  Nxi = np.zeros(4);  Neta = np.zeros(4)
    
    # for n in range(4):
    #     i = n % 2; j = n // 2
    mapping = [(0,0), (1,0), (1,1), (0,1)]
    for n,(i,j) in enumerate(mapping):
        N[n] = N1[i] * N2[j]
        Nxi[n] = dN1[i] * N2[j]
        Neta[n] = N1[i] * dN2[j]
    return N, Nxi, Neta

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


# ================================
# IGA 
# ================================

def quad_bernstein(xi):
    N = np.array([(1-xi)**2, 2*xi*(1-xi), xi**2])
    dN = np.array([-2*(1-xi), 2*(1-2*xi), 2*xi])
    return N, dN

def get_quad_bernstein_hess(xi):
    return np.array([2.0, -4.0, 2.0])

def get_iga2_basis_all(xi, left_bndry: bool, right_bndry: bool):
    # Bernstein + derivs
    B, dB = quad_bernstein(xi)
    B2 = get_quad_bernstein_hess(xi)

    # same linear maps applied to B, dB, B2
    N  = 0.5 * np.array([B[0],  np.sum(B)  + B[1],  B[2]])
    dN = 0.5 * np.array([dB[0], np.sum(dB) + dB[1], dB[2]])
    d2N= 0.5 * np.array([B2[0], np.sum(B2) + B2[1], B2[2]])

    if left_bndry:
        N   += 0.5 * np.array([ B[0],  -B[0],  0.0])
        dN  += 0.5 * np.array([ dB[0], -dB[0], 0.0])
        d2N += 0.5 * np.array([ B2[0], -B2[0], 0.0])

    if right_bndry:
        N   += 0.5 * np.array([0.0, -B[2],  B[2]])
        dN  += 0.5 * np.array([0.0, -dB[2], dB[2]])
        d2N += 0.5 * np.array([0.0, -B2[2], B2[2]])

    return N, dN, d2N

def get_iga2_basis(xi, left_bndry, right_bndry):
    N, dN, _ = get_iga2_basis_all(xi, left_bndry, right_bndry)
    return N, dN

def get_iga2_hess(xi, left_bndry, right_bndry):
    _, _, d2N = get_iga2_basis_all(xi, left_bndry, right_bndry)
    return d2N

# def get_iga2_basis_2d(xi, eta, left_bndry, right_bndry, bot_bndry, top_bndry):
#     N1, dN1 = get_iga2_basis(xi, left_bndry, right_bndry)
#     N2, dN2 = get_iga2_basis(eta, bot_bndry, top_bndry)
#     N = np.zeros(9)
#     Nxi = np.zeros(9)
#     Neta = np.zeros(9)
#     for ii in range(9):
#         i1, i2 = ii % 3; ii // 3
#         N[ii] = N1[i1] * N2[i2]
#         Nxi[ii] = dN1[i1] * N2[i2]
#         Neta[ii] = N1[i1] * dN2[i2]

#     return N, Nxi, Neta


def get_iga2_basis_2d_all(xi, eta, left_bndry, right_bndry, bot_bndry, top_bndry):
    Nx, dNx, d2Nx = get_iga2_basis_all(xi,  left_bndry, right_bndry)
    Ny, dNy, d2Ny = get_iga2_basis_all(eta, bot_bndry,  top_bndry)

    N      = np.zeros(9)
    N_xi   = np.zeros(9)
    N_eta  = np.zeros(9)
    N_xixi = np.zeros(9)
    N_etaeta = np.zeros(9)
    N_xieta  = np.zeros(9)

    for a in range(9):
        i = a % 3
        j = a // 3   # <-- you had a bug here (you overwrote ii)

        N[a]        = Nx[i]   * Ny[j]
        N_xi[a]     = dNx[i]  * Ny[j]
        N_eta[a]    = Nx[i]   * dNy[j]
        N_xixi[a]   = d2Nx[i] * Ny[j]
        N_etaeta[a] = Nx[i]   * d2Ny[j]
        N_xieta[a]  = dNx[i]  * dNy[j]

    return N, N_xi, N_eta, N_xixi, N_etaeta, N_xieta


# third order IGA
def cubic_bernstein(xi):
    """
    Cubic Bernstein basis on [0,1].
    Returns B, dB/dxi
    """
    x = xi
    omx = 1.0 - x

    B0 = omx**3
    B1 = 3.0 * x * omx**2
    B2 = 3.0 * x**2 * omx
    B3 = x**3

    # Derivatives w.r.t. xi
    dB0 = -3.0 * omx**2
    dB1 = 3.0 * omx**2 - 6.0 * x * omx
    dB2 = 6.0 * x * omx - 3.0 * x**2
    dB3 = 3.0 * x**2

    B  = np.array([B0, B1, B2, B3])
    dB = np.array([dB0, dB1, dB2, dB3])
    return B, dB


def cubic_bernstein_hess(xi):
    """
    Second derivatives d2B/dxi2 for cubic Bernstein on [0,1].
    """
    x = xi
    omx = 1.0 - x

    d2B0 = 6.0 * omx
    d2B1 = -12.0 + 18.0 * x
    d2B2 = 6.0 - 18.0 * x
    d2B3 = 6.0 * x

    return np.array([d2B0, d2B1, d2B2, d2B3])


# NOTE : THIS CODE IS NOT RIGHT - see adv_elems/3_cylinder/

# def get_iga3_basis(xi, left_bndry: bool, right_bndry: bool):
#     """
#     Cubic (p=3) 'hierarchic' basis built from Bernstein polynomials.
#     Returns N (len 4) and dN/dxi (len 4).

#     Local DOFs correspond to 4 control-like modes per element.

#     Base hierarchical modes:
#       N0 = B0
#       N1 = (B1 + B2)   (center / partition helper)
#       N2 = (B1 - B2)   (shape mode)
#       N3 = B3

#     Boundary adjustments follow your IGA2 spirit: redistribute interior modes
#     into the end modes when on the first/last element.
#     """
#     B, dB = cubic_bernstein(xi)

#     # hierarchical transform
#     N  = np.array([B[0], (B[1] + B[2]), (B[1] - B[2]), B[3]])
#     dN = np.array([dB[0], (dB[1] + dB[2]), (dB[1] - dB[2]), dB[3]])

#     # boundary adjustments:
#     # On the first element, you typically want an "open" end behavior where
#     # interior content is pulled into the boundary mode to mimic open-uniform knots.
#     if left_bndry:
#         # pull some of N1 into N0, and zero it out correspondingly
#         N[0]  += 0.5 * N[1]
#         N[1]  -= 0.5 * N[1]
#         dN[0] += 0.5 * dN[1]
#         dN[1] -= 0.5 * dN[1]

#         # optionally also damp the antisymmetric mode at the very boundary
#         # (keeps things well-behaved and similar to your p=2 end tweak)
#         N[2]  *= 0.5
#         dN[2] *= 0.5

#     if right_bndry:
#         # pull some of N1 into N3
#         N[3]  += 0.5 * N[1]
#         N[1]  -= 0.5 * N[1]
#         dN[3] += 0.5 * dN[1]
#         dN[1] -= 0.5 * dN[1]

#         # damp antisymmetric mode at boundary
#         N[2]  *= 0.5
#         dN[2] *= 0.5

#     return N, dN


# def get_iga3_hess(xi, left_bndry: bool, right_bndry: bool):
#     """
#     Second derivatives d^2N/dxi^2 for the cubic hierarchical basis above.
#     """
#     B2 = cubic_bernstein_hess(xi)

#     N2 = np.array([B2[0], (B2[1] + B2[2]), (B2[1] - B2[2]), B2[3]])

#     if left_bndry:
#         N2[0] += 0.5 * N2[1]
#         N2[1] -= 0.5 * N2[1]
#         N2[2] *= 0.5

#     if right_bndry:
#         N2[3] += 0.5 * N2[1]
#         N2[1] -= 0.5 * N2[1]
#         N2[2] *= 0.5

#     return N2
