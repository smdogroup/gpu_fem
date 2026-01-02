import numpy as np

# hermite cubic coefficients in xi (a0 + a1*xi + a2*xi^2 + a3*xi^3)
def hermite_cubic_polynomials_1d(ibasis):
    # using standard Hermite basis on [-1,1]:
    if ibasis == 0: # w for node 1
        return [0.5, -0.75, 0.0, 0.25]
    elif ibasis == 1: # dw/dxi for node 1
        return [0.25, -0.25, -0.25, 0.25]
    elif ibasis == 2: # w for node 2
        return [0.5, 0.75, 0.0, -0.25]
    elif ibasis == 3: # dw/dxi for node 2
        return [-0.25, -0.25, 0.25, 0.25]

# linear Lagrange on [-1,1]
def lagrange_polynomials_1d(ibasis):
    if ibasis == 0:
        return [0.5, -0.5]   # 0.5 - 0.5*xi
    elif ibasis == 1:
        return [0.5, 0.5]    # 0.5 + 0.5*xi

def eval_poly(coeffs, xi):
    # coeffs[0] + coeffs[1]*xi + coeffs[2]*xi^2 + ...
    xi_pows = np.array([xi**i for i in range(len(coeffs))])
    return float(np.dot(coeffs, xi_pows))

def poly_derivative(coeffs):
    # returns coefficients of derivative polynomial
    n = len(coeffs)
    if n == 1:
        return [0.0]
    return [coeffs[i+1]*(i+1) for i in range(n-1)]

def poly_second_derivative(coeffs):
    # second derivative coefficients
    return poly_derivative(poly_derivative(coeffs))

def hermite_value(ibasis, xi):
    return eval_poly(hermite_cubic_polynomials_1d(ibasis), xi)

def hermite_d2dx2(ibasis, xi, J):
    # d2/dx2 = (1/J^2) * d2/dxi2
    coeffs = hermite_cubic_polynomials_1d(ibasis)
    d2coeffs = poly_second_derivative(coeffs)
    # print(f"{coeffs=} {d2coeffs=}")
    return (1.0 / (J**2)) * eval_poly(d2coeffs, xi)

def lagrange_value(ibasis, xi):
    return eval_poly(lagrange_polynomials_1d(ibasis), xi)

def lagrange_d1dx(ibasis, xi, J):
    # d/dx = (1/J) * d/dxi
    coeffs = lagrange_polynomials_1d(ibasis)
    dcoeffs = poly_derivative(coeffs)
    return (1.0 / J) * eval_poly(dcoeffs, xi)

def interp_hermite_disp(xi, elem_disp, fine_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""

    # convert rotations back to dw/dxi for interpolation
    hermite_disp = elem_disp[np.array([0, 1, 3, 4])]
    hermite_disp[np.array([1, 3])] *= fine_xscale # dw/dx => dw/dxi
    # hermite disp should interp with dw/dxi smaller (down to fine xscale) to give better conv
    # hard to explain (but in-element rotations typically are exagerrated too much if use coarse xscale)

    w = 0.0
    for ibasis in range(4):
        w += hermite_value(ibasis, xi) * hermite_disp[ibasis]
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

def interp_lagrange_rotation(xi, elem_disp):
    # for some reason the th are much lower when hermite interp (mins energy?) like this (missing high freq error)
    # so trying lagrange basis instead

    thetas = elem_disp[np.array([1,4])]
    theta_shears = elem_disp[np.array([2,5])]
    th, th_s = 0.0, 0.0
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        th += N_i * thetas[ibasis]
        th_s += N_i * theta_shears[ibasis]
    return th, th_s

def interp_hermite_disp_transpose(xi, w_in, fine_xscale):
    """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
    coarse_out = np.zeros(4)
    for ibasis in range(4):
        coarse_out[ibasis] += hermite_value(ibasis, xi) * w_in    
    coarse_out[np.array([1,3])] *= fine_xscale #* 2.0
    coarse_out2 = np.array([coarse_out[0], coarse_out[1], 0.0, coarse_out[2], coarse_out[3], 0.0])
    return coarse_out2

# def interp_hermite_rotation_transpose(xi, th_in, coarse_xscale):
#     """interp the w and th disp here, with elem_disp the hermite cubic DOF [w1, th1, w2, th2]"""
#     w_xi_coarse_in = th_in / coarse_xscale
#     coarse_out = np.zeros(4)
#     for ibasis in range(4):
#         coarse_out[ibasis] += get_hermite_grad(ibasis, xi) * w_xi_coarse_in
#     coarse_out[np.array([1,3])] *= coarse_xscale #* 2.0
#     return coarse_out

def interp_lagrange_rotation_transpose(xi, th_in, th_shear_in):
    theta_out = np.zeros(2)
    theta_shear_out = np.zeros(2)
    
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        theta_out[ibasis] += N_i * th_in
        theta_shear_out[ibasis] += N_i * th_shear_in
    coarse_out = np.array([0.0, theta_out[0], theta_shear_out[0], 0.0, theta_out[1], theta_shear_out[1]])
    return coarse_out

# 3-pt Gauss rule on [-1,1]
def get_quadrature_rule():
    rt35 = np.sqrt(3.0/5.0)
    return [(-rt35, 5.0/9.0), (0.0, 8.0/9.0), (rt35, 5.0/9.0)]

# whether the kirchoff part of the rotation is stored as dw/dxi or dw/dx
# in particular if True, it's stored as dw/dx
# initially putting this to True breaks mesh convergence of this element (but does give right displacements)
# needs to be True if want to use on plate and multigrid
physical_rots = True
# physical_rots = False

def get_kelem(J, EI, GA, k_shear=5.0/6.0, use_reduced_integration_for_shear=True, schur_complement=False):
    """
    J : Jacobian = L/2 (so L = 2*J)
    EI : bending stiffness
    GA : shear rigidity (A*G)  (we will multiply by k_shear)
    k_shear : shear correction factor (default 5/6)
    """
    nquad = 3
    nbasis = 6   # [w1, th1, w2, th2, gs1, gs2] order before reordering
    Kelem = np.zeros((nbasis, nbasis))
    quads = get_quadrature_rule()

    # If doing reduced integration for shear, we evaluate shear at single point (xi=0)
    if use_reduced_integration_for_shear:
        shear_quads = [(0.0, 2.0)]  # one-point Gauss on [-1,1] has weight 2
    else:
        shear_quads = quads

    # bending (EI * int (w_xx - th_s_x)^2 dx) expands into EI*(w_xx*w_xx - 2 w_xx th_s_x + th_s_x th_s_x)
    for xi, weight in quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J
        # build required derivatives/values at xi
        d2phi = [hermite_d2dx2(i, xi, J) for i in range(4)]

        # th_s uses Lagrange; its x-derivative:
        dpsi = [lagrange_d1dx(i, xi, J) for i in range(2)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [lagrange_value(i, xi) for i in range(2)]

        # assemble bending-related terms (w-w, w-ths, ths-ths)
        # index mapping prior to reorder (0..5): w1, th1, w2, th2, gs1, gs2
        for i in range(4):            
            for j in range(4):
                
                # make rot DOF of hermite cubic proper units dw/dx not dw/dxi (for multigrid)
                scaling = 1.0
                if physical_rots and i % 2 == 1: scaling *= J
                if physical_rots and j % 2 == 1: scaling *= J

                Kelem[i,j] += scaling * EI * qfactor * d2phi[i] * d2phi[j]
            
            # coupling w_i with shear DOFs (th_s)
            for j_s in range(2):
                # make rot DOF of hermite cubic proper units dw/dx not dw/dxi (for multigrid)
                scaling = 1.0
                if physical_rots and i % 2 == 1: scaling *= J

                idx_s = 4 + j_s
                Kelem[i, idx_s] -= scaling * EI * qfactor * d2phi[i] * dpsi[j_s]   # -2 folded later (off-diagonal)
                Kelem[idx_s, i] -= scaling * EI * qfactor * dpsi[j_s] * d2phi[i]
        # th_s-th_s part from EI*(th_s_x)^2
        for i_s in range(2):
            for j_s in range(2):
                Kelem[4+i_s, 4+j_s] += EI * qfactor * dpsi[i_s] * dpsi[j_s]

    # shear penalty kGA * ∫ th_s^2 dx
    for xi, weight in shear_quads:
        qfactor = weight * J
        psi_val = [lagrange_value(i, xi) for i in range(2)]
        for i_s in range(2):
            for j_s in range(2):
                Kelem[4+i_s, 4+j_s] += k_shear * GA * qfactor * psi_val[i_s] * psi_val[j_s] * 8.0
                #* 10.0 # NOTE not sure where the 8x is missing here..
    # probably 8x comes from something wrong with dx/2 (in hermite cubic EB case with 1/h**3 and the dx/2 and that is propagating over to Timoshenko)

    # import matplotlib.pyplot as plt
    # # h = J * 2.0
    # # Kelem_exact = 2.0 / h**3 * np.array([
    # #     [6, -3 * h, -6, -3 * h],
    # #     [-3 * h, 2 * h**2, 3 * h, h**2],
    # #     [-6, 3 * h, 6, 3 * h],
    # #     [-3 * h, h**2, 3 * h, 2*h**2],
    # # ])

    # # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # # ax[0].imshow(Kelem[:4,:][:,:4] / EI)
    # # ax[1].imshow(Kelem_exact)
    # # plt.show()

    # plt.imshow(Kelem)
    # # plt.imshow(Kelem / k_shear / GA / (J * 2.0))
    # plt.show()

    if schur_complement:
        # remove the gamma DOF from the system
        Kaa = Kelem[:4, :][:,:4]
        Kab = Kelem[:4,:][:,4:]
        Kba = Kelem[4:,:][:,:4]
        Kbb = Kelem[4:,:][:,4:]

        Kelem = Kaa - Kab @ np.linalg.inv(Kbb) @ Kba
        return Kelem

    else:
        # reorder to your desired ordering: [w1, th1, gam1, w2, th2, gam2]
        new_order = np.array([0, 1, 4, 2, 3, 5])
        Kelem = Kelem[new_order, :][:, new_order]
        return Kelem


def get_felem(J, schur_complement=False):
    """get element load vector"""
    nquad = 3
    nbasis = 6
    quads = get_quadrature_rule()
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = quads[iquad]
        for ibasis in range(nbasis):
            if ibasis < 4:
                scaling = 1.0
                if physical_rots and ibasis % 2 == 1: scaling *= J

                felem[ibasis] += scaling * weight * J * hermite_value(ibasis, xi)
    
    if schur_complement:
        return felem[:4]

    else:

        new_order = np.array([0, 1, 4, 2, 3, 5])
        felem = felem[new_order]
        return felem