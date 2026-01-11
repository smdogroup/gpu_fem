import numpy as np

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

def lagrange_value(ibasis, xi):
    return eval_poly(lagrange_polynomials_1d(ibasis), xi)

def lagrange_d1dx(ibasis, xi, J):
    # d/dx = (1/J) * d/dxi
    coeffs = lagrange_polynomials_1d(ibasis)
    dcoeffs = poly_derivative(coeffs)
    return (1.0 / J) * eval_poly(dcoeffs, xi)

# 3-pt Gauss rule on [-1,1]
def get_quadrature_rule():
    rt35 = np.sqrt(3.0/5.0)
    return [(-rt35, 5.0/9.0), (0.0, 8.0/9.0), (rt35, 5.0/9.0)]

def interp_lagrange(xi, elem_disp):
    w, th = 0.0, 0.0
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        w += N_i * elem_disp[2 * ibasis]
        th += N_i * elem_disp[2 * ibasis + 1]
    return w, th

def interp_lagrange_transpose(xi, nodal_in):
    w_in, th_in = nodal_in[0], nodal_in[1]
    coarse_out = np.zeros(4)
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        coarse_out[2 * ibasis] += N_i * w_in
        coarse_out[2 * ibasis + 1] += N_i * th_in
    return coarse_out

def get_kelem(J, EI, GA, k_shear=5.0/6.0, use_reduced_integration_for_shear=True, use_nondim:bool=False):
    """
    J : Jacobian = L/2 (so L = 2*J)
    EI : bending stiffness
    GA : shear rigidity (A*G)  (we will multiply by k_shear)
    k_shear : shear correction factor (default 5/6)
    """
    nbasis = 4   # [w1, th1, w2, th2] order before reordering
    Kelem = np.zeros((nbasis, nbasis))
    quads = get_quadrature_rule()

    # If doing reduced integration for shear, we evaluate shear at single point (xi=0)
    if use_reduced_integration_for_shear:
        shear_quads = [(0.0, 2.0)]  # one-point Gauss on [-1,1] has weight 2
    else:
        shear_quads = quads

    GA *= 8.0 # correction?

    # bending (EI * int (w_xx - th_s_x)^2 dx) expands into EI*(w_xx*w_xx - 2 w_xx th_s_x + th_s_x th_s_x)
    for xi, weight in shear_quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J

        Le = J * 2.0
        gam = np.sqrt(EI / k_shear / GA / 0.125)
        J2 = J if not(use_nondim) else J / gam #/ Le # since w has changed

        # th_s uses Lagrange; its x-derivative:
        dpsi = [lagrange_d1dx(i, xi, J) * gam for i in range(2)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [lagrange_value(i, xi) for i in range(2)]

        # first 2x2 block Kww before reordering
        for i in range(2):
            for j in range(2):
                
                # kww block
                Kelem[i,j] += k_shear * GA * qfactor * dpsi[i] * dpsi[j]

                # k_[w,th] block
                Kelem[i,2 + j] += -k_shear * GA * qfactor * dpsi[i] * psi_val[j]

                # and transpose
                Kelem[2 + i,j] += -k_shear * GA * qfactor * psi_val[i] * dpsi[j]

                # K_[th,th] block
                Kelem[2 + i, 2 + j] += k_shear * GA * qfactor * psi_val[i] * psi_val[j]

    if gam <= 0.1: 
        cond_num = np.linalg.cond(Kelem)
        print(f"{gam=:.2e} {cond_num=:.2e}")
    
        # import matplotlib.pyplot as plt
        # plt.imshow(Kelem)
        # plt.show()

    for xi, weight in quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J

        # th_s uses Lagrange; its x-derivative:
        dpsi = [lagrange_d1dx(i, xi, J) for i in range(2)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [lagrange_value(i, xi) for i in range(2)]

        for i in range(2):
            for j in range(2):
                Kelem[2 + i, 2 + j] += EI * qfactor * dpsi[i] * dpsi[j]

    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # ax[0].imshow(Kelem[:4,:][:,:4] / EI)
    # ax[1].imshow(Kelem_exact)
    # plt.show()

    # plt.imshow(Kelem)
    # plt.imshow(Kelem / k_shear / GA / (J * 2.0))
    # plt.show()

    # change order from [w1,w2,th1,th2] => [w1, th1, w2, th2]
    new_order = np.array([0, 2, 1, 3])
    Kelem = Kelem[new_order, :][:, new_order]
    return Kelem


def get_felem(J):
    """get element load vector"""
    nquad = 3
    nbasis = 4
    quads = get_quadrature_rule()
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = quads[iquad]
        psi_val = [lagrange_value(i, xi) for i in range(2)]
        
        for i in range(2):
            felem[2 * i] += weight * J * psi_val[i]

    return felem