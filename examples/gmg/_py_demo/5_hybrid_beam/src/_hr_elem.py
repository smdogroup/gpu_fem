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
    w, th, ws = 0.0, 0.0, 0.0
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        w += N_i * elem_disp[3 * ibasis]
        th += N_i * elem_disp[3 * ibasis + 1]
        ws += N_i * elem_disp[3 * ibasis + 2]
    return w, th, ws

def interp_lagrange_transpose(xi, nodal_in):
    w_in, th_in, ws_in = nodal_in[0], nodal_in[1], nodal_in[2]
    coarse_out = np.zeros(6)
    for ibasis in range(2):
        N_i = lagrange_value(ibasis, xi)
        coarse_out[3 * ibasis] += N_i * w_in
        coarse_out[3 * ibasis + 1] += N_i * th_in
        coarse_out[3 * ibasis + 2] += N_i * ws_in
    return coarse_out

def get_kelem(J, EI, GA, k_shear=5.0/6.0, use_reduced_integration_for_shear=False):
    """
    J : Jacobian = L/2 (so L = 2*J)
    EI : bending stiffness
    GA : shear rigidity (A*G)  (we will multiply by k_shear)
    k_shear : shear correction factor (default 5/6)
    """
    nbasis = 6   # [w1, th1, ws1, w2, th2, ws2] order before reordering
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

        # th_s uses Lagrange; its x-derivative:
        dpsi = [lagrange_d1dx(i, xi, J) for i in range(2)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [lagrange_value(i, xi) for i in range(2)]

        for i in range(2):
            for j in range(2):
                
                # w, ws block of Kelem
                Kelem[i,4 + j] -= k_shear * GA * qfactor * dpsi[i] * dpsi[j]
                Kelem[4 + i, j] -= k_shear * GA * qfactor * dpsi[i] * dpsi[j]

                # theta, ws block of Kelem
                Kelem[2 + i,4 + j] -= k_shear * GA * qfactor * psi_val[i] * dpsi[j]
                Kelem[4 + i, 2 + j] -= k_shear * GA * qfactor * dpsi[i] * psi_val[j]

                # ws, ws block of Kelem
                Kelem[4 + i, 4 + j] += k_shear * GA * qfactor * dpsi[i] * dpsi[j]

                # th, th block of Kelem
                Kelem[2 + i, 2 + j] -= EI * qfactor * dpsi[i] * dpsi[j]

    # import matplotlib.pyplot as plt
    # plt.imshow(Kelem)
    # plt.show()

    # change order from [w1,w2,th1,th2,ws1,ws2] => [w1, th1, ws1, w2, th2, ws2]
    new_order = np.array([0, 2, 4, 1, 3, 5])
    Kelem = Kelem[new_order, :][:, new_order]
    Kelem *= -1 # so matching signs to the paper
    return Kelem


def get_felem(J):
    """get element load vector"""
    nquad = 3
    nbasis = 6
    quads = get_quadrature_rule()
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = quads[iquad]
        psi_val = [lagrange_value(i, xi) for i in range(2)]
        
        for i in range(2):
            felem[3 * i] += weight * J * psi_val[i]

    return felem