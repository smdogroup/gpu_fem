import numpy as np

def chebyshev_polynomials(order:int=2):
    # update rule is Tn+1(x) = 2x * Tn(x) - T_{n-1}(x)
    funcs = [
        lambda x : 1.0,
        lambda x : x,
        lambda x : 2.0 * x**2 - 1.0,
        lambda x : 4.0 * x**3 - 3.0 * x,
        lambda x : 8.0 * x**4 - 8.0 * x**2 + 1.0,
    ]

    n = order + 1
    return funcs[:n]

def chebyshev_polynomials_deriv(order:int=2):
    # update rule is Tn+1(x) = 2x * Tn(x) - T_{n-1}(x)
    funcs = [
        lambda x : 0.0,
        lambda x : 1.0,
        lambda x : 4.0 * x,
        lambda x : 12.0 * x**2 - 3.0,
        lambda x : 32.0 * x**3 - 16.0 * x,
    ]

    n = order + 1
    return funcs[:n]

def get_chebyshev_gps(order:int=2):
    # get the gauss points of this order
    n = order + 1
    a = np.cos(np.pi / 2.0 / n)
    xis = -1.0 / a * np.array([np.cos((2.0 * i - 1) * np.pi / 2.0 / n) for i in range(1, n+1)])
    return xis

def chebyshev_value(ibasis, xi, order:int=2):
    # compute the value of ibasis==k => Nk(xi)

    n = order + 1
    T_func = chebyshev_polynomials(order)
    xis = get_chebyshev_gps(order)
    xi_k = xis[ibasis]
    # print(f"{ibasis=} {xi=} {order=} : {xi_k=}")


    Nk_val = 0.0
    a = np.cos(np.pi / 2.0 / n)
    for i in range(n):
        Tik = T_func[i](a * xi_k)
        Ti_in = T_func[i](a * xi)
        # T_denom_vec = T_func[i](a * xis)
        # denom = np.sum(T_denom_vec**2)
        denom = n if i == 0 else n / 2.0
        Nk_val += Tik * Ti_in / denom
    return Nk_val

def chebyshev_d1dx(ibasis, xi, J:float, order:int=2):
    # compute the value of ibasis==k => Nk(xi)

    n = order + 1
    T_func = chebyshev_polynomials(order)
    T_func_derivs = chebyshev_polynomials_deriv(order)
    xis = get_chebyshev_gps(order)
    xi_k = xis[ibasis]
    # print(f"{ibasis=} {xi=} {order=} : {xi_k=}")


    Nk_val = 0.0
    a = np.cos(np.pi / 2.0 / n)
    for i in range(n):
        Tik = T_func[i](a * xi_k)
        Ti_in_deriv = T_func_derivs[i](a * xi) * a
        # T_denom_vec = T_func[i](a * xis)
        # denom = np.sum(T_denom_vec**2)
        denom = n if i == 0 else n / 2.0
        Nk_val += Tik * Ti_in_deriv / denom
    return Nk_val / J


def interp_chebyshev(xi, vals, order:int=2):
    # interp a single variable of interest using chebyshev basis
    n = order + 1
    return np.sum(np.array([
        chebyshev_value(ibasis, xi, order) * vals[ibasis] for ibasis in range(n)
    ]))

def interp_chebyshev_transpose(xi, out_bar, order:int=2):
    # interp a single variable of interest using chebyshev basis
    n = order + 1
    vals = out_bar * np.array([
        chebyshev_value(ibasis, xi, order) for ibasis in range(n)
    ])
    return vals

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

def get_kelem(J, EI, GA, k_shear=5.0/6.0, use_reduced_integration_for_shear=False, order:int=2):
    """
    J : Jacobian = L/2 (so L = 2*J)
    EI : bending stiffness
    GA : shear rigidity (A*G)  (we will multiply by k_shear)
    k_shear : shear correction factor (default 5/6)
    """
    n = order + 1
    Kelem = np.zeros((2 * n, 2 * n))
    quads = get_quadrature_rule(order)

    # If doing reduced integration for shear, we evaluate shear at single point (xi=0)
    if use_reduced_integration_for_shear:
        shear_quads = [(0.0, 2.0)]  # one-point Gauss on [-1,1] has weight 2
    else:
        shear_quads = quads

    # GA *= 8.0 # correction?

    # bending (EI * int (w_xx - th_s_x)^2 dx) expands into EI*(w_xx*w_xx - 2 w_xx th_s_x + th_s_x th_s_x)
    for xi, weight in shear_quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J

        # th_s uses Lagrange; its x-derivative:
        dpsi = [chebyshev_d1dx(i, xi, J, order) for i in range(n)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [chebyshev_value(i, xi, order) for i in range(n)]

        # first nxn block Kww before reordering
        for i in range(n):
            for j in range(n):
                
                # kww block
                Kelem[i,j] += k_shear * GA * qfactor * dpsi[i] * dpsi[j]

                # k_[w,th] block
                Kelem[i,n + j] += -k_shear * GA * qfactor * dpsi[i] * psi_val[j]

                # and transpose
                Kelem[n + i,j] += -k_shear * GA * qfactor * psi_val[i] * dpsi[j]

                # K_[th,th] block
                Kelem[n + i, n + j] += k_shear * GA * qfactor * psi_val[i] * psi_val[j]

    for xi, weight in quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J

        # th_s uses Lagrange; its x-derivative:
        dpsi = [chebyshev_d1dx(i, xi, J, order) for i in range(n)]
        # shear shape values too if needed (for kGA * th_s^2)
        psi_val = [chebyshev_value(i, xi, order) for i in range(n)]

        for i in range(n):
            for j in range(n):
                Kelem[n + i, n + j] += EI * qfactor * dpsi[i] * dpsi[j]

    # reorder the Kelem instead of [w1,...,wn, th1, ..., thn] to [w1,th1,...,wn,thn]
    reorder_ind = []
    for i in range(n):
        reorder_ind += [i, n + i]
    # print(f"{reorder_ind=}")
    reorder_ind = np.array(reorder_ind)
    Kelem = Kelem[reorder_ind, :][:, reorder_ind]

    # import matplotlib.pyplot as plt
    # plt.imshow(Kelem)
    # plt.show()

    return Kelem


def get_felem(J, order:int=2):
    """get element load vector"""
    quads = get_quadrature_rule(order)
    nquad = len(quads)
    n = order + 1
    felem = np.zeros((2 * n,))
    for iquad in range(nquad):
        xi, weight = quads[iquad]
        psi_val = [chebyshev_value(i, xi, order) for i in range(n)]
        
        for i in range(n):
            felem[2 * i] += weight * J * psi_val[i]
    return felem