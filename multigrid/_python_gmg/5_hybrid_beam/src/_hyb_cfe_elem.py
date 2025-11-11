import numpy as np

# TODO : could also do some type of mixed hermite-chebyshev element (but that seems really high order..)


def chebyshev_polynomials(order:int=2):
    # update rule is Tn+1(x) = 2x * Tn(x) - T_{n-1}(x)
    funcs = [
        lambda x : 1.0,
        lambda x : x,
        lambda x : 2.0 * x**2 - 1.0,
        lambda x : 4.0 * x**3 - 3.0 * x,
        lambda x : 8.0 * x**4 - 8.0 * x**2 + 1.0
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
        lambda x : 32.0 * x**3 - 16.0 * x
    ]

    n = order + 1
    return funcs[:n]

def chebyshev_polynomials_deriv2(order:int=2):
    # update rule is Tn+1(x) = 2x * Tn(x) - T_{n-1}(x)
    funcs = [
        lambda x : 0.0,
        lambda x : 0.0,
        lambda x : 4.0,
        lambda x : 24.0 * x,
        lambda x : 96.0 * x**2 - 16.0
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

def chebyshev_d2dx(ibasis, xi, J:float, order:int=2):
    # compute the value of ibasis==k => Nk(xi)

    n = order + 1
    T_func = chebyshev_polynomials(order)
    T_func_derivs2 = chebyshev_polynomials_deriv2(order)
    xis = get_chebyshev_gps(order)
    xi_k = xis[ibasis]
    # print(f"{ibasis=} {xi=} {order=} : {xi_k=}")


    Nk_val = 0.0
    a = np.cos(np.pi / 2.0 / n)
    for i in range(n):
        Tik = T_func[i](a * xi_k)
        Ti_in_deriv2 = T_func_derivs2[i](a * xi) * a**2
        # T_denom_vec = T_func[i](a * xis)
        # denom = np.sum(T_denom_vec**2)
        denom = n if i == 0 else n / 2.0
        Nk_val += Tik * Ti_in_deriv2 / denom
    return Nk_val / J**2

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

# whether the kirchoff part of the rotation is stored as dw/dxi or dw/dx
# in particular if True, it's stored as dw/dx
# initially putting this to True breaks mesh convergence of this element (but does give right displacements)
# needs to be True if want to use on plate and multigrid
physical_rots = True
# physical_rots = False

def get_kelem(J, EI, GA, k_shear=5.0/6.0, order:int=2):
    """
    J : Jacobian = L/2 (so L = 2*J)
    EI : bending stiffness
    GA : shear rigidity (A*G)  (we will multiply by k_shear)
    k_shear : shear correction factor (default 5/6)
    """
    n = order + 1
    Kelem = np.zeros((2 * n, 2 * n))
    quads = get_quadrature_rule(order)

    # bending (EI * int (w_xx - th_s_x)^2 dx) expands into EI*(w_xx*w_xx - 2 w_xx th_s_x + th_s_x th_s_x)
    for xi, weight in quads:
        # quadrature factor dx = J * dxi
        qfactor = weight * J
        # build required derivatives/values at xi
        d2phi = [chebyshev_d2dx(i, xi, J, order) for i in range(n)]
        dphi = [chebyshev_d1dx(i, xi, J, order) for i in range(n)]
        phi = [chebyshev_value(i, xi, order) for i in range(n)]

        
        for i in range(n):
            for j in range(n):

                # bending strain energy part.. 
                # -----------------------------

                # kww block
                Kelem[i,j] += EI * qfactor * d2phi[i] * d2phi[j]

                # k_[w,th] block
                Kelem[i, n + j] -= EI * qfactor * d2phi[i] * dphi[j]

                # k_[th,w] block
                Kelem[n+i, j] -= EI * qfactor * dphi[i] * d2phi[j]

                # k_[th, th] block
                Kelem[n + i, n + j] += EI * qfactor * dphi[i] * dphi[j]
                
                # shear strain energy part 
                # -------------------------
                Kelem[n + i, n + j] += k_shear * GA * qfactor * phi[i] * phi[j]

    # import matplotlib.pyplot as plt
    # plt.imshow(np.log(1.0 + Kelem**2))
    # plt.show()
                
    # reorder the Kelem instead of [w1,...,wn, th1, ..., thn] to [w1,th1,...,wn,thn]
    reorder_ind = []
    for i in range(n):
        reorder_ind += [i, n + i]
    # print(f"{reorder_ind=}")
    reorder_ind = np.array(reorder_ind)
    Kelem = Kelem[reorder_ind, :][:, reorder_ind]
    return Kelem


def get_felem(J, order:int=2):
    """get element load vector"""
    quads = get_quadrature_rule(order)
    nquad = len(quads)
    n = order + 1
    felem = np.zeros(2 * n)
    for iquad in range(nquad):
        xi, weight = quads[iquad]
        psi_val = [chebyshev_value(i, xi, order) for i in range(n)]
        
        for i in range(n):
            felem[2 * i] += weight * J * psi_val[i]
    return felem