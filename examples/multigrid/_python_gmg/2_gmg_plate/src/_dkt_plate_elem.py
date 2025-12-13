"""DKT kirchoff plate triangular element by https://web.mit.edu/kjb/www/Publications_Prior_to_1998/A_Study_of_Three-Node_Triangular_Plate_Bending_Elements.pdf"""

import numpy as np

def triang_basis(xi, eta):
    return np.array([
        2.0 * (1 - xi - eta) * (0.5 - xi - eta),
        xi * (2 * xi - 1.0),
        eta * (2.0 * eta - 1.0),
        4.0 * xi * eta,
        4.0 * eta * (1.0 - xi - eta),
        4.0 * xi * (1.0 - xi - eta),
    ])

def dkt_Hw_shape_funcs(x, y, xi, eta):
    """shape functions for w interp"""
    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    # shift by one so one-based (easier to make y12 equiv to y[3] more readable)
    xij = np.concatenate([np.zeros(1), xij])
    yij = np.concatenate([np.zeros(1), yij])

    Hw = np.array([
        1 - xi - eta,
        0.5 * (1 - xi - eta) * (-yij[3] * xi + yij[2] * eta),
        0.5 * (1 - xi - eta) * (xij[3] * xi - xij[2] * eta),
        xi,
        0.5 * xi * (-yij[1] * eta + yij[3] * (1 - xi - eta)),
        0.5 * xi * (xij[1] * eta - xij[3] * (1 - xi - eta)),
        eta,
        0.5 * eta * (-yij[2] * (1 - xi - eta) + yij[1] * xi),
        0.5 * eta * (xij[2] * (1 - xi - eta) - xij[1] * xi),
    ])
    return Hw

def dkt_H_shape_funcs(x, y, xi, eta):
    N = triang_basis(xi, eta)

    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    lij = np.sqrt(xij**2 + yij**2)
    a = -xij / lij**2
    b = 0.75 * xij * yij / lij**2
    c = (0.25 * xij**2 - 0.5 * yij**2) / lij**2
    d = -yij / lij**2
    e = (0.25 * yij**2 - 0.5 * xij**2) / lij**2

    # now offset stuff so we can type it the same as 4,5,6 here like page 36 of the pdf
    # less error prone
    a = np.concatenate([np.zeros(4), a])
    b = np.concatenate([np.zeros(4), b])
    c = np.concatenate([np.zeros(4), c])
    d = np.concatenate([np.zeros(4), d])
    e = np.concatenate([np.zeros(4), e])
    N = np.concatenate([np.zeros(1), N])

    # Hx and Hy
    Hx, Hy = [], []
    for loop in range(3):
        if loop == 0:
            i, j, m = 5, 6, 1
        elif loop == 1:
            i, j, m = 6, 4, 2
        else:
            i, j, m = 4, 5, 3

        Hx += [
            1.5 * (a[j] * N[j] - a[i] * N[i]),
            b[i] * N[i] + b[j] * N[j],
            N[m] - c[i] * N[i] - c[j] * N[j]
        ]

        Hy += [
            1.5 * (d[j] * N[j] - d[i] * N[i]),
            -N[m] + e[i] * N[i] + e[j] * N[j],
            -(b[i] * N[i] + b[j] * N[j])
        ]

    Hx = np.array(Hx)
    Hy = np.array(Hy)

    return Hx, Hy

def dkt_H_shape_func_grads(x, y, xi, eta):
    """get d[Hx, Hy]/d[xi, eta] derivs"""
    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])
    lij = np.sqrt(xij**2 + yij**2)

    a = -xij / lij**2
    b = 0.75 * xij * yij / lij**2
    c = (0.25 * xij**2 - 0.5 * yij**2) / lij**2
    d = -yij / lij**2
    e = (0.25 * yij**2 - 0.5 * xij**2) / lij**2

    P = 6 * a
    q = 4 * b
    t = 6 * d
    r = 3 * yij**2 / lij**2   

    # now offset stuff so we can type it the same as 4,5,6 here like page 36 of the pdf
    P = np.concatenate([np.zeros(4), P])
    q = np.concatenate([np.zeros(4), q])
    t = np.concatenate([np.zeros(4), t])
    r = np.concatenate([np.zeros(4), r])

    Hx_xi = np.array([
        P[6] * (1 - 2 * xi) + (P[5] - P[6]) * eta,
        q[6] * (1 - 2*xi) - (q[5] + q[6]) * eta,
        -4 + 6 * (xi + eta) + r[6] * (1.0 - 2 * xi) - eta * (r[5] + r[6]),
        -P[6] * (1 - 2 * xi) + eta * (P[4] + P[6]),
        q[6] * (1 - 2 * xi) - eta * (q[6] - q[4]),
        -2 + 6 * xi + r[6] * (1 - 2 * xi) + eta * (r[4] - r[6]),
        -eta * (P[5] + P[4]),
        eta * (q[4] - q[5]),
        -eta * (r[5] - r[4]),
    ])

    Hy_xi = np.array([
        t[6] * (1 - 2 * xi) + (t[5] - t[6]) * eta,
        1.0 + r[6] * (1 - 2*xi) - (r[5] + r[6]) * eta,
        -q[6] * (1 - 2 * xi) + eta * (q[5] + q[6]),
        -t[6] * (1 - 2 * xi) + eta * (t[4] + t[6]),
        -1 + r[6] * (1-2*xi) + eta * (r[4] - r[6]),
        -q[6] * (1 - 2 * xi) - eta * (q[4] - q[6]),
        -eta * (t[4] + t[5]),
        eta * (r[4] - r[5]),
        -eta * (q[4] - q[5]),
    ])

    Hx_eta = np.array([
        -P[5] * (1 - 2 * eta) - xi * (P[6] - P[5]),
        q[5] * (1 - 2 * eta) - xi * (q[5] + q[6]),
        -4 + 6 * (xi + eta) + r[5] * (1 - 2 * eta) - xi * (r[5] + r[6]),
        xi * (P[4] + P[6]),
        xi * (q[4] - q[6]),
        -xi * (r[6] - r[4]),
        P[5] * (1 - 2 * eta) - xi * (P[4] + P[5]),
        q[5] * (1 - 2 * eta) + xi * (q[4] - q[5]),
        -2 + 6 * eta + r[5] * (1 - 2 * eta) + xi * (r[4] - r[5]),
    ])

    Hy_eta = np.array([
        -t[5] * (1 - 2 * eta) - xi * (t[6] - t[5]),
        1 + r[5] * (1 - 2 * eta) - xi * (r[5] + r[6]),
        -q[5] * (1 - 2 * eta) + xi * (q[5] + q[6]),
        xi * (t[4] + t[6]),
        xi * (r[4] - r[6]),
        -xi * (q[4] - q[6]),
        t[5] * (1 - 2 * eta) - xi * (t[4] + t[5]),
        -1 + r[5] * (1 - 2 * eta) + xi * (r[4] - r[5]),
        -q[5] * (1 - 2 * eta) - xi * (q[4] - q[5]),
    ])

    return Hx_xi, Hy_xi, Hx_eta, Hy_eta

def get_strain_mat(x, y, xi, eta):
    """get strain mat B(x,y)"""
    Hx_xi, Hy_xi, Hx_eta, Hy_eta = dkt_H_shape_func_grads(x, y, xi, eta)

    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])

    Hx_xi = np.expand_dims(Hx_xi, axis=0)
    Hy_xi = np.expand_dims(Hy_xi, axis=0)
    Hx_eta = np.expand_dims(Hx_eta, axis=0)
    Hy_eta = np.expand_dims(Hy_eta, axis=0)

    area = 0.5 * (xij[1] * yij[-1] - xij[-1] * yij[1])

    B = 0.5 / area * np.concatenate([
        yij[1] * Hx_xi + yij[-1] * Hx_eta,
        -xij[1] * Hy_xi - xij[-1] * Hy_eta,
        -xij[1] * Hx_xi - xij[-1] * Hx_eta + yij[1] * Hy_xi + yij[-1] * Hy_eta,
    ])
    # B[1] *= -1 # TODO : I think there was wrong sign here, cause curvatures were wrong.. kappa_y
    return B

def get_elem_area(x, y):
    xij = -np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = -np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])

    area = 0.5 * (xij[1] * yij[-1] - xij[-1] * yij[1])
    return area

def add_element_quadpt_jacobian(E, thick, nu, x, y, xi, eta):
    area = get_elem_area(x, y)
    B = get_strain_mat(x, y, xi, eta)
    Db = E * thick**3 / 12.0 / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0],
        [0, 0, 0.5 * (1.0 - nu)]
    ])

    # add Kelem at quadrature point
    K = 2.0 * area * B.T @ Db @ B
    return K

def get_quadpt_curvatures(elem_disp, x, y, xi, eta):
    """get elem moments M(x,y)"""
    B = get_strain_mat(x, y, xi, eta)
    return np.dot(B, elem_disp)

def get_quadpt_moments(elem_disp, E, thick, nu, x, y, xi, eta):
    """get elem moments M(x,y)"""
    B = get_strain_mat(x, y, xi, eta)
    Db = E * thick**3 / 12.0 / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0],
        [0, 0, 0.5 * (1.0 - nu)]
    ])
    return np.dot(Db @ B, elem_disp)

def get_quadpt_laplacian(elem_disp, x, y, xi, eta):
    B = get_strain_mat(x, y, xi, eta)
    curv = np.dot(B, elem_disp)
    # NOTE : I had to change sign here because got laplacian = 0 in middle (that's not right)..
    # v = np.array([-1.0, 1.0, 0.0])
    v = -1.0 * np.array([1.0, 1.0, 0.0])
    return np.dot(v, curv)

def get_weak_moment_kelem_felem(elem_disp, E, thick, nu, x, y, xi:float, eta:float):
    Hx, Hy = dkt_H_shape_funcs(x, y, xi, eta)

    xij = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    yij = np.array([y[2] - y[1], y[0] - y[2], y[1] - y[0]])

    # figure out normal direc
    if xi == 0: # change along eta
        dx, dy = xij[1], yij[1]
    else: # change along xi (one of these two must be zero)
        dx, dy = xij[2], yij[2]

    n = np.array([dy, -dx])
    ds = np.linalg.norm(n)
    n /= ds
    # print(F'{n=}')
    # sign = np.sign(n[direc]) # TODO : need to change signs with +x vs. -x ? prob doesn't matter as long as consistent..

    B = get_strain_mat(x, y, xi, eta)
    D = E * thick**3 / 12.0 / (1.0 - nu**2)
    # D *= 1e3 # stronger penalty..

    # jacobian here..
    # v = np.array([-1.0, 1.0, 0.0]) # gives laplacian from curvatures
    v = -1.0 * np.array([1.0, 1.0, 0.0])
    w = np.dot(v, B) # this part might make it like an energy functional
    a = D * w

    # test function part (for d(delta w)/dn, related to rotations theta_x, theta_y)
    vn = -Hx * n[0] - Hy * n[1]

    # vn *= -1 # DEBUG

    # NOTE : why are we doing normal moments? it makes more sense to me to be tangential moments.. or something different, let's look at the Schwarz PDE they posed again..
    M_kelem = np.outer(vn, a) * ds
    M_felem = np.dot(M_kelem, elem_disp)
    # M_kelem *= 0.0

    return M_kelem, M_felem

def get_kelem(E, thick, nu, x, y):
    """use three point quadrature (since integrand is purely quadratic only need 3 points)"""
    # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    xi_vec = np.array([4.0, 1.0, 1.0]) / 6.0
    eta_vec = np.array([1.0, 4.0, 1.0]) / 6.0
    weights = np.array([1.0]*3) / 3.0

    Kelem = np.zeros((9, 9))

    for i in range(3):
        xi, eta, wt = xi_vec[i], eta_vec[i], weights[i]

        Kelem += wt * add_element_quadpt_jacobian(E, thick, nu, x, y, xi, eta)
    return Kelem

def get_rotations(elem_disp, x, y, xi, eta):
    Hx, Hy = dkt_H_shape_funcs(x, y, xi, eta)
    betax = np.dot(Hx, elem_disp)
    betay = np.dot(Hy, elem_disp)
    return betax, betay

def get_elem_quantities(glob_disps, nxe, E, thick, nu, name="moment"):
    """get the bending moments (at the midpoint of each quad, or pair of tri elements)"""
    assert(name in ["moment", "curvature", "laplacian"])

    nx = nxe + 1
    nelems = nxe**2 * 2 # x2 because we have two triangle elems in each quad element slot
    if name in ["moment", "curvature"]:
        elem_qtys = np.zeros((3, nelems // 2))
    else: # laplacian
        elem_qtys = np.zeros(nelems // 2)
    h = 1.0 / nxe

    for ielem in range(nelems):
        iquad = ielem // 2
        itri = ielem % 2
        ixe = iquad % nxe
        iye = iquad // nxe

        x1 = h * ixe
        x2 = x1 + h
        y1 = h * iye
        y2 = y1 + h
        n1 = nx * iye + ixe
        n3 = n1 + nx
        n2, n4 = n1 + 1, n3 + 1

        if itri == 0: # first tri element in quad slot
            # node 1 must be the vertex C opposite hypotenuse
            x_elem = np.array([x1, x2, x1])
            y_elem = np.array([y1, y1, y2])
            local_nodes = [n1, n2, n3]

        else: # second tri element in quad slot
            # node 1 must be the vertex C opposite hypotenuse
            x_elem = np.array([x2, x1, x2])
            y_elem = np.array([y2, y2, y1])
            local_nodes = [n4, n3, n2]

        local_dof = np.array([3*inode+idof for inode in local_nodes for idof in range(3)])
        elem_disp = glob_disps[local_dof]
        if name == "moment":
            _M_elem = get_quadpt_moments(elem_disp, E, thick, nu, x_elem, y_elem, xi=0.5, eta=0.5)
            elem_qtys[:, iquad] += 0.5 * _M_elem
        elif name == "curvature":
            _curv_elem = get_quadpt_curvatures(elem_disp, x_elem, y_elem, xi=0.5, eta=0.5)
            elem_qtys[:, iquad] += 0.5 * _curv_elem
        else: # laplacian
            laplacian = get_quadpt_laplacian(elem_disp, x_elem, y_elem, xi=0.5, eta=0.5)
            elem_qtys[iquad] += 0.5 * laplacian

    return elem_qtys

def assemble_stiffness_matrix(nxe, E, thick, nu):
    """assemble the global stiffness matrix for a unit square regular grid with triangle elements"""

    nx = nxe + 1
    N = nx**2
    ndof = 3 * N # 3 dof per node

    nelems = nxe**2 * 2 # x2 because we have two triangle elems in each quad element slot

    K = np.zeros((ndof, ndof))

    # unit square grid
    h = 1.0 / nxe

    for ielem in range(nelems):
        iquad = ielem // 2
        itri = ielem % 2
        ixe = iquad % nxe
        iye = iquad // nxe

        x1 = h * ixe
        x2 = x1 + h
        y1 = h * iye
        y2 = y1 + h
        n1 = nx * iye + ixe
        n3 = n1 + nx
        n2, n4 = n1 + 1, n3 + 1

        if itri == 0: # first tri element in quad slot
            # node 1 must be the vertex C opposite hypotenuse
            x_elem = np.array([x1, x2, x1])
            y_elem = np.array([y1, y1, y2])
            local_nodes = [n1, n2, n3]

        else: # second tri element in quad slot
            # node 1 must be the vertex C opposite hypotenuse
            x_elem = np.array([x2, x1, x2])
            y_elem = np.array([y2, y2, y1])
            local_nodes = [n4, n3, n2]

        Kelem = get_kelem(E, thick, nu, x_elem, y_elem)
        local_dof = [3*inode+idof for inode in local_nodes for idof in range(3)]
        # print(f"{iquad=} {itri=} {local_dof=}")

        arr_ind = np.ix_(local_dof, local_dof)
        # plt.imshow(Kelem)
        # plt.show()

        K[arr_ind] += Kelem

    return K

def get_bcs(nxe):
    # now apply bcs to the stiffness matrix and forces
    nx = nxe + 1
    bcs = []
    for iy in range(nx):
        for ix in range(nx):
            inode = nx * iy + ix

            if ix in [0, nx-1] or iy in [0, nx-1]:
                bcs += [3 * inode]
            
            if ix in [0, nx-1]:
                bcs += [3 * inode + 1] # theta y = 0 on y=const edge

            if iy in [0, nx-1]:
                bcs += [3 * inode + 2] # theta x = 0 on x=const edge
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

def add_weak_moment_penalties(K_sd, F_sd, sd_disps, sd_dof, nx, E, thick, nu):
    """apply weak moment penalties on a subdomain problem (actually ends up being laplacian = prev_laplacian condition)"""
    nxe = nx - 1
    h = 1.0 / nxe
    sd_nodes = np.unique(sd_dof // 3)
    ix_list, iy_list = sd_nodes % nx, sd_nodes // nx
    # now shift ix_list, iy_list to subdomain level
    ixn, ixp, iyn, iyp = np.min(ix_list) == 0, np.max(ix_list) == nx-1, np.min(iy_list) == 0, np.max(iy_list) == nx-1
    ix_list -= np.min(ix_list)
    iy_list -= np.min(iy_list)
    # num elements in each direction
    nxe_red, nye_red = np.max(ix_list), np.max(iy_list)
    nx_red, ny_red = nxe_red + 1, nye_red + 1
    # print(f"{nxe_red=} {nye_red=}")

    irt3 = 1.0 / np.sqrt(3)
    ns = nxe_red + nye_red # number of x and y edges (half of them)

    for i_bndry in range(2*ns):
        ii_bndry, i_half = i_bndry % ns, i_bndry // ns
        bndry_case = None
        if ii_bndry < nxe_red: # one of x edges
            ixe = ii_bndry
            bndry_case = 0 if i_half == 0 else 1
            iye = 0 if i_half == 0 else nye_red - 1
        else: # one of y edges
            iye = ii_bndry - nxe_red
            bndry_case = 2 if i_half == 0 else 3
            ixe = 0 if i_half == 0 else nxe_red-1

        if bndry_case == 2: # xneg bc
            lower_elem, changing_eta = True, True
        elif bndry_case == 3: # xpos bc
            lower_elem, changing_eta = False, True
        elif bndry_case == 0: # yneg bc
            lower_elem, changing_eta = True, False
        elif bndry_case == 1: # ypos bc
            lower_elem, changing_eta = False, False
        else:
            continue

        inode = nx_red * iye + ixe

        # print(f"{ixe=} {iye=} {ii_bndry=} {bndry_case=}")

        # NOTE : node 1 is at xi = 0, eta = 0 by defn (so for this reason I always choose node 1 opposite hypotenuse)
        if lower_elem:
            x_elem, y_elem = h * np.array([0.0, 1.0, 0.0]), h * np.array([0.0, 0.0, 1.0])
            elem_nodes = [inode, inode+1, inode + nx_red]
        else:
            x_elem, y_elem = h * np.array([1.0, 0.0, 1.0]), h * np.array([1.0, 1.0, 0.0])
            elem_nodes = [inode + nx_red + 1, inode + nx_red, inode+1]
        elem_dof = np.array([3 * _node + idof for _node in elem_nodes for idof in range(3)])
        elem_disp = sd_disps[elem_dof]

        for _xi in [-irt3, irt3]:

            # set the quadpts (quad weights are just 1.0 btw for 2-point quadrature)
            _xi2 = 0.5 * (1.0 + _xi)
            xi = _xi2 if not(changing_eta) else 0.0
            eta = _xi2 if changing_eta else 0.0

            # print(f"weak moment, {ixe=} {iye=}: {elem_dof=} {x_elem=} {y_elem=} {xi=} {eta=}")
            
            xn = bndry_case == 2 and not(ixn)
            xp = bndry_case == 3 and not(ixp)
            yn = bndry_case == 0 and not(iyn)
            yp = bndry_case == 1 and not(iyp)
            if xn or xp or yn or yp: # aka if not on global boundary (only on transmission bndry, then can do this) 
                elem_disp2 = elem_disp # TODO : how to reduce state drift here, below didn't work
                # elem_disp2 = elem_disp * 0.0 # check if this reduces state drift (basically laplacian on state update = 0 since want to be same as before..)
                M_kelem, M_felem = get_weak_moment_kelem_felem(elem_disp2, E, thick, nu, x_elem, y_elem, xi, eta)

                # exit() # temp debug

                # add into K_sd and F_sd now
                arr_ind = np.ix_(elem_dof, elem_dof)
                K_sd[arr_ind] += M_kelem # TODO: try not adjusting the stiffness matrix..
                # print(F"{ixe=} {iye=} {np.linalg.norm(M_felem)=:.3e}")
                F_sd[elem_dof] += M_felem
    
    return K_sd, F_sd


def apply_subdomain_bcs(nxe, E, thick, nu, K_sd, F_sd, sd_disps, sd_dof, bndry_dof, moment_penalty:bool=True):
    """compute lhs and rhs for subdomain with dirichlet and moment weak bc"""
    # NOTE : moment penalty flag just for debugging (you definitely need it by theorems on Schwarz subdomain conv for 4th order PDEs in this case)

    # get some important dof lists
    int_dof = np.array([_ for _ in sd_dof if not(_ in bndry_dof)])
    int_red_dof = np.array([i for i in range(sd_dof.shape[0]) if not(sd_dof[i] in bndry_dof)])
    bndry_red_dof = np.array([i for i in range(sd_dof.shape[0]) if sd_dof[i] in bndry_dof])

    # need to apply global bcs again? Maybe yes, but only bc of the rot dof (th_x = 0, th_y = 0) as w=0 already included in subdomain bcs
    bcs = get_bcs(nxe)
    glob_bcs = np.array([i for i in range(sd_dof.shape[0]) if sd_dof[i] in bcs])

    # now the weak moment terms on the bndry
    # TODO : only add on interior subdomain boundaries? TBD on this, we'll do it on all boundaries first..
    if moment_penalty:
        K_sd, F_sd = add_weak_moment_penalties(K_sd, F_sd, sd_disps, sd_dof, nxe+1, E, thick, nu)    

    # compute nz dirichlet bc corrections
    x_c = sd_disps[bndry_red_dof]
    # print(F'{x_c=}')
    K_FC = K_sd[int_red_dof, :][:, bndry_red_dof]
    F_C = np.dot(K_FC, x_c) # correction force for nonzero dirichlet bcs

    # apply subdomain w=0 bcs to the matrix (so )
    K_sd, F_sd = _apply_bcs_helper(glob_bcs, K_sd, F_sd)
    K_sd, F_sd = _apply_bcs_helper(bndry_red_dof, K_sd, F_sd)

    # eliminate all constrained DOF and compute reduced problem
    K_red = K_sd[int_red_dof, :][:, int_red_dof]
    F_red = F_sd[int_red_dof] - F_C

    return K_red, F_red, int_dof


def prolongation_operator(nxe_c):
    """go coarse to fine with 2x grid multiplier on coarse disps"""
    nxe_f = nxe_c * 2
    nx_f = nxe_f + 1
    nx_c = nxe_c + 1
    N_c = nx_c**2
    N_f = nx_f**2
    ndof_c, ndof_f = 3 * N_c, 3 * N_f

    H = 1.0 / nxe_c # coarse mesh step size

    P = np.zeros((ndof_f, ndof_c))

    # interpolate
    for i_f in range(ndof_f):
        inode_f = i_f // 3
        idof = i_f % 3
        iy_f = inode_f // nx_f
        ix_f = inode_f % nx_f

        # print(f"{ix_f=} {iy_f=}")

        ix_c, iy_c = ix_f // 2, iy_f // 2
        ix_cr, iy_cr = ix_f % 2, iy_f % 2 
        inode_c = nx_c * iy_c + ix_c

        if ix_cr == 0 and iy_cr == 0:
            # fine node on top of coarse node
            i_c = 3 * inode_c + idof
            P[i_f, i_c] = 1.0

        elif ix_c == nx_c - 1 or iy_c == nx_c - 1:
            # far right or far left elems
            if ix_c == nx_c -1:
                xi, eta = 0.5, 0.0
                _inode_c = inode_c - 1
            elif iy_c == nx_c - 1:
                xi, eta = 0.0, 0.5
                _inode_c = inode_c - nx_c

            elem_nodes = [_inode_c+1, _inode_c+nx_c+1, _inode_c + nx_c]
            elem_dof = np.array([3 * _node + _dof for _node in elem_nodes for _dof in range(3)])
            x_elem = H * np.array([1.0, 1.0, 0.0])
            y_elem = H * np.array([0.0, 1.0, 1.0])
            Hw = dkt_Hw_shape_funcs(x_elem, y_elem, xi, eta)
            Hx, Hy = dkt_H_shape_funcs(x_elem, y_elem, xi, eta)

            if idof == 0: # w dof
                P[i_f, elem_dof] = Hw
            elif idof == 1: # th_x dof
                P[i_f, elem_dof] = -Hy
            elif idof == 2: # th_y dof
                P[i_f, elem_dof] = Hx

        else:
            # fine node on edges of coarse quad
            # since right bndry nodes treated separately each set of bndry nodes is treated first on its left side node
            # NOTE : no need to symmetrize it, I think you can show the interp on an edge only depends upon endpt quantities on that edge?
            xi = 0.5 if ix_cr == 1 else 0.0
            eta = 0.5 if iy_cr == 1 else 0.0

            # print(f"{inode_c=} {ix_c=} {iy_c=} {nx_c=}")

            elem_nodes = [inode_c, inode_c + 1, inode_c + nx_c]
            elem_dof = np.array([3 * _node + _dof for _node in elem_nodes for _dof in range(3)])
            x_elem = H * np.array([0.0, 1.0, 0.0])
            y_elem = H * np.array([0.0, 0.0, 1.0])
            Hw = dkt_Hw_shape_funcs(x_elem, y_elem, xi, eta)
            Hx, Hy = dkt_H_shape_funcs(x_elem, y_elem, xi, eta)
            
            if idof == 0: # w dof
                P[i_f, elem_dof] = Hw
            elif idof == 1: # th_x dof
                P[i_f, elem_dof] = -Hy
            elif idof == 2: # th_y dof
                P[i_f, elem_dof] = Hx

    # get bcs also on each side..
    fine_bcs = get_bcs(nxe_f)
    coarse_bcs = get_bcs(nxe_c)
    P[fine_bcs,:] = 0.0
    P[:,coarse_bcs] = 0.0

    return P


def restriction(nxe_f, u_f):
    assert(nxe_f % 2 == 0)
    nxe_c = nxe_f // 2

