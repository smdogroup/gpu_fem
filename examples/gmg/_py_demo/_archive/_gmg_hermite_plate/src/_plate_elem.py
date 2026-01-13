# check and plot the hermite cubic polynomials in 1d
import matplotlib.pyplot as plt
import numpy as np

def hermite_cubic_polynomials_1d(ibasis):
    # node 1 is xi = -1, node 2 is xi = 1
    if ibasis == 0: # w for node 1
        return [0.5, -0.75, 0.0, 0.25]
    elif ibasis == 1: # dw/dx for node 1
        return [0.25, -0.25, -0.25, 0.25]
    elif ibasis == 2: # w for node 2
        return [0.5, 0.75, 0.0, -0.25]
    elif ibasis == 3: # dw/dx for node 2
        return [-0.25, -0.25, 0.25, 0.25]
    
def eval_polynomial(poly_list, value):
    poly_list_arr = np.array(poly_list)
    var_list_arr = np.array([value**(ind) for ind in range(len(poly_list))])
    return np.dot(poly_list_arr, var_list_arr)

def hermite_cubic_1d(ibasis, xi):
    poly_list = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(poly_list, xi)

def plot_hermite_cubic_1d():
    xi_vec = np.linspace(-1, 1, 100)
    for ibasis in range(4):
        poly = hermite_cubic_polynomials_1d(ibasis)
        h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
        plt.plot(xi_vec, h_vec, label=f"phi_{ibasis}")
    plt.legend()
    plt.show()

# hermite cubic polynomials in 2d
def get_hermite_cubic_2d(ibasis):
    # node 0 (-1,-1), node 1 (1, -1), node 2 (-1, 1), node 3 (1,1)
    if ibasis == 0: # w for node 0
        ixi, ieta = (0, 0)
    elif ibasis == 1: # dw/dx for node 0
        ixi, ieta = (1, 0)
    elif ibasis == 2: # dw/dy for node 0
        ixi, ieta = (0, 1)
    elif ibasis == 3: # w for node 1
        ixi, ieta = (2, 0)
    elif ibasis == 4: # dw/dx for node 1
        ixi, ieta = (3, 0)
    elif ibasis == 5: # dw/dy for node 1
        ixi, ieta = (2, 1)
    elif ibasis == 6: # w for node 2
        ixi, ieta = (0, 2)
    elif ibasis == 7: # dw/dx for node 2
        ixi, ieta = (1, 2)
    elif ibasis == 8: # dw/dy for node 2
        ixi, ieta = (0, 3)
    elif ibasis == 9: # w for node 3
        ixi, ieta = (2, 2)
    elif ibasis == 10: # dw/dx for node 3
        ixi, ieta = (3, 2)
    elif ibasis == 11: # dw/dy for node 3
        ixi, ieta = (2, 3)

    return ixi, ieta

def hermite_cubic_2d(ibasis, xi, eta):
    ixi, ieta = get_hermite_cubic_2d(ibasis)
    return hermite_cubic_1d(ixi, xi)  * hermite_cubic_1d(ieta, eta)

def plot_hermite_cubic_2d():
    # check the hermite cubic polynomials in 4x3 grid
    fig, axs = plt.subplots(4, 3)

    n = 10
    xi_vec = np.linspace(-1,1,n)
    eta_vec = np.linspace(-1,1,n)
    XI, ETA = np.meshgrid(xi_vec, eta_vec)

    for inode in range(4):
        for ihermite in range(3):
            ibasis = 3 * inode + ihermite
            # print(f"{ibasis=}")
            H = np.array([[hermite_cubic_2d(ibasis, my_xi, my_eta) for my_xi in xi_vec] for my_eta in eta_vec])
            axs[inode,ihermite].contourf(XI, ETA, H, levels=20, cmap='viridis', alpha=0.75)  # Filled contours

    plt.show()

def quadrature_rule(iquad):
    # 3x3 gauss quadrature rule
    rt = (3.0/5.0)**0.5
    vec = [-rt, 0, rt]
    # 9 quadrature points, 0-8
    # return the [xi,eta] quad pt location and w the quadrature weight
    odd = 5.0/9.0
    even = 8.0/9.0
    if iquad == 0:
        return [-rt, -rt], odd
    elif iquad == 1:
        return [0, -rt], even
    elif iquad == 2:
        return [rt, -rt], odd
    elif iquad == 3:
        return [-rt, 0], even
    elif iquad == 4:
        return [0, 0], odd
    elif iquad == 5:
        return [rt, 0], even
    elif iquad == 6:
        return [-rt, rt], odd
    elif iquad == 7:
        return [0, rt], even
    elif iquad == 8:
        return [rt, rt], odd

def get_laplacian(ibasis, xi, eta, xscale, yscale):
    ixi, ieta = get_hermite_cubic_2d(ibasis)

    xi_poly = hermite_cubic_polynomials_1d(ixi)
    eta_poly = hermite_cubic_polynomials_1d(ieta)
    xi_factor = eval_polynomial(xi_poly, xi)
    eta_factor = eval_polynomial(eta_poly, eta)

    d2xi_poly = [2.0 * xi_poly[-2], 6.0 * xi_poly[-1]]
    d2xi = eval_polynomial(d2xi_poly, xi)
    d2eta_poly = [2.0 * eta_poly[-2], 6.0 * eta_poly[-1]]
    d2eta = eval_polynomial(d2eta_poly, eta)    
    
    dphi_dxx = 1/xscale**2 * d2xi * eta_factor
    dphi_dyy = 1/yscale**2 * d2eta * xi_factor
    return dphi_dxx + dphi_dyy

def get_gradient(ibasis, xi, eta, xscale, yscale):
    # compute dphi/dx and dphi/dy for the gradient at a given xi, eta point in the isoperimetric element
    ixi, ieta = get_hermite_cubic_2d(ibasis)

    xi_poly = hermite_cubic_polynomials_1d(ixi)
    eta_poly = hermite_cubic_polynomials_1d(ieta)
    xi_factor = eval_polynomial(xi_poly, xi)
    eta_factor = eval_polynomial(eta_poly, eta)

    dxi_poly = [xi_poly[-3], 2.0 * xi_poly[-2], 3.0 * xi_poly[-1]]
    dxi = eval_polynomial(dxi_poly, xi)
    deta_poly = [eta_poly[-3], 2.0 * eta_poly[-2], 3.0 * eta_poly[-1]]
    deta = eval_polynomial(deta_poly, eta)
    
    dphi_dx = 1.0/xscale * dxi * eta_factor
    dphi_dy = 1.0 / yscale * deta * xi_factor

    return [dphi_dx, dphi_dy]

def get_hessians(ibasis, xi, eta, xscale, yscale):
    ixi, ieta = get_hermite_cubic_2d(ibasis)

    xi_poly = hermite_cubic_polynomials_1d(ixi)
    eta_poly = hermite_cubic_polynomials_1d(ieta)
    xi_factor = eval_polynomial(xi_poly, xi)
    eta_factor = eval_polynomial(eta_poly, eta)

    d2xi_poly = [2.0 * xi_poly[-2], 6.0 * xi_poly[-1]]
    d2xi = eval_polynomial(d2xi_poly, xi)
    d2eta_poly = [2.0 * eta_poly[-2], 6.0 * eta_poly[-1]]
    d2eta = eval_polynomial(d2eta_poly, eta)    

    dxi_poly = [xi_poly[-1], 2 * xi_poly[-2], 3 * xi_poly[-1]]
    dxi = eval_polynomial(dxi_poly, xi)   
    deta_poly = [eta_poly[-1], 2 * eta_poly[-2], 3 * eta_poly[-1]]
    deta = eval_polynomial(deta_poly, eta)   
    
    dphi_dxx = 1/xscale**2 * d2xi * eta_factor
    dphi_dyy = 1/yscale**2 * d2eta * xi_factor
    dphi_dxy = 1/xscale * 1/yscale * dxi * deta
    return [dphi_dxx, dphi_dyy, dphi_dxy]

def get_kelem(D, xscale, yscale):
    """
    compute the element stiffness matrix for an element with given x and y dimensions.
    Normally xscale = dx/2 and yscale = dy/2 for jacobian conversion dX/dxi
    """

    # sum over each of the quadrature points and basis functions
    nquad = 9
    nbasis = 12
    Kelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        pt, weight = quadrature_rule(iquad)
        xi = pt[0]
        eta = pt[1]
        for i in range(nbasis):
            for j in range(nbasis):
                Kelem[i,j] += D * weight * xscale * yscale * \
                get_laplacian(i, xi, eta, xscale, yscale) * \
                get_laplacian(j, xi, eta, xscale, yscale)

    # print(f"{Kelem=}")

    return Kelem

def get_gelem(Nxx, Nxy, Nyy, xscale, yscale):
    """
    compute the element stability matrix which is dependent on the in-plane loads
    """
    nquad = 9
    nbasis = 12
    Gelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        pt, weight = quadrature_rule(iquad)
        xi = pt[0]
        eta = pt[1]
        for i in range(nbasis):
            for j in range(nbasis):
                grad_i = get_gradient(i, xi, eta, xscale, yscale)
                grad_j = get_gradient(j, xi, eta, xscale, yscale)
                Gelem[i,j] += weight * xscale * yscale * \
                 (Nxx * grad_i[0] * grad_j[0] + \
                  Nxy * grad_i[0] * grad_j[1] + Nxy * grad_i[1] * grad_j[0] + \
                  Nyy * grad_i[1] * grad_j[1])

    return Gelem

def get_basis_fcn(ibasis, xi, eta):
    return hermite_cubic_2d(ibasis, xi, eta)

def get_felem(xscale, yscale):
    """get element load vector"""
    nquad = 9
    nbasis = 12
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        pt, weight = quadrature_rule(iquad)
        xi = pt[0]; eta = pt[1]
        for ibasis in range(nbasis):
            felem[ibasis] += weight * xscale * yscale * get_basis_fcn(ibasis, xi, eta)
    return felem