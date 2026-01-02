# we choose the hermite cubic polynomials as the basis functions of the element
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
    
def lagrange_polynomials_1d(ibasis):
    if ibasis == 0:
        return [0.5, -0.5]
    elif ibasis == 1:
        return [0.5, 0.5]
    
def eval_polynomial(poly_list, value):
    poly_list_arr = np.array(poly_list)
    var_list_arr = np.array([value**(ind) for ind in range(len(poly_list))])
    return np.dot(poly_list_arr, var_list_arr)

def hermite_cubic_1d(ibasis, xi):
    poly_list = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(poly_list, xi)

def lagrange_1d(ibasis, xi):
    poly_list = lagrange_polynomials_1d(ibasis)
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

def get_lagrange_grad(ibasis, xi, xscale):
    xi_poly = lagrange_polynomials_1d(ibasis)
    return xi_poly[-1] / xscale

def get_basis_fcn(ibasis, xi, xscale):
    xi_poly = hermite_cubic_polynomials_1d(ibasis)
    return eval_polynomial(xi_poly, xi)

def get_kelem(xscale, EI, kGA):
    """get element stiffness matrix"""
    nquad = 3
    nbasis = 6
    Kelem = np.zeros((nbasis, nbasis))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule(iquad)
        for i in range(nbasis):
            for j in range(nbasis):

                if i < 4 and j < 4: #Kww
                    factor = weight * xscale
                    if i % 2 == 1: factor *= xscale
                    if j % 2 == 1: factor *= xscale
                    Kelem[i,j] += EI * factor * get_hess(i, xi, xscale) * get_hess(j, xi, xscale)
                elif i < 4 and j >= 4: #Kw-th
                    factor = weight * xscale
                    if i % 2 == 1: factor *= xscale
                    Kelem[i,j] += EI * factor * get_hess(i, xi, xscale) * get_lagrange_grad(j-4, xi, xscale)
                elif i >= 4 and j < 4: #Kth-w
                    factor = weight * xscale
                    if j % 2 == 1: factor *= xscale
                    dphij_dxx = get_hess(j, xi, xscale)
                    dpsii_dx = get_lagrange_grad(i-4, xi, xscale)
                    # print(f"{iquad=} {i=} {j=} {dphij_dxx=:.2e} {dpsii_dx=:.2e}")
                    Kelem[i,j] += EI * factor * get_lagrange_grad(i-4, xi, xscale) * get_hess(j, xi, xscale)
                elif i >= 4 and j >= 4: #Kth-th
                    factor = weight * xscale
                    Kelem[i,j] += 8.0 * kGA * factor * lagrange_1d(i-4, xi) * lagrange_1d(j-4, xi) # why multiply it by 8 helps?
                    Kelem[i,j] += EI * factor * get_lagrange_grad(i-4, xi, xscale) * get_lagrange_grad(j-4, xi, xscale)

    # plt.imshow(Kelem[:4,:][:,:4])
    # plt.show()
    # plt.imshow(np.sign(Kelem) * np.log(1 + Kelem**2))
    # plt.show()

    # then reorder the Kelem so it goes from [w,th,w2,th2,gam1,gam2] order to [w,th1,gam1,w2,th2,gam2] order
    new_order = np.array([0, 1, 4, 2, 3, 5])
    Kelem = Kelem[new_order, :][:, new_order]
    # plt.imshow(Kelem)
    # plt.show()

    return Kelem

def get_felem(xscale):
    """get element load vector"""
    nquad = 3
    nbasis = 6
    felem = np.zeros((nbasis,))
    for iquad in range(nquad):
        xi, weight = get_quadrature_rule(iquad)
        for ibasis in range(nbasis):
            if ibasis < 4:
                felem[ibasis] += weight * xscale * get_basis_fcn(ibasis, xi, xscale)
    
    new_order = np.array([0, 1, 4, 2, 3, 5])
    felem = felem[new_order]
    return felem