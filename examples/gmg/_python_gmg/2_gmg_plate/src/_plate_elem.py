import numpy as np

def plate_element_stiffness(E, nu, h, coords):
    """
    Returns the 12x12 bending stiffness matrix for a Kirchhoff plate element
    using Lagrange bilinear shape functions.

    Parameters:
    -----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    h : float
        Thickness of the plate
    coords : (4, 2) array
        Nodal coordinates of the quadrilateral element (counter-clockwise)

    Returns:
    --------
    Ke : (12, 12) ndarray
        Element stiffness matrix (bending only)
    """

    # Bending rigidity
    D = E * h**3 / (12 * (1 - nu**2))
    Dmat = D * np.array([
        [1,     nu,     0],
        [nu,    1,      0],
        [0,     0,  (1 - nu)/2]
    ])

    # 2x2 Gauss points for quadrilateral
    gp = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                   [ 1/np.sqrt(3), -1/np.sqrt(3)],
                   [ 1/np.sqrt(3),  1/np.sqrt(3)],
                   [-1/np.sqrt(3),  1/np.sqrt(3)]])
    
    Ke = np.zeros((12, 12))
    
    for xi, eta in gp:
        # Shape functions
        N = 0.25 * np.array([
            (1 - xi)*(1 - eta),
            (1 + xi)*(1 - eta),
            (1 + xi)*(1 + eta),
            (1 - xi)*(1 + eta)
        ])
        
        # Derivatives w.r.t. xi and eta
        dN_dxi = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)]
        ])

        # Jacobian and its inverse
        J = dN_dxi.T @ coords
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        # Derivatives w.r.t. x, y
        dN_dx = dN_dxi @ Jinv

        # Construct B matrix for bending
        B = np.zeros((3, 12))  # (kappa_x, kappa_y, kappa_xy)

        for i in range(4):
            dNi_dx, dNi_dy = dN_dx[i]
            # Rotation DOFs are at index 3*i+1 (theta_x), 3*i+2 (theta_y)
            B[0, 3*i+1] = dNi_dx        # kappa_x = d(theta_x)/dx
            B[1, 3*i+2] = dNi_dy        # kappa_y = d(theta_y)/dy
            B[2, 3*i+1] = dNi_dy        # kappa_xy = d(theta_x)/dy
            B[2, 3*i+2] = dNi_dx        #         + d(theta_y)/dx

        Ke += B.T @ Dmat @ B * detJ

    return Ke
