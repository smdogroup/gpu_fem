# Additive Schwarz smoothers + preconditioners

* had trouble with additive schwarz smoothers in the past as I would try and add constraint forces from surrounding constrained nodes.
* you don't need to do that, just take P some subspace to global prolongation operator and solve the coarse grid system, then correction..

u_i = P * (P^T K P)^{-1} P^T F for each P_i subspace
of the global system K * u = F

Additive schwarz then takes the correction u = sum_i omega * u_i for some omega constant (for convergence)
* then you put this smoother inside a Krylov method

I was often worried about whether the Additive schwarz smoother would have rigid body modes contained in it.
Here is a brief explanation that it does not.
* if single nodes are taken from a beam say P_i = e_i (so that only one DOF comes out of P_i^T).
   Then we get a jacobi smoother (or block-version at that). In this case no translation / rotation rigid body modes occur.
* if two nodes are taken at a time, with P_i^T = [0,...,I_{2x2},...0] and P_i^T is 2 x N restriction matrix,
   then we'll get 2x2 coarse systems which would be as if we took those two nodes of the beam and all others are clamped.
   That system is also free of rigid body modes since element stiffness matrices from surrounding elements were added to these two interior nodes
   making them no longer singular (whereas if it was just those two nodes and one element alone would be singular).
* The same is true for all groups of nodes allowing me to now write coupled smoothers which were shown in the multigrid book
   to be much more powerful and result in more scalable smoothing for very low thickness or high condition number shells.
* I will now first demonstrate these smoothers in python, then on the GPU.
* NOTE : I am not required to choose 3x3 block of 9 nodes or something, I can choose ANY set of coupled nodes such as 5 nodes in cross-like stencil, 3 nodes, even 4 nodes, etc.
   * whatever is scalable, though smoothing strength will vary.
   * can even do line smoothers in each direction if that is easier to invert the 3x3 blocks.

* later will figure out how to efficiently compute the inverses of these larger blocks on the GPU.
* I will begin by demonstrating Additive schwarz smoother as preconditioner for Krylov methods (not as straight up linear solver).