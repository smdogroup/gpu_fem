# MELD - transfer schemes

- [ ] [Efficient and Robust Load and Displacement Transfer Scheme Using Weighted Least Squares](https://arc.aiaa.org/doi/full/10.2514/1.J057318)
    * original MELD, requires jacobian of SVD which may not parallelize well on the GPU
    * try linearized MELD which doesn't need SVD

- [ ] [Load and Displacement Transfer through Quaternion-Based Formulation of the Absolute Orientation Problem](https://arc.aiaa.org/doi/10.2514/6.2024-2415)
    * quaternion based formulation of MELD => no longer requires SVD jacobian, may be easier to parallelize on the GPU

## SVD Utils for original MELD
- [ ] [Eigenvalues of a Symmetric 3 x 3 Matrix](https://dl.acm.org/doi/pdf/10.1145/355578.366316)
    * analytic eigenvalues of a 3x3 sym matrix, I use this for eigenvalue problem of the SVD namely H^* H = A so do eigenvalue problem on A to get singular values.