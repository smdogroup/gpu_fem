# Poisson Geometric Multigrid on GPU

* use float precision, CSR nofill pattern, geometric multigrid with V-cycles
* various smoothers: jacobi, chebyshev-jacobi, gauss-seidel, asynch gauss-seidel, ILU(0), additive schwarz
* the code in this example will be fully self-contained
* measure performance on Milan A100 GPUs to see scalability to billions of DOF or more hopefully..