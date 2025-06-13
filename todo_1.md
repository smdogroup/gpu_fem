# Low-Level TODO List
* "todo_1.md" is all low-level tasks for the current high-level task

*Type Ctrl+Shift+V to view in VScode to view markdown*

* current task is derivatives of structural optimization for tasks 2.1 and 2.2 of geometric programming paper
    * **2.1**: implement derivatives of linear static analysis
    * **2.2**: demo structural optimization with unstiffened panels

Low-level tasks are:
- [ ] some source of high registers and slow computation in the shell elements, maybe templates or interp orders?
- [ ] try implementing a light version, then generalize stuff..
- [ ] the sens functions break the register count in drill strain (if comment out registers drop and then runtime increases by 10x)
- [ ] speedup drill strain again (can't run with more than 64 elems_per_block without it not running it)
    - [ ] hard-code stuff so no templates and clearly low register usage
- [ ] look at krylov enrichment
    - [ ] isolate problematic eigenmode, sick it in the arnoldi
    - [ ] deflated GMRES strategy
    - [ ] other enrichment strategies to improve GMRES speedup
    - [ ] what is A stability and L stability
    - [ ] I can check the condition number of the Kmat on the uCRM case and its eigenvalues
    * need to find a good preconditioner in GMRES, this method is still king
    - [ ] see work by Mark Carpenter in LaRC
- [ ] Dr. K said to pull out TACS CPU Kmat and try to solve with pyAMG to see if superior preconditioners to ILU(k) for GMRES

# Useful literature from NASA CFD Workshop
- [ ] FUN3D uses Jacobian-free Newton-Krylov method and variable preconditioner with ( Linear agglomeration multigrid [a geometric multigrid] + defect-correction Gauss-Seidel iterations )
    - [ ] I could look at implementing a Jacobian-free Newton-Krylov method, where I do compute preconditioner at some nonlinear steps, but don't always update it if soln update is small and then I can do matrix-vec products using an assembly
    * multigrid is very helpful for very large scale problem convergence
- [ ] look at work by Hessam Babae for reduced order models and quantum inspired ROMs
    - [ ] Dynamic Low-Rank Approximation of Matrix Differential equations
    - [ ] tensor train for high-dimensional such as 3-dimenisonal grids ROMs
    - [ ] can even solve high-dimensional nonlinear PDEs (Fokker-Planck, PDF/FDF transport equations)
    - [ ] DEIM-CUR Low-rank approximation



- [ ] verify derivatives for structural optimization
    - [ ] compute ksfailure with simple plate under simple strain distribution
    - [ ] compute mass of a flat plate (so know right answer)
    - [ ] verify mass DVsens against complex step
    - [ ] verify ksfailure DVsens against complex step
    - [ ] verify ksfailure SVsens against complex step
    - [ ] verify ksfailure adjointResProduct against complex step
    - [ ] verify adjoint solution and total derivative

- [ ] verify nonlinear strain case
    - [ ] change add_jacobian, add_residual to assemble calls and zero out data on call (find and replace in all files)
    - [ ] verify with cylindrical panel case against some known literature with Newton solver or Riks solver
