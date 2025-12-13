# Low-Level TODO List
* "todo_1.md" is all low-level tasks for the current high-level task

*Type Ctrl+Shift+V to view in VScode to view markdown*

* current task is derivatives of structural optimization for tasks 2.1 and 2.2 of geometric programming paper
    * **2.1**: implement derivatives of linear static analysis
    * **2.2**: demo structural optimization with unstiffened panels

Low-level tasks are:
- [x] compute stresses
- [x] assert that data and physics are compatible with each other..
- [x] move tying strain computations outside of physics class into shell_utils.h and just check linear or nonlinear from physics template argument
- [x] add nonlinear version of tying strains
    - [x] update calls to have variables in those sections too
- [x] compute KS failure scalar for von mises stress of unstiffened panels
- [x] compute mass routine
- [x] function objects
- [x] design variable objects => decided no DV objects
- [x] evalFunctions routine
- [x] _compute_mass_DVsens
- [x] _compute_ksfailure_DVsens
- [x] _compute_ksfailure_SVsens
- [x] _compute_ksfailure_adjResProduct
- [ ] demo adjoint solve for ksfailure with K^T * psi = -df/du and then total deriv df/dx = (df/dx)_partial + dR/dx^T psi

- [ ] put function declarations at top of each class so easier to use by new users
- [ ] review journal paper

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
