# Current Devel tasks

- [ ] asymptotic isogeometric plate element
    - [x] reused physics and data class from isotropic shell
    - [x] finish outer plate assembler
    - [x] CUDA kernels with asymptotic IGA plate assembler
    - [x] implement StructuredIGAProlongation based on jupyter notebook + ternary operators
    - [ ] asymptotic solution update for plate bending
       - [ ] subtract derivative terms on output to VTK (cause C0-continuity not subset of C1-continuity, C1 is more restrictive)
       - [ ] custom IGA VTK writer with exact x,y,z outputs from spline (not same as control points)
    - [ ] demo + verify plate AIG direct solve on GPU
    - [ ] devel + demo multigrid plate AIG on GPU
    - [ ] asymptotic correctness term of AIG
    - [ ] verify asymptotic correctness of plateAIG
    - [ ] TBD: would it still work if several different thickness components? need rotation forces adjustment between patches or sections?
       * maybe you could rescale by one thickness everywhere?
- [ ] asymptotic isogeometric shell element
- [ ] smoothed aggregation algebraic multigrid
- [ ] qordering ILU(0) based on NASA FUN3D