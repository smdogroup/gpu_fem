# Current Devel tasks

## Qordering ILU(0) NASA
- [x] read through qordering ILU(0) libraries in NASA FUN3D (SFE section is where Kyle's code is)
   * do they do pivoting in each block and how do they do pivoting (YES)
   * do they use cusparse on GPU? (NO)
- [ ] code up qorder "iluk_sse" block size 6 pivoting (slow version in C++)
- [ ] demo iluk_sse qorder on wing case (like in Kyle's paper) first CPU parallel
- [ ] write my own pivoted ILU factorization on GPU with level sets (to replace CuSparse no pivot ILU factor), if possible. will have to see if it is possible.

## Smoothed aggregation AMG
- [ ] demo SA-AMG on plate in python

## AIG Elements

- [ ] AIG9 plate element
    - [x] reused physics and data class from isotropic shell
    - [x] finish outer plate assembler
    - [x] CUDA kernels with asymptotic IGA plate assembler
    - [x] implement StructuredIGAProlongation based on jupyter notebook + ternary operators
    - [ ] asymptotic solution update for plate bending
       - [ ] call asymptotic rhs term.. on fext
       - [ ] subtract derivative terms on output to VTK (cause C0-continuity not subset of C1-continuity, C1 is more restrictive)
       - [ ] custom IGA VTK writer with exact x,y,z outputs from spline (not same as control points)
    - [ ] demo + verify plate AIG direct solve on GPU
       - [ ] fix plate mesh conv loads to work with higher order IGA elements (causes nans if you call that right now)
    - [ ] devel + demo multigrid plate AIG on GPU
      - [x] fix and debug structured IGA prolongation and restriction => just ended up removing / penalizing line searches and it does much better..
    - [ ] verify asymptotic correctness of plateAIG
- [ ] AIG9 shell element
   - [ ] handle multiple matches with T-splines (with each patch one TACS component, use component maps and a new patch bndry array to identify nodes on patch boundaries)
   - [ ] demo T-splines on uCRM wing
- [ ] investigate if weak form is actually correct for asymptotic plate elem
   * currently getting worse performance with higher slenderness, and not sure why..
   * should remove the relation to gamma for condition number or bending to trv shear stiffness disparities..
   * chatgpt originally said slenderness should still appear in cond # (because higher thickness to 1 but the xyz increase so still bad slenderness). However rotation rescaling should change the differential operator and make it solve like less slender plate too.. you could see this for beam case. So need to check my work on the LHS and see if it really is independent to plate slenderness (not just thickness right)?
   - [ ] is the rotation transform done correctly now?
   - [ ] maybe try on beams..
   - [ ] compare to the direct weak form proposed in the original AIG plate element paper.. see if it matches that.. cause I kind of hacked it and it may not be the exact same..
- [ ] lit review and/or devel AIG element for non-constant thickness
- [ ] develop some theory on whether the condition number should improve for AIG elements vs CFI
   * seems like actually condition number is gonna be the same as before..
   * confirm this with numerical experiments
   - [ ] add this to paper showing block 2x2 Kelem OMAG
   - [ ] look at beam case and see if it is possible to recover Kirchoff-like multigrid performance again.. need the strain-gap DOF like HR element?
   - [ ] extend that to plate and shell case..
   - [ ] may need a new element that improves condition number after all?
   * in paper if this is true: explain why asymptotic element would prevent locking but not improve condition number (so restores mesh convergence to high order) but would not necessarily improve multigrid solve times (though IGA may help with that some too as it is better at representing geometries)

# done tasks
