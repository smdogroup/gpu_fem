# Current Devel tasks

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
- [ ] smoothed aggregation algebraic multigrid
- [ ] qordering ILU(0) based on NASA FUN3D

# done tasks
