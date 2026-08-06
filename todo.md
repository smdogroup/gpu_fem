

## Journal paper plan

- [ ] multigrid has these issues with MITC for thin shells
- [ ] show some of the beam multigrid stuff (only subset of elements cite dissertation).. (adding this in) - C1 is important
- [ ] do include plate +/or cylinder very briefly
   * then this shows multigrid good solver but not fully robust to thin shells (with MITC4), and limitations of mixed IGA methods explain.
- [ ] show comparison of runtimes on different hardware (RTX 3060, RTX 3090, A100, H100, B200?)
- [ ] BDDC-LU improves performance but requires wraparound subdomain splitting
   - [ ] add multilevel BDDC dots into the cylinder + wing runtime comparisons
- [ ] empirical / theoretical evidence of wraparound method
   * run a bunch of different multi-patch structures, with wraparound fraction
- [ ] push more on the unstructured mesh BDDC (and comparison to structured BDDC in paper)
   * Dr. K said can I just add a vertex node on subdomain boundaries that are missing a vertex node (even though not 3 or more subdomain connections there). Yes this could be good solution, need robust here.
- [ ] demonstrate GPU (GPU-CPU) speedups + linear solver comparison
- [ ] optimization demonstration


## Journal paper tasks

* see examples/domdec/unstruct_bddc/todo.md for some more tasks

0. [x] multi-GPU direct-LU solver to use for BDDC
   - [x] try single GPU direct solve on root GPU (just with larger subdomains)
   - [x] try CuDSS multi-GPU direct solve again has to be copied to root GPU so may not be great
   - [x] try to write my own distributed multi-GPU direct solve and see if it is better.. multifrontal or MUUMPS? Schur complement-LU? or sequential factorization?
   - [x] double check CuDSS can/can't use distributed matrix formats
   * CUDSS probably won't work that great for high DOF cause must be on root GPU first? So would have to copy it there?
   - [x] could try distributed CSR matrices, then using Lapack level 3 mat-mat operations may be able to assemble Schur complement myself? This is something that can be done for CSR but not BSR with CuSparse
   - [x] Ali said maybe SuperLU_dist has a multi-GPU Schur complement solver (try it out).. https://www.exascaleproject.org/wp-content/uploads/2022/06/LiSherrySparseBofSlides.pdf, also maybe see if hypre can do it. 
   - [x] look at the solver on this paper now, https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059

1. [ ] Finish multi-GPU development
   - [ ] GMG-ASW on multi-GPU
      - [x] do unstructured prolong for wing case now - TacsComponentPartitioner + unstructured prolongation classes (no additional ghost nodes needed probably)
      - [ ] show speedup of multi-GPUs (speedup values)
   - [ ] 2-level BDDC-LU on multi-GPU
#    - [ ] maybe also try multi CPU + GPU (for really high DOF problems)? Worth for more than 4 GPUs?

2. [ ] finish unstructured BDDC (do need this for paper)
   - [ ] finish tasks in domdec/examples/unstruct_bddc/todo.md
   - [ ] make a scatter plot with 4 dots (unstruct-cyl, struct-cyl, unstruct-wing, struct-wing) for thin shells and show how corner violations / whatever metrics influence the runtime
      - [ ] it would show evidence / correlation of a pattern here..
      - [ ] maybe also show diff thicks across diff cases affect sensitivity?
      - [ ] keep wing vertex on patch boundary metric too and show how that influences runtime (again show a bunch of wing cases maybe in scatter plot to show evidence instead off proof!), like Aaron's paper with many optim cases back then
   - [ ] unstructured BDDC on plate/cylinder case
      * try adding subdomain vertices to subdomian interfaces with none (check), near bndry
   - [ ] unstructured wraparound BDDC on wing problem (implement pseudocode)
      * demo on uCRM and HSCT meshes as additional cases in paper (maybe a table and with pictures)

3. [ ] high DOF wing optimization cases
   - [ ] implement smeared stiffener, buckling constraints + multiple load cases
   - [ ] do very high DOF Problems with GMG-ASW vs BDDC-LU for instance (laso show single GPU results too)

4. [ ] writing
   - [ ] add brief element affect on multigrid (beam, plates, shells)
   - [ ] show that iterative solvers don't work well for 2-level BDDC coarse Schur complement
      * previous papers with BDDC-AMG or multilevel BDDC are poisson or incomp N-S, so not as bad ill-cond https://arxiv.org/html/2410.14786v1, https://epubs.siam.org/doi/10.1137/19M1276479
      * show this is true for the BDDC-AMG, BDDC-BDDC (or BDDC3) and BDDC-ASW (in its own plot)
   - [ ] evidence that BDDC wraparound is good
   - [ ] comparison of unstructured + structured BDDC

6. put my GPU code into TACS repo (prob BDDC first, MITC4 shells)
   - [ ] make interface that constructs GPU assembler and classes from CPU assembler
   - [ ] then runs the GPU code as usual
   - [ ] implement BDDC (with these two tasks to make it more practical)
      - [ ] BDDC wraparound for unstructured meshes (and gen single-patch), min # corners and other metrics maybe
      - [ ] BDDC with more general simply supported vs clamped BCs (prob just duplicate node and make some DOF one in each view)

## DONE

- [x] add README.md for gpu_fem to show highly scalable structural analysis pictures (and make people interested in using it)


## Archived

0. Get a scalable multi-GPU coarse solver for BDDC-LU S_VV problem
   - [x] look at this source and high performance BDDC? https://arxiv.org/html/2410.14786v1
      * at least state in paper that they can more easily use multilevel BDDC or stuff like that cause it's poisson? Multilevel BDDC (here https://epubs.siam.org/doi/10.1137/19M1276479 has only been used for incompressible NS, so less ill-cond issues, pattern here)
      * conclusion: iterative solvers do not work for S_VV with thin shells (too ill-cond) need direct solver.
   * maybe: see if edge average constraints make S_VV conditioning more benign?
      * how to make coarse S_VV problem easier to solve
      * maybe try this in python first.. (journal would probably ask about this)
   - [ ] somehow need multi-GPU direct-LU solver
      * must write my own and somehow avoid large inter-GPU copying..
      * try multifrontal solving, do more research on how to do this.. will have to do it myself
   - [ ] show in paper that AMG, ASW and other BDDC solve thick plate but not thin shell well
