* BSR better on GPU, CSR better on CPU
* BSR better with iterative solvers, CSR better with direct solvers
* aeroelastic optimization => coupled analysis with like ~100,500 repeated linear solves that we would like to be very fast
    * direct solve here is nice because it can save LU and then just do triangular solves repeatedly
    * maybe CSR with direct solve would be good for that application
* want more solver options so that we can choose the best solver (problem-specific)

* direct solve faster with many linear solves on same matrix
* iterative solve faster on a single linear solve

* application 1: if you're doing a coupled analysis and you would benefit from direct solve where you retain LU in the matrix
    * may be better to do CSR direct LU solve then
    * downside of CSR is still more data storage (but only more ints, #floats is the same) => so might not be too bad, it still will store 36 * more ints,
     which I think might double the memory size for the matrix
    * even with double memory size, may still be better to retain LU in a coupled analysis
    * can we do with CSR only or BSR? TBD (may have to write our own fillin)
* application 2 : regular structural optimization (no coupling with fluids or anything)
    * assume you just do one linear solve per optimization step
    * then don't want direct solve (no benefit) probably (not solving repeatedly)
    * so instead you'll want to use BSR matrix with iterative solver here (bc BSR should work with iterative solver)

* next steps:
    * make a BSR iterative solve on cusparse for GPU
    * make a CSR direct solve on cusparse for GPU
    * stretch goal, maybe talk with Kevin : BSR direct solve on GPU (hard to get fillin & currently failing)