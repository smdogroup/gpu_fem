# Sparse Approximate Inverse (SPAI) Preconditioners

SPAI preconditioners don't suffer from zero pivots and may be much better for indefinite or high condition number systems

- [ ] implement FSAI based on this paper from Saad, https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf

* implement frobenius norm minimization on plate problem (see bachelor's thesis https://fse.studenttheses.ub.rug.nl/11132/1/Koen_van_Geffen_2013_TWB.pdf)
* try deflation method on SPAI preconditioner (also from https://fse.studenttheses.ub.rug.nl/11132/1/Koen_van_Geffen_2013_TWB.pdf) to help elim near-singular or problematic modes
