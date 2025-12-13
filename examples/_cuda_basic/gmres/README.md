# GMRES example
<<<<<<< HEAD

* comparing GMRES convergence between CSR and BSR matrices
* The CSR + float operations are most supported while BSR is being deprecated (though Kevin sent an email to them about that)

## Observations
* BSR double matrix mult (Dsbsrmv) doesn't give the correct response on 2x2 matrix * 2x1 vec (against truth). Works with float and BSR => may need to write my own BSR double mv method
* CSR + double data type case converges great and matches python (this is the baseline for getting to work with BSR now)
* BSR float case => doesn't converge as well as CSR doesn't match
    * nz pattern different bc of CSR to BSR conversion => led to different LU decomp
    
=======
* TODO:
>>>>>>> 3d8b9704f85882d7188003b5b85f7a195cd75615
