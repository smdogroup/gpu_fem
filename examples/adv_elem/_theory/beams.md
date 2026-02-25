# Beam Theories and Multigrid

## Beam Elements

AIG element 
    * technically considers plates + shells here, but I only used this in beams and wasn't good for multigrid for that
- [x] [Asymptotically accurate and locking-free finite element implementation of first order shear deformation theory for plates](https://www.sciencedirect.com/science/article/pii/S0045794924001160)
- [x] ["Asymptotically accurate and geometric locking-free finite element implementation of a refined shell theory"](https://www.sciencedirect.com/science/article/pii/S0045782525005997)
- [x] [Variational-asymptotic method of constructing a theory of shells](https://www.sciencedirect.com/science/article/pii/0021892879901576)

HHR, HHD and HIGD elements (hierarchic oesterle)
- [x] Oesterle Hierarchic elements: [A shear deformable, rotation-free isogeometric shell formulation](https://www.sciencedirect.com/science/article/pii/S004578251630202X)

DeRham IGA element
- [x] Benzaken plate - [Multigrid Methods for Isogeometric Thin Plate Discretizations](https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf)