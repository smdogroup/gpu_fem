# Plates

## Elements I tried on plate in Scitech paper
* none of these really gave great multigrid performance (above we found that the DRIG element was good)

Review of elements
- [x] Shell theory review: [A Comprehensive Comparative Review of Various Advanced Finite Elements to Alleviate Shear, Membrane and Volumetric Locking](https://link.springer.com/article/10.1007/s11831-023-10050-x)

HIGD element (hierarchic isogeometric displacement)
- [x] Oesterle Hierarchic elements: [A shear deformable, rotation-free isogeometric shell formulation](https://www.sciencedirect.com/science/article/pii/S004578251630202X)

MITC element (mixed interpolation of tensorial components)
- [x] MITC4 element [A new MITC4+ shell element](https://www.sciencedirect.com/science/article/pii/S0045794916309464)

ANS element (Assumed Natural Strain)
- [x] [A Curved C0 Shell Element Based on Assumed Natural-Coordinate Strains](https://asmedigitalcollection.asme.org/appliedmechanics/article/53/2/278/391738/A-Curved-C0-Shell-Element-Based-on-Assumed-Natural)

CFI element (Chebyshev Fully integrated)
- [x] [Improvements in Shear Locking and Spurious Zero Energy Modes Using Chebyshev Finite Element Method](https://asmedigitalcollection.asme.org/computingengineering/article/19/1/011006/367456/Improvements-in-Shear-Locking-and-Spurious-Zero)

HRA element (Hellinger-Reissner Ansatz) 
- [x] [A variational method to avoid locking—independent of the discretization scheme](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.5766)

## New elements

HIGD elements (hierarchic oesterle) - bad multigrid performance
- [x] Oesterle Hierarchic elements: [A shear deformable, rotation-free isogeometric shell formulation](https://www.sciencedirect.com/science/article/pii/S004578251630202X)

DeRham IGA element (DRIG) - great multigrid performance
- [x] Benzaken plate - [Multigrid Methods for Isogeometric Thin Plate Discretizations](https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf)

Do this one too:
Hu-Zhang mixed element
- [ ] [Alleviating shear-locking in the Reissner-Mindlin plate via symmetric Hu-Zhang elements](https://www.researchgate.net/profile/Adam-Sky/publication/372647437_Alleviating_shear-locking_in_the_Reissner-Mindlin_plate_via_symmetric_Hu-Zhang_elements/links/64c130efc41fb852dd9d6889/Alleviating-shear-locking-in-the-Reissner-Mindlin-plate-via-symmetric-Hu-Zhang-elements.pdf)



## New Reading
* don't spend too much time going over plates again (great performance), but do go over this first subgrid or any interesting papers here
- [ ] subgrid formulation is interesting [A variational multiscale stabilized finite element formulation for Reissner–Mindlin plates and Timoshenko beams](https://upcommons.upc.edu/server/api/core/bitstreams/73fce476-07ab-4ea8-84b0-3552f852f9e7/content)
- [ ] [Developments of Mindlin-Reissner Plate Elements](https://onlinelibrary.wiley.com/doi/10.1155/2015/456740)
- [ ] D. Boffi, F. Brezzi, and M. Fortin, Mixed Finite Element Methods and Applications (Springer
Series in Computational Mathematics), 1st ed. Berlin, Heidelberg: Springer Berlin Heidel-
berg, 2013, vol. 44, isbn: 978-3-642-36518-8.
- [ ] [Numerical Approximation of Mindlin-Reissner Plates](https://www.jstor.org/stable/2008086?seq=1)
- [ ] [Least-squares Finite Element Approximations for the Reissner–Mindlin Plate](https://www.math.purdue.edu/~caiz/pdf-paper/99CaYeZh.pdf)
- [ ] [Preconditioning discrete approximations of the Reissner-Mindlin plate model]()
- [ ] [An edge-based smoothed finite element method (ES-FEM) with stabilized discrete shear gap technique for analysis of Reissner–Mindlin plates](https://www.sciencedirect.com/science/article/pii/S0045782509002990)
- [ ] ["An isogeometric method for the Reissner-Mindlin plate bending problem"](https://www.sciencedirect.com/science/article/abs/pii/S0045782511003215)
- [ ] [Legendre spectral finite elements for Reissner–Mindlin composite plates](https://www.sciencedirect.com/science/article/pii/S0168874X15000943)
