# Advanced Shell Elements


## Plates
* don't spend too much time going over plates again (great performance), but do go over this
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


## Standard Shells

- [ ] TODO : get simo and other shell theory papers (see which include Christoffels and which don't, Dr. K's does not?)
- [ ] D. Chapelle and K.-J. Bathe, The Finite Element Analysis of Shells: Fundamentals (Compu-
tational Fluid and Solid Mechanics), 2nd ed. Berlin, Heidelberg: Springer Berlin Heidelberg,
2011, isbn: 978-3-642-16408-8

## Multigrid with Shells?
- [ ] Benzaken plate - [Multigrid Methods for Isogeometric Thin Plate Discretizations](https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf)
    - [ ] write down shell theories in folder / markdown below

## C1-continuous Mindlin Shells

- [ ] [A novel continuity finite element based on Mindlin theory for doubly-curved laminated composite shells](https://www.sciencedirect.com/science/article/pii/S0263823121004286)
    * get other strain-based elements from this ref?
- [ ] [A Consistent Finite Element Formulation of the Geometrically Non-linear Reissner-Mindlin Shell Model](https://link.springer.com/article/10.1007/s11831-021-09702-7)
    * would this one be multigrid friendly? may just have to implement it
- [ ] [An efficient C1 finite element with continuity requirements for multilayered/sandwich shell structures](https://hal.science/hal-00087622/document)

## Isogeometric Mindlin shells

- [ ] [An isogeometric Reissner-Mindlin shell element based on Bézier dual basis functions: overcoming locking and improved coarse mesh accuracy](https://coreform.com/papers/isogeometric_reissner-mindlin_element_20200602.pdf)
    - [ ] this is great paper to look at has continuous vs. discrete I want to implement all these shell types.. IGA
    
- [ ] [Isogeometric shell analysis: The Reissner–Mindlin shell](https://www.sciencedirect.com/science/article/pii/S0045782509001820)
- [ ] [Adaptive isogeometric analysis by local h-refinement with T-splines](https://www.sciencedirect.com/science/article/pii/S0045782508002569)
- [ ] [Isogeometric analysis using T-splines](https://www.sciencedirect.com/science/article/pii/S0045782509000875)
- [ ] J. Pitk¨aranta, “The problem of membrane locking in finite element analysis of cylindrical
shells,” Numerische Mathematik, vol. 61, no. 1, pp. 523–542, 1992.
- [ ] G. Kikis and S. Klinkel, “Two-field formulations for isogeometric reissner–mindlin plates
and shells with global and local condensation,” Computational Mechanics, vol. 69, no. 1,
pp. 1–21, 2022.
- [ ] G. Kikis, W. Dornisch, and S. Klinkel, “Adjusted approximation spaces for the treatment
of transverse shear locking in isogeometric reissner–mindlin shell analysis,” Computer
Methods in Applied Mechanics and Engineering, vol. 354, pp. 850–870, 2019.
- [ ] [Isogeometric Reissner–Mindlin shell analysis with exactly calculated director vectors](https://kluedo.ub.rptu.de/frontdoor/deliver/index/docId/4447/file/Dornisch+et+al.+-+Isogeometric+Reissner-Mindlin+shell+analysis+with+exactly+calculated+director+vectors.pdf)

## Isogeometric Kirchoff Shells
- [ ] R. A. Sauer, Z. Zou, and T. J. Hughes, “A simple and efficient hybrid discretization approach
to alleviate membrane locking in isogeometric thin shells,” Computer Methods in Applied
Mechanics and Engineering, vol. 424, p. 116 869, 2024.
- [ ] J. Kiendl, K.-U. Bletzinger, J. Linhard, and R. W¨uchner, “Isogeometric shell analysis
with kirchhoff–love elements,” Computer Methods in Applied Mechanics and Engineering,
vol. 198, no. 49, pp. 3902–3914, 2009.
