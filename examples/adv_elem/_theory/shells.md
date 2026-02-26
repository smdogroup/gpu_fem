# Advanced Shell Elements

## Elements library
DeRham IGA element (DRIG)
- [x] Benzaken plate - [Multigrid Methods for Isogeometric Thin Plate Discretizations](https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf)
MITC element (mixed interpolation of tensorial components)
- [x] MITC4 element [A new MITC4+ shell element](https://www.sciencedirect.com/science/article/pii/S0045794916309464)
MITC-LP vs MITC-EP
- [x] new method of smoothing locking energy only vs global energy in prolongator, see [Energy Optimization of Algebraic Multigrid Bases](https://link.springer.com/article/10.1007/s006070050022) 

## New Nedelec style Shells (already published)
- [ ] 2024 Phd thesis - [Mixed Finite Element Methods For Nonlinear Continuum Mechanics And Shells](https://repositum.tuwien.at/handle/20.500.12708/17043)
    * REVIEW THIS, auto locking-free elements for shells, may solved mem locking too? 
- [ ] paper on conforming shells 2025 - [THE HELLAN–HERRMANN–JOHNSON AND TDNNS METHOD FOR LINEAR AND NONLINEAR SHELLS](https://arxiv.org/pdf/2304.13806)
    * may have solved mem locking?
- [ ] [Avoiding membrane locking with Regge interpolation](https://www.sciencedirect.com/science/article/pii/S004578252030709X)
- [ ] NGsolve mixed finite element library - https://jschoeberl.github.io/iFEM/plates_shells/4_rollup.html
- [ ] [Finite element discretizations of curvature tensors](https://meetings.ams.org/math/jmm2024/meetingapp.cgi/Paper/29833)

## Standard Shells
- [ ] [On a stress resultant geometrically exact shell model. Part I: Formulation and optimal parametrization](https://www.sciencedirect.com/science/article/pii/0045782589900029)
- [ ] [On a stress resultant geometrically exact shell model. Part II: The linear theory; Computational aspects](https://www.sciencedirect.com/science/article/pii/0045782589900984)
- [ ] [On a stress resultant geometrically exact shell model. Part III: Computational aspects of the nonlinear theory](https://www.sciencedirect.com/science/article/pii/0045782590900943)
- [ ] D. Chapelle and K.-J. Bathe, The Finite Element Analysis of Shells: Fundamentals (Compu-
tational Fluid and Solid Mechanics), 2nd ed. Berlin, Heidelberg: Springer Berlin Heidelberg,
2011, isbn: 978-3-642-16408-8


## C1-continuous Mindlin Shells

- [ ] [A novel continuity finite element based on Mindlin theory for doubly-curved laminated composite shells](https://www.sciencedirect.com/science/article/pii/S0263823121004286)
    * get other strain-based elements from this ref?
- [ ] [A Consistent Finite Element Formulation of the Geometrically Non-linear Reissner-Mindlin Shell Model](https://link.springer.com/article/10.1007/s11831-021-09702-7)
    * would this one be multigrid friendly? may just have to implement it
- [ ] [An efficient C1 finite element with continuity requirements for multilayered/sandwich shell structures](https://hal.science/hal-00087622/document)

## Isogeometric Mindlin shells

- [ ] [An isogeometric Reissner-Mindlin shell element based on Bézier dual basis functions: overcoming locking and improved coarse mesh accuracy](https://coreform.com/papers/isogeometric_reissner-mindlin_element_20200602.pdf)
    - [ ] this is great paper to look at has continuous vs. discrete I want to implement all these shell types.. IGA

- [ ] [Two-field formulations for isogeometric Reissner–Mindlin plates and shells with global and local condensation](https://link.springer.com/article/10.1007/s00466-021-02080-8)
- [ ] [Isogeometric shell analysis: The Reissner–Mindlin shell](https://www.sciencedirect.com/science/article/pii/S0045782509001820)
- [ ] J. Pitk¨aranta, “The problem of membrane locking in finite element analysis of cylindrical
shells,” Numerische Mathematik, vol. 61, no. 1, pp. 523–542, 1992.
- [ ] G. Kikis and S. Klinkel, “Two-field formulations for isogeometric reissner–mindlin plates
and shells with global and local condensation,” Computational Mechanics, vol. 69, no. 1,
pp. 1–21, 2022.
- [ ] G. Kikis, W. Dornisch, and S. Klinkel, “Adjusted approximation spaces for the treatment
of transverse shear locking in isogeometric reissner–mindlin shell analysis,” Computer
Methods in Applied Mechanics and Engineering, vol. 354, pp. 850–870, 2019.
- [ ] [Isogeometric Reissner–Mindlin shell analysis with exactly calculated director vectors](https://kluedo.ub.rptu.de/frontdoor/deliver/index/docId/4447/file/Dornisch+et+al.+-+Isogeometric+Reissner-Mindlin+shell+analysis+with+exactly+calculated+director+vectors.pdf)

## T-spline isogeometric
- [ ] T-spline for uCRM wingbox [Geometrically consistent static aeroelastic simulation using isogeometric analysis](https://www.sciencedirect.com/science/article/pii/S0045782518302779)
- [ ] [Adaptive isogeometric analysis by local h-refinement with T-splines](https://www.sciencedirect.com/science/article/pii/S0045782508002569)
- [ ] [Isogeometric analysis using T-splines](https://www.sciencedirect.com/science/article/pii/S0045782509000875)

## Isogeometric Kirchoff Shells
- [ ] R. A. Sauer, Z. Zou, and T. J. Hughes, “A simple and efficient hybrid discretization approach
to alleviate membrane locking in isogeometric thin shells,” Computer Methods in Applied
Mechanics and Engineering, vol. 424, p. 116 869, 2024.
- [ ] J. Kiendl, K.-U. Bletzinger, J. Linhard, and R. W¨uchner, “Isogeometric shell analysis
with kirchhoff–love elements,” Computer Methods in Applied Mechanics and Engineering,
vol. 198, no. 49, pp. 3902–3914, 2009.

## Other
- [ ] [A four-field mixed formulation for incompressible finite elasticity](https://www.researchgate.net/publication/389547895_A_four-field_mixed_formulation_for_incompressible_finite_elasticity)