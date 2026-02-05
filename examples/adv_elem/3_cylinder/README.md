# RM curvilinear shells

* starting to do cylidner case, general KL-shell analysis

## Lit Review
- [ ] read Kiendl et al. https://web.me.iastate.edu/jmchsu/files/Kiendl_et_al-2015-CMAME.pdf
    * On the canonical equations of Kirchhoff-Love theory of shells
    * 411 citations, has good refs for geomNL KL-shell analysis with C1-cont IGA
    * should show me how to make fully consistent KL-shell solve with IGA, and then maybe I can extend that to RM-case 
        respecting shear and mem locking physics (zero shear strains of trv and membrane variety)
- [ ] read Kiendl et al. 2009, https://www.sciencedirect.com/science/article/pii/S0045782509002680
    * Isogeometric shell analysis with Kirchhoff–Love elements
    * how to do rotation-free IGA KL shells
- [ ] Thanh et al., https://www.sciencedirect.com/science/article/pii/S0045782511002696
    * Rotation free isogeometric thin shell analysis using PHT-splines
- [ ] Benston et al. 2011, https://www.sciencedirect.com/science/article/pii/S0045782510003488
    * A large deformation, rotation-free, isogeometric shell
- [ ] Uhm et al. 2009, https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2648
    * T-spline finite element method for the analysis of shell structures
- [ ] Benson et al. 2010, https://www.sciencedirect.com/science/article/pii/S0045782509001820
    * Isogeometric shell analysis
- [ ] Dornish 2013, https://www.sciencedirect.com/science/article/pii/S0045782512002927
    * Isogeometric Reissner–Mindlin shell analysis with exactly calculated director vectors
- [ ] D.J. Benson, S. Hartmann, Y. Bazilevs, M.-C. Hsu, and T.J.R. Hughes. Blended isogeometric
shells. Computer Methods in Applied Mechanics and Engineering, 255:133–146, 2013
- [ ] Lu and C. Zheng. Dynamic cloth simulation by isogeometric analysis. Computer Methods
in Applied Mechanics and Engineering, 268:475–493, 2014
- [ ] N. Nguyen-Thanh, N. Valizadeh, M.N. Nguyen, H. Nguyen-Xuan, X. Zhuang, P. Areias,
G. Zi, Y. Bazilevs, L. De Lorenzis, and T. Rabczuk. An extended isogeometric thin shell
analysis based on Kirchhoff–Love theory. Computer Methods in Applied Mechanics and
Engineering, 284:265–291, 2015.
- [ ] J. Kiendl. Isogeometric Analysis and Shape Optimal Design of Shell Structures. PhD thesis,
Technische Universtit¨at M¨unchen, 2011.
- [ ]  J. Kiendl. Isogeometric Analysis and Shape Optimal Design of Shell Structures. PhD thesis,
Technische Universtit¨at M¨unchen, 2011.
- [ ] M.A. Scott, M.J. Borden, C.V. Verhoosel, T.W. Sederberg, and T.J.R. Hughes. Isogeometric
finite element data structures based on B´ezier extraction of T-splines. International Journal
for Numerical Methods in Engineering, 88:126–156, 2011
- [ ] J. Kiendl, Y. Bazilevs, M.-C. Hsu, R. W¨uchner, and K.-U. Bletzinger. The bending strip
method for isogeometric analysis of Kirchhoff-Love shell structures comprised of multiple
patches. Computer Methods in Applied Mechanics and Engineering, 199:2403–2416, 2010
- [ ] A. Apostolatos, R. Schmidt, R. W¨uchner, and K.-U. Bletzinger. A Nitsche-type formula-
tion and comparison of the most common domain decomposition methods in isogeometric
analysis. International Journal for Numerical Methods in Engineering, 97:473–504, 2013.
- [ ] Guo and M. Ruess. Nitsche’s method for a coupling of isogeometric thin shells and blended
shell structures. Computer Methods in Applied Mechanics and Engineering, 284:881–905,
2015.
- [ ] https://link.springer.com/article/10.1007/s10778-007-0115-6
- [ ] W. Dornisch and S. Klinkel. Treatment of Reissner-Mindlin shells with kinks without the
need for drilling rotation stabilization in an isogeometric framework. Computer Methods in
Applied Mechanics and Engineering, 276:35–66, 2014.
- [ ] D.J. Benson, S. Hartmann, Y. Bazilevs, M.-C. Hsu, and T.J.R. Hughes. Blended isogeometric
shells. Computer Methods in Applied Mechanics and Engineering, 255:133–146, 2013.
- [ ] S. Hosseini, J.J.C. Remmers, C.V. Verhoosel, and R. de Borst. An isogeometric solid-like
shell element for nonlinear analysis. International Journal for Numerical Methods in Engi-
neering, 95:238–256, 2013
- [ ] R. Bouclier, T. Elguedj, and A. Combescure. Efficient isogeometric NURBS-based solid-
shell elements: Mixed formulation and B-bar-method. Computer Methods in Applied Me-
chanics and Engineering, 267:86–110, December 2013

## TODO for thick-ind multigrid
- [ ] try rotation-free RM beam+plate based on prev rot-free KL shells (maybe integration promote rotations?)
- [ ] then extend to cylinder and other shell cases..
- [ ] include geomNL too