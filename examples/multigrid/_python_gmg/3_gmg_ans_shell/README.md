* MITC element coarse-fine interp or prolongation doesn't respect tying point lines (near thin plate limit) => can lead to huge defects
* trying here a consistent interpolation shell element used in J. Fish work 
    * namely "unstructured multigrid method for shells" by J. Fish
    * expect this to respect tying strain interps better and support grid hierarchy better than MITC element for geometric multigrid
* Curved C0 shell element with consistent interp "A Curved C Shell Element Based on Assumed Natural-Coordinate Strains" by K.C. Park