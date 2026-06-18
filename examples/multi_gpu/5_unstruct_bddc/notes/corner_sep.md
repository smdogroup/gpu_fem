# Notes on CornerSeparatingSplitter

Has very few violations for a few special cases, lower corner violations should be a good thing, cause that means fewer corner / vertex nodes are on same element. This would lead to very poor condition number at thin shell for each violation (HYPOTHESIS).

Yielding great thin shell performance (t=10^-3)!
nxe = 100, subdomain = 5x5 => 12 violations, conv in 136 iterations
nxe = 200, subdomain = 5x5 => 25 violations, conv in 99 iterations
nxe = 200, subdomain = 2x2 => 0 violations, conv in 57 iterations

Poor performance
nxe = 100, subdomain = 4x4 => 217 violations, conv in 404 iterations
nxe = 200, subdomain = 4x4 => 917 violations, conv in 352 iterations