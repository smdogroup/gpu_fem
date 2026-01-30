
    """
    # De Rham (cohomology) IGA plate element
    # based on paper: https://www.sciencedirect.com/science/article/abs/pii/S0045782511003215
    # "An isogeometric method for the Reissner-Mindlin plate bending problem" by Veiga

    # see also the thickness-independent Schwarz multigrid smoothers for it in Benzaken et al.,
    # https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf
    # "Multigrid Methods for Isogeometric Thin Plate Discretizations"

    # multi-patch and domain decomp methods from this book on mixed variational FEA Methods
    # https://dmvn.mexmat.net/content/books/brezzi-fortin-mixed-and-hybrid-finite-elements-methods.pdf

    # except here I'm just doing a beam case..
    """
