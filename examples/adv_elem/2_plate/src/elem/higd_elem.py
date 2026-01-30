


# based on Oesterle paper, "A shear deformable, rotation-free isogeometric shell formulation"
# https://www.sciencedirect.com/science/article/pii/S004578251630202X


class HierarchicIsogeometricDispElement9:
    """hierarchic displacement element HIGD with 2nd order IGA basis to allow 2nd derivatives in weak form"""
    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 2
        self.nodes_per_elem = 9
        self.reduced_integrated = reduced_integrated
        self.clamped = True
        self.ORDER = 2 # 2nd order IGA