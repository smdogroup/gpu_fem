if args.elem == 'drig':
    ELEMENT = DeRhamIsogeometricCylinderElement(
        r=R, axial_factor=axial_factor,
        curvature_on=True, # curvature terms lead to mem locking
        # curvature_on=False,
    )
    ASSEMBLER = DeRhamIGACylinderAssembler
elif args.elem == 'mdrig':
    ELEMENT = MixedDeRhamIGACylinderElement(
        r=R, axial_factor=axial_factor,
        curvature_on=True, # curvature terms lead to mem locking
        # curvature_on=False,
    )
    ASSEMBLER = MixedDeRhamIGACylinderAssembler