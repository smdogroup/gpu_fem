from pyNastran.bdf.bdf import BDF

# Read in the model (pyNastran will follow INCLUDE statements)
model = BDF()
model.read_bdf("capsStruct_0/Scratch/tacs/tacs.dat", xref=True)

# Write out as a single combined BDF
model.write_bdf("merged.bdf", size=16)
