import numpy as np

hdl = open("_thicks.txt", "r")

comp_names = []
for line in hdl.readlines():
    # print(f"{line=}")
    if 'Femap' in line:
        after_colon = line.split(":")[-1]
        print(f"{after_colon=}")
        comp_name = after_colon.split(" ")[1] #[:-2]
        comp_name = comp_name.split("\n")[0]
        # print(f"{comp_name=}")
        comp_names += [comp_name]

print(f"{comp_names=}")

# now get thicknesses for them
int_thicks = [0.005, 0.003]
skin_thicks = [0.02, 0.005]

comp_prefixes = ['lOML', 'uOML', 'spTE', 'spLE', 'rib']
is_int_struct = [0, 0, 1, 1, 1]
thickDVs = []
for comp_name in comp_names:
    for iprefix, prefix in enumerate(comp_prefixes):
        if prefix in comp_name:
            comp_ind = int(comp_name.split(prefix)[1])
            # print(f"{comp_ind=}")
            span_frac = (comp_ind - 1) / 22.0

            if is_int_struct[iprefix]:
                thick = int_thicks[0] * (1.0 - span_frac) + int_thicks[1] * span_frac
            else:
                thick = skin_thicks[0] * (1.0 - span_frac) + skin_thicks[1] * span_frac
            thickDVs += [thick]

print(F"{thickDVs=}")
print(f"{len(thickDVs)=}")


