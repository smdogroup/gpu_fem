import numpy as np

hdl = open("ML-sizing-1.txt", "r")
lines = hdl.readlines()
hdl.close()

comp_names = [f"lOML{iOML}" for iOML in range(1, 23)]
comp_names += [f"uOML{iOML}" for iOML in range(1, 23)]
comp_names += [f"rib{irib}" for irib in range(1, 24)]
comp_names += [f"spLE{ispar}" for ispar in range(1, 23)]
comp_names += [f"spTE{ispar}" for ispar in range(1, 23)]

components = {comp_name:{
    "ref_axis" : np.zeros(3),
    "LENGTH" : 1.0,
    "T" : 1.0,
    "sheight" : 1.0,
    "sthick" : 1.0,
    "spitch" : 1.0,
} for comp_name in comp_names}


past_struct = False
for line in lines:

    if "structural" in line:
        # get past first aerodynamic discipline section
        past_struct = True; continue
    
    if past_struct:
        # now can check DVs
        chunks = line.split(" ") 
        # print(f"{chunks=}")
        name_var = chunks[1]
        nvar_chunks = name_var.split("-")
        name, var = nvar_chunks[0], nvar_chunks[1]
        value = float(chunks[-1].strip())

        # print(f"{name=} {var=} {value=}")
        # print(line)
        components[name][var] = value
        _ref_axis = None
        if "rib" in name or "sp" in name:
            _ref_axis = np.array([0, 0, 1])
        else: # OML is in name
            iOML = int(name[4:])
            # print(f"{name=} {iOML=}")
            if iOML < 4: # 1,2,3
                # wing span is in y direction and z is vertical, x is downstream
                _ref_axis = [0, 1, 0]
            else:
                _ref_axis = np.array([0.23018, 0.61677, 0.0])
                _ref_axis /= np.linalg.norm(_ref_axis)

        components[name]["ref_axis"] = np.array(_ref_axis)

# print(components), debug
# for icomp in range(len(comp_names)):
#     _name = comp_names[icomp]
#     _comp_dict = components[_name]
#     print(f"{_name} : {_comp_dict}")

out_hdl = open("AOB-design.txt", "w")
for icomp in range(len(comp_names)):
    _name = comp_names[icomp]
    _comp_dict = components[_name]
    _ref_axis = _comp_dict["ref_axis"]
    out_hdl.write(f"{_name} : {_ref_axis[0]:.4f} {_ref_axis[1]:.4f} {_ref_axis[2]:.4f},")
    for var in ["LENGTH", "T", "sheight", "sthick", "spitch"]:
        _value = _comp_dict[var]
        out_hdl.write(f" {_value:.5f}")
    out_hdl.write("\n")

out_hdl.close()