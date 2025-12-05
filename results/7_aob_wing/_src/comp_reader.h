#pragma once

#include <array>
#include <string>

struct ComponentConfig {
    std::string name;  // substring match
    std::array<double, 3> ref_axis;
    double panel_length = 0.0;  // optional
    double pthick = 0.0;
    double sheight = 0.0;
    double sthick = 0.0;
    double spitch = 0.0;
};

inline std::vector<ComponentConfig> loadComponentConfig(const std::string &filename) {
    std::vector<ComponentConfig> configs;
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Could not open component config file");
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);

        ComponentConfig c;
        char comma;

        ss >> c.name >> comma >> c.ref_axis[0] >> c.ref_axis[1] >> c.ref_axis[2] >> comma >>
            c.panel_length >> c.pthick >> c.sheight >> c.sthick >> c.spitch;

        configs.push_back(c);
    }
    return configs;
}

inline const ComponentConfig &matchComponentConfig(const std::string &descr,
                                                   const std::vector<ComponentConfig> &configs) {
    for (const auto &c : configs) {
        if (descr.find(c.name) != std::string::npos) {
            return c;
        }
    }

    // fallback to the last entry, assuming it's "default"
    return configs.back();
}

template <class T, class Data>
void build_AOB_component_data(TACSMeshLoader &mesh_loader, HostVec<Data> &comp_data,
                              const std::string &filename = "design/AOB-design.txt") {
    int ncomp = comp_data.getSize();
    // read original design file for DVs, length, ref axis, etc.
    std::vector<ComponentConfig> configs = loadComponentConfig(filename);

    for (int icomp = 0; icomp < ncomp; icomp++) {
        const char *descript_c = mesh_loader.getComponentDescript(icomp);
        // printf("descript_c '%s'\n", descript_c);
        std::string descript = (descript_c ? descript_c : "");

        // match by substring
        ComponentConfig cfg = matchComponentConfig(descript, configs);

        // create a new data object (for aluminum first)
        T E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
        // debug..
        // if (icomp < 20)
        //     printf(
        //         "icomp %d, name %s : length %.2e, pthick %.2e, sheight %.2e, refAxis %.2e, %.2e,
        //         "
        //         "%.2e\n",
        //         icomp, descript.c_str(), cfg.panel_length, cfg.pthick, cfg.sheight,
        //         cfg.ref_axis[0], cfg.ref_axis[1], cfg.ref_axis[2]);
        comp_data[icomp] = Data(E, nu, cfg.pthick, cfg.sheight, cfg.sthick, cfg.spitch,
                                &cfg.ref_axis[0], rho, ys, 0.0, cfg.panel_length);
    }
}