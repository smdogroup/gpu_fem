#include "mesh/TACSMeshLoader.h"

int main() {
    TACSMeshLoader<double> loader{};
    loader.scanBDFFile("CRM_box_2nd.bdf");

    int num_nodes = loader.getNumNodes();
    printf("num_nodes = %d\n", num_nodes);
};