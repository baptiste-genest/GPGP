#ifndef PLOT_H
#define PLOT_H

#include "StochasticCalculus.h"
#include "polyscope/volume_grid.h"

namespace SGP {

polyscope::VolumeGrid* PlotNarrowBand(const SparseNarrowBand& narrow,std::string label = "grid") {
#ifdef SGP2D
    return nullptr;
#else
    auto bbsize = narrow.bbox;
    glm::uvec3 res(bbsize[0],bbsize[1],bbsize[2]);
    glm::vec3 low(0);
    glm::vec3 high(0);

    vec xlow = narrow.embedder * vec::Zero();
    vec xhigh = narrow.embedder*(narrow.bbox - GridElement::Constant(1)).cast<scalar>();
    for (auto d : range(dim)) {
        low[d] = xlow[d];
        high[d] = xhigh[d];
    }

    return polyscope::registerVolumeGrid(label,res,low,high);
#endif
}

polyscope::SurfaceMesh* ExtractIsoSurface(polyscope::VolumeGrid* pcgrid,const ScalarGrid& phi,scalar isoval) {
    auto pcg = pcgrid->addNodeScalarQuantity("phi",phi.data());
    pcg->setIsosurfaceVizEnabled(true);
    pcg->setIsosurfaceLevel(isoval);
    auto iso = pcg->registerIsosurfaceAsMesh("iso");
    iso->setEnabled(true);
    return iso;
}



}


#endif // PLOT_H
