#ifndef PLOT_H
#define PLOT_H

#include "StochasticCalculus.h"
#include "polyscope/volume_grid.h"
#include "polyscope/sparse_volume_grid.h"

namespace SGP {

inline polyscope::SparseVolumeGrid* PlotNarrowBand(const StochasticCalculus& calc,std::string label = "narrow band") {
#ifdef SGP2D
    return nullptr;
#else
    const auto& nodes = calc.getNodes();
    scalar h = calc.getDx();

    std::vector<glm::ivec3> cells;
    std::vector<scalar> mu,field;
    cells.reserve(nodes.size());
    mu.reserve(nodes.size());
    field.reserve(nodes.size());

    for (const auto& [hash,node] : nodes) {
        SliceIndex I = calc.getGridCoord(hash);
        cells.emplace_back(I(0),I(1),I(2));
        mu.push_back(node.mu);
        field.push_back(node.field);
    }

    Vector<dim> o = calc.getEmbedder() * Vector<dim>::Zero();
    glm::vec3 origin(o(0) - h/2,o(1) - h/2,o(2) - h/2);
    glm::vec3 width(h,h,h);

    auto* grid = polyscope::registerSparseVolumeGrid(label,origin,width,cells);
    grid->addCellScalarQuantity("mu",mu)->setEnabled(true);
    grid->addCellScalarQuantity("GPIS mean",field);
    return grid;
#endif
}

inline polyscope::VolumeGrid* RegisterBBoxGrid(const SparseNarrowBand& narrow,std::string label = "grid") {
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

}

#endif // PLOT_H
