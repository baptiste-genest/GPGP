#ifndef NARROWBAND_H
#define NARROWBAND_H

#include "BarnesHuttSPSR.h"
#include "StochasticCalculus.h"

namespace SGP {


scalar StencilReachHeuristic(const PosList<dim>& GD,
                           const GPIS& BH,
                           scalar feature_size);

SparseNarrowBand BuildAdaptiveNarrowBand(const PosList<dim>& GD,
                                             const GPIS& BH,
                                             scalar h, scalar eps = 1e-3);



}
#endif // NARROWBAND_H
