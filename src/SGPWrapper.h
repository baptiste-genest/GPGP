#ifndef SGPWRAPPER_H
#define SGPWRAPPER_H

#include "StochasticCalculus.h"
#include "BarnesHuttSPSR.h"
#include "StochasticBarnesHutt.h"

namespace SGP {

std::pair<Tensor<dim,pred>,scalar> ComputeRandomFieldPSR(const PointGrid& G,scalar reg,scalar beta,const GaussianDipoles<dim>& GD);

}

#endif // SGPWRAPPER_H
