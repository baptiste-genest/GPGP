#include "SGPWrapper.h"

std::pair<SGP::Tensor<SGP::dim, SGP::pred>, SGP::scalar> SGP::ComputeRandomFieldPSR(const PointGrid &G, scalar reg, scalar beta, const GaussianDipoles<dim> &GD)
{
    auto time = Time::now();
    StochasticBarnesHutt BHSPSR(reg,beta,GD.size());

    for (auto i : range(GD.size()))
        BHSPSR.insert(GD[i]);

    BHSPSR.precomputeMoments();
    BHSPSR.computeRadius();
    scalar iso = GetAverageIso(GD,BHSPSR);
    auto SPSR_field = Apply<Vector<dim>,pred>(G, [&](const Vector<dim>& x) {
        return BHSPSR.predict(x);
    });
    return {SPSR_field,iso};
}
