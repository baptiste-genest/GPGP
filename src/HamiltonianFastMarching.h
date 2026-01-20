#ifndef HAMILTONIANFASTMARCHING_H
#define HAMILTONIANFASTMARCHING_H

#include "StochasticCalculus.h"
#include <queue>
#include "Grid.h"

namespace SGP {

class HamiltonianFastMarching : public StochasticCalculus
{
private:

    struct arrow {
        int target;
        scalar alpha;
        SliceIndex offset;
    };
    using arrows = std::vector<arrow>;

    std::unordered_map<int,arrows> reversedStencils;

    struct Trial {
        scalar Up;
        int id;
        bool operator<(const Trial& o) const {return Up > o.Up;}
    };

    std::priority_queue<Trial,std::vector<Trial>> pq;

    Vec U;

    enum state {
        Accepted,
        Trial,
        Far
    };

    std::vector<state> states;

    scalar getLargestRoot(const vec& P) const;

    void preProcess(int p);
    void postProcess(int p);
public:
    using StochasticCalculus::StochasticCalculus;

    void precomputeStencils();

    VectorGrid computeGeodesicVelocityField() const;

    std::vector<vec> integrateGeodesic(const vec& target,scalar dt,const ScalarGrid3& U, const VectorGrid3 &V) const;

    Vec run(SliceIndex source);
    Vec run(int node_idx);

};

}

#endif // HAMILTONIANFASTMARCHING_H
