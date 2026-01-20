#include "HamiltonianFastMarching.h"

SGP::scalar SGP::HamiltonianFastMarching::getLargestRoot(const vec &P) const
{
    // solve quadratic equation
    // P = P.x*t^2 + P.y*t + P.z
    scalar a = P(0), b = P(1), c = P(2);
    scalar disc = b*b - 4*a*c;
    if (disc < 0 || a < 1e-10){
        // std::cerr << "invalid polynomial " << P.transpose() << std::endl;
        return 1000;
        // throw std::runtime_error("No real root found in HamiltonianFastMarching");
    }
    scalar r1 = (-b + std::sqrt(disc)) / (2*a);
    scalar r2 = (-b - std::sqrt(disc)) / (2*a);
    return std::max(r1,r2);
}

void SGP::HamiltonianFastMarching::preProcess(int p)
{

}

void SGP::HamiltonianFastMarching::postProcess(int p)
{

}

void SGP::HamiltonianFastMarching::precomputeStencils()
{
    computeVoronoiStencil();

    auto m = getNumVariables();
// #pragma omp parallel for
    for (const auto& [h,node] : nodes) {
        int id = node.id;
        const auto& decomp = node.stencil;
        SliceIndex I = getGridCoord(h);
        for (auto s : signs()) {
            for (const auto & [h,c] : decomp.elements) {
                SliceIndex y = I + s*h;
                auto idy = getRestrictionIndex(y);
                if (!idy)
                    continue;
                if (visibilityCheck(I,y))
                    reversedStencils[*idy].push_back({id, c, s*h});
            }
        }
    }
}


SGP::VectorGrid SGP::HamiltonianFastMarching::computeGeodesicVelocityField() const
{
    if (states.empty())
        throw std::runtime_error("You must call run() before computing the geodesic velocity field");
    VectorGrid V = VectorGrid(bbox);
    // V.fill(Vector<dim>::Zero());

    scalar h = 1.0 / getDx();

    for (const auto& [h,node] : nodes) {
        Vector<dim> v;v.setZero();
        SliceIndex I = getGridCoord(h);
        auto id = node.id;
        for (const auto& [e,a] : node.stencil.elements) {
            for (auto s : signs()) {
                SliceIndex J = I - e*s;
                auto jid = getRestrictionIndex(J);
                if (!jid)
                    continue;
                if (!visibilityCheck(I,J))
                    continue;
                scalar dU = (U(id) - U(*jid));
                if (dU > 0)
                    v += a*dU*e.cast<scalar>()/h*s;
            }
        }
        V(I) = v;
    }

    return V;
}

SGP::Vec SGP::HamiltonianFastMarching::run(SliceIndex s)
{
    auto hs = getHash(s);

    int source;

    if (hs)
        try {
            source = nodes.at(*hs).id;
        } catch (...) {
            throw std::runtime_error("Source out of bounds");
        }
    else
        throw std::runtime_error("Source out of bounds");

    return run(source);
}

SGP::Vec SGP::HamiltonianFastMarching::run(int source)
{
    if (reversedStencils.empty())
        throw std::runtime_error("You must call precomputeStencils() before running the fast marching");

    int m = getNumVariables();


    U = Vec::Constant(m,1e10);
    states = std::vector<state>(m, Far);

    U(source) = 0;
    states[source] = Trial;
    pq.push({0, source});

    std::vector<vec> polynomials(m,vec(0,0,-1));

    while (!pq.empty()) {
        auto [u,q] = pq.top();
        pq.pop();
        if (states[q] == Accepted)
            continue;
        states[q] = Accepted;
        // preProcess(q);

        vec dP = vec(1,-2*u,u*u);

        for (const auto& p : reversedStencils[q]) {
            if (states[p.target] == Accepted)
                continue;
            if (states[p.target] == Far) {
                states[p.target] = Trial;
                preProcess(p.target);
            }
            polynomials[p.target] += dP*p.alpha;
            scalar newU = getLargestRoot(polynomials[p.target]);
            if (newU < U(p.target)) {
                U(p.target) = newU;
                pq.push({newU, p.target});
            }
            // else {
            //     spdlog::warn("value larger than before {} {}",U(p.target),newU);
            // }
        }
    }

    int not_reached = 0;
    for (auto i : range(m))
        if (states[i] != Accepted) {
            U(i) = -1;
            not_reached++;
        }
    if (not_reached)
        spdlog::warn("{} variables were not reached by the fast marching", not_reached);

    return U;

}
