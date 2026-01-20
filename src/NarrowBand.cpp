#include "NarrowBand.h"


SGP::SparseNarrowBand SGP::BuildAdaptiveNarrowBand(const PosList<dim> &GD, const GPIS &BH, scalar h, scalar eps) {

    scalar iso = GetAverageIso(GD,BH);

    Vector<dim> x0 = GD.getPos(0) + Vector<dim>::Constant(h/2);

    auto ToSpace = [h,x0] (const GridElement& x) -> Vector<dim> {
        return x0 + x.cast<scalar>() * h;
    };
    auto ToGrid = [h,x0] (const Vector<dim>& x) -> GridElement {
        return ((x - x0)/h).array().floor().cast<int>();
    };

    std::queue<GridElement> todo;
    for (auto i : range(GD.size())) {
        GridElement x = ToGrid(GD.getPos(i));
        todo.push(x);
    }

    GridSet visited;
    std::mutex visited_mtx;

    GridMap<StochasticCalculus::ComputeNode> accepted;
    std::mutex accepted_mtx;

    GridElement bbmin = GridElement::Zero(), bbmax = GridElement::Zero();
    std::mutex bb_mtx;

    std::mutex queue_mtx;


#pragma omp parallel
    {
        while (true) {
            GridElement x;
            {
                std::lock_guard<std::mutex> qlock(queue_mtx);
                if (todo.empty()) break;
                x = todo.front();
                todo.pop();
            }

            {
                std::lock_guard<std::mutex> vlock(visited_mtx);
                if (visited.find(x) != visited.end()) {
                    continue;
                }
                visited.insert(x);
            }

            auto p = ToSpace(x);
            auto bhp = BH.predict(p);
            auto mu = StochasticCalculus::computeSurfaceProb(bhp, iso);

            if (mu > eps) {
                {
                    std::lock_guard<std::mutex> bclock(bb_mtx);
                    for (auto d : range(dim)) {
                        bbmin(d) = std::min(bbmin(d), x(d));
                        bbmax(d) = std::max(bbmax(d), x(d));
                    }
                }

                auto G = StochasticCalculus::computeDiffusionTensor(bhp, iso);
                auto red = ComputeVoronoiReduction(G);

                {
                    std::lock_guard<std::mutex> alock(accepted_mtx);
                    StochasticCalculus::ComputeNode node;
                    node.G = G;
                    node.mu = mu;
                    node.stencil = red;
                    node.id = accepted.size();
                    node.field = bhp.mean(0);
                    node.pred = bhp;
                    accepted[x] = node;
                }

                {
                    std::lock_guard<std::mutex> qlock2(queue_mtx);
                    for (const auto& [e,_] : red.elements) {
                        for (auto s : signs()) {
                            GridElement y = x + s*e;
                            todo.push(y);
                        }
                    }
                }
            }
        }
    }

    // GridElement bbox = GridElement::Constant((bbmax - bbmin).maxCoeff()  + 1);
    GridElement bbox = bbmax - bbmin + GridElement::Constant(1);
    GridMap<StochasticCalculus::ComputeNode> rslt;
    rslt.reserve(accepted.size());
    for (const auto& [I,node] : accepted) {
        GridElement J = I - bbmin;
        rslt[J] = node;
    }

    GridEmbedder embedder;
    embedder = Translation(x0)*Scaling(h)*Translation(bbmin.cast<scalar>());
    //                         = [h,x0,bbmin](const GridElement& I) -> Vector<dim> {
    //     return Vector<dim>(x0 + (I + bbmin).cast<scalar>() * h);
    // };

    return {rslt, bbox, h, iso, embedder};
}

SGP::scalar SGP::StencilReachHeuristic(const PosList<dim> &GD, const GPIS &BH,scalar feature_size)
{
    scalar iso = GetAverageIso(GD,BH);
    int max_res = 0;
#pragma omp parallel
    {
        int local_max = 0;
#pragma omp for nowait
        for (int i = 0; i < (int)GD.size(); ++i) {
            auto G = StochasticCalculus::computeDiffusionTensor(BH.predict(GD.getPos(i)), iso);
            auto red = ComputeVoronoiReduction(G);
            for (const auto& [e, _] : red.elements) {
                local_max = std::max(local_max, e.lpNorm<1>());
                // local_max = std::max(local_max, e.lpNorm<Eigen::Infinity>());
            }
        }
#pragma omp critical
        max_res = std::max(max_res, local_max);
    }
    return feature_size/std::max(max_res,2);
}
