#include "StochasticCalculus.h"

int SGP::StochasticCalculus::NQMC = 512;


SGP::StochasticCalculus::StochasticCalculus(const GaussianGrid &random_field,
                                            scalar iso, scalar eps, bool padding) : iso(iso)
{
    bbox = random_field.getSizes();

    dx = 2./(bbox.maxCoeff() - 1);

    nodes.clear();

    const int N = (int)random_field.getSize();
    int max_threads = omp_get_max_threads();

    std::vector<std::vector<std::pair<decltype(random_field.getSliceIndexes(0)), ComputeNode>>> local_results;
    local_results.resize(max_threads);

// #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();

        auto field_value = random_field.at(i);

        scalar mu = computeSurfaceProb(field_value, iso);
        if (mu > eps) {
            ComputeNode node;
            node.mu = mu;
            node.G = computeDiffusionTensor(field_value, iso);
            node.id = -1;
            node.pred = field_value;

            auto I = random_field.getSliceIndexes(i);

            local_results[tid].emplace_back(I, std::move(node));
        }
    }

    size_t total = 0;
    for (auto &v : local_results) total += v.size();
    nodes.reserve(total ? (size_t)(total * 1.3) : 0);

    for (auto &v : local_results) {
        for (auto &p : v) {
            auto &I = p.first;
            auto &node = p.second;
            node.id = (int)nodes.size();
            node.mu += min_mu;
            node.G += SquareMatrix<dim>::Identity()*min_mu;
            auto h = getHash(I);
            if (!h)
                throw std::runtime_error("Node out of bounds");
            nodes[*h] = std::move(node);
        }
    }

    auto S = random_field.getSizes();
    // embedder = [S] (const GridElement& I) -> Vector<dim> {
    //     // map grid index to [-1,1]^dim
    //     Vector<dim> x;
    //     for (int i = 0; i < dim; i++)
    //         x(i) = -1. + 2.*I(i)/scalar(S(i)-1);
    //     return x;
    // };
    scalar s = S.maxCoeff();
    embedder = Translation(-Vector<dim>::Constant(1))*Scaling(2./(s-1));

}

SGP::StochasticCalculus::StochasticCalculus(const SparseNarrowBand &input): bbox(input.bbox),embedder(input.embedder),iso(input.iso)
{
    dx = input.dx;
    nodes.reserve(input.elements.size());
    for (auto [I,node] : input.elements) {
        auto id = getHash(I);
        if (!id)
            throw std::runtime_error("Input node out of bounds");
        node.mu += min_mu;
        node.G += SquareMatrix<dim>::Identity()*min_mu;
        nodes[*id] = node;
    }
}

SGP::smat SGP::StochasticCalculus::buildMassMatrix()
{
    return buildDiagMu(false)*std::pow(getDx(),dim);
}

SGP::smat SGP::StochasticCalculus::buildDiagInvMu(bool vec)
{
    auto key = vec ? "inv_area_vec" : "inv_area";
    if (operatorCache.contains(key))
        return operatorCache[key];
    int m = nodes.size();
    smat M = buildDiagMu(vec);
    for (auto k=0; k<M.outerSize(); ++k)
        for (smat::InnerIterator it(M,k); it; ++it)
            it.valueRef() = 1./it.value();
    operatorCache[key] = M;
    return M;
}


SGP::scalar SGP::StochasticCalculus::max_mu = 20;
SGP::scalar SGP::StochasticCalculus::min_mu = 0;

SGP::scalar SGP::StochasticCalculus::computeSurfaceProb(const ValueGradientGaussian &x,scalar iso) {
    scalar rho = Gaussian::pdf(x.getMean()(0),x.getCov()(0,0),iso);
    auto G = x.conditionOnFirstCoord(iso);
    scalar mean_grad_norm = G.evalQMC(NQMC,[] (const Vector<dim>& n) ->scalar {
        return n.norm();
    });
    return std::min(rho*mean_grad_norm,max_mu);
}

SGP::SquareMatrix<SGP::dim> SGP::StochasticCalculus::computeDiffusionTensor(const ValueGradientGaussian &p,scalar iso) {
    auto G = p.conditionOnFirstCoord(iso);
    scalar rho = Gaussian::pdf(p.getMean()(0),p.getCov()(0,0),iso);
    SquareMatrix<dim> rslt = G.evalQMC<SquareMatrix<dim>>(NQMC,[](const Vector<dim>& x) {
        Vector<dim> n = x.stableNormalized();
        SquareMatrix<dim> T = SquareMatrix<dim>::Identity() - n*n.transpose();
        return SquareMatrix<dim>(T*x.norm());
    });
    rslt *= rho;
    scalar norm = rslt.norm();
    if (norm > max_mu)
        rslt = rslt/norm*max_mu;
    return rslt;
}


SGP::smat SGP::StochasticCalculus::buildTangentTensorMatrix() {
    if (operatorCache.contains("T"))
        return operatorCache["T"];
    int m = nodes.size();
    Triplets triplets;
    triplets.reserve(m*9);
    for (const auto& [I,node] : nodes) {
        int id = node.id;
        SquareMatrix<dim> pi = node.G;
        for (auto i : range(dim))
            for (auto j : range(dim))
                triplets.push_back({dim*id+i, dim*id+j,pi(i,j)});
    }
    smat rslt(dim*m, dim*m);
    rslt.setFromTriplets(triplets.begin(),triplets.end());
    operatorCache["T"] = rslt;
    return rslt;
}

SGP::SliceIndex SGP::StochasticCalculus::delta(int i) {
    SliceIndex D = SliceIndex::Zero();
    D(i) = 1;
    return D;
}

SGP::smat SGP::StochasticCalculus::buildFiniteDifferenceGradient(bool forward,BoundaryCondition bc) {
    std::string key = forward ? "FDG_forward_"+std::to_string(bc) : "FDG_backward_"+std::to_string(bc);
    if (operatorCache.contains(key))
        return operatorCache[key];
    int m = nodes.size();
    Triplets triplets;
    triplets.reserve(m*7);

    int s = forward ? 1 : -1;

    for (const auto& [I,node] : nodes) {
        int idx = node.id;
        for (int l = 0; l < dim; l++) {
            SliceIndex D_index = getGridCoord(I) + s*delta(l);
            auto idy = getRestrictionIndex(D_index);
            if (idy){
                triplets.push_back(Triplet(dim*idx+l, *idy, s));
                if (bc == Neumann)
                    triplets.push_back(Triplet(dim*idx+l,idx, -s));
            }
            if (bc == Dirichlet)
                triplets.push_back(Triplet(dim*idx+l,idx, -s));
        }
    }

    smat rslt(dim*m, m);
    rslt.setFromTriplets(triplets.begin(), triplets.end());
    operatorCache[key] = rslt;

    return rslt;
}

SGP::smat SGP::StochasticCalculus::buildDiagMu(bool vec) {
    if (vec && operatorCache.contains("area_vec"))
        return operatorCache["area_vec"];
    if (!vec && operatorCache.contains("area"))
        return operatorCache["area"];
    int m = nodes.size();
    Triplets triplets;
    if (vec)
        triplets.reserve(m*dim);
    else
        triplets.reserve(m);
    for (const auto& [I,node] : nodes) {
        int id = node.id;
        scalar rg = node.mu;
        if (vec)
            for (auto i : range(dim))
                triplets.push_back({dim*id + i, dim*id+i, rg});
        else
            triplets.push_back({id, id, rg});
    }
    if (vec){
        smat rslt(dim*m, dim*m);
        rslt.setFromTriplets(triplets.begin(),triplets.end());
        operatorCache["area_vec"] = rslt;
        return rslt;
    } else {
        smat rslt(m, m);
        rslt.setFromTriplets(triplets.begin(),triplets.end());
        operatorCache["area"] = rslt;
        return rslt;
    }
}




SGP::smat SGP::StochasticCalculus::buildGradient(BoundaryCondition bc) {
    smat GF = buildFiniteDifferenceGradient(true,bc);
    smat T = buildTangentTensorMatrix();
    smat imu = buildDiagInvMu(true);
    smat pi = imu*T;
    smat G = pi*GF;
    G /= getDx();
    G.makeCompressed();

    operatorCache["grid"] = G;
    return G;
}

SGP::smat SGP::StochasticCalculus::buildIntegratedDivergence(BoundaryCondition bc){
    if (operatorCache.contains("div"))
        return operatorCache["div"];
    smat div = buildFiniteDifferenceGradient(true,bc).transpose();
    smat dA = buildDiagMu(true);
    smat ID = div*dA;
    ID /= getDx();
    ID.makeCompressed();

    operatorCache["div"] = ID;
    return ID;
}


SGP::smat SGP::StochasticCalculus::buildNaiveLaplacian(scalar offset,BoundaryCondition bc) {

    if (operatorCache.contains("naivelap"))
        return operatorCache["naivelap"];

    smat mass = buildDiagMu(false);

    smat grad = buildGradient(bc);
    smat div = buildIntegratedDivergence(bc);
    smat L = div*grad;

    smat I = smat(getNumVariables(),getNumVariables());
    I.setIdentity();
    scalar eps = offset*getTikhonovParameter(L);
    L += I*eps;
    L.makeCompressed();

    operatorCache["naivelap"] = L;

    return L;
}

SGP::ScalarGrid SGP::StochasticCalculus::extend(const Vec &x, scalar fill) {
    ScalarGrid rslt(bbox);// = Vec::Constant(bboxSize(),fill);
    rslt.fill(fill);
    for (const auto& [I,node] : nodes) {
        rslt(getGridCoord(I)) = x(node.id);
    }
    return rslt;
}

SGP::VectorGrid SGP::StochasticCalculus::extendVec(const Vec &x) {
    VectorGrid rslt(bbox);
    for (const auto& [I,node] : nodes) {
        int id = node.id;
        for (auto i : range(dim))
            rslt(getGridCoord(I))(i) = x(dim*id + i);
    }
    return rslt;
}

SGP::Vec SGP::StochasticCalculus::restrictVec(const VectorGrid &x) {
    Vec rslt = Vec::Zero(dim*nodes.size());
    for (const auto& [I,node] : nodes) {
        for (auto i : range(dim)) {
            rslt(dim*node.id + i) = x(getGridCoord(I))(i);
        }
    }
    return rslt;
}

SGP::Vec SGP::StochasticCalculus::restrictMat(const TensorGrid &x)
{
    Vec rslt = Vec::Zero(dim*dim*nodes.size());
    for (const auto& [I,node] : nodes) {
        for (auto i : range(dim))
            for (auto j : range(dim))
                rslt(dim*dim*node.id + dim*i + j) = x(getGridCoord(I))(i,j);
    }
    return rslt;

}

SGP::Vec SGP::StochasticCalculus::restrict(const ScalarGrid &x) {
    Vec rslt = Vec::Zero(nodes.size());
    for (const auto& [I,node] : nodes) {
        rslt(node.id) = x(getGridCoord(I));
    }
    return rslt;
}

void SGP::StochasticCalculus::computeVoronoiStencil()
{
    scalar max_norm = 0;
    SGP::GridElement max_element = SGP::GridElement::Zero();
    for (auto& [I,node] : nodes) {
        node.stencil = ComputeVoronoiReduction<dim>(node.G);
        for (const auto& e : node.stencil.elements) {
            if (e.first.norm() > max_norm) {
                max_norm = e.first.lpNorm<1>();
                max_element = e.first;
            }
        }
    }
}

SGP::Vec SGP::StochasticCalculus::getGPISMean() const
{
    Vec rslt = Vec::Zero(nodes.size());
    for (const auto& [I,node] : nodes) {
        rslt(node.id) = node.pred.mean(0);
    }
    return rslt;
}

SGP::ScalarGrid SGP::StochasticCalculus::extendFill(const Vec &x)
{
    // where nodes exist, fill with x.
    // for other nodes, use a BFS to fill in values from neighbors, where the value
    // at an unknown node is the value at the closest node

    ScalarGrid rslt(bbox);
    rslt.fill(0);
    std::queue<SGP::GridElement> to_process;
    GridSet processed;
    for (const auto& [I,node] : nodes) {
        GridElement G = getGridCoord(I);
        rslt(G) = x(node.id);
        to_process.push(G);
        processed.insert(G);
    }
    while (!to_process.empty()) {
        GridElement current = to_process.front();
        to_process.pop();
        for (auto d : range(dim)) {
            for (auto s : signs()) {
                GridElement neighbor = current;
                neighbor(d) += s;
                if (checkInBbox(neighbor) && !processed.contains(neighbor)) {
                    rslt(neighbor) = rslt(current);
                    to_process.push(neighbor);
                    processed.insert(neighbor);
                }
            }
        }
    }
    return rslt;
}


SGP::ScalarGrid SGP::StochasticCalculus::getValueField(scalar fill) const
{
    ScalarGrid P(bbox);
    P.fill(fill);
    for (const auto& [I,node] : nodes) {
        P(getGridCoord(I)) = node.field;
    }
    return P;
}

SGP::Points<SGP::dim> SGP::StochasticCalculus::embedNarrowBand() const
{
    Points<dim> rslt(dim,getNumVariables());
    for (const auto& [I,node] : nodes) {
        rslt.col(node.id) = embedder * getGridCoord(I).cast<scalar>();
    }
    return rslt;
}

SGP::ScalarGrid SGP::StochasticCalculus::getProbField() const
{
    ScalarGrid P(bbox);
    P.fill(0);
    for (const auto& [I,node] : nodes) {
        P(getGridCoord(I)) = node.mu;
    }
    return P;
}

SGP::smat SGP::StochasticCalculus::buildVoronoiLaplace(scalar offset,BoundaryCondition bc)
{
    if (operatorCache.contains("voronoilap_"+std::to_string(bc)))
        return operatorCache["voronoilap_"+std::to_string(bc)];
    if (!nodes.begin()->second.stencil.elements.size())
        computeVoronoiStencil();
    Triplets triplets;
    int m = nodes.size();

    triplets.reserve(m*12*2);
    Vec diag = Vec::Zero(m);

    int total = 0,out = 0;
    for (const auto& [I,node] : nodes) {
        GridElement x = getGridCoord(I);

        const auto& decomp = node.stencil;

        for (auto s : signs()) {
            for (const auto & [h,c] : decomp.elements) {
                GridElement y = x + s*h;
                int idx = node.id;
                if (bc == Dirichlet)
                    diag(idx) += c;
                total++;
                auto idy = getRestrictionIndex(y);
                if (!idy) {
                    out++;
                    // y not in domain
                    continue;
                }

                if (!visibilityCheck(x,y)){
                    out++;
                    continue;
                }

                if (bc == Neumann)
                    diag(idx) += c;

                triplets.push_back(Triplet{idx,int(*idy),-c});
                triplets.push_back(Triplet{int(*idy),idx,-c});
                diag(*idy) += c;
            }
        }
    }
    for (auto i : range(m))
        triplets.push_back({i,i,diag(i)});

    // spdlog::info(" out of {} total elements, {}% were not in the nodes. ",total,out/scalar(total)*100);


    smat L(m,m);
    L.setFromTriplets(triplets.begin(),triplets.end());
    L/= 2.;

    scalar lmb = getTikhonovParameter(L)*offset;

    smat I(m,m);I.setIdentity();
    L += lmb*I;
    L.makeCompressed();
    operatorCache["voronoilap_"+std::to_string(bc)] = L;
    return L;
}

bool SGP::StochasticCalculus::visibilityCheck(const SliceIndex &Ip, const SliceIndex &Iq) const
{
    // check that the integer segment between p and q does not leave the domain of variables
    SliceIndex D = Iq - Ip;

    auto pred = [&] (SliceIndex xi) -> bool {
        return getRestrictionIndex(xi).has_value();
    };

    SliceIndex abs_D = D.cwiseAbs();
    SliceIndex step;
    for (int i = 0; i < dim; ++i) step[i] = (D[i] > 0 ? 1 : (D[i] < 0 ? -1 : 0));

    // number of steps = maximum absolute D
    const int L = abs_D.maxCoeff();
    // degenerate: same point
    if (L == 0) {
        return pred(Ip);
    }

    SliceIndex pos = Ip;
    SliceIndex acc = SliceIndex::Zero(); // accumulators

    // loop includes both endpoints: iterate L+1 times (k = 0..L)
    for (int k = 0; k <= L; ++k) {
        if (!pred(pos)) return false;      // if any visited point is blocked -> no LOS
        if (k == L) break;                 // done after checking endpoint

        // update accumulators and step coordinates when accumulator crosses threshold
        for (int i = 0; i < dim; ++i) {
            acc[i] += abs_D[i];
            // when accumulator *2 >= L, step in that axis
            // Using >= guarantees symmetry and matches common generalized Bresenham behaviour.
            if (2 * acc[i] >= L) {
                pos[i] += step[i];
                acc[i] -= L;
            }
        }
    }

    return true; // all visited points passed predicate

    /*
    int steps = 0;
    for (auto d : range(D.size()))
        steps = std::max(steps, std::abs(D(d)));
    if (steps == 0)
        return true;
    for (int s = 1; s < steps; s++) {
        SliceIndex I = Ip;
        for (auto d : range(D.size())) {
            I(d) += D(d)*s/steps;
        }
        if (!P.validIndex(I))
            return false;
        if (!idmap.contains(P.getIndex(I)))
            return false;
    }
    return true;
*/
}

SGP::scalar SGP::StochasticCalculus::lerp(const Vector<dim> &x, const Vec &f) const
{
    auto getValue = [&] (const SliceIndex &I) -> scalar {
        auto id = getRestrictionIndex(I);
        if (id)
            return f(*id);
        else
            return 0;
    };
    AffineMap inv = embedder.inverse();
#ifdef SGP2D
    vec2 xi = inv * x;
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar c00 = getValue(SliceIndex(i0,j0)) * (1 - xd) + getValue(SliceIndex(i0+1,j0)) * xd;
    scalar c01 = getValue(SliceIndex(i0,j0+1)) * (1 - xd) + getValue(SliceIndex(i0+1,j0+1)) * xd;
    return c00 * (1 - yd) + c01 * yd;
#else
    vec xi = inv * x;
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;


    scalar c000 = getValue(SliceIndex(i0,j0,k0));
    scalar c001 = getValue(SliceIndex(i0,j0,k0+1));
    scalar c010 = getValue(SliceIndex(i0,j0+1,k0));
    scalar c011 = getValue(SliceIndex(i0,j0+1,k0+1));
    scalar c100 = getValue(SliceIndex(i0+1,j0,k0));
    scalar c101 = getValue(SliceIndex(i0+1,j0,k0+1));
    scalar c110 = getValue(SliceIndex(i0+1,j0+1,k0));
    scalar c111 = getValue(SliceIndex(i0+1,j0+1,k0+1));
    scalar c00 = c000 * (1 - xd) + c100 * xd;
    scalar c01 = c001 * (1 - xd) + c101 * xd;
    scalar c10 = c010 * (1 - xd) + c110 * xd;
    scalar c11 = c011 * (1 - xd) + c111 * xd;
    scalar c0 = c00 * (1 - yd) + c10 * yd;
    scalar c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
#endif
}

SGP::Vector<SGP::dim> SGP::StochasticCalculus::lerpVec(const Vector<dim> &x, const Vec &f) const
{
    auto getValue = [&] (const SliceIndex &I) -> Vector<dim> {
        auto id = getRestrictionIndex(I);
        Vector<dim> rslt = Vector<dim>::Zero();
        if (id) {
            int vid = *id;
            for (auto i : range(dim))
                rslt(i) = f(dim*vid + i);
        }
        return rslt;
    };
    AffineMap inv = embedder.inverse();
#ifdef SGP2D
    vec2 xi = inv * x;
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    Vector<dim> c00 = getValue(SliceIndex(i0,j0)) * (1 - xd) + getValue(SliceIndex(i0+1,j0)) * xd;
    Vector<dim> c01 = getValue(SliceIndex(i0,j0+1)) * (1 - xd) + getValue(SliceIndex(i0+1,j0+1)) * xd;
    return c00 * (1 - yd) + c01 * yd;
#else
    vec xi = inv * x;
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;


    Vector<dim> c000 = getValue(SliceIndex(i0,j0,k0));
    Vector<dim> c001 = getValue(SliceIndex(i0,j0,k0+1));
    Vector<dim> c010 = getValue(SliceIndex(i0,j0+1,k0));
    Vector<dim> c011 = getValue(SliceIndex(i0,j0+1,k0+1));
    Vector<dim> c100 = getValue(SliceIndex(i0+1,j0,k0));
    Vector<dim> c101 = getValue(SliceIndex(i0+1,j0,k0+1));
    Vector<dim> c110 = getValue(SliceIndex(i0+1,j0+1,k0));
    Vector<dim> c111 = getValue(SliceIndex(i0+1,j0+1,k0+1));
    Vector<dim> c00 = c000 * (1 - xd) + c100 * xd;
    Vector<dim> c01 = c001 * (1 - xd) + c101 * xd;
    Vector<dim> c10 = c010 * (1 - xd) + c110 * xd;
    Vector<dim> c11 = c011 * (1 - xd) + c111 * xd;
    Vector<dim> c0 = c00 * (1 - yd) + c10 * yd;
    Vector<dim> c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
#endif
}
