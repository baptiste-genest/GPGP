#include "MeshSampling.h"
// #include <polyscope/polyscope.h>

// polyscope::PointCloud *SGP::display(std::string label, const Mesh &M, const SurfacePoints &X)
// {
//     return polyscope::registerPointCloud(label,toPositions(M,X).transpose());
// }


SGP::SurfacePoints SGP::sampleMesh(const SGP::Mesh &M, int sampleNum, const scalars &face_weights) {
    SurfacePoints sampleList(sampleNum);

    const auto& pos = M.geometry->vertexPositions;

    int triNum = M.topology->nFaces();
    if (triNum != face_weights.size()){
        spdlog::error("invalid face weights");
        return sampleList;
    }

    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::discrete_distribution<> d(face_weights.begin(), face_weights.end());

    auto randomUnit = [] () -> scalar {
        std::uniform_real_distribution<scalar> gen(0,1);
        static thread_local std::random_device rd;
        return gen(rd);
    };

    for (int j = 0; j < sampleNum; j++) {
        // double sample;
        size_t faceidx = d(gen);

        //random point generation within previously selected face area
        vecs V;
        for (auto v : M.topology->face(faceidx).adjacentVertices()){
            V.push_back(toVec(pos[v]));
        }

        scalar alpha, beta;
        alpha = (1 - 0) * randomUnit() + 0;
        beta = (1 - 0) *  randomUnit() + 0;

        scalar a, b, c;
        a = 1 - std::sqrt(beta);
        b = (std::sqrt(beta)) * (1 - alpha);
        c = std::sqrt(beta) * alpha;

        //resulting sample
        // vec P = a*V[0] + b*V[1] + c*V[2];
        //PointOnMesh pom;
        //sampleList[j] = {P,faceidx,vec(a,b,c)};
        sampleList[j] = SurfacePoint(M.topology->face(faceidx),Vector3(a,b,c));
    }

    return sampleList;
}
