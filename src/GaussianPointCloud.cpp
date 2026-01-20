#include "GaussianPointCloud.h"


SGP::GaussianDipoles<3> SGP::GaussianDipolesFromTriangleSoup(const Mesh &M, scalar sigma)
{
    int n = M.topology->nFaces();
    GaussianDipoles<3> S;
    S.dipoles.resize(n);
    M.geometry->requireFaceAreas();
    M.geometry->requireFaceNormals();
    for (auto f : M.topology->faces() ){
        S.dipoles[f.getIndex()].p = M.faceBarycenter(f);
        S.dipoles[f.getIndex()].n = toVec(M.geometry->faceNormal(f)*M.geometry->faceArea(f));
        // S.dipoles[f.getIndex()].CovMoment = mat::Zero();
        // S.dipoles[f.getIndex()].CovPos = mat::Identity()*sigma*sigma/9;
    }

    constexpr auto crossMat = [](const vec& x) -> mat {
        mat rslt;
        rslt << 0, -x(2), x(1),
            x(2), 0, -x(0),
            -x(1), x(0), 0;
        return rslt;
    };

    using noise_triplet = Matrix<2*3,3*3>;

    for (auto f : M.topology->faces() ){
        int i = f.getIndex();
        noise_triplet T = noise_triplet::Zero();
        SquareMatrix<3*3> noises_cov = SquareMatrix<3*3>::Zero();

        int j = 0;
        for (auto h : f.adjacentHalfedges()){
            if (h.face().getIndex() != i)
                continue;
            auto vj = h.tipVertex();
            auto vk = h.next().tipVertex();
            Vector<3> d = M.vertex(vk) - M.vertex(vj);
            SquareMatrix<3> C = crossMat(d)*0.5;
            SquareMatrix<3> Ct = C.transpose();
            SquareMatrix<3> CtC = C * Ct;
            noises_cov.block(3*j,3*j,3,3) = SquareMatrix<3>::Identity()*sigma*sigma;

            T.block(0,j*3,3,3) = SquareMatrix<3>::Identity() / 3;
            T.block(3,j*3,3,3) = C;

            // S.dipoles[i].CovMoment += CtC*sigma*sigma;
            j++;
        }
        S.dipoles[i].FullCov = T*noises_cov*T.transpose();
    }
    return S;

}

SGP::scalar SGP::GetAverageIso(const PosList<dim> &P, const GPIS &gp)
{
    scalar iso = 0;
    for (auto i : range(P.size())) {
        iso += gp.predict(P.getPos(i)).getMean()(0);
    }
    return iso / P.size();
}

SGP::GaussianDipoles<2> SGP::GaussianDipolesFromPolyline(const Points<2>& P, scalar sigma)
{
    int N = P.cols();

    GaussianDipoles<2> S;
    S.dipoles.resize(N-1);

    mat2 I = mat2::Identity();
    mat2 J = mat2::Zero();
    J << 0, -1,
         1,  0;


    SquareMatrix<2*2> T = SquareMatrix<2*2>::Zero();
    T.block(0,0,2,2) = I / 2;
    T.block(2,0,2,2) = I / 2;
    T.block(0,2,2,2) = J;
    T.block(2,2,2,2) =-J;

    SquareMatrix<2*2> F = T*T.transpose()*sigma*sigma;


    for (auto i : range(N-1) ){
        S.dipoles[i].p = (P.col(i) + P.col(i+1))/2;
        Vector<2> n = P.col(i+1) - P.col(i);
        n = J*n;
        S.dipoles[i].n = n;
        S.dipoles[i].FullCov = F;
    }

    return S;
}

SGP::GaussianDipoles<3> SGP::GaussianDipolesFromLidarScan(const Points<3> &P, const Points<3> &W, const Vec &S, scalar sig_min, scalar sig_max)
{
    SquareMatrix<2*dim> cov = SquareMatrix<2*dim>::Identity();
    std::vector<SquareMatrix<2*dim>> covs(P.cols(),cov);

    scalar lfs = 0;
    for (int i = 0; i < P.cols(); i++)
        lfs += std::sqrt(W.col(i).norm());
    lfs /= P.cols();

    for (int i = 0; i < P.cols(); i++) {
        scalar v = S(i);
        scalar s = std::lerp(sig_min*lfs,sig_max*lfs,1-v);
        covs[i].block(0,0,3,3) = SquareMatrix<dim>::Identity()*s*s/9;
        covs[i].block(3,3,3,3) = SquareMatrix<dim>::Identity()*s*s/4*W.col(i).norm();
    }
    return GaussianDipoles(P,W,covs);
}
