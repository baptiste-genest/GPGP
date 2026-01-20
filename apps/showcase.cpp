#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_grid.h"

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/exact_geodesics.h"
#include "geometrycentral/surface/flip_geodesics.h"

#include "../src/StochasticGeometryProcessing.h"
#include "../src/Mesh.h"
#include "../src/MeshSampling.h"
#include "../src/StochasticPoissonSurfaceReconstruction.h"
#include "../src/HamiltonianFastMarching.h"
#include "eigen3/Eigen/IterativeLinearSolvers"

#include "../src/BarnesHuttSPSR.h"
#include "../src/StochasticBarnesHutt.h"

#include "../src/Grid.h"
#include "../src/NarrowBand.h"
#include "../src/SGPWrapper.h"
#include "../src/utils.h"
#include "../src/Plot.h"

#include "../extern/CLI11.hpp"
#include "../extern/json.hpp"

#include <cstdlib>
#include <fenv.h>

using namespace SGP;

int N = 100;

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>


std::pair<Mat, Vec> computeEigenModesSPD(const smat &A, const smat &M, int nb) {
    using namespace Spectra;

    Spectra::SparseSymMatProd<scalar> Aop(M);
    Spectra::SparseRegularInverse<scalar> Bop(A);

    int nev = std::min(2*nb,(int)A.cols());

    SymGEigsSolver<Spectra::SparseSymMatProd<scalar>, Spectra::SparseRegularInverse<scalar>, GEigsMode::RegularInverse> eigs(Aop,Bop,nb,nev);

    eigs.init();
    eigs.compute(SortRule::LargestAlge);

    if (eigs.info() != CompInfo::Successful){
        exit(1);
    }
    Mat rslt = eigs.eigenvectors(nb);

    return {rslt,eigs.eigenvalues()};
}


vecs integrateGeodesic(const vec& target,scalar dt,const ScalarGrid& U, const VectorGrid &V) {
    vecs path;
    path.push_back(target);

    vec p = target;

    scalar val = Grid3D::lerp(U,p);

    int max_iter = 100000;
    for (auto iter : range(max_iter)) {
        p -= dt*Grid3D::lerp(V,p);
        path.push_back(p);
        // std::cout << Grid3D::lerp(U,p) << " " << Grid3D::lerp(V,p) << std::endl;
        if (std::abs(Grid3D::lerp(U,p)) < 1e-6)
            break;
    }

    return path;
}



Vec format(const Vec& x) {
    scalar min_x = x.minCoeff();
    scalar max_x = x.maxCoeff();
    Vec rslt = (x - Vec::Constant(x.size(),min_x)) / (max_x - min_x + 1e-10);
    return rslt;
}

Vec format_log(const Vec& y) {
    // format between 0 and 1
    Vec x = y.array().log();
    scalar min_x = x.minCoeff();
    scalar max_x = x.maxCoeff();
    Vec rslt = (x - Vec::Constant(x.size(),min_x)) / (max_x - min_x + 1e-6);
    return rslt;
}


HamiltonianFastMarching calc;
int eigenfunction = 0;

std::string input;
scalar reg = 0.01;
scalar beta = 2.3;
scalar eps = 0.01;

Mat EigenVectors;
polyscope::SurfaceMesh* surf_iso = nullptr;


bool naive_laplace = false;
bool adaptive = false;
bool dense = false;
bool HFM = false;
bool green = false;


void init () {

    GaussianDipoles GD = GaussianDipoles(input);
    GD.normalize(1.5);

    scalar lfs = GD.estimateScale();



    polyscope::VolumeGridNodeScalarQuantity* field;
    polyscope::VolumeGrid* pcgrid;
    polyscope::PointCloud* narrow_band;

    scalar iso;

    StopWatch profiler;
    profiler.start();
    StochasticBarnesHutt BHSPSR(GD,reg*lfs,beta);
    profiler.tick("build Barnes-Hut",true);

    auto pc_input = polyscope::registerPointCloud("input",GD.getPositions().transpose());
    pc_input->addVectorQuantity("normals",GD.getMoments().transpose())->setEnabled(true);
    pc_input->setEnabled(false);


    scalar l = 1;

    scalar h;
    if (N < 0){
        h = StencilReachHeuristic(GD,BHSPSR,lfs);
        N = std::ceil(2*l/h);
        spdlog::info("adaptive resolution h = {}, grid width {}",h,N);
    }
    else
        h = 2.*l/N;

    if (dense){
        PointGrid G = Grid3D::getGrid(N);
        pcgrid = polyscope::registerVolumeGrid("grid",{N,N,N},glm::vec3(-l,-l,-l),glm::vec3(l,l,l));

        profiler.start();
        auto SPSR_field = G.apply<pred>([&BHSPSR](const Vector<dim>& x) {
            return BHSPSR.predict(x);
        });
        iso = GetAverageIso(GD,BHSPSR);
        profiler.tick("query GPIS",true);

        calc = HamiltonianFastMarching(SPSR_field,iso , eps);
        profiler.tick("compute fields mu & T",true);

        auto value_field = Apply<pred,scalar>(SPSR_field, [iso](const pred& x) {
            return x.getMean()(0);
        });
        field = pcgrid->addNodeScalarQuantity("GPIS mean",value_field.data());
    }
    else {

        auto narrow = BuildAdaptiveNarrowBand(GD,BHSPSR,h,eps);
        pcgrid = PlotNarrowBand(narrow,"narrow");
        profiler.tick("build narrow band and fields",true);
        calc = HamiltonianFastMarching(narrow);

        field = pcgrid->addNodeScalarQuantity("GPIS mean",calc.extendFill(calc.getGPISMean()).data());
        profiler.tick("extend mean field to bounding box grid (for plot only)",true);
    }

    // extract mean iso
    field->setGridcubeVizEnabled(false);
    field->setIsosurfaceLevel(iso);
    field->setIsosurfaceVizEnabled(true);
    field->setEnabled(true);
    surf_iso = field->registerIsosurfaceAsMesh("mean surface");
    surf_iso->setShadeStyle(polyscope::MeshShadeStyle::Smooth);

    auto NB = calc.embedNarrowBand();

    narrow_band = polyscope::registerPointCloud("narrow band",NB.transpose());

    profiler.start();
    smat M = calc.buildMassMatrix();
    smat L = calc.buildVoronoiLaplace();

    // smat grad = calc.buildGradient();
    // smat div = calc.buildIntegratedDivergence();
    profiler.tick("build operators",true);


    using SPDSolver = Eigen::ConjugateGradient<smat, Eigen::Lower|Eigen::Upper>;

    SPDSolver solver(L);

    if (eigenfunction) {
        EigenVectors = computeEigenModesSPD(L,M,eigenfunction).first;
        auto rslt_eig = calc.extend(EigenVectors.col(eigenfunction-1));
        surf_iso->addVertexScalarQuantity("eig",format(sampleOnMesh(surf_iso,rslt_eig)));
        pcgrid->addNodeScalarQuantity("eig",rslt_eig.data())->setEnabled(false);
    }

    if (HFM) {
        int s = MostAlignedPoint(NB,vec(1,0,0));
        profiler.start();
        calc.precomputeStencils();
        profiler.tick("compute stencils",true);
        Vec U = calc.run(s);
        auto V = calc.computeGeodesicVelocityField();

        auto rslt = calc.extend(U,U.maxCoeff()+1);

        // vec x = MostAlignedVertex(surf_iso,vec(0,1,0));
        vec x = ClosestVertex(surf_iso,vec(-0.337,-0.131,-0.212));
        // vec x = ClosestVertex(surf_iso,vec(-0.232,-0.0259,0.5959));

        auto path = integrateGeodesic(x,100,rslt,V);
        auto curve = polyscope::registerCurveNetworkLine("geodesic",path);

        auto rslt_iso = format(sampleOnMesh(surf_iso,rslt));
        auto pc_dist = surf_iso->addVertexDistanceQuantity("input solution",rslt_iso);
        pc_dist->setIsolinesEnabled(true);
        // pc_dist->setIsolinePeriod(0.005,true);
        pc_dist->setIsolineWidth(0.005,true);
        // curve->setEnabled(false);
    }


    if (green) {
        Vec S = Vec::Zero(calc.getNumVariables());
        auto s = MostAlignedVertex(surf_iso,vec(1,0,0.1));
        // auto s = ClosestVertex(surf_iso,vec(-0.337,-0.131,-0.212));
        S(GetClosestPoint(NB,s)) = 1;

        SPDSolver poisson_solver(L);

        Vec RHS = M*S;

        Vec green = format(sampleOnMesh(surf_iso,calc.extend(poisson_solver.solve(RHS))));
        Vec green_log = format_log(sampleOnMesh(surf_iso,calc.extend(poisson_solver.solve(RHS))));

        auto pc_green = surf_iso->addVertexScalarQuantity("green's function",green);
        surf_iso->addVertexScalarQuantity("green's function log",green_log);
        pc_green->setColorMap("magma");
        pc_green->setIsolinesEnabled(true);
    }


}

int eig_nb = 0;
void myCallBack() {
    if (eigenfunction) {
        if (ImGui::SliderInt("eigenvector number",&eig_nb,0,EigenVectors.cols()-1)){
            auto rslt_eig = calc.extend(EigenVectors.col(eig_nb));
            surf_iso->addVertexScalarQuantity("eig",format(sampleOnMesh(surf_iso,rslt_eig)));
        }
    }
}

int main(int argc, char** argv) {
    CLI::App app("Uncertainty aware geometry processing on GPIS") ;

    app.add_option("--grid_size", N, "resolution of the grid, if negative adaptive resolution (default -1)");
    app.add_option("--reg", reg, "regularized winding number parameter (default 0.4)");
    app.add_option("--beta", beta, "barnes hutt approx parameter (default 3)");
    app.add_option("--epsilon", eps, "narrow band epsilon (default 0.01)");
    app.add_flag("--spectra",eigenfunction,"number of eigenfunctions to compute (default 0) WARNING CAN BE LONG");
    // app.add_flag("--hodge",hodge,"Hodge decomposition");
    app.add_flag("--geodesic",HFM,"compute geodesic distance");
    app.add_flag("--green",green,"green's function computation");
    app.add_flag("--naive",naive_laplace,"use naive laplacian instead");
    app.add_flag("--dense",dense,"if true then don't use BFS narrow band");

    CLI11_PARSE(app, argc, argv);

    // polyscope::options::allowHeadlessBackends = !viz;
    polyscope::init();
    init();

    polyscope::state::userCallback = myCallBack;
    polyscope::show();
    return 0;
}
