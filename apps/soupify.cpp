#include "../src/StochasticGeometryProcessing.h"
#include "../src/Mesh.h"
#include "../src/MeshSampling.h"
#include "../src/StochasticPoissonSurfaceReconstruction.h"
#include "../src/HamiltonianFastMarching.h"
#include "Eigen/IterativeLinearSolvers"

#include "../src/BarnesHuttSPSR.h"
#include "../src/StochasticBarnesHutt.h"

#include "../src/Grid.h"
#include "../src/NarrowBand.h"
#include "../src/SGPWrapper.h"
#include "../src/utils.h"
#include "../src/Plot.h"

#include "../extern/CLI11.hpp"
#include "../extern/json.hpp"


using namespace SGP;

std::string meshsrc;
std::string out = "dipoles.gdp";
scalar assumed_noise = 1;

int main(int argc, char** argv) {
    CLI::App app("Soupify") ;

    app.add_option("--input", meshsrc, "input mesh (can be triangle soup)");
    app.add_option("--output", out, "output file");
    app.add_option("--assumed_noise", assumed_noise, "assumed noise sigma on positions");

    CLI11_PARSE(app, argc, argv);

    Mesh M(meshsrc);

    M.normalize(1.8);
    M.geometry->requireMeshLengthScale();

    scalar lfs = M.geometry->meshLengthScale;
    spdlog::info("Local feature size {}",lfs);

    GaussianDipoles<dim> GD = GaussianDipolesFromTriangleSoup(M,assumed_noise*lfs);
    GD.save(out);

    return 0;
}
