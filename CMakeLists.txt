project(UncertaintyAwareGeometryProcessing)
cmake_minimum_required(VERSION 3.12)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CPM)
include(openmp)
include(eigen)
include(polyscope)
include(geometry-central)
include(spdlog)

file(GLOB SOURCE
    "src/*"
)

add_executable(showcase apps/showcase.cpp ${SOURCE} )
target_link_libraries(showcase polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX geometry-central)
