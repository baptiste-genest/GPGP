cmake_minimum_required(VERSION 3.20)
project(UncertaintyAwareGeometryProcessing LANGUAGES CXX)

# ---- Global settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ---- Dependencies ----
include(CPM)
include(openmp)
include(eigen)
include(polyscope)
include(geometry-central)
include(spdlog)
include(spectra)

# ---- Sources ----
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

add_library(core STATIC ${CORE_SOURCES})

target_link_libraries(core
    PUBLIC
        polyscope
        spdlog
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        Spectra
        geometry-central
)

target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(showcase apps/showcase.cpp)
target_link_libraries(showcase PRIVATE core)

add_executable(soupify apps/soupify.cpp)
target_link_libraries(soupify PRIVATE core)
