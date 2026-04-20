
# Uncertainty-Aware Geometry processing on Gaussian Process Implicit Surfaces -- Source code

This repository contains the source code of the SIGGRAPH 2026 paper *"[Uncertainty-Aware Geometry processing on Gaussian Process Implicit Surfaces](https://baptiste-genest.github.io/publications/#gpgp)", Baptiste Genest, David Coeurjolly.*
<img width="1993" height="729" alt="teaser" src="https://github.com/user-attachments/assets/a412d37e-df5b-46a7-bbfd-c6fa776eaee3" />



This guide explains how to compile the project using CMake, focusing on the top-level `CMakeLists.txt`. It also lists all dependencies and how to obtain them.

## Requirements

- **CMake** >= 3.12
- **C++20** compatible compiler (e.g., GCC 10+, Clang 10+, MSVC 2019+)
- **git** (for fetching some dependencies)
- **Internet connection** (for fetching dependencies using CPM)

### Dependencies

The following libraries are required and are automatically handled by the CMake build system via CPM or included CMake scripts:

- [Eigen3](https://gitlab.com/libeigen/eigen) (version 3.4.0, downloaded automatically)
- [OpenMP](https://www.openmp.org/) (for parallelization, usually provided by your compiler)
- [Polyscope](https://github.com/nmwsharp/polyscope)
- [geometry-central](https://github.com/nmwsharp/geometry-central)
- [spdlog](https://github.com/gabime/spdlog)
- [Spectra](https://github.com/yixuan/spectra)

All dependencies except Eigen3 are also included via CMake include scripts (see the `cmake/` directory).

## Step-by-Step Compilation

1. **Go in code folder**
   ```bash
   cd GPGP
   ```

2. **Create a build directory**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the project with CMake**
   ```bash
   cmake ..
   ```
   - This will download and configure all dependencies using CPM and the scripts in `cmake/`.

4. **Build the project**
   ```bash
   cmake --build .
   ```
   - This will build the executable defined in the main `CMakeLists.txt`.

The executable `showcase` should be built, it corresponds to a source file in the `apps/` directory.

## Example

Note that all parameters can be described with the help command :
```bash
./exe --help
```
the parameters are described with names that match the ones used in the paper.

to reproduce the figure 8, you can execute
```bash
./showcase --input ../data/spot_low_variance.gdp --eta 0.001 --grid_size 300 --geodesic 
./showcase --input ../data/spot_high_variance.gdp --eta 1 --geodesic 
```
