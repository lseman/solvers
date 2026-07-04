cmake_minimum_required(VERSION 3.20)

project(solvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Module build options
option(BUILD_OSPQ "Build OSQP solver extension" OFF)
option(BUILD_PIQP "Build PIQP solver extension" OFF)
option(BUILD_QDLDL "Build QDLDL extension" OFF)
option(BUILD_IPM "Build IPM solver extension" ON)

include(FetchContent)

if(NOT COMMAND CPMAddPackage)
    FetchContent_Declare(
        CPM
        GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
        GIT_TAG v0.42.1
    )
    FetchContent_MakeAvailable(CPM)
    include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    OPTIONS
    "EIGEN_BUILD_DOC OFF"
    "BUILD_TESTING OFF"
)

cpmaddpackage(
    NAME stdexec
    GITHUB_REPOSITORY NVIDIA/stdexec
    GIT_TAG main
    OPTIONS "STDEXEC_BUILD_TESTS OFF" "STDEXEC_BUILD_EXAMPLES OFF" "STDEXEC_BUILD_BENCHMARKS OFF" "STDEXEC_BUILD_DOCS OFF" "BUILD_TESTING OFF"
)

CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG v2.9.2
)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

add_library(solvers_headers INTERFACE)
target_include_directories(
    solvers_headers
    INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
target_link_libraries(solvers_headers INTERFACE Eigen3::Eigen)
target_compile_features(solvers_headers INTERFACE cxx_std_20)

if(BUILD_OSPQ)
    nanobind_add_module(osqp_ext python/osqp_ext.cpp)
    target_link_libraries(osqp_ext PRIVATE solvers_headers)
    set_target_properties(
        osqp_ext
        PROPERTIES
        OUTPUT_NAME osqp
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS osqp_ext)
endif()

if(BUILD_PIQP)
    nanobind_add_module(piqp_ext python/piqp_ext.cpp)
    target_link_libraries(piqp_ext PRIVATE solvers_headers)
    set_target_properties(
        piqp_ext
        PROPERTIES
        OUTPUT_NAME piqp
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS piqp_ext)
endif()

if(BUILD_QDLDL)
    nanobind_add_module(qdldl_ext python/qdldl_ext.cpp)
    target_link_libraries(qdldl_ext PRIVATE solvers_headers)
    set_target_properties(
        qdldl_ext
        PROPERTIES
        OUTPUT_NAME qdldl
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS qdldl_ext)
endif()

if(BUILD_IPM)
    nanobind_add_module(
        ipm_ext
        python/ipm_module.cpp
        src/ipm/IPSolver.cpp
        src/ipm/ipm_linear_solver.cpp
    )
    target_link_libraries(ipm_ext PRIVATE solvers_headers stdexec)
    set_target_properties(
        ipm_ext
        PROPERTIES
        OUTPUT_NAME ipm_solver
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS ipm_ext)
endif()

if(SOLVER_TARGETS)
    install(TARGETS ${SOLVER_TARGETS} LIBRARY DESTINATION solvers)
endif()
install(FILES python/solvers/__init__.py DESTINATION solvers)
