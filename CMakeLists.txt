cmake_minimum_required(VERSION 3.20)

project(solvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SIMD and optimization flags for SOTA performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /fp:fast")
else()
    # GCC/Clang: haswell baseline (covers AVX2/FMA/AVX512 on modern CPUs)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell")

    # Optimization: -O3, LTO
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-signed-zeros")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")

    # LTO for smaller, faster binaries
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ── Highway SIMD (optional, runtime ISA dispatch) ──────────────────────────
option(WITH_HIGHWAY "Enable Google Highway for runtime SIMD dispatch (AVX512/AVX2/SSE2/NEON)" OFF)
if(WITH_HIGHWAY)
    message(STATUS "Highway SIMD enabled — runtime ISA dispatch")
    set(LDTLT_WITH_HIGHWAY ON)
endif()

# Module build options
option(BUILD_OSPQ "Build OSQP solver extension" ON)
option(BUILD_PIQP "Build PIQP solver extension" ON)
option(BUILD_QDLDL "Build QDLDL extension" ON)
option(BUILD_LDLT "Build custom LDLT extension" ON)
option(BUILD_IPM "Build IPM solver extension" ON)
option(BUILD_PDLP "Build PDLP first-order LP solver extension" ON)
option(WITH_CUDA "Enable CUDA backend for PDLP" OFF)
option(BUILD_SOLVER_TESTS "Build solver C++ tests" OFF)

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

# Highway: portable SIMD with runtime ISA dispatch
if(WITH_HIGHWAY)
    CPMAddPackage(
        NAME highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG v1.2.0
        OPTIONS
            "HWY_ENABLE_TESTS OFF"
            "HWY_ENABLE_EXAMPLES OFF"
            "HWY_ENABLE_CONTRIB OFF"
            "HWY_ENABLE_INSTALL OFF"
    )
    if(highway_ADDED)
        message(STATUS "Highway fetched via CPM")
    endif()
endif()

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
    if(WITH_HIGHWAY)
        target_compile_definitions(qdldl_ext PRIVATE LDLT_WITH_HIGHWAY)
        target_include_directories(qdldl_ext PRIVATE ${highway_SOURCE_DIR}/src)
        target_link_libraries(qdldl_ext PRIVATE highway::highway)
    endif()
    set_target_properties(
        qdldl_ext
        PROPERTIES
        OUTPUT_NAME qdldl
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS qdldl_ext)
endif()

if(BUILD_LDLT)
    nanobind_add_module(ldlt_ext python/ldlt_ext.cpp)
    target_link_libraries(ldlt_ext PRIVATE solvers_headers)
    if(WITH_HIGHWAY)
        target_compile_definitions(ldlt_ext PRIVATE LDLT_WITH_HIGHWAY)
        target_include_directories(ldlt_ext PRIVATE ${highway_SOURCE_DIR}/src)
        target_link_libraries(ldlt_ext PRIVATE highway::highway)
    endif()
    set_target_properties(
        ldlt_ext
        PROPERTIES
        OUTPUT_NAME ldlt
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS ldlt_ext)

    nanobind_add_module(ldlt_bk_ext python/ldlt_bk_ext.cpp)
    target_link_libraries(ldlt_bk_ext PRIVATE solvers_headers)
    if(WITH_HIGHWAY)
        target_compile_definitions(ldlt_bk_ext PRIVATE LDLT_WITH_HIGHWAY)
        target_include_directories(ldlt_bk_ext PRIVATE ${highway_SOURCE_DIR}/src)
        target_link_libraries(ldlt_bk_ext PRIVATE highway::highway)
    endif()
    set_target_properties(
        ldlt_bk_ext
        PROPERTIES
        OUTPUT_NAME ldlt_bk
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS ldlt_bk_ext)
endif()

if(BUILD_IPM)
    nanobind_add_module(
        ipm_ext
        python/ipm_module.cpp
        src/ipm/IPSolver.cpp
        # src/ipm/ipm_linear_solver.cpp
    )
    target_link_libraries(ipm_ext PRIVATE solvers_headers stdexec)
    set_target_properties(
        ipm_ext
        PROPERTIES
        OUTPUT_NAME ipm_solver
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS ipm_ext)

    # Quasi-definite augmented-system solver (BK pivoting + iterative refinement)
    nanobind_add_module(ipm_qd python/ipm_qd_ext.cpp)
    target_link_libraries(ipm_qd PRIVATE solvers_headers)
    if(WITH_HIGHWAY)
        target_compile_definitions(ipm_qd PRIVATE LDLT_WITH_HIGHWAY)
        target_include_directories(ipm_qd PRIVATE ${highway_SOURCE_DIR}/src)
        target_link_libraries(ipm_qd PRIVATE highway::highway)
    endif()
    set_target_properties(
        ipm_qd
        PROPERTIES
        OUTPUT_NAME ipm_qd
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS ipm_qd)
endif()

if(BUILD_PDLP)
    nanobind_add_module(pdlp_ext python/pdlp_ext.cpp)
    target_link_libraries(pdlp_ext PRIVATE solvers_headers)
    if(WITH_CUDA)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        find_package(CUDAToolkit REQUIRED)
        target_sources(pdlp_ext PRIVATE src/pdlp/pdlp_cuda.cu)
        target_compile_definitions(pdlp_ext PRIVATE PDLP_WITH_CUDA)
        # host code in .cu must use the same arch as the C++ TUs — Eigen's
        # allocator alignment is decided per-TU and mixing is an ODR violation
        if(NOT MSVC)
            target_compile_options(pdlp_ext PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=haswell>)
        endif()
        target_link_libraries(pdlp_ext PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
        # nvcc device link steps don't mix with host LTO
        set_target_properties(pdlp_ext PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    set_target_properties(
        pdlp_ext
        PROPERTIES
        OUTPUT_NAME pdlp
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND SOLVER_TARGETS pdlp_ext)
endif()

if(SOLVER_TARGETS)
    install(TARGETS ${SOLVER_TARGETS} LIBRARY DESTINATION solvers)
endif()
install(FILES python/solvers/__init__.py DESTINATION solvers)

if(BUILD_SOLVER_TESTS)
    enable_testing()

    add_executable(test_supernodes tests/test_supernodes.cpp)
    target_link_libraries(test_supernodes PRIVATE solvers_headers)

    add_test(NAME supernodes COMMAND test_supernodes)
endif()
