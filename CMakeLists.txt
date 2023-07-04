cmake_minimum_required(VERSION 3.20)
include (${CMAKE_SOURCE_DIR}/CMake/HipSOLVERVersion.cmake)
project(HipSOLVER
    VERSION ${HipSOLVER_VERSION}
    LANGUAGES CXX)


find_package(hip REQUIRED)

include(GNUInstallDirs)
file(RELATIVE_PATH relRPath
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relRPath})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(HipSOLVERCommonConfig INTERFACE)
target_compile_features(HipSOLVERCommonConfig
    INTERFACE
        cxx_std_17
    )

option(H4I_USE_ROCM_HIPBLAS "Whether to use ROCm-installed hipBLAS" OFF)
if(NOT H4I_USE_ROCM_HIPBLAS)
    # We want to use the H4I implementation.
    # Build our library.
    add_subdirectory(src)
endif()

