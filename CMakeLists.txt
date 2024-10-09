cmake_minimum_required(VERSION 3.20)
project(HipSOLVER
    VERSION ${HipSOLVER_VERSION}
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STANDALONE_BUILD ON)
else()
    set(STANDALONE_BUILD OFF)
endif()

# set the default CMAKE_INSTALL_PREFIX to the current directory/install
if(STANDALONE_BUILD AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# print where the library will be installed
message(STATUS "HipSOLVER will be installed to: ${CMAKE_INSTALL_PREFIX}")

# Find HIP
if(STANDALONE_BUILD)
  # Build H4I-MKLShim from submodule first
  add_subdirectory(H4I-MKLShim REQUIRED)
  
  # Find hip package, but exclude its H4I-MKLShim
  set(CMAKE_IGNORE_PATH ${CMAKE_IGNORE_PATH} ${hip_DIR}/../H4I-MKLShim)
  find_package(hip CONFIG REQUIRED)
  
  # Explicitly set the path to our built H4I-MKLShim
  set(H4I_MKLShim_DIR ${CMAKE_CURRENT_BINARY_DIR}/H4I-MKLShim)
endif()

include(GNUInstallDirs)
file(RELATIVE_PATH relRPath
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relRPath})

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
