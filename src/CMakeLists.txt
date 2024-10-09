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

include(ExternalProject)
option(H4I_ROCM_HIPSOLVER_TAG "Tag to use from the ROCm hipBLAS repository when obtaining its hipsolver.h header" "rocm-5.3.3")
ExternalProject_Add(ROCmHipsolver
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/hipsolver
    GIT_TAG ${H4I_ROCM_HIPSOLVER_TAG}

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E env SHELL=/bin/bash
                    bash ${CMAKE_CURRENT_SOURCE_DIR}/../Scripts/install-hipsolver-header.sh
                    <SOURCE_DIR> ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
)

# Add a custom target that depends on the external project
add_custom_target(ROCmHipsolverHeaders
    DEPENDS ROCmHipsolver
)

set(PROJECT_VERSION_MAJOR 5)
set(PROJECT_VERSION_MINOR 3)
set(PROJECT_VERSION_PATCH 3)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/internal/hipsolver-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-version.h
    @ONLY)

add_library(hipsolver SHARED
    util.cpp
    hipsolver.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
# Add dependency on the custom target
add_dependencies(hipsolver ROCmHipsolverHeaders)
target_include_directories(hipsolver
  PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include;${CMAKE_BINARY_DIR}/include>"
    $<INSTALL_INTERFACE:include>)
# Check if hip::host target exists
if(NOT TARGET hip::host)
    message(FATAL_ERROR "hip::host target not found. Please ensure HIP is properly installed and found by CMake.")
endif()

target_link_libraries(hipsolver
	PRIVATE
    MKLShim
  PUBLIC
    hip::host
  )

include (GNUInstallDirs)
install(TARGETS hipsolver
    EXPORT hipsolver
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/hipsolver.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-version.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-types.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-functions.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-compat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/internal/hipsolver-export.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/internal/hipsolver-refactor.h 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/internal/
)


# install(EXPORT hipsolver
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver
#   NAMESPACE H4I::
# )

