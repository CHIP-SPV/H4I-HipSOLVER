find_package(MKLShim REQUIRED)

include(ExternalProject)
option(H4I_ROCM_HIPSOLVER_TAG "Tag to use from the ROCm hipBLAS repository when obtaining its hipsolver.h header" "rocm-5.3.3")
ExternalProject_Add(ROCmHipsolver
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/hipsolver
    GIT_TAG ${H4I_ROCM_HIPSOLVER_TAG}

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND sh ${CMAKE_SOURCE_DIR}/Scripts/install-hipsolver-header.sh <SOURCE_DIR> ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
)

configure_file(
    ${CMAKE_SOURCE_DIR}/include/internal/hipsolver-version.h.in
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-version.h
    @ONLY)

add_library(hipsolver SHARED
    util.cpp
    hipsolver.cpp)
target_include_directories(hipsolver
  PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include;${CMAKE_BINARY_DIR}/include>"
    $<INSTALL_INTERFACE:include>)
target_link_libraries(hipsolver
	PRIVATE
    H4I::MKLShim
  PUBLIC
    hip::host
  )

include (GNUInstallDirs)
install(TARGETS hipsolver
    EXPORT hipsolver
)
install(FILES
    ${CMAKE_BINARY_DIR}/include/hipsolver.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
)

install(FILES
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-version.h
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-types.h
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-functions.h
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-compat.h
    ${CMAKE_SOURCE_DIR}/include/internal/hipsolver-export.h
    ${CMAKE_BINARY_DIR}/include/internal/hipsolver-refactor.h 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/internal/
)


install(EXPORT hipsolver
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver
  NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/HipSOLVERConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/hipsolverConfig.cmake
	PATH_VARS CMAKE_INSTALL_INCLUDEDIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/hipsolverConfig.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver
)
