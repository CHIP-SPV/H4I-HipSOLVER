// Copyright 2021-2023 UT-Battelle
// See LICENSE in the root of the source distribution for license info.

#include <iostream>
#include <hip/hip_runtime.h>

// Define minimal types needed for the context test
typedef void* hipsolverHandle_t;

typedef enum {
    HIPSOLVER_STATUS_SUCCESS = 0,
    HIPSOLVER_STATUS_NOT_INITIALIZED = 1,
    HIPSOLVER_STATUS_ALLOC_FAILED = 2,
    HIPSOLVER_STATUS_INVALID_VALUE = 3,
    HIPSOLVER_STATUS_MAPPING_ERROR = 4,
    HIPSOLVER_STATUS_EXECUTION_FAILED = 5,
    HIPSOLVER_STATUS_INTERNAL_ERROR = 6,
    HIPSOLVER_STATUS_NOT_SUPPORTED = 7,
    HIPSOLVER_STATUS_ARCH_MISMATCH = 8,
    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9
} hipsolverStatus_t;

// Function declarations needed for the test
extern "C" {
hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle);
hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle);
hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId);

// Dn interface
hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle);
hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle);
hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle, hipStream_t streamId);
}

// Simple macro to check HipSOLVER status
#define CHECK_HIPSOLVER_STATUS(status) \
    if (status != HIPSOLVER_STATUS_SUCCESS) { \
        std::cerr << "HipSOLVER error at line " << __LINE__ << ": " << status << std::endl; \
        return EXIT_FAILURE; \
    }

int main() {
    std::cout << "Testing HipSOLVER Context Creation and Basic Functions" << std::endl;
    
    // Test context creation
    hipsolverHandle_t handle = nullptr;
    hipsolverStatus_t status = hipsolverCreate(&handle);
    CHECK_HIPSOLVER_STATUS(status);
    
    if (handle == nullptr) {
        std::cerr << "Failed to create HipSOLVER handle" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Successfully created HipSOLVER handle" << std::endl;
    
    // Test setting stream
    hipStream_t stream;
    hipError_t hipStatus = hipStreamCreate(&stream);
    if (hipStatus != hipSuccess) {
        std::cerr << "Failed to create HIP stream" << std::endl;
        hipsolverDestroy(handle);
        return EXIT_FAILURE;
    }
    
    status = hipsolverSetStream(handle, stream);
    CHECK_HIPSOLVER_STATUS(status);
    std::cout << "Successfully set stream" << std::endl;
    
    // Also test the Dn interface
    hipsolverHandle_t dnHandle = nullptr;
    status = hipsolverDnCreate(&dnHandle);
    CHECK_HIPSOLVER_STATUS(status);
    
    if (dnHandle == nullptr) {
        std::cerr << "Failed to create HipSOLVER Dn handle" << std::endl;
        hipStreamDestroy(stream);
        hipsolverDestroy(handle);
        return EXIT_FAILURE;
    }
    
    status = hipsolverDnSetStream(dnHandle, stream);
    CHECK_HIPSOLVER_STATUS(status);
    std::cout << "Successfully set stream for Dn handle" << std::endl;
    
    // Clean up
    hipStreamDestroy(stream);
    status = hipsolverDnDestroy(dnHandle);
    CHECK_HIPSOLVER_STATUS(status);
    
    // Test destruction
    status = hipsolverDestroy(handle);
    CHECK_HIPSOLVER_STATUS(status);
    
    std::cout << "All tests passed successfully!" << std::endl;
    return EXIT_SUCCESS;
} 