// Copyright 2021-2023 UT-Battelle
// See LICENSE in the root of the source distribution for license info.

#pragma once

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
hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle);
hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle);
hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId);

// Dn interface
hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle);
hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle);
hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle, hipStream_t streamId); 