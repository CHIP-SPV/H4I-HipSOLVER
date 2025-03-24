#include <iostream>
#include <vector>
#include <cmath>
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "hipsolver.h"

#define HIP_CHECK(stat)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = stat;                                          \
        if(err != hipSuccess)                                           \
        {                                                               \
            std::cerr << "HIP error: " << hipGetErrorString(err)        \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

#define HIPSOLVER_CHECK(stat)                                           \
    do                                                                  \
    {                                                                   \
        hipsolverStatus_t err = stat;                                   \
        if(err != HIPSOLVER_STATUS_SUCCESS)                             \
        {                                                               \
            std::cerr << "hipSOLVER error: " << err                     \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

int main() {
    std::cout << "======== H4I-HipSOLVER Basic Test ========" << std::endl;
    
    // Test context creation
    std::cout << "Testing hipSOLVER context creation..." << std::endl;
    hipsolverHandle_t handle = nullptr;
    HIPSOLVER_CHECK(hipsolverCreate(&handle));
    
    if (handle) {
        std::cout << "Context creation successful!" << std::endl;
    } else {
        std::cerr << "Failed to create context!" << std::endl;
        return -1;
    }

    // Test stream creation and setting
    std::cout << "Testing hipSOLVER stream handling..." << std::endl;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIPSOLVER_CHECK(hipsolverSetStream(handle, stream));
    std::cout << "Stream handling successful!" << std::endl;

    // Test a simple solver operation (getrf - LU factorization)
    std::cout << "Testing hipSOLVER operation (getrf)..." << std::endl;
    
    const int n = 3;
    const int lda = n;
    
    // Example matrix A (3x3)
    // [ 3.0, 1.0, 2.0 ]
    // [ 2.0, 4.0, 1.0 ]
    // [ 1.0, 2.0, 5.0 ]
    std::vector<float> hA = {
        3.0f, 2.0f, 1.0f,  // First column
        1.0f, 4.0f, 2.0f,  // Second column
        2.0f, 1.0f, 5.0f   // Third column
    };
    
    // Expected result after LU factorization
    // (L and U combined, with diagonal of L implied to be 1)
    std::vector<float> expected_LU = {
        3.0f, 2.0f, 1.0f,
        0.333333f, 3.333333f, 1.666667f,
        0.666667f, 0.2f, 4.3333333f
    };
    
    float* dA;
    int* dInfo;
    int* dIpiv;
    int h_info = 0;
    std::vector<int> h_ipiv(n);
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&dA, n * n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dInfo, sizeof(int)));
    HIP_CHECK(hipMalloc(&dIpiv, n * sizeof(int)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(dA, hA.data(), n * n * sizeof(float), hipMemcpyHostToDevice));
    
    // Get buffer size for getrf
    int lwork;
    HIPSOLVER_CHECK(hipsolverSgetrf_bufferSize(handle, n, n, dA, lda, &lwork));
    
    float* dWork;
    HIP_CHECK(hipMalloc(&dWork, lwork * sizeof(float)));
    
    // Execute LU factorization
    HIPSOLVER_CHECK(hipsolverSgetrf(handle, n, n, dA, lda, dWork, lwork, dIpiv, dInfo));
    
    // Get results
    std::vector<float> result(n * n);
    HIP_CHECK(hipMemcpy(result.data(), dA, n * n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_ipiv.data(), dIpiv, n * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_info, dInfo, sizeof(int), hipMemcpyDeviceToHost));
    
    // Print results
    std::cout << "LU factorization info: " << h_info << std::endl;
    std::cout << "Pivots: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_ipiv[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "LU Factorization Matrix:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[j * n + i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Validate results - check h_info
    if (h_info == 0) {
        std::cout << "LU factorization successful!" << std::endl;
    } else {
        std::cerr << "LU factorization failed with info = " << h_info << std::endl;
    }
    
    // Clean up
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dInfo));
    HIP_CHECK(hipFree(dIpiv));
    HIP_CHECK(hipFree(dWork));
    HIP_CHECK(hipStreamDestroy(stream));
    HIPSOLVER_CHECK(hipsolverDestroy(handle));
    
    std::cout << "======== Test Complete ========" << std::endl;
    return 0;
} 