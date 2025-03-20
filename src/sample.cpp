#include <hip/hip_runtime.h>
#include <hipsolver.h>
#include <internal/hipsolver-types.h>
#include <iostream>
#include <vector>

// Error checking macro
#define CHECK_HIP(err) if (err != hipSuccess) { \
    printf("HIP error: %s\n", hipGetErrorString(err)); \
    exit(1); \
}

#define CHECK_HIPSOLVER(err) if (err != HIPSOLVER_STATUS_SUCCESS) { \
    printf("HIPSOLVER error: %d\n", err); \
    exit(1); \
}

int main() {
    // Initialize problem size
    const int n = 3;  // Size of the matrix
    const int nrhs = 1;  // Number of right hand sides
    const int lda = n;  // Leading dimension of A
    const int ldb = n;  // Leading dimension of B

    // Initialize input matrices on host
    std::vector<double> A_h = {
        3.0, 1.0, 1.0,
        1.0, 3.0, 1.0,
        1.0, 1.0, 3.0
    };
    std::vector<double> B_h = {1.0, 2.0, 3.0};  // Right-hand side
    std::vector<int> ipiv_h(n);  // Pivot indices

    // Allocate device memory
    double *A_d, *B_d;
    int *ipiv_d;
    CHECK_HIP(hipMalloc((void**)&A_d, n * n * sizeof(double)));
    CHECK_HIP(hipMalloc((void**)&B_d, n * sizeof(double)));
    CHECK_HIP(hipMalloc((void**)&ipiv_d, n * sizeof(int)));

    // Copy data to device
    CHECK_HIP(hipMemcpy(A_d, A_h.data(), n * n * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B_d, B_h.data(), n * sizeof(double), hipMemcpyHostToDevice));

    // Initialize HipSOLVER
    hipsolverHandle_t handle;
    CHECK_HIPSOLVER(hipsolverCreate(&handle));

    // Calculate workspace size
    int lwork;
    CHECK_HIPSOLVER(hipsolverDgetrf_bufferSize(
        handle, n, n, A_d, lda, &lwork));

    // Allocate workspace
    double* work;
    CHECK_HIP(hipMalloc((void**)&work, lwork * sizeof(double)));

    // Perform LU factorization
    int* info;
    CHECK_HIP(hipMalloc((void**)&info, sizeof(int)));
    
    CHECK_HIPSOLVER(hipsolverDgetrf(
        handle, n, n, A_d, lda, work, lwork, ipiv_d, info));

    // Solve the system using the factorization
    CHECK_HIPSOLVER(hipsolverDgetrs(
        handle, HIPSOLVER_OP_N, n, nrhs,
        A_d, lda, ipiv_d, B_d, ldb,
        work, lwork, info));

    // Copy solution back to host
    CHECK_HIP(hipMemcpy(B_h.data(), B_d, n * sizeof(double), hipMemcpyDeviceToHost));

    // Print solution
    std::cout << "Solution x:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << B_h[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    hipsolverDestroy(handle);
    hipFree(A_d);
    hipFree(B_d);
    hipFree(ipiv_d);
    hipFree(work);
    hipFree(info);

    return 0;
}
