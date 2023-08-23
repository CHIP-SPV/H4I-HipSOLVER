#include <iostream>
#include <hipsolver.h>
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/types.h"

#define HIP_CHECK(m) \
  if(m != hipSuccess){ return HIPSOLVER_STATUS_INVALID_VALUE;}

#define HIPSOLVER_TRY \
  if (handle == nullptr) {\
    return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;\
  }\
  try {

#define HIPSOLVER_CATCH(msg) \
  } catch(std::exception const& e) {\
    std::cerr <<msg<<" exception: " << e.what() << std::endl;\
    return HIPSOLVER_STATUS_EXECUTION_FAILED;\
  }\
  return HIPSOLVER_STATUS_SUCCESS;

bool isValid(hipsolverSideMode_t s) {
    if (s != HIPSOLVER_SIDE_LEFT && s != HIPSOLVER_SIDE_RIGHT){
        return false;
    }
    return true;
}

inline H4I::MKLShim::onemklGen convertToGen(hipsolverSideMode_t s) {
    switch(s){
        case HIPSOLVER_SIDE_LEFT: return H4I::MKLShim::ONEMKL_GEN_Q;
        case HIPSOLVER_SIDE_RIGHT: return H4I::MKLShim::ONEMKL_GEN_P;
    }
}

inline H4I::MKLShim::onemklSideMode convert(hipsolverSideMode_t s) {
    switch(s){
        case HIPSOLVER_SIDE_LEFT: return H4I::MKLShim::ONEMKL_SIDE_LEFT;
        case HIPSOLVER_SIDE_RIGHT: return H4I::MKLShim::ONEMKL_SIDE_RIGHT;
    }
}

bool isValid(hipsolverEigMode_t j){
    if (j == HIPSOLVER_EIG_MODE_NOVECTOR || j == HIPSOLVER_EIG_MODE_VECTOR)
        return true;
    return false;
}

inline H4I::MKLShim::onemklJob convert(hipsolverEigMode_t job) {
  switch(job) {
    case HIPSOLVER_EIG_MODE_NOVECTOR: return H4I::MKLShim::ONEMKL_JOB_NOVEC;
    case HIPSOLVER_EIG_MODE_VECTOR: return H4I::MKLShim::ONEMKL_JOB_VEC;
  }
}

bool isValid(hipsolverFillMode_t v){
    if (v == HIPSOLVER_FILL_MODE_UPPER || v == HIPSOLVER_FILL_MODE_LOWER)
        return true;
    return false;
}

inline H4I::MKLShim::onemklUplo convert(hipsolverFillMode_t val) {
    switch(val) {
        case HIPSOLVER_FILL_MODE_UPPER:
            return H4I::MKLShim::ONEMKL_UPLO_UPPER;
        case HIPSOLVER_FILL_MODE_LOWER:
            return H4I::MKLShim::ONEMKL_UPLO_LOWER;
    }
}

bool isValid(hipsolverOperation_t t){
    if (t == HIPSOLVER_OP_T || t == HIPSOLVER_OP_C || t == HIPSOLVER_OP_N)
        return true;
    return false;
}
inline H4I::MKLShim::onemklTranspose convert(hipsolverOperation_t val) {
    switch(val) {
        case HIPSOLVER_OP_T:
            return H4I::MKLShim::ONEMKL_TRANSPOSE_TRANS;
        case HIPSOLVER_OP_C:
            return H4I::MKLShim::ONEMLK_TRANSPOSE_CONJTRANS;
        case HIPSOLVER_OP_N:
        default:
            return H4I::MKLShim::ONEMKL_TRANSPOSE_NONTRANS;
    }
}
bool isValid(hipsolverEigType_t t) {
    if (t == HIPSOLVER_EIG_TYPE_1 || t == HIPSOLVER_EIG_TYPE_2 || t == HIPSOLVER_EIG_TYPE_3)
        return true;
    return false;
}
inline int64_t convert(hipsolverEigType_t t) {
  switch(t){
    case HIPSOLVER_EIG_TYPE_1:
      return 1;
    case HIPSOLVER_EIG_TYPE_2:
      return 2;
    case HIPSOLVER_EIG_TYPE_3:
      return 3;
    default:
      return -1; // error: Never come here
  }
}
// gebrd
hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Sgebrd_ScPadSz(ctxt, m, n, *lwork);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgebrd_scratchpad")
}
hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Dgebrd_ScPadSz(ctxt, m, n, *lwork);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgebrd_scratchpad")
}
hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Cgebrd_ScPadSz(ctxt, m, n, *lwork);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgebrd_scratchpad")
}
hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Zgebrd_ScPadSz(ctxt, m, n, *lwork);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgebrd_scratchpad")
}

hipsolverStatus_t hipsolverSgebrd(hipsolverHandle_t handle,
                                 int               m,
                                 int               n,
                                 float*            A,
                                 int               lda,
                                 float*            D,
                                 float*            E,
                                 float*            tauq,
                                 float*            taup,
                                 float*            work,
                                 int               lwork,
                                 int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = lda;
    auto status = hipsolverSgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sgebrd(ctxt, m, n, A, lda, D, E, tauq, taup, work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgebrd")
}
hipsolverStatus_t hipsolverDgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                double*           A,
                                int               lda,
                                double*           D,
                                double*           E,
                                double*           tauq,
                                double*           taup,
                                double*           work,
                                int               lwork,
                                int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = lda;
    auto status = hipsolverDgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dgebrd(ctxt, m, n, A, lda, D, E, tauq, taup, work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }

  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgebrd")
}
hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                hipFloatComplex*  A,
                                int               lda,
                                float*            D,
                                float*            E,
                                hipFloatComplex*  tauq,
                                hipFloatComplex*  taup,
                                hipFloatComplex*  work,
                                int               lwork,
                                int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = lda;
    auto status = hipsolverCgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cgebrd(ctxt, m, n, (float _Complex*)A, lda, D, E,
                 (float _Complex*)tauq, (float _Complex*)taup, (float _Complex*)work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgebrd")
}
hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                hipDoubleComplex* A,
                                int               lda,
                                double*           D,
                                double*           E,
                                hipDoubleComplex* tauq,
                                hipDoubleComplex* taup,
                                hipDoubleComplex* work,
                                int               lwork,
                                int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = lda;
    auto status = hipsolverZgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zgebrd(ctxt, m, n, (double _Complex*)A, lda, D, E,
                 (double _Complex*)tauq, (double _Complex*)taup, (double _Complex*)work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgebrd")
}

//syevd/heevd
hipsolverStatus_t hipsolverSsyevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              float*              A,
                                              int                 lda,
                                              float*              D,
                                              int*                lwork){
  HIPSOLVER_TRY
  if (!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (lwork == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Ssyevd_ScPadSz(ctxt, convert(jobz), convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssyevd_scratchpad")
}

hipsolverStatus_t hipsolverDsyevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              double*             A,
                                              int                 lda,
                                              double*             D,
                                              int*                lwork){
  HIPSOLVER_TRY
  if (!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (lwork == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Dsyevd_ScPadSz(ctxt, convert(jobz), convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssyevd_scratchpad")
}

hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipFloatComplex*    A,
                                              int                 lda,
                                              float*              D,
                                              int*                lwork){
  HIPSOLVER_TRY
  if (!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (lwork == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Cheevd_ScPadSz(ctxt, convert(jobz), convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cheevd_scratchpad")
}

hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipDoubleComplex*   A,
                                            int                 lda,
                                            double*             D,
                                            int*                lwork){
  HIPSOLVER_TRY
  if (!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (lwork == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  auto size = H4I::MKLShim::Zheevd_ScPadSz(ctxt, convert(jobz), convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zheevd_scratchpad")
}

hipsolverStatus_t hipsolverSsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              D,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(jobz) || !isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr ){
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    auto status = hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo,0,sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Ssyevd(ctxt, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssyevd")
}

hipsolverStatus_t hipsolverDsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             D,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(jobz) || !isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr ){
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    auto status = hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo,0,sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dsyevd(ctxt, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dsyevd")
}

hipsolverStatus_t hipsolverCheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              D,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(jobz) || !isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr ){
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    auto status = hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo,0,sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cheevd(ctxt, convert(jobz), convert(uplo), n, (float _Complex*)A, lda, D, (float _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));  
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cheevd")
}

hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             D,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(jobz) || !isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr ){
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    auto status = hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo,0,sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zheevd(ctxt, convert(jobz), convert(uplo), n, (double _Complex*)A, lda, D, (double _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work)); 
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zheevd")
}

// orgtr/ungtr
hipsolverStatus_t hipsolverSorgtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            float*              A,
                                            int                 lda,
                                            float*              tau,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sorgtr_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sorgtr_scratchpad")
}

hipsolverStatus_t hipsolverDorgtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            double*             A,
                                            int                 lda,
                                            double*             tau,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dorgtr_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dorgtr_scratchpad")
}

hipsolverStatus_t hipsolverCungtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipFloatComplex*    A,
                                            int                 lda,
                                            hipFloatComplex*    tau,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cungtr_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cungtr_scratchpad")
}

hipsolverStatus_t hipsolverZungtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipDoubleComplex*   A,
                                            int                 lda,
                                            hipDoubleComplex*   tau,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
      return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
      return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
      return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zungtr_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungtr_scratchpad")
}

// orgbr/ungbr
hipsolverStatus_t hipsolverSorgbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             float*              A,
                                             int                 lda,
                                             float*              tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM; 
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sorgbr_ScPadSz(ctxt, convertToGen(side), m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sorgbr_scratchpad")
}
hipsolverStatus_t hipsolverDorgbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             double*             A,
                                             int                 lda,
                                             double*             tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM; 
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dorgbr_ScPadSz(ctxt, convertToGen(side), m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dorgbr_scratchpad")
}
hipsolverStatus_t hipsolverCungbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             hipFloatComplex*    tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM; 
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cungbr_ScPadSz(ctxt, convertToGen(side), m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cungbr_scratchpad")
}
hipsolverStatus_t hipsolverZungbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             hipDoubleComplex*   tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM; 
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zungbr_ScPadSz(ctxt, convertToGen(side), m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungbr_scratchpad")
}

// orgqr/ungqr
hipsolverStatus_t hipsolverSorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sorgqr_ScPadSz(ctxt, m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sorgqr_scratchpad")
}
hipsolverStatus_t hipsolverDorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dorgqr_ScPadSz(ctxt, m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dorgqr_scratchpad")
}
hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               k,
                                             hipFloatComplex*  A,
                                             int               lda,
                                             hipFloatComplex*  tau,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cungqr_ScPadSz(ctxt, m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cungqr_scratchpad")
}
hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               k,
                                             hipDoubleComplex* A,
                                             int               lda,
                                             hipDoubleComplex* tau,
                                             int*              lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zungqr_ScPadSz(ctxt, m, n, k, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungqr_scratchpad")
}
hipsolverStatus_t hipsolverSorgqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  float*            A,
                                  int               lda,
                                  float*            tau,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero.
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sorgqr(ctxt, m, n, k, A, lda, tau, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sorgqr")
}
hipsolverStatus_t hipsolverDorgqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  double*           A,
                                  int               lda,
                                  double*           tau,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dorgqr(ctxt, m, n, k, A, lda, tau, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dorgqr")
}
hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  tau,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cungqr(ctxt, m, n, k, (float _Complex*)A, lda, (float _Complex*)tau, (float _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cungqr")
}
hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* tau,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
      return status;
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zungqr(ctxt, m, n, k, (double _Complex*)A, lda, (double _Complex*)tau, (double _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungqr")
}

// ormqr/unmqr
hipsolverStatus_t hipsolverSormqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             float*               A,
                                             int                  lda,
                                             float*               tau,
                                             float*               C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sormqr_ScPadSz(ctxt, convert(side), convert(trans), m, n, k, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sormqr_scratchpad")
}
hipsolverStatus_t hipsolverDormqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             double*              A,
                                             int                  lda,
                                             double*              tau,
                                             double*              C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dormqr_ScPadSz(ctxt, convert(side), convert(trans), m, n, k, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dormqr_scratchpad")
}
hipsolverStatus_t hipsolverCunmqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             hipFloatComplex*     tau,
                                             hipFloatComplex*     C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cunmqr_ScPadSz(ctxt, convert(side), convert(trans), m, n, k, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cunmqr_scratchpad")
}
hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             hipDoubleComplex*    tau,
                                             hipDoubleComplex*    C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zunmqr_ScPadSz(ctxt, convert(side), convert(trans), m, n, k, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zunmqr_scratchpad")
}
hipsolverStatus_t hipsolverSormqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  float*               A,
                                  int                  lda,
                                  float*               tau,
                                  float*               C,
                                  int                  ldc,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side) || !isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = 0;
    auto status = hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sormqr(ctxt, convert(side), convert(trans), m, n, k, A, lda, tau, C, ldc, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));  
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sormqr")
}
hipsolverStatus_t hipsolverDormqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  double*              A,
                                  int                  lda,
                                  double*              tau,
                                  double*              C,
                                  int                  ldc,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side) || !isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = 0;
    auto status = hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dormqr(ctxt, convert(side), convert(trans), m, n, k, A, lda, tau, C, ldc, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));  
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dormqr")
}
hipsolverStatus_t hipsolverCunmqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  hipFloatComplex*     tau,
                                  hipFloatComplex*     C,
                                  int                  ldc,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side) || !isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = 0;
    auto status = hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cunmqr(ctxt, convert(side), convert(trans), m, n, k, (float _Complex*)A, lda, (float _Complex*)tau,
                       (float _Complex*)C, ldc, (float _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));  
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cunmqr")
}
hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  hipDoubleComplex*    tau,
                                  hipDoubleComplex*    C,
                                  int                  ldc,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(side) || !isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    lwork = 0;
    auto status = hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zunmqr(ctxt, convert(side), convert(trans), m, n, k, (double _Complex*)A, lda, (double _Complex*)tau,
                       (double _Complex*)C, ldc, (double _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work)); 
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zunmqr")
}

// ormtr/unmtr
hipsolverStatus_t hipsolverSormtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             float*               A,
                                             int                  lda,
                                             float*               tau,
                                             float*               C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sormtr_ScPadSz(ctxt, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sormtr_scratchpad")
}
hipsolverStatus_t hipsolverDormtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             double*              A,
                                             int                  lda,
                                             double*              tau,
                                             double*              C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dormtr_ScPadSz(ctxt, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dormtr_scratchpad")
}
hipsolverStatus_t hipsolverCunmtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             hipFloatComplex*     tau,
                                             hipFloatComplex*     C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cunmtr_ScPadSz(ctxt, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cunmtr_scratchpad")
}
hipsolverStatus_t hipsolverZunmtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             hipDoubleComplex*    tau,
                                             hipDoubleComplex*    C,
                                             int                  ldc,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans) || !isValid(side) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zunmtr_ScPadSz(ctxt, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zunmtr_scratchpad")
}

// geqrf
hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sgeqrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgeqrf_scratchpad")
}
hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dgeqrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgeqrf_scratchpad")
}
hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cgeqrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgeqrf_scratchpad")
}
hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zgeqrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgeqrf_scratchpad")
}
hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            tau,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
      lwork = lda;
      auto status = hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
      if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
      HIP_CHECK(hipMalloc(&work, lwork));
      allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sgeqrf(ctxt, m, n, A, lda, tau, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgeqrf")
}
hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           tau,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
      lwork = lda;
      auto status = hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
      if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
      HIP_CHECK(hipMalloc(&work, lwork));
      allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dgeqrf(ctxt, m, n, A, lda, tau, work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgeqrf")
}
hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  tau,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
      lwork = lda;
      auto status = hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
      if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
      HIP_CHECK(hipMalloc(&work, lwork));
      allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cgeqrf(ctxt, m, n, (float _Complex*)A, lda, (float _Complex*)tau, (float _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgeqrf")
}
hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* tau,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || tau == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
      lwork = lda;
      auto status = hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
      if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
      HIP_CHECK(hipMalloc(&work, lwork));
      allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zgeqrf(ctxt, m, n, (double _Complex*)A, lda, (double _Complex*)tau, (double _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgeqrf")
}

// getrf
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sgetrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgetrf_scratchpad")
}
hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dgetrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgetrf_scratchpad")
}
hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cgetrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgetrf_scratchpad")
}
hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zgetrf_ScPadSz(ctxt, m, n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgetrf_scratchpad")
}
hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    hipsolverSgetrf_bufferSize(handle, m, n, A, lda, &lwork);
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing crash as
  //     MKL's requirement is more.
  //     Note: It can have performance impact as there are extra copies and element wise copies are involved
  int64_t* local_dIpiv;
  auto no_of_elements = max(1, min(m,n));
  // Allocating it on host with device access to avoid extra copy needed while accessing it from host
  HIP_CHECK(hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sgetrf(ctxt, m, n, A, lda, local_dIpiv, work, lwork);

  int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
  for(auto i=0; i< no_of_elements; ++i){
      local_hIpiv[i] = (int)local_dIpiv[i];
  }
  // copy back the data to out param devIpiv
  HIP_CHECK(hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* no_of_elements, hipMemcpyHostToDevice));

  // release the memory allocated in the WA
  HIP_CHECK(hipFree(local_dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgetrf")
}
hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    hipsolverDgetrf_bufferSize(handle, m, n, A, lda, &lwork);
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing crash as
  //     MKL's requirement is more.
  //     Note: It can have performance impact as there are extra copies and element wise copies are involved
  int64_t* local_dIpiv;
  auto no_of_elements = max(1, min(m,n));
  // Allocating it on host with device access to avoid extra copy needed while accessing it from host
  HIP_CHECK(hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dgetrf(ctxt, m, n, A, lda, local_dIpiv, work, lwork);

  int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
  for(auto i=0; i< no_of_elements; ++i){
      local_hIpiv[i] = (int)local_dIpiv[i];
  }
  // copy back the data to out param devIpiv
  HIP_CHECK(hipMemcpy(devIpiv, local_hIpiv, sizeof(int)*no_of_elements, hipMemcpyHostToDevice));

  // release the memory allocated in the WA
  HIP_CHECK(hipFree(local_dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgetrf")
}
hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    hipsolverCgetrf_bufferSize(handle, m, n, A, lda, &lwork);
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing crash as
  //     MKL's requirement is more.
  //     Note: It can have performance impact as there are extra copies and element wise copies are involved
  int64_t* local_dIpiv;
  auto no_of_elements = max(1, min(m,n));
  // Allocating it on host with device access to avoid extra copy needed while accessing it from host
  HIP_CHECK(hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cgetrf(ctxt, m, n, (float _Complex*)A, lda, local_dIpiv, (float _Complex*)work, lwork);

  int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
  for(auto i=0; i< no_of_elements; ++i){
      local_hIpiv[i] = (int)local_dIpiv[i];
  }
  // copy back the data to out param devIpiv
  HIP_CHECK(hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* no_of_elements, hipMemcpyHostToDevice));

  // release the memory allocated in the WA
  HIP_CHECK(hipFree(local_dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgetrf")
}
hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    hipsolverZgetrf_bufferSize(handle, m, n, A, lda, &lwork);
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing crash as
  //     MKL's requirement is more.
  //     Note: It can have performance impact as there are extra copies and element wise copies are involved
  int64_t* local_dIpiv;
  auto no_of_elements = max(1, min(m,n));
  // Allocating it on host with device access to avoid extra copy needed while accessing it from host
  HIP_CHECK(hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zgetrf(ctxt, m, n, (double _Complex*)A, lda, local_dIpiv, (double _Complex*)work, lwork);

  int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
  for(auto i=0; i< no_of_elements; ++i){
      local_hIpiv[i] = (int)local_dIpiv[i];
  }
  // copy back the data to out param devIpiv
  HIP_CHECK(hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* no_of_elements, hipMemcpyHostToDevice));

  // release the memory allocated in the WA
  HIP_CHECK(hipFree(local_dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgetrf")
}

// getrs
hipsolverStatus_t hipsolverSgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             float*               A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             float*               B,
                                             int                  ldb,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Sgetrs_ScPadSz(ctxt, convert(trans), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgetrs_scratchpad")
}
hipsolverStatus_t hipsolverDgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             double*              A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             double*              B,
                                             int                  ldb,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dgetrs_ScPadSz(ctxt, convert(trans), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgetrs_scratchpad")
}
hipsolverStatus_t hipsolverCgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             hipFloatComplex*     B,
                                             int                  ldb,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cgetrs_ScPadSz(ctxt, convert(trans), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgetrs_scratchpad")
}
hipsolverStatus_t hipsolverZgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             hipDoubleComplex*    B,
                                             int                  ldb,
                                             int*                 lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if(lwork == nullptr)
    return HIPSOLVER_STATUS_INVALID_VALUE;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zgetrs_ScPadSz(ctxt, convert(trans), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgetrs_scratchpad")
}
hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  float*               A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  float*               B,
                                  int                  ldb,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || B == nullptr || devIpiv == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate =true;
  }
  // WA: MKL does not use devinfo hence setting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing result mismatch
  //     Note: It can have performance impact as there are extra copies and
  //           element wise copies are involved between Host <-> device memory
  auto no_of_elements = max(1, n);
  int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
  HIP_CHECK(hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost));

  int64_t* dIpiv;
  HIP_CHECK(hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements));

  for(auto i=0; i<no_of_elements; ++i) {
      dIpiv[i] = local_hIpiv[i];
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sgetrs(ctxt, convert(trans), n, nrhs, A, lda, dIpiv, B, ldb, work, lwork);

  HIP_CHECK(hipFree(dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sgetrs")
}
hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  double*              A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  double*              B,
                                  int                  ldb,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || B == nullptr || devIpiv == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate =true;
  }
  // WA: MKL does not use devinfo hence setting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing result mismatch
  //     Note: It can have performance impact as there are extra copies and
  //           element wise copies are involved between Host <-> device memory
  auto no_of_elements = max(1, n);
  int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
  HIP_CHECK(hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost));

  int64_t* dIpiv;
  HIP_CHECK(hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements));

  for(auto i=0; i<no_of_elements; ++i) {
      dIpiv[i] = local_hIpiv[i];
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dgetrs(ctxt, convert(trans), n, nrhs, A, lda, dIpiv, B, ldb, work, lwork);

  HIP_CHECK(hipFree(dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dgetrs")
}
hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipFloatComplex*     B,
                                  int                  ldb,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || B == nullptr || devIpiv == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate =true;
  }
  // WA: MKL does not use devinfo hence setting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing result mismatch
  //     Note: It can have performance impact as there are extra copies and
  //           element wise copies are involved between Host <-> device memory
  auto no_of_elements = max(1, n);
  int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
  HIP_CHECK(hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost));

  int64_t* dIpiv;
  HIP_CHECK(hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements));

  for(auto i=0; i<no_of_elements; ++i) {
      dIpiv[i] = local_hIpiv[i];
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cgetrs(ctxt, convert(trans), n, nrhs, (float _Complex*)A, lda, dIpiv,
                       (float _Complex*)B, ldb, (float _Complex*)work, lwork);

  HIP_CHECK(hipFree(dIpiv));
  free(local_hIpiv);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cgetrs")
}
hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipDoubleComplex*    B,
                                  int                  ldb,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(trans))
    return HIPSOLVER_STATUS_INVALID_ENUM;
  if (A == nullptr || B == nullptr || devIpiv == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0) {
    lwork = 0;
    auto status = hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate =true;
  }
  // WA: MKL does not use devinfo hence setting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
  //     hence need special handling here. Force type cast was causing result mismatch
  //     Note: It can have performance impact as there are extra copies and
  //           element wise copies are involved between Host <-> device memory
  auto no_of_elements = max(1, n);
  int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
  HIP_CHECK(hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost));

  int64_t* dIpiv;
  HIP_CHECK(hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements));

  for(auto i=0; i<no_of_elements; ++i) {
      dIpiv[i] = local_hIpiv[i];
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zgetrs(ctxt, convert(trans), n, nrhs, (double _Complex*)A, lda, dIpiv,
                       (double _Complex*)B, ldb, (double _Complex*)work, lwork);
  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zgetrs")
}

// potrf
hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Spotrf_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotrf_scratchpad")
}
hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dpotrf_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotrf_scratchpad")
}
hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cpotrf_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotrf_scratchpad")
}
hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zpotrf_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotrf_scratchpad")
}
hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Spotrf(ctxt, convert(uplo), n, A, lda, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotrf")
}
hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dpotrf(ctxt, convert(uplo), n, A, lda, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotrf")
}
hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cpotrf(ctxt, convert(uplo), n, (float _Complex*)A, lda, (float _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotrf")
}
hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zpotrf(ctxt, convert(uplo), n, (double _Complex*)A, lda, (double _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotrf")
}

// potri
hipsolverStatus_t hipsolverSpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Spotri_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotri_scratchpad")
}
hipsolverStatus_t hipsolverDpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dpotri_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotri_scratchpad")
}
hipsolverStatus_t hipsolverCpotri_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cpotri_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotri_scratchpad")
}
hipsolverStatus_t hipsolverZpotri_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zpotri_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotri_scratchpad")
}
hipsolverStatus_t hipsolverSpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Spotri(ctxt, convert(uplo), n, A, lda, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotri")
}
hipsolverStatus_t hipsolverDpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dpotri(ctxt, convert(uplo), n, A, lda, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotri")
}
hipsolverStatus_t hipsolverCpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cpotri(ctxt, convert(uplo), n, (float _Complex*)A, lda, (float _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotri")
}
hipsolverStatus_t hipsolverZpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zpotri(ctxt, convert(uplo), n, (double _Complex*)A, lda, (double _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotri")
}

// potrs
hipsolverStatus_t hipsolverSpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             float*              A,
                                             int                 lda,
                                             float*              B,
                                             int                 ldb,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Spotrs_ScPadSz(ctxt, convert(uplo), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotrs_scratchpad")
}
hipsolverStatus_t hipsolverDpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             double*             A,
                                             int                 lda,
                                             double*             B,
                                             int                 ldb,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dpotrs_ScPadSz(ctxt, convert(uplo), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotrs_scratchpad")
}
hipsolverStatus_t hipsolverCpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             hipFloatComplex*    B,
                                             int                 ldb,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Cpotrs_ScPadSz(ctxt, convert(uplo), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotrs_scratchpad")
}
hipsolverStatus_t hipsolverZpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             hipDoubleComplex*   B,
                                             int                 ldb,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zpotrs_ScPadSz(ctxt, convert(uplo), n, nrhs, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotrs_scratchpad")
}
hipsolverStatus_t hipsolverSpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  float*              A,
                                  int                 lda,
                                  float*              B,
                                  int                 ldb,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Spotrs(ctxt, convert(uplo), n, nrhs, A, lda, B, ldb, work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Spotrs")
}
hipsolverStatus_t hipsolverDpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  double*             A,
                                  int                 lda,
                                  double*             B,
                                  int                 ldb,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dpotrs(ctxt, convert(uplo), n, nrhs, A, lda, B, ldb, work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dpotrs")
}
hipsolverStatus_t hipsolverCpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    B,
                                  int                 ldb,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cpotrs(ctxt, convert(uplo), n, nrhs, (float _Complex*)A, lda, (float _Complex*)B, ldb, (float _Complex*)work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cpotrs")
}
hipsolverStatus_t hipsolverZpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   B,
                                  int                 ldb,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }
  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zpotrs(ctxt, convert(uplo), n, nrhs, (double _Complex*)A, lda, (double _Complex*)B, ldb, (double _Complex*)work, lwork);
  if (allocate){
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zpotrs")
}

// sytrd/hetrd
hipsolverStatus_t hipsolverSsytrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A,
                                             int                 lda,
                                             float*              D,
                                             float*              E,
                                             float*              tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Ssytrd_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssytrd_scratchpad")
}
hipsolverStatus_t hipsolverDsytrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A,
                                             int                 lda,
                                             double*             D,
                                             double*             E,
                                             double*             tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dsytrd_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dsytrd_scratchpad")
}
hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             float*              D,
                                             float*              E,
                                             hipFloatComplex*    tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Chetrd_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Chetrd_scratchpad")
}
hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             double*             D,
                                             double*             E,
                                             hipDoubleComplex*   tau,
                                             int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zhetrd_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zhetrd_scratchpad")
}
hipsolverStatus_t hipsolverSsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  float*              tau,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Ssytrd(ctxt, convert(uplo), n, A, lda, D, E, tau, work, lwork);
  if (allocate) {
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssytrd")
}
hipsolverStatus_t hipsolverDsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  double*             tau,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dsytrd(ctxt, convert(uplo), n, A, lda, D, E, tau, work, lwork);
  if (allocate) {
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dsytrd")
}
hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  hipFloatComplex*    tau,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Chetrd(ctxt, convert(uplo), n, (float _Complex*)A, lda, D, E, (float _Complex*)tau, (float _Complex*)work, lwork);
  if (allocate) {
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Chetrd")
}
hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  hipDoubleComplex*   tau,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zhetrd(ctxt, convert(uplo), n, (double _Complex*)A, lda, D, E, (double _Complex*)tau, (double _Complex*)work, lwork);
  if (allocate) {
    HIP_CHECK(hipFree(work));
  }
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zhetrd")
}

// sygvd/hegvd
hipsolverStatus_t hipsolverSsygvd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigType_t  itype,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            float*              A,
                                            int                 lda,
                                            float*              B,
                                            int                 ldb,
                                            float*              W,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo) || !isValid(jobz) || !isValid(itype))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Ssygvd_ScPadSz(ctxt, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssytrd_scratchpad")
}
hipsolverStatus_t hipsolverDsygvd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigType_t  itype,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            double*             A,
                                            int                 lda,
                                            double*             B,
                                            int                 ldb,
                                            double*             W,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo) || !isValid(jobz) || !isValid(itype))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Dsygvd_ScPadSz(ctxt, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dsytrd_scratchpad")
}
hipsolverStatus_t hipsolverChegvd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigType_t  itype,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipFloatComplex*    A,
                                            int                 lda,
                                            hipFloatComplex*    B,
                                            int                 ldb,
                                            float*              W,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo) || !isValid(jobz) || !isValid(itype))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Chegvd_ScPadSz(ctxt, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Chetrd_scratchpad")
}
hipsolverStatus_t hipsolverZhegvd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigType_t  itype,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipDoubleComplex*   A,
                                            int                 lda,
                                            hipDoubleComplex*   B,
                                            int                 ldb,
                                            double*             W,
                                            int*                lwork){
  HIPSOLVER_TRY
  if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
  if (!isValid(uplo) || !isValid(jobz) || !isValid(itype))
  {
    return HIPSOLVER_STATUS_INVALID_ENUM;
  }
  if (lwork == nullptr ) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zhegvd_ScPadSz(ctxt, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zhetrd_scratchpad")
}
hipsolverStatus_t hipsolverSsygvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              B,
                                  int                 ldb,
                                  float*              W,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(itype) || !isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Ssygvd(ctxt, convert(itype), convert(jobz), convert(uplo), n, A, lda, B, ldb, W, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Ssygvd")
}
hipsolverStatus_t hipsolverDsygvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*              A,
                                  int                 lda,
                                  double*              B,
                                  int                 ldb,
                                  double*              W,
                                  double*              work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(itype) || !isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dsygvd(ctxt, convert(itype), convert(jobz), convert(uplo), n, A, lda, B, ldb, W, work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dsygvd")
}
hipsolverStatus_t hipsolverChegvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    B,
                                  int                 ldb,
                                  float*              W,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(itype) || !isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Chegvd(ctxt, convert(itype), convert(jobz), convert(uplo), n, (float _Complex*)A, lda,
                      (float _Complex*)B, ldb, W, (float _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Chegvd")
}
hipsolverStatus_t hipsolverZhegvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   B,
                                  int                 ldb,
                                  double*             W,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo){
  HIPSOLVER_TRY
  if(!handle)
    return HIPSOLVER_STATUS_NOT_INITIALIZED;

  if (!isValid(itype) || !isValid(jobz) || !isValid(uplo))
    return HIPSOLVER_STATUS_INVALID_ENUM;

  if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }

  bool allocate = false;
  if (work == nullptr || lwork == 0){
    auto status = hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return status;
    }
    HIP_CHECK(hipMalloc(&work, lwork));
    allocate = true;
  }

  // WA: MKL does not use devInfo hence resetting it to zero
  HIP_CHECK(hipMemset(devInfo, 0, sizeof(int)));

  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zhegvd(ctxt, convert(itype), convert(jobz), convert(uplo), n, (double _Complex*)A, lda,
                      (double _Complex*)B, ldb, W, (double _Complex*)work, lwork);

  if (allocate)
    HIP_CHECK(hipFree(work));
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zhegvd")
}