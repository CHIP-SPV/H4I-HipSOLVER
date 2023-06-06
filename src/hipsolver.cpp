#include <iostream>
#include <hipsolver.h>
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklsolver.h"
#include "h4i/mklshim/types.h"

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

inline H4I::MKLShim::onemklJob convert(hipsolverEigMode_t job) {
  switch(job) {
    case HIPSOLVER_EIG_MODE_NOVECTOR: return H4I::MKLShim::ONEMKL_JOB_NOVEC;
    case HIPSOLVER_EIG_MODE_VECTOR: return H4I::MKLShim::ONEMKL_JOB_VEC;
  }
}

inline H4I::MKLShim::onemklUplo convert(hipsolverFillMode_t val) {
    switch(val) {
        case HIPSOLVER_FILL_MODE_UPPER:
            return H4I::MKLShim::ONEMKL_UPLO_UPPER;
        case HIPSOLVER_FILL_MODE_LOWER:
            return H4I::MKLShim::ONEMKL_UPLO_LOWER;
    }
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

// gebrd
hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork){
  HIPSOLVER_TRY
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
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = lda;
  auto status = hipsolverSgebrd_bufferSize(handle, m, n, &lwork);
  if (status != HIPSOLVER_STATUS_SUCCESS)
    return status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sgebrd(ctxt, m, n, A, lda, D, E, tauq, taup, work, lwork);
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
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = lda;
  auto status = hipsolverDgebrd_bufferSize(handle, m, n, &lwork);
  if (status != HIPSOLVER_STATUS_SUCCESS)
    return status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dgebrd(ctxt, m, n, A, lda, D, E, tauq, taup, work, lwork);
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
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = lda;
  auto status = hipsolverCgebrd_bufferSize(handle, m, n, &lwork);
  if (status != HIPSOLVER_STATUS_SUCCESS)
    return status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cgebrd(ctxt, m, n, (float _Complex*)A, lda, D, E,
                 (float _Complex*)tauq, (float _Complex*)taup, (float _Complex*)work, lwork);
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
  if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = lda;
  auto status = hipsolverZgebrd_bufferSize(handle, m, n, &lwork);
  if (status != HIPSOLVER_STATUS_SUCCESS)
    return status;
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zgebrd(ctxt, m, n, (double _Complex*)A, lda, D, E,
                 (double _Complex*)tauq, (double _Complex*)taup, (double _Complex*)work, lwork);
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
  if (A == nullptr || D == nullptr || lwork == nullptr) {
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
  if (A == nullptr || D == nullptr || lwork == nullptr) {
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
  if (A == nullptr || D == nullptr || lwork == nullptr) {
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
  if (A == nullptr || D == nullptr || lwork == nullptr) {
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
  if (A == nullptr || D == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Ssyevd(ctxt, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
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
  if (A == nullptr || D == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dsyevd(ctxt, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
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
  if (A == nullptr || D == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cheevd(ctxt, convert(jobz), convert(uplo), n, (float _Complex*)A, lda, D, (float _Complex*)work, lwork);
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
  if (A == nullptr || D == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zheevd(ctxt, convert(jobz), convert(uplo), n, (double _Complex*)A, lda, D, (double _Complex*)work, lwork);
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
  if (A == nullptr || tau == nullptr || lwork == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
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
  if (A == nullptr || tau == nullptr || lwork == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
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
  if (A == nullptr || tau == nullptr || lwork == nullptr) {
    return HIPSOLVER_STATUS_INVALID_VALUE;
  }
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
  if (A == nullptr || tau == nullptr || lwork == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

  auto size = H4I::MKLShim::Zungtr_ScPadSz(ctxt, convert(uplo), n, lda);
  *lwork = (int)size;
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungtr_scratchpad")
}

hipsolverStatus_t hipsolverSorgtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                float*              A,
                                int                 lda,
                                float*              tau,
                                float*              work,
                                int                 lwork,
                                int*                devInfo){
  HIPSOLVER_TRY
  if (A == nullptr || tau == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = 0;
  hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Sorgtr(ctxt, convert(uplo), n, A, lda, tau, work, lwork);
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Sorgtr")
}

hipsolverStatus_t hipsolverDorgtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                double*             A,
                                int                 lda,
                                double*             tau,
                                double*             work,
                                int                 lwork,
                                int*                devInfo){
  HIPSOLVER_TRY
  if (A == nullptr || tau == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = 0;
  hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Dorgtr(ctxt, convert(uplo), n, A, lda, tau, work, lwork);
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Dorgtr")
}

hipsolverStatus_t hipsolverCungtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                hipFloatComplex*    A,
                                int                 lda,
                                hipFloatComplex*    tau,
                                hipFloatComplex*    work,
                                int                 lwork,
                                int*                devInfo){
  HIPSOLVER_TRY
  if (A == nullptr || tau == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = 0;
  hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Cungtr(ctxt, convert(uplo), n, (float _Complex*)A, lda,
                (float _Complex*)tau, (float _Complex*)work, lwork);
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Cungtr")
}

hipsolverStatus_t hipsolverZungtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                hipDoubleComplex*   A,
                                int                 lda,
                                hipDoubleComplex*   tau,
                                hipDoubleComplex*   work,
                                int                 lwork,
                                int*                devInfo){
  HIPSOLVER_TRY
  if (A == nullptr || tau == nullptr || work == nullptr) {
      return HIPSOLVER_STATUS_INVALID_VALUE;
  }
  lwork = 0;
  hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
  auto* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
  H4I::MKLShim::Zungtr(ctxt, convert(uplo), n, (double _Complex*)A, lda,
                (double _Complex*)tau, (double _Complex*)work, lwork);
  return HIPSOLVER_STATUS_SUCCESS;
  HIPSOLVER_CATCH("Zungtr")
}

