#include <iostream>
#include <hipsolver.h>
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklsolver.h"

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
