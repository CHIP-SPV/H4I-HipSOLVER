#include "hipsolver.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle){
  if(handle != nullptr)
  {
    // Obtain the handles to the back handlers.
    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)NULL, 0, &nHandles);
    unsigned long handles[nHandles];
    hipGetBackendNativeHandles((uintptr_t)NULL, handles, 0);
    *handle = H4I::MKLShim::Create(handles, nHandles);
  }
  return (*handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle){
  if(handle != nullptr)
  {
    H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
    H4I::MKLShim::Destroy(ctxt);
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle,
                                     hipStream_t       stream) {
  if(handle != nullptr)
  {
      H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

      int nHandles = H4I::MKLShim::nHandles;
      std::array<uintptr_t, H4I::MKLShim::nHandles> nativeHandles;
      //todo: add return error check
      hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream),
              nativeHandles.data(), 0);

      H4I::MKLShim::SetStream(ctxt, nativeHandles);
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}
