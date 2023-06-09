#include "hipsolver.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle){
  if(handle != nullptr)
  {
    // HIP supports mutile backends hence query current backend name
    auto backendName = hipGetBackendName();
    // Obtain the handles to the back handlers.
    unsigned long handles[4];
    int           nHandles = 4;
    hipGetBackendNativeHandles((uintptr_t)NULL, handles, &nHandles);
    *handle = H4I::MKLShim::Create(handles, nHandles, backendName);
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
              nativeHandles.data(), &nHandles);

      H4I::MKLShim::SetStream(ctxt, nativeHandles);
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}
