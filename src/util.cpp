#include "hipsolver.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle){
  if(handle != nullptr)
  {
    #ifdef hipGetBackendName
    std::cerr << "Error: The hipGetBackendName API is deprecated. Please update your H4I-MKLShim to use the latest API." << std::endl;
    return HIPSOLVER_STATUS_INTERNAL_ERROR;
    #endif

    // Get native handles
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
    #ifdef hipGetBackendName
    std::cerr << "Error: The hipGetBackendName API is deprecated. Please update your H4I-MKLShim to use the latest API." << std::endl;
    return HIPSOLVER_STATUS_INTERNAL_ERROR;
    #endif

    H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);

    // Get native handles from stream
    int nHandles;
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), 0, &nHandles);
    unsigned long handles[nHandles];
    hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), handles, 0);
    // Backend name is already at index BACKEND_NAME (0), no need to modify handles array
    H4I::MKLShim::SetStream(ctxt, handles, nHandles);
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

// helpers
hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle)
{
    return hipsolverCreate(handle);
}

hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle)
{
    return hipsolverDestroy(handle);
}

hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle, hipStream_t streamId)
{
    return hipsolverSetStream(handle, streamId);
}
