#include "hipsolver.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle){
  if(handle != nullptr)
  {
    #ifndef hipGetBackendName    
      // Obtain the handles to the back handlers.
      int nHandles;
      hipGetBackendNativeHandles((uintptr_t)NULL, 0, &nHandles);
      unsigned long handles[nHandles];
      hipGetBackendNativeHandles((uintptr_t)NULL, handles, 0);
      char* backendName = (char*)handles[0];
      // New implementation of hipGetBackendNativeHandles keep backend name in the Native handles
      // Removing backend name from the list to make it sync to older native handle. This will help Shim layer remains unchanged
      for(auto i=1; i<nHandles; ++i) {
          handles[i-1] = handles[i];
      }
      handles[nHandles-1] = 0;
      nHandles--;
      *handle = H4I::MKLShim::Create(handles, nHandles, backendName);
    #else 
      // HIP supports mutile backends hence query current backend name
      auto backendName = hipGetBackendName();
      // Obtain the handles to the back handlers.
      unsigned long handles[4];
      int           nHandles = 4;
      hipGetBackendNativeHandles((uintptr_t)NULL, handles, &nHandles);
      *handle = H4I::MKLShim::Create(handles, nHandles, backendName);
    #endif
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
    //todo: add return error check
#ifndef hipGetBackendName
    //this is a context with the native handles and the NULL stream
    H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
      // Obtain the backendnativehandles for the stream we want to use
      int nHandles;
      hipGetBackendNativeHandles( reinterpret_cast<uintptr_t>(stream), 0, &nHandles);
      unsigned long handles[nHandles];
      hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream),
              handles, 0);
      char* backendName = (char*)handles[0];
      // New implementation of hipGetBackendNativeHandles keep backend name in the Native handles
      // Removing backend name from the list to make it sync to older native handle. This will help Shim layer remains unchanged
      for(auto i=1; i<nHandles; ++i) {
          handles[i-1] = handles[i];
      }
      handles[nHandles-1] = 0;
      nHandles--;

      //set ctext to use handles from stream
      //      H4I::MKLShim::Update(ctxt, handles, nHandles, backendName);
#else
      H4I::MKLShim::Context* ctxt = static_cast<H4I::MKLShim::Context*>(handle);
      // HIP supports mutile backends hence query current backend name
      auto backendName = hipGetBackendName();
      // Obtain the handles to the back handlers.
      unsigned long handles[4];
      int           nHandles = 4;
      hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), handles, &nHandles);
      //      H4I::MKLShim::Update(ctxt,handles, nHandles, backendName);

#endif      
      H4I::MKLShim::SetStream(ctxt, handles, nHandles, backendName);

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
