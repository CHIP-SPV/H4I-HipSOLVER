#include "hipsolver.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/hiputils/HandleManager.h"

static H4I::HIPUtils::HandleManager<hipsolverHandle_t,
                                    hipsolverStatus_t,
                                    HIPSOLVER_STATUS_SUCCESS,
                                    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR> hmgr;

hipsolverStatus_t
hipsolverCreate(hipsolverHandle_t* handle)
{
    return hmgr.Create(handle);
}

hipsolverStatus_t
hipsolverDestroy(hipsolverHandle_t handle)
{
    return hmgr.Destroy(handle);
}

hipsolverStatus_t
hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
    return hmgr.SetStream(handle, stream);
}

hipsolverStatus_t
hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* stream)
{
    return hmgr.GetStream(handle, stream);
}

