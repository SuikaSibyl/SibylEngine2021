#include "CudaModulePCH.h"
#include "CudaContext.h"

namespace SIByL
{
	CudaContext::CudaDeviceInfo CudaContext::s_DeviceInfo;

	void CudaContext::Init()
	{
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceCount returned %d\n-> %s\n",
                static_cast<int>(error_id), cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
            printf("There are no available device(s) that support CUDA\n");
        }
        else {
            printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        }
	}


}