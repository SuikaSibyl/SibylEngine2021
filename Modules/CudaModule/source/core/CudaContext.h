#pragma once

class cudaDeviceProp;

namespace SIByL
{
	class CudaContext
	{
	public:
		static void Init();



		struct CudaDevicesInfo
		{
			int DeviceCount = 0;
			std::vector<cudaDeviceProp> Devices;
		};

		static CudaDevicesInfo s_DeviceInfo;

	private:
		static void DeviceQuery();
		static bool CUDASupport;
	};
}