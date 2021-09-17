#pragma once

namespace SIByL
{
	class CudaContext
	{
	public:
		static void Init();


		struct CudaDeviceInfo
		{
			int DeviceCount = 0;
		};

		static CudaDeviceInfo s_DeviceInfo;
	};
}