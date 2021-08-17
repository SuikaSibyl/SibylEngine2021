#pragma once

namespace SIByL
{
	class DX12Synchronizer
	{
	public:
		DX12Synchronizer();
		void ForceSynchronize();

	private:
		ComPtr<ID3D12Fence> m_GpuFence;
		int m_CpuCurrentFence = 0;
	};
}