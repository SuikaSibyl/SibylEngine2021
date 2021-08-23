#pragma once
#include "Sibyl/Renderer/Synchronizer.h"

namespace SIByL
{
	class DX12Synchronizer : public Synchronizer
	{
	public:
		DX12Synchronizer();
		virtual void ForceSynchronize() override;

	private:
		ComPtr<ID3D12Fence> m_GpuFence;
		int m_CpuCurrentFence = 0;
	};
}