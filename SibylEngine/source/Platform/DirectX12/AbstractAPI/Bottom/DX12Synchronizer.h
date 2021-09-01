#pragma once
#include "Sibyl/Graphic/AbstractAPI/Bottom/Synchronizer.h"

namespace SIByL
{
	class DX12Synchronizer : public Synchronizer
	{
	public:
		DX12Synchronizer();
		virtual void ForceSynchronize() override;
		virtual void StartFrame() override;
		virtual bool CheckFinish(UINT64 fence) override;
		virtual void EndFrame() override;

	private:
		void ForceSynchronize(UINT64 fence);

	private:
		ComPtr<ID3D12Fence> m_GpuFence;
		UINT64 m_CpuCurrentFence = 0;
	};
}