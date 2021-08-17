#include "SIByLpch.h"
#include "DX12Synchronizer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

namespace SIByL
{
	DX12Synchronizer::DX12Synchronizer()
	{
		DXCall(DX12Context::GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_GpuFence)));
	}

	void DX12Synchronizer::ForceSynchronize()
	{
		m_CpuCurrentFence++;
		// Send current cpu fence to gpu
		DX12Context::GetCommandQueue()->Signal(m_GpuFence.Get(), m_CpuCurrentFence);
		// If GPU did not finish all the commands
		if (m_GpuFence->GetCompletedValue() < m_CpuCurrentFence)
		{
			// Create an event, and stop cpu thread until the gpu catch up
			HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");
			m_GpuFence->SetEventOnCompletion(m_CpuCurrentFence, eventHandle);
			WaitForSingleObject(eventHandle, INFINITE);
			CloseHandle(eventHandle);
		}
	}
}