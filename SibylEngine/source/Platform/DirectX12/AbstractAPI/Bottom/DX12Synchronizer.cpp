#include "SIByLpch.h"
#include "DX12Synchronizer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "DX12FrameResources.h"

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
		if (!CheckFinish(m_CpuCurrentFence))
		{
			ForceSynchronize(m_CpuCurrentFence);
		}
	}

	void DX12Synchronizer::ForceSynchronize(UINT64 fence)
	{
		// Create an event, and stop cpu thread until the gpu catch up
		HANDLE eventHandle = CreateEvent(nullptr, false, false, L"FenceSetDone");
		m_GpuFence->SetEventOnCompletion(fence, eventHandle);
		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}

	bool DX12Synchronizer::CheckFinish(UINT64 fence)
	{
		if (m_GpuFence->GetCompletedValue() < fence)
			return false;
		else
			return true;
	}

	void DX12Synchronizer::StartFrame()
	{
		DX12FrameResourcesManager::UseNextFrameResource();
		UINT64 fence = DX12FrameResourcesManager::GetCurrentFence();
		// If GPU did not finish all the commands
		if (!CheckFinish(fence))
		{
			ForceSynchronize(fence);
		}
	}

	void DX12Synchronizer::EndFrame()
	{
		// Update CPU Fence
		m_CpuCurrentFence++;
		// Send current cpu fence to gpu
		DX12Context::GetCommandQueue()->Signal(m_GpuFence.Get(), m_CpuCurrentFence);
		DX12FrameResourcesManager::SetCurrentFence(m_CpuCurrentFence);
	}
}