#include "SIByLpch.h"
#include "DX12FrameResources.h"

namespace SIByL
{
	DX12FrameResourcesManager* DX12FrameResourcesManager::m_Instance;
	int DX12FrameResourcesManager::m_FrameResourcesCount;
	int DX12FrameResourcesManager::m_CurrentFrameIndex;

	DX12FrameResourcesManager::
		DX12FrameResourcesManager(int frameResourcesCount)
	{
		SIByL_CORE_ASSERT(!m_Instance, "DX12FrameResourcesManager Instance Redefinition");
		m_Instance = this;
		m_FrameResourcesCount = frameResourcesCount;
		// Create Upload Buffers for All Frame Resources
		m_UploadBuffers = new Ref<DX12UploadBuffer>[frameResourcesCount];
		for (int i = 0; i < frameResourcesCount; i++)
		{
			m_UploadBuffers[i].reset(new DX12UploadBuffer);
		}
	}

	DX12FrameResourcesManager* DX12FrameResourcesManager::Get()
	{
		return m_Instance;
	}

	DX12FrameResourcesManager::~DX12FrameResourcesManager()
	{
		m_UploadBuffers = nullptr;
	}
}