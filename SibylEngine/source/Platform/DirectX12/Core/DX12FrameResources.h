#pragma once

#include "Sibyl/Renderer/ShaderBinder.h"
#include "Platform/DirectX12/Core/UploadBuffer.h"
#include <iostream>
#include <memory>

namespace SIByL
{
	class DX12FrameResourcesManager;

	template<typename T>
	class DX12FrameResource
	{
	public:
		friend class DX12FrameResourcesManager;
		DX12FrameResource()
		{
			DX12FrameResourcesManager::Get()->RegistFrameResource(this);
		}

		Ref<T> GetCurrentBuffer() { return m_CpuBuffers[DX12FrameResourcesManager::GetCurrentIndex()]; }
		void UploadCurrentBuffer() { return UploadBufferToGpu(DX12FrameResourcesManager::GetCurrentIndex()); }
		D3D12_GPU_VIRTUAL_ADDRESS GetCurrentGPUAddress() 
			{ return m_GpuBuffers.get()[DX12FrameResourcesManager::GetCurrentIndex()].GPU; }

	private:
		Ref<DX12UploadBuffer::Allocation> m_GpuBuffers;
		Ref<T>* m_CpuBuffers;

	private:
		void UploadBufferToGpu(int index)
		{
			memcpy(m_GpuBuffers.get()[index].CPU
				, m_CpuBuffers[index].get()
				, sizeof(T));
		}
	};

	class DX12FrameResourcesManager
	{
	public:
		DX12FrameResourcesManager(int frameResourcesCount = 3);
		~DX12FrameResourcesManager();
		static DX12FrameResourcesManager* Get();
		static int GetCurrentIndex() { return m_CurrentFrameIndex; }

		template<typename T>
		void RegistFrameResource(DX12FrameResource<T>* frameResource)
		{
			frameResource->m_CpuBuffers = new Ref<T>[m_FrameResourcesCount];
			frameResource->m_GpuBuffers.reset(new DX12UploadBuffer::Allocation[m_FrameResourcesCount]);
			for (int i = 0; i < m_FrameResourcesCount; i++)
			{
				frameResource->m_CpuBuffers[i].reset(new T);

				// Allocate Upload Buffer for resource
				frameResource->m_GpuBuffers.get()[i] =
					m_UploadBuffers[i]->Allocate(sizeof(T), true);
			}
		}

	private:
		Ref<DX12UploadBuffer>* m_UploadBuffers;
		static int m_FrameResourcesCount;
		static int m_CurrentFrameIndex;
		static DX12FrameResourcesManager* m_Instance;
	};
}