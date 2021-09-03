#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12UploadBuffer.h"
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
		~DX12FrameResource()
		{
			for (int i = 0; i < m_FrameResourcesCount; i++)
				m_CpuBuffers[i] = nullptr;
			delete m_CpuBuffers;

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

	class DX12FrameResourceBuffer
	{
	public:
		friend class DX12FrameResourcesManager;
		DX12FrameResourceBuffer(uint32_t size);
		~DX12FrameResourceBuffer()
		{
			delete m_CpuBuffer;
		}

		void CopyMemoryToConstantsBuffer
			(void* data, uint32_t offset, uint32_t length)
		{
			void* target = (void*)((char*)m_CpuBuffer + offset);
			memcpy((target)
				, data
				, length);
		}

		void* GetCurrentBuffer();
		void UploadCurrentBuffer();
		D3D12_GPU_VIRTUAL_ADDRESS GetCurrentGPUAddress();

		uint32_t GetSizeInByte() { return m_SizeByte; }
	private:
		Ref<DX12UploadBuffer::Allocation> m_GpuBuffers;
		void* m_CpuBuffer;
		uint32_t m_SizeByte;

	private:
		void UploadBufferToGpu(int index)
		{
			memcpy(m_GpuBuffers.get()[index].CPU
				, (m_CpuBuffer)
				, m_SizeByte);
		}
	};

	class DX12FrameResourcesManager
	{
	public:
		DX12FrameResourcesManager(int frameResourcesCount = 3);
		~DX12FrameResourcesManager();
		static DX12FrameResourcesManager* Get();
		static void UseNextFrameResource() { m_CurrentFrameIndex = (m_CurrentFrameIndex + 1) % m_FrameResourcesCount; }
		static int GetCurrentIndex() { return m_CurrentFrameIndex; }
		static UINT64 GetCurrentFence() { return m_Fence[m_CurrentFrameIndex]; }
		static void SetCurrentFence(UINT64 cpuFence);
		static ComPtr<ID3D12CommandAllocator> GetCurrentAllocator() { return m_CommandAllocators[m_CurrentFrameIndex]; }

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

		void RegistFrameResource(DX12FrameResourceBuffer* frameResource)
		{
			frameResource->m_CpuBuffer = new byte[frameResource->GetSizeInByte()];
			frameResource->m_GpuBuffers.reset(new DX12UploadBuffer::Allocation[m_FrameResourcesCount]);
			for (int i = 0; i < m_FrameResourcesCount; i++)
			{
				// Allocate Upload Buffer for resource
				frameResource->m_GpuBuffers.get()[i] =
					m_UploadBuffers[i]->Allocate(frameResource->GetSizeInByte(), true);
			}
		}

	private:
		Ref<DX12UploadBuffer>* m_UploadBuffers;
		static int m_FrameResourcesCount;
		static int m_CurrentFrameIndex;
		static DX12FrameResourcesManager* m_Instance;
		static UINT64* m_Fence;
		static ComPtr<ID3D12CommandAllocator>* m_CommandAllocators;
	};
}