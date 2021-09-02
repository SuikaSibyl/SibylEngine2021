#pragma once
#include "SIByLpch.h"

// DX12DescriptorAllocation: This class wraps an allocation that is returned from the
// DescriptorAllocator::Allocate method.
// The DX12DescriptorAllocation class also stores a pointer back to the page it came from
// and will automatically free itself if the descriptor(s) are no longer required.

namespace SIByL
{
	class DX12DescriptorAllocatorPage;

	class DX12DescriptorAllocation
	{
	public:
		// Creates a NULL descriptor.
		DX12DescriptorAllocation();

		DX12DescriptorAllocation(
			D3D12_CPU_DESCRIPTOR_HANDLE descriptor, 
			D3D12_GPU_DESCRIPTOR_HANDLE descriptorGpu,
			uint32_t numHandles, uint32_t descriptorSize, std::shared_ptr<DX12DescriptorAllocatorPage> page,
			bool gpuVisible);

		// The destructor will automatically free the allocation.
		~DX12DescriptorAllocation();

		// Copies are not allowed.
		DX12DescriptorAllocation(const DX12DescriptorAllocation&) = delete;
		DX12DescriptorAllocation& operator=(const DX12DescriptorAllocation&) = delete;

		// Move is allowed.
		DX12DescriptorAllocation(DX12DescriptorAllocation&& allocation);
		DX12DescriptorAllocation& operator=(DX12DescriptorAllocation&& other);

		// Check if this a valid descriptor.
		bool IsNull() const;

		// Get a descriptor at a particular offset in the allocation.
		D3D12_CPU_DESCRIPTOR_HANDLE GetDescriptorHandle(uint32_t offset = 0) const;
		D3D12_GPU_DESCRIPTOR_HANDLE GetDescriptorHandleGpu(uint32_t offset = 0) const;

		// Get the number of (consecutive) handles for this allocation.
		uint32_t GetNumHandles() const;

		// Get the heap that this allocation came from.
		// (For internal use only).
		std::shared_ptr<DX12DescriptorAllocatorPage> GetDescriptorAllocatorPage() const;

	private:
		// Free the descriptor back to the heap it came from.
		void Free();

		// The base descriptor.
		D3D12_CPU_DESCRIPTOR_HANDLE m_Descriptor;
		D3D12_GPU_DESCRIPTOR_HANDLE m_DescriptorGpu;
		// The number of descriptors in this allocation.
		uint32_t m_NumHandles;
		// The offset to the next descriptor.
		uint32_t m_DescriptorSize;

		// A pointer back to the original page where this allocation came from.
		std::shared_ptr<DX12DescriptorAllocatorPage> m_Page;
	};
}