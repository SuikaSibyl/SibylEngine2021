#pragma once
#include "SIByLpch.h"
#include "Platform/DirectX12/Core/DescriptorAllocation.h"

// DescriptorAllocator class is used to allocate (CPU visible) descriptors.
// CPU visible descriptors are used to create views for resources
// (for example Render Target Views (RTV), Depth-Stencil Views (DSV), Constant Buffer Views (CBV),
// Shader Resource Views (SRV), Unordered Access Views (UAV), and Samplers).
// Before a CBV, SRV, UAV, or Sampler can be used in a shader,
// it must be copied to a GPU visible descriptor.

// CPU visible descriptors are used for describing:
// - Render Target Views(RTV)
// - Depth - Stencil Views(DSV)
// - Constant Buffer Views(CBV)
// - Shader Resource Views(SRV)
// - Unordered Access Views(UAV)
// - Samplers

// The DescriptorAllocator class is used to allocate descriptors to the application
// when loading new resources (like textures).
// In a typical game engine, resources may need to be loaded and unloaded from memory
// at sporadic moments while the player moves around the level.
// To support large dynamic worlds, it may be necessary to initially load some resources,
// unload them from memory, and reload different resources.
// The DescriptorAllocator manages all of the descriptors that are required to describe those resources.
// Descriptors that are no longer used (for example, when a resource is unloaded from memory)
// will be automatically returned back to the descriptor heap for reuse.

// DescriptorAllocator: This is the main interface to the application for requesting descriptors.
// The DescriptorAllocator class manages the descriptor pages.

namespace SIByL
{
	class DescriptorAllocatorPage;

	class DescriptorAllocator
	{
	public:
		DescriptorAllocator(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptorsPerHeap = 256);
		virtual ~DescriptorAllocator();

		// DescriptorAllocator::Allocate:
		// Allocates a number of contiguous descriptors from a CPU visible descriptor heap.
		DescriptorAllocation Allocate(uint32_t numDescriptors = 1);

		// DescriptorAllocator::ReleaseStaleDescriptors:
		// Frees any stale descriptors that can be returned to the list of available descriptors for reuse.
		// This method should only be called after any of the descriptors that were freed
		// are no longer being referenced by the command queue.
		void ReleaseStaleDescriptors(uint64_t frameNumber);

	private:
		using DescriptorHeapPool = std::vector< std::shared_ptr<DescriptorAllocatorPage> >;

		// Create a new heap with a specific number of descriptors.
		std::shared_ptr<DescriptorAllocatorPage> CreateAllocatorPage();

		D3D12_DESCRIPTOR_HEAP_TYPE m_HeapType;
		uint32_t m_NumDescriptorsPerHeap;

		DescriptorHeapPool m_HeapPool;
		// Indices of available heaps in the heap pool.
		std::set<size_t> m_AvailableHeaps;

		std::mutex m_AllocationMutex;
	}; 

}

// DescriptorAllocatorPage: This class is a wrapper for a ID3D12DescriptorHeap.
// The DescriptorAllocatorPage also keeps track of the free list for the page.