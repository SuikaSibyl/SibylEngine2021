#pragma once
#include "SIByLpch.h"

/**
 * Macro defines.
 */

#define _KB(x) (x * 1024)
#define _MB(x) (x * 1024 * 1024)

#define _64KB _KB(64)
#define _1MB _MB(1)
#define _2MB _MB(2)
#define _4MB _MB(4)
#define _8MB _MB(8)
#define _16MB _MB(16)
#define _32MB _MB(32)
#define _64MB _MB(64)
#define _128MB _MB(128)
#define _256MB _MB(256)

namespace SIByL
{
	class DX12UploadBuffer
	{
	public:
		// Bind to upload data to the GPU
		struct Allocation
		{
			void* CPU;
			D3D12_GPU_VIRTUAL_ADDRESS GPU;
			ID3D12Resource* Page;
			size_t Offset;
		};
		/**
		 * @param pageSize The size to use to allocate new pages in GPU memory.
		 */
		explicit DX12UploadBuffer(size_t pageSize = _2MB);
		size_t GetPageSize() const { return m_PageSize; }
		Allocation Allocate(size_t sizeInBytes, size_t alignment);
		/**
		 * Release all allocated pages. This should only be done when the command list
		 * is finished executing on the Command Queue.
		 */
		void Reset();
	private:
		// A single page for the allocator.
		struct Page
		{
			Page(size_t sizeInBytes);
			~Page();
			// Check to see if the page has room to satisfy the requested
			// allocation.
			bool HasSpace(size_t sizeInBytes, size_t alignment) const;
			// Allocate memory from the page.
			// Throws std::bad_alloc if the the allocation size is larger
			// that the page size or the size of the allocation exceeds the 
			// remaining space in the page.
			Allocation Allocate(size_t sizeInBytes, size_t alignment);
			// Reset the page for reuse.
			void Reset();
		private:
			// Store the GPU Memory for the page
			ComPtr<ID3D12Resource> m_d3d12Resource;
			// CPU Based pointer.
			void* m_CPUPtr;
			// GPU Based pointer.
			D3D12_GPU_VIRTUAL_ADDRESS m_GPUPtr;
			// Allocated page size.
			size_t m_PageSize;
			// Current allocation offset in bytes.
			size_t m_Offset;
		};
		// A pool of memory pages.
		using PagePool = std::deque< std::shared_ptr<Page> >;
		// Request a page from the pool of available pages
		// or create a new page if there are no available pages.
		std::shared_ptr<Page> RequestPage();
		// Hold all of the pages that have ever been created
		PagePool m_PagePool;
		// A pool of pages that are available for allocation
		PagePool m_AvailablePages;
		// store a pointer to the current memory page
		std::shared_ptr<Page> m_CurrentPage;
		// The size of each page of memory.
		size_t m_PageSize;
	};
}