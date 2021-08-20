#include "SIByLpch.h"
#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "UploadBuffer.h"

namespace SIByL
{
	DX12UploadBuffer::DX12UploadBuffer(size_t pageSize)
		: m_PageSize(pageSize)
	{}

    // UPLOADBUFFER::ALLOCATE
    // The Allocate method is used to allocate a chunk(or block) of memory from a memory page.
    // This method returns an DX12UploadBuffer::Allocation struct that was defined in the header file.
    DX12UploadBuffer::Allocation DX12UploadBuffer::Allocate(size_t sizeInBytes, size_t alignment)
    {
        if (sizeInBytes > m_PageSize)
        {
            throw std::bad_alloc();
        }

        // If there is no current page, or the requested allocation exceeds the
        // remaining space in the current page, request a new page.
        if (!m_CurrentPage || !m_CurrentPage->HasSpace(sizeInBytes, alignment))
        {
            m_CurrentPage = RequestPage();
        }

        return m_CurrentPage->Allocate(sizeInBytes, alignment);
    }

    // UPLOADBUFFER::REQUESTPAGE
    // If either the allocator does not have a page to make an allocation from, 
    // or the current page does not have the available space to satisfy the allocation request, 
    // a new page must be retrieved from the list of available pages or a new page must be created.
    // The RequestPage method will return a memory page that can be used to 
    // satisfy allocation requests.
    std::shared_ptr<DX12UploadBuffer::Page> DX12UploadBuffer::RequestPage()
    {
        std::shared_ptr<Page> page;

        if (!m_AvailablePages.empty())
        {
            page = m_AvailablePages.front();
            m_AvailablePages.pop_front();
        }
        else
        {
            page = std::make_shared<Page>(m_PageSize);
            m_PagePool.push_back(page);
        }

        return page;
    }


    // UPLOADBUFFER::RESET
    // The Reset method is used to reset all of the memory allocations 
    // so that they can be reused for the next frame
    // (or more specifically, the next command list recording).
    void DX12UploadBuffer::Reset()
    {
        m_CurrentPage = nullptr;
        // Reset all available pages.
        m_AvailablePages = m_PagePool;

        for (auto page : m_AvailablePages)
        {
            // Reset the page for new allocations.
            page->Reset();
        }
    }

    // UPLOADBUFFER::PAGE::PAGE
    // The constructor for a Page takes the size of the page as its only argument.
    DX12UploadBuffer::Page::Page(size_t sizeInBytes)
        : m_PageSize(sizeInBytes)
        , m_Offset(0)
        , m_CPUPtr(nullptr)
        , m_GPUPtr(D3D12_GPU_VIRTUAL_ADDRESS(0))
    {
        ID3D12Device* device = DX12Context::Main->GetDevice();

        DXCall(device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(m_PageSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&(m_d3d12Resource))
        ));

        m_GPUPtr = m_d3d12Resource->GetGPUVirtualAddress();
        m_d3d12Resource->Map(0, nullptr, &m_CPUPtr);
    }

    DX12UploadBuffer::Page::~Page()
    {
        m_d3d12Resource->Unmap(0, nullptr);
        m_CPUPtr = nullptr;
        m_GPUPtr = D3D12_GPU_VIRTUAL_ADDRESS(0);
    }

    bool DX12UploadBuffer::Page::HasSpace(size_t sizeInBytes, size_t alignment) const
    {
        size_t alignedSize = Math::AlignUp(sizeInBytes, alignment);
        size_t alignedOffset = Math::AlignUp(m_Offset, alignment);

        return alignedOffset + alignedSize <= m_PageSize;
    }

    // UPLOADBUFFER::PAGE::ALLOCATE
    // The Page::Allocate method is where the actual allocation occurs.
    // This method returns an Allocation structure that can be used to 
    // directly copy(using memcpy for example) CPU data to the GPUand bind that GPU address to the pipeline.
    DX12UploadBuffer::Allocation DX12UploadBuffer::Page::Allocate(size_t sizeInBytes, size_t alignment)
    {
        if (!HasSpace(sizeInBytes, alignment))
        {
            // Can't allocate space from page.
            throw std::bad_alloc();
        }

        size_t alignedSize = Math::AlignUp(sizeInBytes, alignment);
        m_Offset = Math::AlignUp(m_Offset, alignment);

        Allocation allocation;
        allocation.CPU = static_cast<uint8_t*>(m_CPUPtr) + m_Offset;
        allocation.GPU = m_GPUPtr + m_Offset;
        allocation.Offset = m_Offset;
        allocation.Page = m_d3d12Resource.Get();

        m_Offset += alignedSize;

        return allocation;
    }

    // UPLOADBUFFER::PAGE::RESET
    // The Page::Reset method simply resets the page¡¯s pointer offset to 0
    // so that it can be used to make new allocations.
    void DX12UploadBuffer::Page::Reset()
    {
        m_Offset = 0;
    }
}