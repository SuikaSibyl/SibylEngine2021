#include "SIByLpch.h"
#include "DescriptorAllocator.h"

#include "DescriptorAllocatorPage.h"

namespace SIByL
{
    DescriptorAllocator::DescriptorAllocator(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptorsPerHeap)
        : m_HeapType(type)
        , m_NumDescriptorsPerHeap(numDescriptorsPerHeap)
    {
    }

    // DESCRIPTORALLOCATOR::CREATEALLOCATORPAGE
    // The CreateAllocatorPage method is used to create a new page of descriptors.
    // The DescriptorAllocatorPage class (which will be shown later) is a wrapper for
    // the ID3D12DescriptorHeapand manages the actual descriptors.
    std::shared_ptr<DescriptorAllocatorPage> DescriptorAllocator::CreateAllocatorPage()
    {
        auto newPage = std::make_shared<DescriptorAllocatorPage>(m_HeapType, m_NumDescriptorsPerHeap);

        m_HeapPool.emplace_back(newPage);
        m_AvailableHeaps.insert(m_HeapPool.size() - 1);

        return newPage;
    }

    // DESCRIPTORALLOCATOR::ALLOCATE
    // The Allocate method allocates a contiguous block of descriptors from a descriptor heap.
    // The method iterates through the available descriptor heap (pages) and tries to allocate
    // the requested number of descriptors until a descriptor heap (page) is able to fulfill the requested allocation. 
    DescriptorAllocation DescriptorAllocator::Allocate(uint32_t numDescriptors)
    {
        std::lock_guard<std::mutex> lock(m_AllocationMutex);

        DescriptorAllocation allocation;

        for (auto iter = m_AvailableHeaps.begin(); iter != m_AvailableHeaps.end(); ++iter)
        {
            auto allocatorPage = m_HeapPool[*iter];

            allocation = allocatorPage->Allocate(numDescriptors);

            if (allocatorPage->NumFreeHandles() == 0)
            {
                iter = m_AvailableHeaps.erase(iter);
            }

            // A valid allocation has been found.
            if (!allocation.IsNull())
            {
                break;
            }
        }

        // No available heap could satisfy the requested number of descriptors.
        if (allocation.IsNull())
        {
            m_NumDescriptorsPerHeap = max(m_NumDescriptorsPerHeap, numDescriptors);
            auto newPage = CreateAllocatorPage();

            allocation = newPage->Allocate(numDescriptors);
        }

        return allocation;
    }

    // DESCRIPTORALLOCATOR::RELEASESTALEDESCRIPTORS
    // The ReleaseStaleDescriptors method iterates over all of the descriptor heap pages
    // and calls the page¡¯s ReleaseStaleDescriptors method.
    void DescriptorAllocator::ReleaseStaleDescriptors(uint64_t frameNumber)
    {
        std::lock_guard<std::mutex> lock(m_AllocationMutex);

        for (size_t i = 0; i < m_HeapPool.size(); ++i)
        {
            auto page = m_HeapPool[i];

            page->ReleaseStaleDescriptors(frameNumber);

            if (page->NumFreeHandles() > 0)
            {
                m_AvailableHeaps.insert(i);
            }
        }
    }
}