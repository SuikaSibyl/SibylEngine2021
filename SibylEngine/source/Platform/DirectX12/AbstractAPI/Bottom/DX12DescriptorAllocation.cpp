#include "SIByLpch.h"
#include "DX12DescriptorAllocation.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "DX12DescriptorAllocatorPage.h"

namespace SIByL
{
    DX12DescriptorAllocation::DX12DescriptorAllocation()
        : m_Descriptor{ 0 }
        , m_NumHandles(0)
        , m_DescriptorSize(0)
        , m_Page(nullptr)
    {}

    DX12DescriptorAllocation::DX12DescriptorAllocation(
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor,
        D3D12_GPU_DESCRIPTOR_HANDLE descriptorGpu,
        uint32_t numHandles, uint32_t descriptorSize, std::shared_ptr<DX12DescriptorAllocatorPage> page, bool gpuVisible)
        : m_Descriptor(descriptor)
        , m_DescriptorGpu(descriptorGpu)
        , m_NumHandles(numHandles)
        , m_DescriptorSize(descriptorSize)
        , m_Page(page)
    {}

    DX12DescriptorAllocation::~DX12DescriptorAllocation()
    {
        Free();
    }

    DX12DescriptorAllocation::DX12DescriptorAllocation(DX12DescriptorAllocation&& allocation)
        : m_Descriptor(allocation.m_Descriptor)
        , m_NumHandles(allocation.m_NumHandles)
        , m_DescriptorSize(allocation.m_DescriptorSize)
        , m_Page(std::move(allocation.m_Page))
    {
        allocation.m_Descriptor.ptr = 0;
        allocation.m_NumHandles = 0;
        allocation.m_DescriptorSize = 0;
    }

    DX12DescriptorAllocation& DX12DescriptorAllocation::operator=(DX12DescriptorAllocation&& other)
    {
        // Free this descriptor if it points to anything.
        Free();

        m_Descriptor = other.m_Descriptor;
        m_NumHandles = other.m_NumHandles;
        m_DescriptorSize = other.m_DescriptorSize;
        m_Page = std::move(other.m_Page);

        other.m_Descriptor.ptr = 0;
        other.m_NumHandles = 0;
        other.m_DescriptorSize = 0;

        return *this;
    }

    void DX12DescriptorAllocation::Free()
    {
        if (!IsNull() && m_Page)
        {
            m_Page->Free(std::move(*this), DX12Context::GetFrameCount());

            m_Descriptor.ptr = 0;
            m_NumHandles = 0;
            m_DescriptorSize = 0;
            m_Page.reset();
        }
    }

    // Check if this a valid descriptor.
    bool DX12DescriptorAllocation::IsNull() const
    {
        return m_Descriptor.ptr == 0;
    }

    // Get a descriptor at a particular offset in the allocation.
    D3D12_CPU_DESCRIPTOR_HANDLE DX12DescriptorAllocation::GetDescriptorHandle(uint32_t offset) const
    {
        assert(offset < m_NumHandles);
        return { m_Descriptor.ptr + (m_DescriptorSize * offset) };
    }

    // Get a descriptor at a particular offset in the allocation.
    D3D12_GPU_DESCRIPTOR_HANDLE DX12DescriptorAllocation::GetDescriptorHandleGpu(uint32_t offset) const
    {
        assert(offset < m_NumHandles);
        return { m_DescriptorGpu.ptr + (m_DescriptorSize * offset) };
    }

    uint32_t DX12DescriptorAllocation::GetNumHandles() const
    {
        return m_NumHandles;
    }

    std::shared_ptr<DX12DescriptorAllocatorPage> DX12DescriptorAllocation::GetDescriptorAllocatorPage() const
    {
        return m_Page;
    }
}