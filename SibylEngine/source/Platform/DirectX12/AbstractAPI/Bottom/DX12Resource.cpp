#include "SIByLpch.h"
#include "DX12Resource.h"

#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Common/DX12Utility.h"

#include "DX12ResourceStateTracker.h"

namespace SIByL
{
    DX12Resource::DX12Resource(const std::wstring& name)
        : m_ResourceName(name)
    {}

    DX12Resource::DX12Resource(const D3D12_RESOURCE_DESC& resourceDesc, const D3D12_CLEAR_VALUE* clearValue, const std::wstring& name)
    {
        auto device = DX12Context::GetDevice();

        if (clearValue)
        {
            m_d3d12ClearValue = std::make_unique<D3D12_CLEAR_VALUE>(*clearValue);
        }

        DXCall(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            m_d3d12ClearValue.get(),
            IID_PPV_ARGS(&m_d3d12Resource)
        ));

        DX12ResourceStateTracker::AddGlobalResourceState(m_d3d12Resource.Get(), D3D12_RESOURCE_STATE_COMMON);

        SetName(name);
    }

    DX12Resource::DX12Resource(ComPtr<ID3D12Resource> resource, const std::wstring& name)
        : m_d3d12Resource(resource)
    {
        DX12ResourceStateTracker::AddGlobalResourceState(m_d3d12Resource.Get(), D3D12_RESOURCE_STATE_COMMON);

        SetName(name);
    }

    DX12Resource::DX12Resource(const DX12Resource& copy)
        : m_d3d12Resource(copy.m_d3d12Resource)
        , m_ResourceName(copy.m_ResourceName)
        , m_d3d12ClearValue(std::make_unique<D3D12_CLEAR_VALUE>(*copy.m_d3d12ClearValue))
    {
        int i = 3;
    }

    DX12Resource::DX12Resource(DX12Resource&& copy)
        : m_d3d12Resource(std::move(copy.m_d3d12Resource))
        , m_ResourceName(std::move(copy.m_ResourceName))
        , m_d3d12ClearValue(std::move(copy.m_d3d12ClearValue))
    {
    }

    DX12Resource& DX12Resource::operator=(const DX12Resource& other)
    {
        if (this != &other)
        {
            m_d3d12Resource = other.m_d3d12Resource;
            m_ResourceName = other.m_ResourceName;
            if (other.m_d3d12ClearValue)
            {
                m_d3d12ClearValue = std::make_unique<D3D12_CLEAR_VALUE>(*other.m_d3d12ClearValue);
            }
        }

        return *this;
    }

    DX12Resource& DX12Resource::operator=(DX12Resource&& other)
    {
        if (this != &other)
        {
            m_d3d12Resource = other.m_d3d12Resource;
            m_ResourceName = other.m_ResourceName;
            m_d3d12ClearValue = std::move(other.m_d3d12ClearValue);

            other.m_d3d12Resource.Reset();
            other.m_ResourceName.clear();
        }

        return *this;
    }

    DX12Resource::~DX12Resource()
    {
    }

    void DX12Resource::SetD3D12Resource(ComPtr<ID3D12Resource> d3d12Resource, const D3D12_CLEAR_VALUE* clearValue)
    {
        m_d3d12Resource = d3d12Resource;
        if (m_d3d12ClearValue)
        {
            m_d3d12ClearValue = std::make_unique<D3D12_CLEAR_VALUE>(*clearValue);
        }
        else
        {
            m_d3d12ClearValue.reset();
        }
        SetName(m_ResourceName);
    }

    void DX12Resource::SetName(const std::wstring& name)
    {
        m_ResourceName = name;
        if (m_d3d12Resource && !m_ResourceName.empty())
        {
            m_d3d12Resource->SetName(m_ResourceName.c_str());
        }
    }

    void DX12Resource::Reset()
    {
        m_d3d12Resource.Reset();
        m_d3d12ClearValue.reset();
    }
}