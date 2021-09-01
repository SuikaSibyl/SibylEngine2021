#include "SIByLpch.h"
#include "DX12CommonResource.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

namespace SIByL
{
	DX12DepthStencilResource::DX12DepthStencilResource(const uint32_t& width, const uint32_t& height)
		:m_Width(width), m_Height(height)
	{
		// Allocate a descriptor from HeapAllocator
		Ref<DescriptorAllocator> dsvDespAllocator = DX12Context::GetDsvDescriptorAllocator();
		m_DescriptorAllocation = dsvDespAllocator->Allocate(1);

		// Create the actual buffer
		CreateResource();
	}

    void DX12DepthStencilResource::Reset()
    {
        m_Resource->Reset();
    }

    void DX12DepthStencilResource::Resize(const uint32_t& width, const uint32_t& height)
    {
        // Ignore unormal size input
        if (width <= 0 || height <= 0) return;

        Reset();
        CreateResource();
    }

	void DX12DepthStencilResource::CreateResource()
	{
        D3D12_RESOURCE_DESC dsvResourceDesc;
        dsvResourceDesc.Alignment = 0;
        dsvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dsvResourceDesc.DepthOrArraySize = 1;
        dsvResourceDesc.Width = m_Width;
        dsvResourceDesc.Height = m_Height;
        dsvResourceDesc.MipLevels = 1;
        dsvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        dsvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        dsvResourceDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsvResourceDesc.SampleDesc.Count = 1;
        dsvResourceDesc.SampleDesc.Quality = 0;

        CD3DX12_CLEAR_VALUE optClear;
        optClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        optClear.DepthStencil.Depth = 1;
        optClear.DepthStencil.Stencil = 0;

        ComPtr<ID3D12Resource> resource;

        DXCall(DX12Context::GetDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &dsvResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            &optClear,
            IID_PPV_ARGS(&resource)));

        DX12Context::GetDevice()->CreateDepthStencilView(
            resource.Get(),
            nullptr,
            m_DescriptorAllocation.GetDescriptorHandle());
        
        m_Resource.reset(new DX12Resource(resource, L"DepthStencilBuffer"));
	}

}