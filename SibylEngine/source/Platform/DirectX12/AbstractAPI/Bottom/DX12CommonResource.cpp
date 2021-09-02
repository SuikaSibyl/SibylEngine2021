#include "SIByLpch.h"
#include "DX12CommonResource.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"

namespace SIByL
{
    ///////////////////////////////////////////////////////////////////////////////////
    //																				 //
    //						     	Swap Chain Resource								 //
    //																				 //
    ///////////////////////////////////////////////////////////////////////////////////

    DX12SwapChainResource::DX12SwapChainResource(const uint32_t& width, const uint32_t& height, ComPtr<ID3D12Resource> resource)
        :m_Width(width), m_Height(height)
    {
        // Allocate a descriptor from HeapAllocator
        Ref<DescriptorAllocator> rtvDespAllocator = DX12Context::GetRtvDescriptorAllocator();
        m_RTVDescriptorAllocation = rtvDespAllocator->Allocate(1);

        // Create the actual buffer
        CreateResource(resource);
    }

    void DX12SwapChainResource::Reset()
    {
        m_Resource->Reset();
    }

    void DX12SwapChainResource::Resize(const uint32_t& width, const uint32_t& height, ComPtr<ID3D12Resource> resource)
    {
        // Ignore unormal size input
        if (width <= 0 || height <= 0) return;
        m_Width = width; m_Height = height;
        Reset();
        CreateResource(resource);
    }

    void DX12SwapChainResource::CreateResource(ComPtr<ID3D12Resource> resource)
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE  handle(m_RTVDescriptorAllocation.GetDescriptorHandle());

        // Get buffer of the swap chain
        m_Resource = CreateScope<DX12Resource>(resource, L"SwapChainRenderTarget");

        // Create RTV in heap
        DX12Context::GetDevice()->CreateRenderTargetView(
            resource.Get(),
            nullptr,
            handle);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DX12SwapChainResource::GetRTVCpuHandle()
    {
        return m_RTVDescriptorAllocation.GetDescriptorHandle();
    }

    ///////////////////////////////////////////////////////////////////////////////////
    //																				 //
    //							Depth Stencil Resource								 //
    //																				 //
    ///////////////////////////////////////////////////////////////////////////////////

	DX12DepthStencilResource::DX12DepthStencilResource(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList)
		:m_Width(width), m_Height(height)
	{
		// Allocate a descriptor from HeapAllocator
		Ref<DescriptorAllocator> dsvDespAllocator = DX12Context::GetDsvDescriptorAllocator();
		m_DSVDescriptorAllocation = dsvDespAllocator->Allocate(1);

		// Create the actual buffer
		CreateResource(pSCommandList);
	}

    void DX12DepthStencilResource::Reset()
    {
        m_Resource->Reset();
    }

    void DX12DepthStencilResource::Resize(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList)
    {
        // Ignore unormal size input
        if (width <= 0 || height <= 0) return;
        m_Width = width; m_Height = height;
        Reset();
        CreateResource(pSCommandList);
    }

	void DX12DepthStencilResource::CreateResource(Ref<DX12CommandList> pSCommandList)
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
            m_DSVDescriptorAllocation.GetDescriptorHandle());
        
        m_Resource.reset(new DX12Resource(resource, L"DepthStencilBuffer"));

        pSCommandList->TransitionBarrier(*m_Resource, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    
    D3D12_CPU_DESCRIPTOR_HANDLE DX12DepthStencilResource::GetDSVCpuHandle()
    {
        return m_DSVDescriptorAllocation.GetDescriptorHandle();
    }

    ///////////////////////////////////////////////////////////////////////////////////
    //																				 //
    //							Render Target Resource								 //
    //																				 //
    ///////////////////////////////////////////////////////////////////////////////////

    DX12RenderTargetResource::DX12RenderTargetResource(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList)
        :m_Width(width), m_Height(height)
    {
        // Allocate a descriptor from HeapAllocator
        Ref<DescriptorAllocator> rtvDespAllocator = DX12Context::GetRtvDescriptorAllocator();
        Ref<DescriptorAllocator> srvDespAllocator = DX12Context::GetSrvDescriptorAllocator();
        Ref<DescriptorAllocator> srvDespAllocatorGpu = DX12Context::GetSrvDescriptorAllocatorGpu();
        m_RTVDescriptorAllocation = rtvDespAllocator->Allocate(1);
        m_SRVDescriptorAllocationCpu = srvDespAllocator->Allocate(1);
        m_SRVDescriptorAllocationGpu = srvDespAllocatorGpu->Allocate(1);

        // Create the actual buffer
        CreateResource(pSCommandList);
    }

    void DX12RenderTargetResource::Reset()
    {
        m_Resource->Reset();
    }

    void DX12RenderTargetResource::Resize(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList)
    {
        // Ignore unormal size input
        if (width <= 0 || height <= 0) return;
        m_Width = width; m_Height = height;
        Reset();
        CreateResource(pSCommandList);
    }

    void DX12RenderTargetResource::CreateResource(Ref<DX12CommandList> pSCommandList)
    {
        D3D12_RESOURCE_DESC rtvResourceDesc;
        rtvResourceDesc.Alignment = 0;
        rtvResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rtvResourceDesc.DepthOrArraySize = 1;
        rtvResourceDesc.Width = m_Width;
        rtvResourceDesc.Height = m_Height;
        rtvResourceDesc.MipLevels = 1;
        rtvResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rtvResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        rtvResourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtvResourceDesc.SampleDesc.Count = 1;
        rtvResourceDesc.SampleDesc.Quality = 0;

        CD3DX12_CLEAR_VALUE optClear;
        optClear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        float ClearColor[4] = { 0, 0, 0, 0 };
        memcpy(optClear.Color, &ClearColor, 4 * sizeof(float));

        ComPtr<ID3D12Resource> resource;

        DXCall(DX12Context::GetDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &rtvResourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            &optClear,
            IID_PPV_ARGS(&resource)));

        DX12Context::GetDevice()->CreateRenderTargetView(
            resource.Get(),
            nullptr,
            m_RTVDescriptorAllocation.GetDescriptorHandle());
        
        DX12Context::GetDevice()->CreateShaderResourceView(
            resource.Get(),
            nullptr,
            m_SRVDescriptorAllocationCpu.GetDescriptorHandle());
        
        DX12Context::GetDevice()->CreateShaderResourceView(
            resource.Get(),
            nullptr,
            m_SRVDescriptorAllocationCpu.GetDescriptorHandle());

        m_ImGuiAllocation = ImGuiLayerDX12::Get()->RegistShaderResource();

        DX12Context::GetDevice()->CreateShaderResourceView(
            resource.Get(),
            nullptr,
            m_ImGuiAllocation.m_CpuHandle);

        m_Resource.reset(new DX12Resource(resource, L"RenderTargetBuffer"));
        
        pSCommandList->TransitionBarrier(*m_Resource, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DX12RenderTargetResource::GetRTVCpuHandle()
    {
        return m_RTVDescriptorAllocation.GetDescriptorHandle();
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DX12RenderTargetResource::GetSRVCpuHandle()
    {
        return m_SRVDescriptorAllocationCpu.GetDescriptorHandle();
    }

    D3D12_GPU_DESCRIPTOR_HANDLE DX12RenderTargetResource::GetSRVGpuHandle()
    {
        return m_SRVDescriptorAllocationGpu.GetDescriptorHandleGpu();
    }

    D3D12_GPU_DESCRIPTOR_HANDLE DX12RenderTargetResource::GetImGuiGpuHandle()
    {
        return m_ImGuiAllocation.m_GpuHandle;
    }
}