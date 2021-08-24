#include "SIByLpch.h"
#include "DX12Context.h"
#include "DX12Utility.h"
#include "Sibyl/Renderer/GraphicContext.h"
#include "Platform/DirectX12/Renderer/DX12SwapChain.h"

namespace SIByL
{
	DX12Context*  DX12Context::Main;

	void DX12Context::Init()
	{
		SIByL_CORE_ASSERT(!Main, "DX12 Environment already exists!");
		Main = this;
		EnableDebugLayer();
		CreateDevice();
		GetDescriptorSize();
		CreateCommandQueue();
		CreateGraphicCommandList();
		CreateDescriptorAllocator();
		CreateRenderPipeline();
		CreateSynchronizer();
		CreateUploadBuffer();
		CreateFrameResourcesManager();

		m_CommandList->Restart();
		CreateSwapChain();
		m_CommandList->Execute();


		SIByL_CORE_INFO("DirectX 12 Init finished");
	}

	void DX12Context::EnableDebugLayer()
	{
#if defined(_DEBUG)
		// Always enable the debug layer before doing anything DX12 related
		// so all possible errors generated while creating DX12 objects
		// are caught by the debug layer.
		DXCall(D3D12GetDebugInterface(IID_PPV_ARGS(&m_DebugInterface)));
		m_DebugInterface->EnableDebugLayer();
#endif
	}

	void DX12Context::CreateDevice()
	{
		DXCall(CreateDXGIFactory1(IID_PPV_ARGS(&m_DxgiFactory)));
		DXCall(D3D12CreateDevice(nullptr,
			D3D_FEATURE_LEVEL_12_0,
			IID_PPV_ARGS(&m_D3dDevice)));
	}

	void DX12Context::GetDescriptorSize()
	{
		m_RtvDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_DsvDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		m_Cbv_Srv_UavDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	void DX12Context::CreateCommandQueue()
	{
		D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
		commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		DXCall(m_D3dDevice->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&m_CommandQueue)));
	}

	void DX12Context::CreateGraphicCommandList()
	{
		m_GraphicCommandList = std::make_unique<DX12GraphicCommandList>();
		m_CommandList = m_GraphicCommandList.get();
	}

	void DX12Context::CreateDescriptorAllocator()
	{
		m_RtvDescriptorAllocator = std::make_unique<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_DsvDescriptorAllocator = std::make_unique<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		m_SrvDescriptorAllocator = std::make_unique<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	void DX12Context::CreateSwapChain()
	{
		m_SwapChain = std::make_unique<DX12SwapChain>();
		m_SwapChain->BindRenderTarget();
	}

	void DX12Context::CreateRenderPipeline()
	{
		m_RenderPipeline = std::make_unique<DX12RenderPipeline>();
	}

	void DX12Context::CreateSynchronizer()
	{
		m_Synchronizer = std::make_unique<DX12Synchronizer>();
	}

	void DX12Context::CreateUploadBuffer()
	{
		m_UploadBuffer = std::make_unique<DX12UploadBuffer>();
	}

	void DX12Context::CreateFrameResourcesManager()
	{
		m_FrameResourcesManager = std::make_unique<DX12FrameResourcesManager>();
	}

	ID3D12DescriptorHeap* DX12Context::CreateSRVHeap()
	{
		ID3D12DescriptorHeap* g_pd3dSrvDescHeap;
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		desc.NumDescriptors = 1;
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		DXCall(m_D3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)));

		return g_pd3dSrvDescHeap;
	}
}