#include "SIByLpch.h"
#include "DX12Context.h"
#include "DX12Utility.h"
#include "Platform/DirectX12/Renderer/DX12SwapChain.h"

namespace SIByL
{
	DX12Context*  DX12Context::Main;

	void DX12Context::Init()
	{
		SIByL_CORE_ASSERT(!Main, "DX12 Environment already exists!");
		Main = this;
		CreateDevice();
		CreateCommandQueue();
		CreateGraphicCommandList();
		SIByL_CORE_INFO("DirectX 12 Init finished");
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
		UINT rtvDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		UINT dsvDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		UINT cbv_srv_uavDescriptorSize = m_D3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
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
		DXCall(m_D3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator)));
		DXCall(m_D3dDevice->CreateCommandList(0,
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			m_CommandAllocator.Get(),
			nullptr,
			IID_PPV_ARGS(&m_GraphicCmdList)));
		m_GraphicCmdList->Close();
	}

	void DX12Context::CreateSwapChain(int width, int height)
	{
		m_SwapChain = std::make_unique<DX12SwapChain>(width, height);
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