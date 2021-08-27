#include "SIByLpch.h"
#include "DX12Context.h"
#include "DX12Utility.h"

#include "Sibyl/Core/Application.h"
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

	DX12Context::~DX12Context()
	{
		//Application::Get().OnResourceDestroy();
		m_Synchronizer->ForceSynchronize();

		m_SwapChain = nullptr;
		m_CommandQueue = nullptr;

		// Release Main Descriptor Allocators
		m_RtvDescriptorAllocator = nullptr;
		m_DsvDescriptorAllocator = nullptr;
		m_SrvDescriptorAllocator = nullptr;

		m_CommandList = nullptr;
		m_Synchronizer = nullptr;
		m_FrameResourcesManager = nullptr;
		m_UploadBuffer = nullptr;

		m_D3dDevice = nullptr;

#if defined(_DEBUG)
		if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&m_DxgiDebug))))
		{
			m_DxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
		}
#endif
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
		m_CommandList = std::make_shared<DX12GraphicCommandList>();
	}

	void DX12Context::CreateDescriptorAllocator()
	{
		m_RtvDescriptorAllocator = std::make_shared<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		m_DsvDescriptorAllocator = std::make_shared<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		m_SrvDescriptorAllocator = std::make_shared<DescriptorAllocator>(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	void DX12Context::CreateSwapChain()
	{
		m_SwapChain = std::make_shared<DX12SwapChain>();
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
		m_UploadBuffer = std::make_shared<DX12UploadBuffer>();
	}

	void DX12Context::CreateFrameResourcesManager()
	{
		m_FrameResourcesManager = std::make_shared<DX12FrameResourcesManager>();
	}

	ComPtr<ID3D12DescriptorHeap> DX12Context::CreateSRVHeap()
	{
		ComPtr<ID3D12DescriptorHeap> g_pd3dSrvDescHeap;
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		desc.NumDescriptors = 1;
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		DXCall(m_D3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)));
		return g_pd3dSrvDescHeap;
	}

	std::array<CD3DX12_STATIC_SAMPLER_DESC, 6> DX12Context::GetStaticSamplers()
	{
		//过滤器POINT,寻址模式WRAP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC pointWarp(0,	//着色器寄存器
			D3D12_FILTER_MIN_MAG_MIP_POINT,		//过滤器类型为POINT(常量插值)
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

		//过滤器POINT,寻址模式CLAMP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC pointClamp(1,	//着色器寄存器
			D3D12_FILTER_MIN_MAG_MIP_POINT,		//过滤器类型为POINT(常量插值)
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

		//过滤器LINEAR,寻址模式WRAP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC linearWarp(2,	//着色器寄存器
			D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//过滤器类型为LINEAR(线性插值)
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

		//过滤器LINEAR,寻址模式CLAMP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC linearClamp(3,	//着色器寄存器
			D3D12_FILTER_MIN_MAG_MIP_LINEAR,		//过滤器类型为LINEAR(线性插值)
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

		//过滤器ANISOTROPIC,寻址模式WRAP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC anisotropicWarp(4,	//着色器寄存器
			D3D12_FILTER_ANISOTROPIC,			//过滤器类型为ANISOTROPIC(各向异性)
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//U方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP,	//V方向上的寻址模式为WRAP（重复寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_WRAP);	//W方向上的寻址模式为WRAP（重复寻址模式）

		//过滤器LINEAR,寻址模式CLAMP的静态采样器
		CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(5,	//着色器寄存器
			D3D12_FILTER_ANISOTROPIC,			//过滤器类型为ANISOTROPIC(各向异性)
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//U方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	//V方向上的寻址模式为CLAMP（钳位寻址模式）
			D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	//W方向上的寻址模式为CLAMP（钳位寻址模式）

		return{ pointWarp, pointClamp, linearWarp, linearClamp, anisotropicWarp, anisotropicClamp };
	}
}