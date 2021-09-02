#pragma once

#include "SIByLpch.h"

#include "Sibyl/Graphic/AbstractAPI/Top/GraphicContext.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DescriptorAllocator.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12UploadBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Synchronizer.h"
#include "Platform/DirectX12/AbstractAPI/Top/DX12RenderPipeline.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12FrameResources.h"


namespace SIByL
{
	class DX12CommandQueue;
	class DX12CommandList;

	class DX12Context :public GraphicContext
	{
	public:
		static DX12Context* Main;
		~DX12Context();

		virtual void Init() override;
		virtual void OnWindowResize(uint32_t width, uint32_t height) override;

		inline static uint64_t GetFrameCount() { return Main->m_FrameCount; };

	public:
		inline static ID3D12Device* GetDevice() { return Main->m_D3dDevice.Get(); }
		inline static SwapChain* GetSwapChain() { return Main->m_SwapChain.get(); }
		inline static IDXGIFactory4* GetDxgiFactory() { return Main->m_DxgiFactory.Get(); }
		inline static DX12UploadBuffer* GetUploadBuffer() { return Main->m_UploadBuffer.get(); }
		inline static DX12Synchronizer* GetSynchronizer() { return dynamic_cast<DX12Synchronizer*>(Main->m_Synchronizer.get()); }
		static std::array<CD3DX12_STATIC_SAMPLER_DESC, 6> GetStaticSamplers();


	private:
		void EnableDebugLayer();
		void CreateDevice();
		void GetDescriptorSize();
		void CreateCommandQueue();
		void CreateGraphicCommandList();
		void CreateDescriptorAllocator();
		void CreateSwapChain();
		void CreateRenderPipeline();
		void CreateSynchronizer();
		void CreateUploadBuffer();
		void CreateFrameResourcesManager();

	public:
		ComPtr<ID3D12DescriptorHeap> CreateSRVHeap();

	private:
		ComPtr<ID3D12Debug>			m_DebugInterface;
		ComPtr<IDXGIFactory4>		m_DxgiFactory;
		ComPtr<ID3D12Device>		m_D3dDevice;
		ComPtr<IDXGIDebug1>			m_DxgiDebug;

		std::unique_ptr<DX12RenderPipeline>			m_RenderPipeline;
		Ref<DX12UploadBuffer>			m_UploadBuffer;
		Ref<DX12FrameResourcesManager>	m_FrameResourcesManager;


		// Command Submission System
		// ====================================================================
	public:
		static ID3D12CommandQueue* GetCommandQueue();
		static Ref<DX12CommandQueue> GetSCommandQueue() { return Main->m_SGraphicQueue; }
		static ID3D12GraphicsCommandList* GetInFlightDXGraphicCommandList();
		static Ref<DX12CommandList> GetInFlightSCmdList() { return Main->m_InFlightSCmdList; }
		static void SetInFlightSCmdList(Ref<DX12CommandList> cmdList) { Main->m_InFlightSCmdList = cmdList; }
	private:
		Ref<DX12CommandQueue>		m_SGraphicQueue;
		Ref<DX12CommandList>		m_InFlightSCmdList;

		// Descriptor Sizes
		// ====================================================================
	public:
		inline static UINT GetRtvDescriptorSize() { return Main->m_RtvDescriptorSize; }
		inline static UINT GetDsvDescriptorSize() { return Main->m_DsvDescriptorSize; }
		inline static UINT GetCbvSrvUavDescriptorSize() { return Main->m_Cbv_Srv_UavDescriptorSize; }

	private:
		UINT m_RtvDescriptorSize;
		UINT m_DsvDescriptorSize;
		UINT m_Cbv_Srv_UavDescriptorSize;

		// Descriptor Allocators
		// ====================================================================
	public:
		inline static Ref<DescriptorAllocator> GetRtvDescriptorAllocator() { return Main->m_RtvDescriptorAllocator; }
		inline static Ref<DescriptorAllocator> GetDsvDescriptorAllocator() { return Main->m_DsvDescriptorAllocator; }
		inline static Ref<DescriptorAllocator> GetSrvDescriptorAllocator() { return Main->m_SrvDescriptorAllocator; }
		inline static Ref<DescriptorAllocator> GetSrvDescriptorAllocatorGpu() { return Main->m_SrvDescriptorAllocatorGpu; }

	private:
		Ref<DescriptorAllocator> m_RtvDescriptorAllocator;
		Ref<DescriptorAllocator> m_DsvDescriptorAllocator;
		Ref<DescriptorAllocator> m_SrvDescriptorAllocator;
		Ref<DescriptorAllocator> m_SrvDescriptorAllocatorGpu;

		uint64_t m_FrameCount;
	};
}