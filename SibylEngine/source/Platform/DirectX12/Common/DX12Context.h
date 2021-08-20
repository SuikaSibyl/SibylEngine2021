#pragma once

#include "SIByLpch.h"
#include "Sibyl/Renderer/GraphicContext.h"
#include "Platform/OpenGL/Renderer/OpenGLSwapChain.h"
#include "Platform/DirectX12/Core/DescriptorAllocator.h"
#include "Platform/DirectX12/Core/DX12CommandList.h"
#include "Platform/DirectX12/Core/UploadBuffer.h"
#include "Platform/DirectX12/Core/DX12Synchronizer.h"
#include "Platform/DirectX12/Renderer/DX12RenderPipeline.h"

namespace SIByL
{
	class DX12Context :public GraphicContext
	{
	public:
		static DX12Context* Main;
		virtual void Init() override;

		inline static uint64_t GetFrameCount() { return Main->m_FrameCount; };

	public:
		inline static ID3D12Device* GetDevice() { return Main->m_D3dDevice.Get(); }
		inline static DX12GraphicCommandList* GetGraphicCommandList() { return Main->m_GraphicCommandList.get(); }
		inline static SwapChain* GetSwapChain() { return Main->m_SwapChain.get(); }
		inline static DX12Synchronizer* GetSynchronizer() { return Main->m_Synchronizer.get(); }
		inline static ID3D12GraphicsCommandList* GetDXGraphicCommandList() { return Main->m_GraphicCommandList->Get(); }
		inline static IDXGIFactory4* GetDxgiFactory() { return Main->m_DxgiFactory.Get(); }
		inline static ID3D12CommandQueue* GetCommandQueue() { return Main->m_CommandQueue.Get(); }
		inline static DX12UploadBuffer* GetUploadBuffer() { return Main->m_UploadBuffer.get(); }

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

	public:
		ID3D12DescriptorHeap* CreateSRVHeap();

	private:
		ComPtr<ID3D12Debug>			m_DebugInterface;
		ComPtr<IDXGIFactory4>		m_DxgiFactory;
		ComPtr<ID3D12Device>		m_D3dDevice;
		ComPtr<ID3D12CommandQueue>	m_CommandQueue;

		std::unique_ptr<DX12GraphicCommandList> m_GraphicCommandList;
		std::unique_ptr<DX12RenderPipeline> m_RenderPipeline;
		std::unique_ptr<DX12Synchronizer> m_Synchronizer;
		std::unique_ptr<DX12UploadBuffer> m_UploadBuffer;

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
		inline static DescriptorAllocator* GetRtvDescriptorAllocator() { return Main->m_RtvDescriptorAllocator.get(); }
		inline static DescriptorAllocator* GetDsvDescriptorAllocator() { return Main->m_DsvDescriptorAllocator.get(); }
		inline static DescriptorAllocator* GetSrvDescriptorAllocator() { return Main->m_SrvDescriptorAllocator.get(); }

	private:
		std::unique_ptr<DescriptorAllocator> m_RtvDescriptorAllocator;
		std::unique_ptr<DescriptorAllocator> m_DsvDescriptorAllocator;
		std::unique_ptr<DescriptorAllocator> m_SrvDescriptorAllocator;

		uint64_t m_FrameCount;
	};
}