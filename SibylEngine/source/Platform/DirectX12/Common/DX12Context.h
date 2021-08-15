#pragma once

#include "SIByLpch.h"
#include "Sibyl/Renderer/SwapChain.h"

namespace SIByL
{
	class DX12Context
	{
	public:
		static DX12Context* Main;
		void Init();

	public:
		inline static ID3D12Device* GetDevice() { return Main->m_D3dDevice.Get(); }
		inline static ID3D12GraphicsCommandList* GetGraphicCommandList() { return Main->m_GraphicCmdList.Get(); }
		inline static IDXGIFactory4* GetDxgiFactory() { return Main->m_DxgiFactory.Get(); }
		inline static ID3D12CommandQueue* GetCommandQueue() { return Main->m_CommandQueue.Get(); }

	private:
		void CreateDevice();
		void GetDescriptorSize();
		void CreateCommandQueue();
		void CreateGraphicCommandList();
		void CreateSwapChain(int width, int height);

	public:
		ID3D12DescriptorHeap* CreateSRVHeap();

	private:
		ComPtr<IDXGIFactory4>		m_DxgiFactory;
		ComPtr<ID3D12Device>		m_D3dDevice;
		ComPtr<ID3D12CommandQueue>	m_CommandQueue;
		ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
		ComPtr<ID3D12GraphicsCommandList> m_GraphicCmdList;

		std::unique_ptr<SwapChain> m_SwapChain;
	};
}