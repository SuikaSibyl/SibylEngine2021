#pragma once

#include "SIByLpch.h"

namespace SIByL
{
	class DX12Environment
	{
	public:
		static DX12Environment* Main;
		void Init();

	public:
		inline ID3D12Device* GetDevice() { return m_D3dDevice.Get(); }
		inline ID3D12GraphicsCommandList* GetGraphicCommandList() { return m_GraphicCmdList.Get(); }

	private:
		void CreateDevice();
		void GetDescriptorSize();
		void CreateCommandQueue();
		void CreateGraphicCommandList();

	public:
		ID3D12DescriptorHeap* CreateSRVHeap();

	private:
		ComPtr<IDXGIFactory4>		m_DxgiFactory;
		ComPtr<ID3D12Device>		m_D3dDevice;
		ComPtr<ID3D12CommandQueue>	m_CommandQueue;
		ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
		ComPtr<ID3D12GraphicsCommandList> m_GraphicCmdList;

	};
}