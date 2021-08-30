#pragma once

#include "SIByLpch.h"
#include "Sibyl/Renderer/SwapChain.h"
#include "Platform/DirectX12/Core/DescriptorAllocation.h"

namespace SIByL
{
	class DX12SwapChain :public SwapChain
	{
	public:
		DX12SwapChain();
		DX12SwapChain(int width, int height);
		~DX12SwapChain() { }
		void CreateSwapChain(int width, int height);
		void RecreateRenderTarget();
		void CreateDepthStencil(int width, int height);
		void RecreateDepthStencil(int width, int height);

		virtual void BindRenderTarget() override;
		virtual void SetRenderTarget() override;
		virtual void PreparePresent() override;
		virtual void Present() override;
		virtual void Reisze(uint32_t width, uint32_t height) override;

	private:
		void SetViewportRect(int width, int height);

	private:
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D12Resource> m_SwapChainBuffer[2];
		ComPtr<ID3D12Resource> m_SwapChainDepthStencil;
		DescriptorAllocation m_DescriptorAllocation;
		DescriptorAllocation m_DSVDespAllocation;
		UINT m_CurrentBackBuffer = 0;

	private:
		D3D12_VIEWPORT viewPort;
		D3D12_RECT scissorRect;
	};
}