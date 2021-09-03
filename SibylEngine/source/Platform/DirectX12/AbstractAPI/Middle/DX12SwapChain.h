#pragma once

#include "SIByLpch.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/SwapChain.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DescriptorAllocation.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12CommonResource.h"

namespace SIByL
{
	class DX12SwapChain :public SwapChain
	{
	public:
		DX12SwapChain();
		DX12SwapChain(int width, int height);
		~DX12SwapChain() { }
		void CreateSwapChain(int width, int height);
		void RecreateRenderTarget(int width, int height);
		void CreateDepthStencil(int width, int height);
		void RecreateDepthStencil(int width, int height);

		virtual void BindRenderTarget() override {};
		virtual void SetRenderTarget() override;
		virtual void PreparePresent() override;
		virtual void Present() override;
		virtual void Reisze(uint32_t width, uint32_t height) override;

	private:
		void SetViewportRect(int width, int height);

	private:
		ComPtr<IDXGISwapChain> m_SwapChain;

		DX12DescriptorAllocation m_DescriptorAllocation;
		DX12DescriptorAllocation m_DSVDespAllocation;
		UINT m_CurrentBackBuffer = 0;
		
		Ref<DX12DepthStencilResource> m_DepthStencilResource;
		Ref<DX12SwapChainResource> m_SwapChainResource[2];

	private:
		D3D12_VIEWPORT viewPort;
		D3D12_RECT scissorRect;
	};
}