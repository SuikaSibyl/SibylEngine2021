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
		void CreateSwapChain(int width, int height);

		virtual void BindRenderTarget() override;
		virtual void SetRenderTarget() override;
		virtual void PreparePresent() override;
		virtual void Present() override;

	private:
		void SetViewportRect(int width, int height);

	private:
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D12Resource> m_SwapChainBuffer[2];
		DescriptorAllocation m_DescriptorAllocation;
		UINT m_CurrentBackBuffer = 0;

	private:
		D3D12_VIEWPORT viewPort;
		D3D12_RECT scissorRect;
	};
}