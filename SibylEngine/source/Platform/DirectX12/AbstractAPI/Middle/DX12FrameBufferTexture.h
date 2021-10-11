#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBufferTexture.h"

namespace SIByL
{
	class DX12Resource;
	class DX12DepthStencilResource;
	class DX12RenderTargetResource;

	class DX12RenderTarget :public RenderTarget
	{
	public:
		DX12RenderTarget(const FrameBufferTextureDesc& descriptor);

		virtual void Resize(uint32_t width, uint32_t height) override;

		DX12Resource* GetResource();

		Ref<DX12RenderTargetResource> m_RenderTargetResource;
	};

	class DX12StencilDepth :public StencilDepth
	{
	public:
		DX12StencilDepth(const FrameBufferTextureDesc& descriptor);

		virtual void Resize(uint32_t width, uint32_t height) override;

		DX12Resource* GetResource();

		Ref<DX12DepthStencilResource> m_DepthStencilResource;
	};
}