#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBufferTexture.h"

namespace SIByL
{
	class DX12DepthStencilResource;
	class DX12RenderTargetResource;

	class DX12RenderTarget :public RenderTarget
	{
	public:
		DX12RenderTarget(const FrameBufferTextureDesc& descriptor);

		Ref<DX12DepthStencilResource> m_DepthStencilResource;
	};

	class DX12StencilDepth :public StencilDepth
	{
	public:
		DX12StencilDepth(const FrameBufferTextureDesc& descriptor);

		Ref<DX12RenderTargetResource> m_RenderTargetResource;
	};
}