#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/RenderTarget.h"

namespace SIByL
{
	class DX12DepthStencilResource;
	class DX12RenderTargetResource;

	class DX12RenderTarget :public RenderTarget
	{
	public:
		Ref<DX12DepthStencilResource> m_DepthStencilResource;
	};

	class DX12StencilDepth :public StencilDepth
	{
	public:
		Ref<DX12RenderTargetResource> m_RenderTargetResource;
	};
}