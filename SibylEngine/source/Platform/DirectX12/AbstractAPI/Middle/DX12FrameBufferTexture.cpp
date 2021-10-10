#include "SIByLpch.h"
#include "DX12FrameBufferTexture.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12CommonResource.h"

namespace SIByL
{
	DX12RenderTarget::DX12RenderTarget(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;

	}

	DX12StencilDepth::DX12StencilDepth(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;

	}

}