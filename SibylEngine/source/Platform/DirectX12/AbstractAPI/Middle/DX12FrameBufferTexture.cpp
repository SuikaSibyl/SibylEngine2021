#include "SIByLpch.h"
#include "DX12FrameBufferTexture.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12CommonResource.h"

namespace SIByL
{
	DX12RenderTarget::DX12RenderTarget(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;

		DXGI_FORMAT format;
		switch (Descriptor.Format)
		{
		case FrameBufferTextureFormat::None:
			break;
		case FrameBufferTextureFormat::RGB8:
			format = DXGI_FORMAT_R8G8B8A8_UNORM;
			break;
		//case FrameBufferTextureFormat::RGBA16:
		//	format = ;
		//	break;
		case FrameBufferTextureFormat::DEPTH24STENCIL8:
			format = DXGI_FORMAT_D24_UNORM_S8_UINT;
			break;
		default:
			break;
		}

		m_RenderTargetResource = CreateRef<DX12RenderTargetResource>(descriptor.Width, descriptor.Height, format, DX12Context::GetInFlightSCmdList());
	}

	void DX12RenderTarget::Resize(uint32_t width, uint32_t height)
	{
		m_RenderTargetResource->Resize(width, height, DX12Context::GetInFlightSCmdList());
	}

	DX12Resource* DX12RenderTarget::GetResource()
	{
		return m_RenderTargetResource->GetResource();
	}

	DX12StencilDepth::DX12StencilDepth(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;

		DXGI_FORMAT format;
		switch (Descriptor.Format)
		{
		case FrameBufferTextureFormat::None:
			break;
		case FrameBufferTextureFormat::RGB8:
			format = DXGI_FORMAT_R8G8B8A8_UNORM;
			break;
			//case FrameBufferTextureFormat::RGBA16:
			//	format = ;
			//	break;
		case FrameBufferTextureFormat::DEPTH24STENCIL8:
			format = DXGI_FORMAT_D24_UNORM_S8_UINT;
			break;
		default:
			break;
		}

		m_DepthStencilResource = CreateRef<DX12DepthStencilResource>(descriptor.Width, descriptor.Height, format, DX12Context::GetInFlightSCmdList());
	}

	DX12Resource* DX12StencilDepth::GetResource()
	{
		return m_DepthStencilResource->GetResource();
	}

	void DX12StencilDepth::Resize(uint32_t width, uint32_t height)
	{
		m_DepthStencilResource->Resize(width, height, DX12Context::GetInFlightSCmdList());
	}
}