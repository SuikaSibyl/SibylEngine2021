#include "SIByLpch.h"
#include "FrameBuffer.h"

#include "Sibyl/Renderer/Renderer.h"

#include "Platform/OpenGL/Renderer/OpenGLFrameBuffer.h"
#include "Platform/DirectX12/Renderer/DX12FrameBuffer.h"

namespace SIByL
{
	Ref<FrameBuffer> FrameBuffer::Create(const FrameBufferDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLFrameBuffer>(desc); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12FrameBuffer>(desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
	}
}