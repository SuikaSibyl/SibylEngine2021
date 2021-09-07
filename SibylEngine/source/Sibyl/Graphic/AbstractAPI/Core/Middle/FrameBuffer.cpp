#include "SIByLpch.h"
#include "FrameBuffer.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Sibyl/Graphic/AbstractAPI/Library/FrameBufferLibrary.h"

#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLFrameBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12FrameBuffer.h"

namespace SIByL
{
	Ref<FrameBuffer> FrameBuffer::Create(const FrameBufferDesc& desc, const std::string& key)
	{
		Ref<FrameBuffer> result = nullptr;
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: result = std::make_shared<OpenGLFrameBuffer>(desc); FrameBufferLibrary::Register(key,result); break;
		case RasterRenderer::DirectX12: result = std::make_shared<DX12FrameBuffer>(desc); FrameBufferLibrary::Register(key, result); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return result;
	}
}