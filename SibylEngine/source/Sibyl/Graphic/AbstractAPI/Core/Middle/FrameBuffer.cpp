#include "SIByLpch.h"
#include "FrameBuffer.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Library/FrameBufferLibrary.h"

#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLFrameBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12FrameBuffer.h"

namespace SIByL
{
	Ref<FrameBuffer_v1> FrameBuffer_v1::Create(const FrameBufferDesc_v1& desc, const std::string& key)
	{
		Ref<FrameBuffer_v1> result = nullptr;
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: result = std::make_shared<OpenGLFrameBuffer_v1>(desc); FrameBufferLibrary::Register(key,result); break;
		case RasterRenderer::DirectX12: result = std::make_shared<DX12FrameBuffer_v1>(desc); FrameBufferLibrary::Register(key, result); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return result;
	}

	Ref<FrameBuffer> FrameBuffer::Create(const FrameBufferDesc& desc, const std::string& key)
	{
		Ref<FrameBuffer> result = nullptr;
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: result = std::make_shared<OpenGLFrameBuffer>(desc); Library<FrameBuffer>::Push(key, result); break;
		case RasterRenderer::DirectX12: return nullptr; break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return result;
	}
}