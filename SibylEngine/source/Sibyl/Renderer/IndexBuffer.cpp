#include "SIByLpch.h"
#include "IndexBuffer.h"

#include "Sibyl/Renderer/Renderer.h"
#include "Platform/OpenGL/Renderer/OpenGLIndexBuffer.h"

namespace SIByL
{
	IndexBuffer* IndexBuffer::Create(unsigned int* indices, uint32_t iCount)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new OpenGLIndexBuffer(indices, iCount); break;
		case RasterRenderer::DirectX12: return nullptr; break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}

		return nullptr;
	}
}