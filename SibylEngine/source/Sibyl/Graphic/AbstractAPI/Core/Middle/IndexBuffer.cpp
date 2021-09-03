#include "SIByLpch.h"
#include "IndexBuffer.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLIndexBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12IndexBuffer.h"

namespace SIByL
{
	IndexBuffer* IndexBuffer::Create(unsigned int* indices, uint32_t iCount)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new OpenGLIndexBuffer(indices, iCount); break;
		case RasterRenderer::DirectX12: return new DX12IndexBuffer(indices, iCount); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}

		return nullptr;
	}
}