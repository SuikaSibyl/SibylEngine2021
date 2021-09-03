#include "SIByLpch.h"
#include "VertexBuffer.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLVertexBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12VertexBuffer.h"

namespace SIByL
{
	VertexBuffer* VertexBuffer::Create(float* vertices, uint32_t floatCount, Type type)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new OpenGLVertexBuffer(vertices, floatCount, type); break;
		case RasterRenderer::DirectX12: return new DX12VertexBuffer(vertices, floatCount, type); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

}