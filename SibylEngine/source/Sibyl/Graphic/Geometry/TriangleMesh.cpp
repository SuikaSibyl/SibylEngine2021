#include "SIByLpch.h"
#include "TriangleMesh.h"

#include "Sibyl/Renderer/Renderer.h"
#include "Platform/OpenGL/Graphic/Geometry/OpenGLTriangleMesh.h"

namespace SIByL
{
	TriangleMesh* TriangleMesh::Create(
		float* vertices, uint32_t vCount,
		unsigned int* indices, uint32_t iCount,
		VertexBufferLayout layout)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: 
			return new OpenGLTriangleMesh(vertices, vCount, indices, iCount, layout);
		case RasterRenderer::DirectX12: return nullptr; break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}
}