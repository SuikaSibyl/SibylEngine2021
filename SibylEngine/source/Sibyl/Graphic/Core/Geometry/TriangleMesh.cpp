#include "SIByLpch.h"
#include "TriangleMesh.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Platform/OpenGL/AbstractAPI/Top/OpenGLTriangleMesh.h"
#include "Platform/DirectX12/AbstractAPI/Top/DX12TriangleMesh.h"

namespace SIByL
{
	Ref<TriangleMesh> TriangleMesh::Create(
		float* vertices, uint32_t vCount,
		unsigned int* indices, uint32_t iCount,
		VertexBufferLayout layout)
	{
		uint32_t floatCount = layout.GetStide() * vCount * 0.25;

		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: 
			return std::make_shared<OpenGLTriangleMesh>(vertices, floatCount, indices, iCount, layout);
		case RasterRenderer::DirectX12:
			return std::make_shared<DX12TriangleMesh>(vertices, floatCount, indices, iCount, layout);
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}
}