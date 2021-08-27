#include "SIByLpch.h"
#include "ShaderBinder.h"

#include "Sibyl/Renderer/Renderer.h"

#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"

namespace SIByL
{
	ShaderBinder* ShaderBinder::Create(const ShaderBinderDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return nullptr; break;
		case RasterRenderer::DirectX12: return new DX12ShaderBinder(desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

}