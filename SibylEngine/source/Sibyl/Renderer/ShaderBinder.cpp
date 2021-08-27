#include "SIByLpch.h"
#include "ShaderBinder.h"

#include "Sibyl/Renderer/Renderer.h"

#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"

namespace SIByL
{
	Ref<ShaderBinder> ShaderBinder::Create(const ShaderBinderDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return nullptr; break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12ShaderBinder>(desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	void ShaderBinder::InitMappers(const ShaderBinderDesc& desc)
	{

	}
}