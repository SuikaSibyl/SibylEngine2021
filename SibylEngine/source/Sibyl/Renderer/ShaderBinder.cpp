#include "SIByLpch.h"
#include "ShaderBinder.h"

#include "SIByLsettings.h"
#include "Sibyl/Renderer/Renderer.h"

#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"

namespace SIByL
{
	ShaderBinder* ShaderBinder::Create()
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return nullptr; break;
		case RasterRenderer::DirectX12: return new DX12ShaderBinder(); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

}