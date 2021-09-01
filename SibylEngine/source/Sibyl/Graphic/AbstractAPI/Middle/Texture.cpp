#include "SIByLpch.h"
#include "Texture.h"

#include "Sibyl/Renderer/Renderer.h"
#include "SIByLsettings.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"

namespace SIByL
{
	Ref<Texture2D> Texture2D::Create(const std::string& path)
	{
		std::string totalPath = TexturePath + path;
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLTexture2D>(totalPath); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Texture2D>(totalPath); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

}