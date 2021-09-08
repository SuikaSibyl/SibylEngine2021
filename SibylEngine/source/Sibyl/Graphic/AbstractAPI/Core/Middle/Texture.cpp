#include "SIByLpch.h"
#include "Texture.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "SIByLsettings.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"

namespace SIByL
{
	Ref<Texture2D> Texture2D::Create(const std::string& path)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLTexture2D>(path); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Texture2D>(path); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	Ref<Texture2D> Texture2D::Create(Ref<Image> image)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLTexture2D>(image); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Texture2D>(image); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}
}