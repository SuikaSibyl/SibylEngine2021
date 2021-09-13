#include "SIByLpch.h"
#include "Texture.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "SIByLsettings.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"

namespace SIByL
{
	Ref<Texture2D> Texture2D::Create(const std::string& path)
	{
		std::string id = GetPathID(path);

		Ref<Texture2D> image = Library<Texture2D>::Fetch(id);

		if (image == nullptr)
		{
			switch (Renderer::GetRaster())
			{
			case RasterRenderer::OpenGL: image = std::make_shared<OpenGLTexture2D>(AssetRoot + path); break;
			case RasterRenderer::DirectX12: image = std::make_shared<DX12Texture2D>(AssetRoot + path); break;
			case RasterRenderer::CpuSoftware: image = nullptr; break;
			case RasterRenderer::GpuSoftware: image = nullptr; break;
			default: image = nullptr; break;
			}

			Library<Texture2D>::Push(id, image);
		}

		return image;
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