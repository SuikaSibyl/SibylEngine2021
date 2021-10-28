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
		
		Ref<Texture2D> image = nullptr;

		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: image = std::make_shared<OpenGLTexture2D>(AssetRoot + path); break;
		case RasterRenderer::DirectX12: image = std::make_shared<DX12Texture2D>(AssetRoot + path); break;
		case RasterRenderer::CpuSoftware: image = nullptr; break;
		case RasterRenderer::GpuSoftware: image = nullptr; break;
		default: image = nullptr; break;
		}

		image->Identifer = id;
		Library<Texture2D>::Push(id, image);

		return image;
	}

	Ref<Texture2D> Texture2D::Create(Ref<Image> image, const std::string ID)
	{
		std::string id = "PROCEDURE=" + ID;
		Ref<Texture2D> texture = Library<Texture2D>::Fetch(id);

		if (texture == nullptr)
		{
			switch (Renderer::GetRaster())
			{
			case RasterRenderer::OpenGL: texture = std::make_shared<OpenGLTexture2D>(image); break;
			case RasterRenderer::DirectX12: texture = std::make_shared<DX12Texture2D>(image); break;
			case RasterRenderer::CpuSoftware: texture = nullptr; break;
			case RasterRenderer::GpuSoftware: texture = nullptr; break;
			default: texture = nullptr; break;
			}

			texture->Identifer = id;
			Library<Texture2D>::Push(id, texture);
		}

		return texture;
	}

	Ref<TextureCubemap> TextureCubemap::Create(const std::string& path)
	{
		std::string id = GetPathID(path);
		Ref<TextureCubemap> texture = nullptr;

		if (texture == nullptr)
		{
			switch (Renderer::GetRaster())
			{
			case RasterRenderer::OpenGL: texture = std::make_shared<OpenGLTextureCubemap>(path); break;
			case RasterRenderer::DirectX12: texture = nullptr; break;
			case RasterRenderer::CpuSoftware: texture = nullptr; break;
			case RasterRenderer::GpuSoftware: texture = nullptr; break;
			default: texture = nullptr; break;
			}

			texture->Identifer = id;
			Library<TextureCubemap>::Push(id, texture);
		}

		return texture;
	}

}