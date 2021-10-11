#include "SIByLpch.h"
#include "Shader.h"

#include "SIByLsettings.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"

#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLShader.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Shader.h"

namespace SIByL
{
	Ref<Shader> Shader::Create()
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLShader>(); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Shader>(); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	Ref<Shader> Shader::Create(std::string file, const ShaderDesc& desc, const ShaderBinderDesc& binderDesc)
	{
		std::string id = GetPathID(file);
		Ref<Shader> shader = Library<Shader>::Fetch(id);

		if (shader == nullptr)
		{
			switch (Renderer::GetRaster())
			{
			case RasterRenderer::OpenGL: shader = std::make_shared<OpenGLShader>(AssetRoot + file + ".glsl", desc, binderDesc); break;
			case RasterRenderer::DirectX12: shader = std::make_shared<DX12Shader>(AssetRoot + file + ".hlsl", binderDesc, desc); break;
			case RasterRenderer::CpuSoftware: shader = nullptr; break;
			case RasterRenderer::GpuSoftware: shader = nullptr; break;
			default: shader = nullptr; break;
			}

			Library<Shader>::Push(id, shader);
		}

		shader->ShaderID = file;
		return shader;
	}
	
	Ref<Shader> Shader::Create(std::string vFile, std::string pFile, const ShaderDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLShader>(ShaderPath + vFile + ".vert", ShaderPath + pFile + ".frag", desc); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Shader>(ShaderPath + vFile + ".hlsl", ShaderPath + pFile + ".hlsl", desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	////////////////////////////////////////////////////////////////
	//						Compute Shader						 ///
	////////////////////////////////////////////////////////////////
	Ref<ComputeShader> ComputeShader::Create(std::string file, const ShaderBinderDesc& binderDesc)
	{
		std::string id = GetPathID(file);
		Ref<ComputeShader> shader = Library<ComputeShader>::Fetch(id);

		if (shader == nullptr)
		{
			switch (Renderer::GetRaster())
			{
			case RasterRenderer::OpenGL: shader = std::make_shared<OpenGLComputeShader>(AssetRoot + file + ".glsl", binderDesc); break;
			case RasterRenderer::DirectX12: shader = nullptr; break;
			case RasterRenderer::CpuSoftware: shader = nullptr; break;
			case RasterRenderer::GpuSoftware: shader = nullptr; break;
			default: shader = nullptr; break;
			}

			Library<ComputeShader>::Push(id, shader);
		}

		shader->ShaderID = file;
		return shader;
	}

}