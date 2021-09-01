#include "SIByLpch.h"
#include "Shader.h"

#include "SIByLsettings.h"
#include "Sibyl/Renderer/Renderer.h"

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
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLShader>(ShaderPath + file + ".glsl", desc, binderDesc); break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12Shader>(ShaderPath + file + ".hlsl", binderDesc, desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
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

}