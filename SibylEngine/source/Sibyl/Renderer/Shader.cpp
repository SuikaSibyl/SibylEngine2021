#include "SIByLpch.h"
#include "Shader.h"

#include "SIByLsettings.h"
#include "Sibyl/Renderer/Renderer.h"

#include "Platform/OpenGL/Renderer/OpenGLShader.h"
#include "Platform/DirectX12/Renderer/DX12Shader.h"

namespace SIByL
{
	Shader* Shader::Create()
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new OpenGLShader(); break;
		case RasterRenderer::DirectX12: return new DX12Shader(); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	Shader* Shader::Create(std::string vFile, std::string pFile)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new OpenGLShader(ShaderPath + vFile, ShaderPath + pFile); break;
		case RasterRenderer::DirectX12: return new DX12Shader(ShaderPath + vFile, ShaderPath + pFile); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

}