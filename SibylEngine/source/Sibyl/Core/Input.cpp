#include "SIByLpch.h"
#include "Input.h"

#include "Sibyl/Renderer/Renderer.h"
#include "Platform/OpenGL/Window/GLFWInput.h"
#include "Platform/Windows/Window/WindowsInput.h"

namespace SIByL
{
	Input* Input::s_Instance = nullptr;

	void Input::Init()
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: s_Instance = new GLFWInput(); break;
		case RasterRenderer::DirectX12: s_Instance = new WindowsInput(); break;
		case RasterRenderer::CpuSoftware: return; break;
		case RasterRenderer::GpuSoftware: return; break;
		default: return; break;
		}
	}
}