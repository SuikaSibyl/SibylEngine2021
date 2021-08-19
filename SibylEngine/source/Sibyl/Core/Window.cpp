#include "SIByLpch.h"
#include "Window.h"

#include "Sibyl/Renderer/Renderer.h"
#include "Platform/OpenGL/Window/GLFWWindow.h"
#include "Platform/Windows/Window/WindowsWindow.h"

namespace SIByL
{
	Window* Window::Create(const WindowProps& props)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new GLFWWindow(props); break;
		case RasterRenderer::DirectX12: return new WindowsWindow(props); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
	}
}