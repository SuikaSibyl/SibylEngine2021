#include "SIByLpch.h"
#include "Window.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include "Platform/OpenGL/Window/GLFWWindow.h"
#include "Platform/Windows/Window/WindowsWindow.h"

namespace SIByL
{
	Ref<Window> Window::Create(const WindowProps& props)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<GLFWWindow>(props); break;
		case RasterRenderer::DirectX12: return std::make_shared<WindowsWindow>(props); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
	}
}