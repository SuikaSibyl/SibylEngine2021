module;
#include <cstdint>
#include <functional>
#include <string_view>
#include <functional>
module Core.Window;
import Core.Enums;
import Core.Event;
import Core.Input;
import Core.Layer;
import Core.SObject;
import Core.Window.GLFW;

namespace SIByL::Core
{
	IWindow::IWindow(const wchar_t* title)
	{

	}

	IWindow* IWindowFactory(
		EWindowVendor vendor,
		uint32_t const& width,
		uint32_t const& height,
		std::string_view name)
	{
		switch (vendor)
		{
		case SIByL::Core::EWindowVendor::GLFW:
			return new IWindowGLFW(width, height, name);
			break;
		case SIByL::Core::EWindowVendor::WINDOWS:
			break;
		default:
			break;
		}

		return nullptr;
	}

	WindowLayer::WindowLayer(
		EWindowVendor vendor,
		EventCallbackFn event_callback,
		uint32_t const& width,
		uint32_t const& height,
		std::string_view name)
	{
		window.reset(IWindowFactory(vendor, width, height, name));
		window->setEventCallback(event_callback);
	}

	void WindowLayer::onUpdate()
	{
		window->onUpdate();
	}

	void WindowLayer::onShutdown()
	{
		window.reset(nullptr);
	}

	IWindow* WindowLayer::getWindow()
	{
		return window.get();
	}

}