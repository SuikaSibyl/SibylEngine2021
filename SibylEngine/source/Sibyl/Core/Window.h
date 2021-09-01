#pragma once

#include "Core.h"
#include "Sibyl/Events/Event.h"
#include "Sibyl/Graphic/AbstractAPI/Top/GraphicContext.h"

namespace SIByL
{
	struct WindowProps
	{
		std::string Title;
		unsigned int Width;
		unsigned int Height;

		WindowProps(const std::string& title = "SIByL Engine",
			unsigned int width = 1280,
			unsigned int height = 720)
			:Title(title), Width(width), Height(height)
		{}
	};

	// Interface representing a desktop system based Window
	class SIByL_API Window
	{
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() {
		
		}

		virtual void OnUpdate() = 0;

		virtual unsigned int GetWidth() const = 0;
		virtual unsigned int GetHeight() const = 0;

		// Window attributes
		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		virtual void* GetNativeWindow() const = 0;

		static Ref<Window> Create(const WindowProps& props = WindowProps());
		GraphicContext* GetGraphicContext() { return m_GraphicContext; }

	protected:
		GraphicContext* m_GraphicContext;
	};
}