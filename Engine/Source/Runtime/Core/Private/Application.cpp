module;
#include <vector>
#include <string_view>
#include <unordered_map>
#include <functional>
#include <Macros.h>
module Core.Application;

import Core.Event;
import Core.Layer;
import Core.Enums;

namespace SIByL::Core
{
	IApplication::IApplication()
	{
		onEventCallbackFn = BIND_EVENT_FN(IApplication::onEvent);
	}

	IApplication::~IApplication()
	{

	}

	void IApplication::awake()
	{
		is_running = true;
		onAwake();
	}

	void IApplication::mainLoop()
	{
		while (is_running)
		{
			onUpdate();
			// Update
			for (int i = 0; i < layer_stack.layer_stack.size(); i++)
			{
				layer_stack.layer_stack[i]->onUpdate();
			}
		}
	}

	void IApplication::shutdown()
	{
		onShutdown();
		for (ILayer* layer : layer_stack)
		{
			layer->onShutdown();
		}
	}

	void IApplication::onEvent(Event& e)
	{
		// application handling
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(IApplication::onWindowClose));
		dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_FN(IApplication::onWindowResizeSafe));

		if (e.handled)
			return;

		// layer_stack handling
		for (auto it = layer_stack.end(); it != layer_stack.begin();)
		{
			(*--it)->onEvent(e);
			if (e.handled)
				break;
		}
	}

	bool IApplication::onWindowClose(WindowCloseEvent& e)
	{
		return false;
	}

	auto IApplication::onWindowResizeSafe(WindowResizeEvent& e) -> bool
	{
		if (e.GetWidth() == 0 && e.GetHeight() == 0)
		{
			((IWindow*)e.GetWindowPtr())->waitUntilNotMinimized(*e.GetWidthPtr(), *e.GetHeightPtr());
		}
		return onWindowResize(e);
	}

	auto IApplication::onWindowResize(WindowResizeEvent& e) -> bool
	{
		return false;
	}

	void IApplication::pushLayer(ILayer* layer)
	{
		layer_stack.pushLayer(layer);
	}

	void IApplication::pushOverlay(ILayer* overlay)
	{
		layer_stack.pushOverlay(overlay);
	}

	void IApplication::popLayer(ILayer* layer)
	{
		layer_stack.popLayer(layer);
	}

	void IApplication::popOverlay(ILayer* overlay)
	{
		layer_stack.popOverlay(overlay);
	}
}