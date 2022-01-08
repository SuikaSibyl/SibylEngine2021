module;
#include <vector>
#include <string_view>
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

	}

	IApplication::~IApplication()
	{

	}

	void IApplication::awake()
	{
		is_running = true;
	}

	void IApplication::mainLoop()
	{
		while (is_running)
		{
			// Update
			for (int i = 0; i < layer_stack.layers.size(); i++)
			{
				layer_stack.layers[i]->onUpdate();
			}
		}
	}

	void IApplication::shutdown()
	{
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
		for (auto iter = window_layers.begin(); iter != window_layers.end(); iter++)
		{
			if ((*iter)->getWindow() == (IWindow*)e.window)
			{
				delete (*iter);
				layer_stack.popLayer((ILayer*) *iter);
				window_layers.erase(iter);
				break;
			}
		}
		
		if (window_layers.size() == 0)
			is_running = false;
		return true;
	}

	void IApplication::addWindow(DescWindow const& desc)
	{
		WindowLayer* window_layer = new WindowLayer(
			desc.vendor,
			BIND_EVENT_FN(IApplication::onEvent), 
			desc.width, 
			desc.height, 
			desc.name);

		pushLayer(window_layer);
		window_layers.push_back(window_layer);
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