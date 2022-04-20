module;
#include <vector>
#include <functional>
#include <Macros.h>
#include <string_view>
module UAT.IUniversalApplication;
import Core.Application;
import Core.Window;
import Core.Layer;
import Core.LayerStack;
import Core.Enums;
import Core.Event;

namespace SIByL
{
	inline namespace UAT
	{
		auto IUniversalApplication::attachWindowLayer(WindowLayerDesc const& desc) -> WindowLayer*
		{
			WindowLayer* window_layer = new WindowLayer(
				SIByL::EWindowVendor::GLFW,
				onEventCallbackFn,
				1280,
				720,
				"Hello");

			pushLayer(window_layer);
			windowLayers.push_back(window_layer);

			return window_layer;
		}

		auto IUniversalApplication::onWindowClose(WindowCloseEvent& e) -> bool
		{
			for (auto iter = windowLayers.begin(); iter != windowLayers.end(); iter++)
			{
				if ((*iter)->getWindow() == (IWindow*)e.window)
				{
					//delete (*iter);
					//layer_stack.popLayer((ILayer*)*iter);
					windowLayers.erase(iter);
					break;
				}
			}

			if (windowLayers.size() == 0)
				is_running = false;
			return true;
		}
	}
}