module;
#include <string_view>
#include <vector>
#include <unordered_map>
#include <functional>
export module Core.Application;

import Core.Layer;
import Core.LayerStack;
import Core.Event;
import Core.Window;
import Core.Enums;

namespace SIByL
{
	inline namespace Core
	{
		export class IApplication
		{
		public:
			IApplication();
			virtual ~IApplication();

			virtual void onAwake() {}
			virtual void onUpdate() {}
			virtual void onShutdown() {}
			virtual bool onWindowClose(WindowCloseEvent& e);
			virtual bool onKeyPressedEvent(KeyPressedEvent& e);
			auto onWindowResizeSafe(WindowResizeEvent& e) -> bool;
			virtual auto onWindowResize(WindowResizeEvent& e) -> bool;

			void awake();
			void mainLoop();
			void shutdown();

			void onEvent(Event& e);

			void pushLayer(ILayer* layer);
			void pushOverlay(ILayer* overlay);
			void popLayer(ILayer* layer);
			void popOverlay(ILayer* overlay);

		protected:
			bool is_running = false;
			LayerStack layer_stack;
			EventCallbackFn onEventCallbackFn;
		};
	}
}