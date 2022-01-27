module;
#include <string_view>
#include <vector>
#include <unordered_map>
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
		export struct DescWindow
		{
			EWindowVendor vendor;
			uint32_t const& width;
			uint32_t const& height;
			std::string_view name;
		};

		export class IApplication
		{
		public:
			IApplication();
			virtual ~IApplication();

			void awake();
			void mainLoop();
			void shutdown();

			void onEvent(Event& e);
			bool onWindowClose(WindowCloseEvent& e);

			auto addWindow(DescWindow const& desc) noexcept -> WindowLayer*;

			void pushLayer(ILayer* layer);
			void pushOverlay(ILayer* overlay);
			void popLayer(ILayer* layer);
			void popOverlay(ILayer* overlay);

		private:
			bool is_running = false;
			LayerStack layer_stack;
			std::vector<WindowLayer*> window_layers;
		};
	}
}