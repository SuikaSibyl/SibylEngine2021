module;
#include <vector>
#include <functional>
#include <Macros.h>
#include <string_view>
export module UAT.IUniversalApplication;
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
		export class IUniversalApplication :public IApplication
		{
		public:
			auto attachWindowLayer(WindowLayerDesc const& desc) -> WindowLayer*;
			auto onWindowClose(WindowCloseEvent& e) -> bool;

			
		private:
			std::vector<WindowLayer*> windowLayers;
		};
	}
}