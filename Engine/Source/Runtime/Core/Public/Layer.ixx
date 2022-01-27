module;
#include <string_view>
export module Core.Layer;

import Core.Event;

namespace SIByL
{
	inline namespace Core
	{
		export class ILayer
		{
		public:
			ILayer(std::string_view name = "NewLayer")
				:layerName(name) {}

			virtual ~ILayer() = default;

			virtual void onAwake();
			virtual void onShutdown();
			virtual void onUpdate();
			virtual void onAttach();
			virtual void onDetach();
			virtual void onDraw();
			virtual void onEvent(Event& event);

			inline std::string_view getName() const { return layerName; }

		protected:
			std::string_view layerName;
		};
	}
}