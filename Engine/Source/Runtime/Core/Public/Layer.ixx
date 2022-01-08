module;
#include <vector>
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

			virtual void onAwake() {}
			virtual void onShutdown() {}
			virtual void onUpdate() {}
			virtual void onAttach() {}
			virtual void onDetach() {}
			virtual void onDraw() {}
			virtual void onEvent(Event& event) {}

			inline std::string_view getName() const { return layerName; }

		protected:
			std::string_view layerName;
		};

		export class LayerStack final
		{
		public:
			LayerStack() = default;
			~LayerStack();

			void pushLayer(ILayer* layer);
			void pushOverlay(ILayer* overlay);
			void popLayer(ILayer* layer);
			void popOverlay(ILayer* overlay);

			//using itestd::vector<ILayer*>::iteratorrator = std::vector<ILayer*>::iterator;
			std::vector<ILayer*>::iterator begin() { return layers.begin(); }
			std::vector<ILayer*>::iterator end() { return layers.end(); }

		//private:
			std::vector<ILayer*> layers;
			unsigned int layer_insert_index = 0;
		};
	}
}