module;
#include <vector>
#include <string_view>
export module Core.LayerStack;

import Core.Event;
import Core.Layer;

namespace SIByL
{
	inline namespace Core
	{
		export class LayerStack
		{
		public:
			LayerStack() = default;
			~LayerStack();

			void pushLayer(ILayer* layer);
			void pushOverlay(ILayer* overlay);
			void popLayer(ILayer* layer);
			void popOverlay(ILayer* overlay);

			std::vector<ILayer*> layer_stack;
			unsigned int layer_insert_index = 0;

			//using itestd::vector<ILayer*>::iteratorrator = std::vector<ILayer*>::iterator;
			std::vector<ILayer*>::iterator begin() { return layer_stack.begin(); }
			std::vector<ILayer*>::iterator end() { return layer_stack.end(); }
		};
	}
}