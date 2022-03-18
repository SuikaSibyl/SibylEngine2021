module;
#include <vector>
#include <string_view>
module Core.LayerStack;

import Core.Event;
import Core.Layer;

using namespace std;

namespace SIByL::Core
{
	LayerStack::~LayerStack()
	{}

	void LayerStack::pushLayer(ILayer* layer)
	{
		layer_stack.emplace(layer_stack.begin() + layer_insert_index, layer);
		layer_insert_index++;
	}

	void LayerStack::pushOverlay(ILayer* overlay)
	{
		layer_stack.emplace_back(overlay);
	}

	void LayerStack::popLayer(ILayer* layer)
	{
		for (auto iter = layer_stack.begin(); iter != layer_stack.end(); iter++)
		{
			if (*iter == layer)
			{
				layer_stack.erase(iter);
				break;
			}
		}
		//auto it = std::find(layers.begin(), layers.end(), layer);
		//if (it != layers.end())
		//{
		//	it = layers.erase(it);
		//	layer_insert_index--;
		//}
	}

	void LayerStack::popOverlay(ILayer* overlay)
	{
		auto it = std::find(layer_stack.begin(), layer_stack.end(), overlay);
		if (it != layer_stack.end())
			layer_stack.erase(it);
	}
}