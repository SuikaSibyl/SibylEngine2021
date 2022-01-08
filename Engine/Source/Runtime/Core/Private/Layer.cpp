module;
#include <vector>
module Core.Layer;

namespace SIByL::Core
{
	LayerStack::~LayerStack()
	{
		for (ILayer* layer : layers)
			delete layer;
	}

	void LayerStack::pushLayer(ILayer* layer)
	{
		layers.emplace(layers.begin() + layer_insert_index, layer);
		layer_insert_index++;
	}

	void LayerStack::pushOverlay(ILayer* overlay)
	{
		layers.emplace_back(overlay);
	}

	void LayerStack::popLayer(ILayer* layer)
	{
		for (auto iter = layers.begin(); iter != layers.end(); iter++)
		{
			if (*iter == layer)
			{
				layers.erase(iter);
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
		auto it = std::find(layers.begin(), layers.end(), overlay);
		if (it != layers.end())
			layers.erase(it);
	}
}