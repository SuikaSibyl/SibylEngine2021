module;
#include <string_view>
export module Asset.AssetLayer;
import Core.Layer;
import Asset.RuntimeAssetManager;

namespace SIByL::Asset
{
	export struct AssetLayer :public ILayer
	{
		AssetLayer() :ILayer("Asset Layer") {}

		RuntimeAssetManager runtimeManager;
	};
}