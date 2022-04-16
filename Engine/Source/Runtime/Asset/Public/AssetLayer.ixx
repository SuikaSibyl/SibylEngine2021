module;
#include <string_view>
export module Asset.AssetLayer;
import Core.Layer;
import RHI.IFactory;
import Asset.RuntimeAssetManager;
import Asset.RuntimeAssetLibrary;
import Asset.Asset;
import Asset.Mesh;

namespace SIByL::Asset
{
	export struct AssetLayer :public ILayer
	{
		AssetLayer(RHI::IResourceFactory* factory)
			:ILayer("Asset Layer"), resourceFactory(factory) { runtimeManager.initialize(); }
		~AssetLayer() { runtimeManager.exit(); }

		auto mesh(GUID guid) noexcept -> Mesh*;

		RHI::IResourceFactory* resourceFactory;
		RuntimeAssetManager runtimeManager;
		RuntimeAssetLibrary runtimeLibrary;
	};
}