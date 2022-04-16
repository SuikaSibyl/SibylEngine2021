module;
#include <unordered_map>
export module Asset.RuntimeAssetLibrary;
import Core.MemoryManager;
import Asset.Asset;
import Asset.Mesh;

namespace SIByL::Asset
{
	export
	template <class T>
	struct DedicateAssetLibrary
	{
		auto tryFind(GUID const& guid) noexcept -> T*;
		auto emplace(GUID const& guid, MemScope<T>&& rv) noexcept -> T*;

		std::unordered_map<GUID, MemScope<T>> library;
	};

	template <class T>
	auto DedicateAssetLibrary<T>::tryFind(GUID const& guid) noexcept -> T*
	{
		auto iter = library.find(guid);
		if (iter != library.end()) return iter->second.get();
		else return nullptr;
	}

	template <class T>
	auto DedicateAssetLibrary<T>::emplace(GUID const& guid, MemScope<T>&& rv) noexcept -> T*
	{
		library.emplace(guid, std::move(rv));
		return tryFind(guid);
	}

	export struct RuntimeAssetLibrary
	{

		DedicateAssetLibrary<Mesh> meshLib;
	};
}