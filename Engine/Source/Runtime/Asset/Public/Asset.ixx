module;
#include <cstdint>
export module Asset.Asset;

namespace SIByL::Asset
{
	export using GUID = std::uint64_t;

	export enum struct AssetKind
	{
		Texture,
		Mesh,
	};

	export struct Asset
	{
	public:
		auto guid() noexcept ->GUID { return _guid; }

	private:
		GUID _guid;
	};
}