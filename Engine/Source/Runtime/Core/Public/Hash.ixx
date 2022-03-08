module;
#include <filesystem>
export module Core.Hash;

namespace SIByL
{
	inline namespace Core
	{
		export class Hash
		{
			static auto path2hash(std::filesystem::path path) noexcept -> uint64_t;
		};
	}
}