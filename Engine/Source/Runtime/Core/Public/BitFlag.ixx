module;
#include <cstdint>
export module Core.BitFlag;

namespace SIByL
{
	inline namespace Core
	{
		export template <class T>
		auto addBit(T const& bit) { return (uint32_t)bit; }

		export template <class T>
		auto hasBit(uint32_t flags, T const& bit) { return flags & (uint32_t)bit; }
	}
}