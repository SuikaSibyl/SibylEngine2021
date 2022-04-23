module;
#include <string>
export module Core.String;

namespace SIByL
{
	export inline auto to_hex_string(uint64_t const& i) noexcept -> std::string
	{
		std::string result = "0x";
		uint64_t number = i;
		uint64_t nibble = 0;
		char hexes[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
		for (int i = 64 / 4 - 1; i >= 0; i--)
		{
			nibble = number >> i * 4;
			nibble = nibble & 0xf;
			result.push_back(hexes[nibble]);
		}
		return result;
	}
}