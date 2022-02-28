module;
#include <cstdint>
module RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		inline auto sizeofDataType(DataType type) noexcept -> uint32_t
		{
			switch (type)
			{
			case SIByL::RHI::DataType::None: return 0;
				break;
			case SIByL::RHI::DataType::Bool: return 1;
				break;
			case SIByL::RHI::DataType::Int: return 4 * 1;
				break;
			case SIByL::RHI::DataType::Int2: return 4 * 2;
				break;
			case SIByL::RHI::DataType::Int3: return 4 * 3;
				break;
			case SIByL::RHI::DataType::Int4: return 4 * 4;
				break;
			case SIByL::RHI::DataType::Mat3: return 4 * 3 * 3;
				break;
			case SIByL::RHI::DataType::Mat4: return 4 * 4 * 4;
				break;
			case SIByL::RHI::DataType::Float: return 4;
				break;
			case SIByL::RHI::DataType::Float2: return 4 * 2;
				break;
			case SIByL::RHI::DataType::Float3: return 4 * 3;
				break;
			case SIByL::RHI::DataType::Float4: return 4 * 4;
				break;
			default:
				break;
			}
			return 0;
		};

		inline auto flagBitSwitch(uint32_t const& input, uint32_t const& flag, uint32_t const& vendor_flag, uint32_t& target) noexcept -> void
		{
			if (input & flag)
			{
				target |= vendor_flag;
			}
		}
	}
}