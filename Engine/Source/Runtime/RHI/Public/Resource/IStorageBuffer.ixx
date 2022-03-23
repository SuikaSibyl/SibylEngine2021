module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
export module RHI.IStorageBuffer;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IStorageBuffer
		{
		public:
			IStorageBuffer() = default;
			virtual ~IStorageBuffer() = default;

			virtual auto getSize() noexcept -> uint32_t = 0;
			virtual auto getIBuffer() noexcept -> IBuffer* = 0;
		};
	}
}