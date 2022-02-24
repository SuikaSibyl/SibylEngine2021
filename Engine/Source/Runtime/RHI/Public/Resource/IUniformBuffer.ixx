module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
export module RHI.IUniformBuffer;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IUniformBuffer
		{
		public:
			IUniformBuffer() = default;
			virtual ~IUniformBuffer() = default;

			virtual auto updateBuffer(Buffer* buffer) noexcept -> void = 0;
			virtual auto getSize() noexcept -> uint32_t = 0;
		};
	}
}