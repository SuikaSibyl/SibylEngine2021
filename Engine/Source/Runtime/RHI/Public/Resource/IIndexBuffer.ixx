module;
#include <cstdint>
export module RHI.IIndexBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IIndexBuffer
		{
		public:
			virtual ~IIndexBuffer() = default;
			virtual auto getElementSize() noexcept -> uint32_t = 0;
			virtual auto getIndicesCount() noexcept -> uint32_t = 0;

		private:

		};
	}
}