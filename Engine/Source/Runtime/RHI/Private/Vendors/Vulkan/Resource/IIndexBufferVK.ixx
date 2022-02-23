module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
export module RHI.IIndexBuffer.VK;
import RHI.IIndexBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IIndexBufferVK :public IIndexBuffer
		{
		public:
			virtual ~IIndexBufferVK() = default;

		private:

		};
	}
}