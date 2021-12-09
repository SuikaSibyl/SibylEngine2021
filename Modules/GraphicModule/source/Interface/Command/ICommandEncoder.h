#pragma once

namespace SIByL
{
	namespace Graphic
	{
		class ICommandEncoder
		{
		public:
			virtual ~ICommandEncoder() noexcept = default;

			virtual void endEncoding() noexcept = 0;
			//virtual void writeTimestamp(IQueryPool* queryPool, SlangInt queryIndex) noexcept  = 0;

		};
	}
}