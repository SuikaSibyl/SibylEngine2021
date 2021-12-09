#pragma once

namespace SIByL
{
	namespace Graphic
	{
		class IResourceView
		{
		public:
			virtual ~IResourceView() noexcept = default;

			enum class Type
			{
				Unknown,

				RenderTarget,
				DepthStencil,
				ShaderResource,
				UnorderedAccess,
				AccelerationStructure,
			};


		};
	}
}