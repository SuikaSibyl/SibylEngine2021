#pragma once

namespace SIByL
{
	namespace Graphic
	{
		class IFramebuffer
		{
		public:
			virtual ~IFramebuffer() noexcept = default;

			struct Desc
			{
				//uint32_t renderTargetCount;
				//IResourceView* const* renderTargetViews;
				//IResourceView* depthStencilView;
				//IFramebufferLayout* layout;
			};



		};
	}
}