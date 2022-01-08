module;
#include <vulkan/vulkan.h>
export module Core.GraphicContext.VK;
import Core.GraphicContext;

namespace SIByL
{
	inline namespace Core
	{
		export class IGraphicContextVK
		{
		public:
			auto createInstance() -> void;
			auto checkExtension() -> void;
			auto cleanUp() -> void;
			VkInstance instance;
		};
	}
}