module;
#include <vulkan/vulkan.h>
export module RHI.IResource.VK;
import RHI.IResource;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export class IResourceVK : public IResource
		{
		public:
			auto getVKFormat() noexcept -> VkFormat;
			auto getVKImageViewType() noexcept -> VkImageViewType;

			static auto VKFormat2IFormat(VkFormat format) noexcept -> ResourceFormat;
			static auto VKImageViewType2IType(VkImageViewType type) noexcept -> ResourceType;

			auto setVKFormat(VkFormat format) noexcept -> bool;
			auto setVKImageViewType(VkImageViewType type) noexcept -> bool;
		};
	}
}