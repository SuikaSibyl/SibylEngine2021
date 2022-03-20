module;
#include <utility>
module Editor.ImFactory;

import Core.MemoryManager;
import RHI.IEnum;
import RHI.ISampler;
import RHI.ITextureView;
import Editor.ImGuiLayer;
import Editor.ImImage;

import RHI.IEnum.VK;
import RHI.ISampler.VK;
import RHI.ITextureView.VK;
import Editor.ImImage.VK;

namespace SIByL::Editor
{
	auto ImFactory::createImImage(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout) noexcept -> MemScope<ImImage>
	{
		MemScope<ImImage> imimage = nullptr;
		switch (layer->api)
		{
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ImImageVK> imimage_vk = MemNew<ImImageVK>(sampler, view, layout);
			imimage = MemCast<ImImage>(imimage_vk);
		}
		break;
		default:
			break;
		}
		return imimage;
	}
}