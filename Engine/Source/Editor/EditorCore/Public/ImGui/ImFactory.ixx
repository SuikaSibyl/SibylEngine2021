module;
export module Editor.ImFactory;

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
	export struct ImFactory
	{
		ImFactory(ImGuiLayer* _layer) :layer(_layer) {}

		auto createImImage(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout) noexcept -> MemScope<ImImage>;

		ImGuiLayer* layer;
	};
}