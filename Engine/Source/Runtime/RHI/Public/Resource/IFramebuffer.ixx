module;
#include <cstdint>
#include <vector>
export module RHI.IFramebuffer;
import Core.Color;
import RHI.IResource;
import RHI.ITextureView;
import RHI.IRenderPass;

namespace SIByL
{
	namespace RHI
	{
		// Frame Buffers Are groups of output textures used during a raster based graphics pipeline execution as outputs.
		// ╭──────────────┬─────────────────────╮
		// │  Vulkan	  │   vk::Framebuffer   │
		// │  DirectX 12  │   ID3D12Resource    │
		// │  OpenGL      │   GLuint            │
		// ╰──────────────┴─────────────────────╯

		export struct FramebufferDesc
		{
			uint32_t width, height;
			IRenderPass* renderPass;
			std::vector<ITextureView*> attachments;
		};

		export class IFramebuffer :public IResource
		{
		public:
			IFramebuffer() = default;
			IFramebuffer(IFramebuffer&&) = default;
			virtual ~IFramebuffer() = default;


		};
	}
}
