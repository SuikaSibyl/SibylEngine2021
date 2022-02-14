module;

export module RHI.IFramebuffer;
import RHI.IResource;

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

		export class IFramebuffer :public IResource
		{
		public:
			IFramebuffer() = default;
			IFramebuffer(IFramebuffer&&) = default;
			virtual ~IFramebuffer() = default;


		};
	}
}
