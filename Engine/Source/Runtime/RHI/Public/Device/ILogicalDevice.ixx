module;

export module RHI.ILogicalDevice;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// A Device gives you access to the core inner functions of the API
		// such as creating graphics data structures like textures, buffers, queues, pipelines, etc. 
		// ╭──────────────┬─────────────────╮
		// │  Vulkan	  │   vk::Device    │
		// │  DirectX 12  │   ID3D12Device  │
		// │  OpenGL      │   N/A		    │
		// ╰──────────────┴─────────────────╯

		export class ILogicalDevice :public SObject
		{
		public:
			virtual ~ILogicalDevice() = default;

		};
	}
}
