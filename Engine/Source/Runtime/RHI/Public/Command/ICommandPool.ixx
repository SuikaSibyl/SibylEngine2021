module;

export module RHI.ICommandPool;
import Core.SObject;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		// A Command Pool is a data structure that allows you to create command buffers.
		// ╭──────────────┬────────────────────────────╮
		// │  Vulkan	  │   vk::CommandPool	       │
		// │  DirectX 12  │   ID3D12CommandAllocator   │
		// │  OpenGL      │   N/A					   │
		// ╰──────────────┴────────────────────────────╯

		export class ICommandPool :public SObject
		{
		public:
			ICommandPool() = default;
			ICommandPool(ICommandPool const&) = delete;
			ICommandPool(ICommandPool&&) = delete;
			virtual ~ICommandPool() = default;
		};
	}
}
