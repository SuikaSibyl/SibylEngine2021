export module RHI.IPhysicalDevice;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// Physical Devices allow you to query for important device specific details such as memory size and feature support.
		// ╭──────────────┬────────────────────────────╮
		// │  Vulkan	  │   vk::PhysicalDevice       │
		// │  DirectX 12  │   IDXGIAdapter             │
		// │  OpenGL      │   glGetString(GL_VENDOR)   │
		// ╰──────────────┴────────────────────────────╯

		export class IPhysicalDevice :public SObject
		{
		public:
			virtual ~IPhysicalDevice() = default;

			virtual auto isDebugLayerEnabled() noexcept -> bool { return true; }
		};
	}
}
