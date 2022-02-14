module;

export module RHI.IBarrier;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// Barriers: A more granular form of synchronization, inside command buffers.
		// ╭──────────────┬───────────────────────────╮
		// │  Vulkan	  │   vkCmdPipelineBarrier    │
		// │  DirectX 12  │   D3D12_RESOURCE_BARRIER  │
		// │  OpenGL      │   glMemoryBarrier         │
		// ╰──────────────┴───────────────────────────╯

		export class IBarrier :public SObject
		{
		public:
			IBarrier() = default;
			IBarrier(IBarrier&&) = default;
			virtual ~IBarrier() = default;

		private:
		};
	}
}
