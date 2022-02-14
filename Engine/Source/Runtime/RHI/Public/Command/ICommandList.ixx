module;

export module RHI.ICommandList;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		// Command Lists are groups of command buffers pushed in batches to the GPU. 
		// The reason for doing this is to keep the GPU constantly busy, 
		// leading to less de-synchronization between the CPU and GPU [Foley 2015].
		//
		// ╭──────────────┬───────────────────────────────────────────────────╮						 
		// │  Vulkan	  │   vk::SubmitInfo                                  │
		// │  DirectX 12  │   ID3D12CommandList[]                             │
		// │  OpenGL      │   Intenal to Driver or with GL_NV_command_list    │
		// ╰──────────────┴───────────────────────────────────────────────────╯

		export class ICommandList :public IResource
		{
		public:
			ICommandList() = default;
			ICommandList(ICommandList&&) = default;
			virtual ~ICommandList() = default;


		};
	}
}
