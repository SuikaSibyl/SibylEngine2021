module;

export module RHI.IBuffer;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		// A Buffer is an array of data, such as a mesh's positional data, color data, 
		// index data, etc. Similar rules for images apply to buffers in Vulkan and WebGPU.
		// ╭──────────────┬──────────────────────────────────╮
		// │  Vulkan	  │   vk::Buffer & vk::BufferView    │
		// │  DirectX 12  │   ID3D12Resource                 │
		// │  OpenGL      │   Varies by OS                   │
		// ╰──────────────┴──────────────────────────────────╯

		export class IBuffer :public IResource
		{
		public:
			IBuffer() = default;
			IBuffer(IBuffer&&) = default;
			virtual ~IBuffer() = default;


		};
	}
}
