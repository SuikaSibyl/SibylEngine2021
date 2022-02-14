module;

export module RHI.IShaderBinding;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		// Most modern graphics APIs feature a binding data structure to help connect
		// uniform buffers and textures to graphics pipelines that need that data. 
		// ╭──────────────┬────────────────────────────────────────────╮
		// │  Vulkan	  │   vk::PipelineLayout & vk::DescriptorSet   │
		// │  DirectX 12  │   ID3D12RootSignature                      │
		// │  OpenGL      │   GLuint                                   │
		// ╰──────────────┴────────────────────────────────────────────╯

		export class IShaderBinding :public IResource
		{
		public:
			IShaderBinding() = default;
			IShaderBinding(IShaderBinding&&) = default;
			virtual ~IShaderBinding() = default;


		};
	}
}
