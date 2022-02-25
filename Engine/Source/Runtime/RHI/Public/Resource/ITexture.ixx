module;
#include <vulkan/vulkan.h>
export module RHI.ITexture;
import Core.SObject;
import Core.MemoryManager;
import RHI.ITextureView;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		// Textures are arrays of data that store color information, 
		// and serve as inputs/outputs for rendering. 
		// 
		// Vulkan, DirectX 12, and WebGPU introduce the idea of 
		// having multiple views of a given texture that can view that texture 
		// in different encoded formats or color spaces. 
		// 
		// Vulkan introduces the idea of managed memory for Images and buffers, 
		// thus a texture is a triplet of an Image, 
		// Image View when used (there can be multiple of these), 
		// and Memory in either device only or in CPU-GPU accessible space.
		// 
		// ╭──────────────┬───────────────────────────────╮
		// │  Vulkan	  │   vk::Image & vk::ImageView   │
		// │  DirectX 12  │   ID3D12Resource              │
		// │  OpenGL      │   GLuint                      │
		// ╰──────────────┴───────────────────────────────╯

		export struct TextureDesc
		{
			ResourceType type;
			ResourceFormat format;
			ImageTiling tiling;
			ImageUsageFlags usages;
			BufferShareMode shareMode;
			SampleCount sampleCount;
			ImageLayout layout;
			uint32_t width;
			uint32_t height;
		};

		export class ITexture :public SObject
		{
		public:
			ITexture() = default;
			ITexture(ITexture&&) = default;
			ITexture(ITexture const&) = delete;
			virtual ~ITexture() = default;

			virtual auto transitionImageLayout(ImageLayout new_layout) noexcept -> void = 0;
			virtual auto createView(TextureViewDesc const& desc) noexcept -> MemScope<ITextureView> = 0;
		};
	}
}
