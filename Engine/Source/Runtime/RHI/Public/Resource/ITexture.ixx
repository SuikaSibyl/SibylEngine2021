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

			virtual auto transitionImageLayout(ImageLayout old_layout, ImageLayout new_layout) noexcept -> void = 0;
			virtual auto createView(TextureViewDesc const& desc) noexcept -> MemScope<ITextureView> = 0;
			virtual auto getDescription() noexcept -> TextureDesc const& = 0;
		};

		export enum class ImageAspectFlagBits :uint32_t
		{
			COLOR_BIT = 0x00000001,
			DEPTH_BIT = 0x00000002,
			STENCIL_BIT = 0x00000004,
			METADATA_BIT = 0x00000008,
			PLANE_0_BIT = 0x00000010,
			PLANE_1_BIT = 0x00000020,
			PLANE_2_BIT = 0x00000040,
			MEMORY_PLANE_0_BIT = 0x00000080,
			MEMORY_PLANE_1_BIT = 0x00000100,
			MEMORY_PLANE_2_BIT = 0x00000200,
			MEMORY_PLANE_3_BIT = 0x00000400,
		};
		export using ImageAspectFlags = uint32_t;

		export struct ImageSubresourceRange {
			ImageAspectFlags aspectMask;
			uint32_t baseMipLevel;
			uint32_t levelCount;
			uint32_t baseArrayLayer;
			uint32_t layerCount;
		};

		export struct BufferImageCopyDesc
		{
			// src from buffer
			uint64_t bufferOffset;
			uint32_t bufferRowLength;
			uint32_t bufferImageHeight;
			// dst to image
			// - subresource
			ImageAspectFlags aspectMask;
			uint32_t mipLevel;
			uint32_t baseArrayLayer;
			uint32_t layerCount;
			int32_t imageOffsetX, imageOffsetY, imageOffsetZ;
			uint32_t imageExtentX, imageExtentY, imageExtentZ;
		};
		export struct IBufferImageCopy
		{};
	}
}
