module;
#include <cstdint>
#include <vector>
export module RHI.IRenderPass;
import Core.SObject;
import Core.Color;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		// Render Passes : the attachments referenced by the pipeline stages and their usage
		// specify how many color and depth buffers there will be,
		// how many samples to use for each of them 
		// and how their contents should be handled throughout the rendering operations
		// 
		// ╭──────────────┬─────────────────╮
		// │  Vulkan	  │   VkRenderPass  │
		// │  DirectX 12  │   render pass   │
		// │  OpenGL      │                 │
		// ╰──────────────┴─────────────────╯

		export class IRenderPass :public SObject
		{
		public:
			IRenderPass() = default;
			IRenderPass(IRenderPass&&) = default;
			virtual ~IRenderPass() = default;


		};

		export enum class AttachmentLoadOp :uint32_t
		{
			LOAD,
			CLEAR,
			DONT_CARE,
		};
		
		export enum class AttachmentStoreOp :uint32_t
		{
			STORE,
			DONT_CARE,
		};

		export struct AttachmentDesc
		{
			SampleCount samples;
			ResourceFormat format;
			AttachmentLoadOp loadOp;
			AttachmentStoreOp storeOp;
			AttachmentLoadOp stencilLoadOp;
			AttachmentStoreOp stencilStoreOp;
			ImageLayout initalLayout;
			ImageLayout finalLayout;
			ColorFloat4 clearColor;
		};

		export struct RenderPassDesc
		{
			std::vector<AttachmentDesc> colorAttachments;
			std::vector<AttachmentDesc> depthstencialAttachments;
		};
	}
}
