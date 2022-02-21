module;
export module RHI.IRenderPass;
import Core.SObject;
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

		export struct RenderPassDesc
		{
			SampleCount samples;
			ResourceFormat format;
		};

		export class IRenderPass :public SObject
		{
		public:
			IRenderPass() = default;
			IRenderPass(IRenderPass&&) = default;
			virtual ~IRenderPass() = default;


		};
	}
}
