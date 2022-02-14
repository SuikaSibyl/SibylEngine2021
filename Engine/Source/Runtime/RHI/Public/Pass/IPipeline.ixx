module;
#include <vector>
export module RHI.IPipeline;
import RHI.IResource;
import RHI.IShader;
import RHI.IFixedFunctions;
import RHI.IPipelineLayout;
import RHI.IRenderPass;
import RHI.ILogicalDevice;

namespace SIByL
{
	namespace RHI
	{
		// Pipelines are an overarching description of what will be executed 
		// when performing a raster draw call, compute dispatch, or ray tracing dispatch.
		// 
		// DirectX 11 and OpenGL are unique here where they don't have a dedicated object for the graphics pipeline, 
		// but instead use calls to set the pipeline state in between executing draw calls.
		// 
		// ╭──────────────┬──────────────────────────╮
		// │  Vulkan	  │   vk::Pipeline           │
		// │  DirectX 12  │   ID3D12PipelineState    │
		// │  OpenGL      │   Various State Calls    │
		// ╰──────────────┴──────────────────────────╯
		//
		// Shader stages		: the shader modules that define the functionality of the programmable stages of the graphics pipeline
		// Fixed-function state : all of the structures that define the fixed - function stages of the pipeline, like input assembly, rasterizer, viewportand color blending
		// Pipeline layout		: the uniformand push values referenced by the shader that can be updated at draw time
		// Render pass			: the attachments referenced by the pipeline stagesand their usage
		export struct IPipelineDesc
		{
			ILogicalDevice* logicalDevice;
			std::vector<IShader*> shaders;
			IVertexLayout* vertexLayout;
			IInputAssembly* inputAssembly;
			IViewportsScissors* viewportsScissors;
			IRasterizer* rasterizer;
			IMultisampling* multisampling;
			IDepthStencil* depthStencil;
			IColorBlending* colorBlending;
			IDynamicState* dynamicState;
			IPipelineLayout* pipelineLayout;
			IRenderPass* renderPass;
		};

		export class IPipeline :public IResource
		{
		public:
			IPipeline() = default;
			IPipeline(IPipeline&&) = default;
			virtual ~IPipeline() = default;

		protected:
			IPipelineDesc desc;
		};
	}
}
