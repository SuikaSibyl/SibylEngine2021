module;
#include <vector>
#include <unordered_map>
module GFX.RDG.RasterNodes;
import Core.MemoryManager;
import RHI.GraphicContext;
import RHI.IPhysicalDevice;
import RHI.ILogicalDevice;
import RHI.IPipelineLayout;
import RHI.ISwapChain;
import RHI.ICompileSession;
import RHI.IEnum;
import RHI.IFactory;
import RHI.IShader;
import RHI.IFixedFunctions;
import RHI.IPipeline;
import RHI.IFramebuffer;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ISemaphore;
import RHI.IFence;
import RHI.IVertexBuffer;
import RHI.IBuffer;
import RHI.IDeviceGlobal;
import RHI.IIndexBuffer;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorPool;
import RHI.IDescriptorSet;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IRenderPass;
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.IBarrier; 
import RHI.ICommandBuffer;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	auto RasterDrawCall::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		

	}

	auto RasterMaterialScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		//RHI::IDescriptorSet* set = raster_pass->descriptorSets[flight_idx].get();
		//commandbuffer->cmdBindDescriptorSets(
		//	RHI::PipelineBintPoint::GRAPHICS,
		//	raster_pass->pipelineLayout.get(),
		//	0, 1, &set, 0, nullptr);
	}

	auto RasterPipelineScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		commandbuffer->cmdBindPipeline(pipeline.get());

	}
	
	auto RasterPipelineScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;
		// vertex buffer layout
		vertexLayout = factory->createVertexLayout(vertexBufferLayout);
		// input assembly
		inputAssembly = factory->createInputAssembly(topologyKind);
		// viewport scissors
		viewportScissors = factory->createViewportsScissors(viewportExtend, viewportExtend);
		// raster
		rasterizer = factory->createRasterizer({
			polygonMode,
			lineWidth,
			cullMode,
			});
		// multisample
		multisampling = factory->createMultisampling({
			false,
			});
		// depth stencil
		depthstencil = factory->createDepthStencil({
			true,
			false,
			RHI::CompareOp::LESS
			});
		// color blending 
		colorBlending = factory->createColorBlending({
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			true,
			});
		// pipeline state
		dynamicStates = factory->createDynamicState({
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
			});

	}

	auto RasterPassScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		// begin render pass, actually bind the FrameBuffer
		commandbuffer->cmdBeginRenderPass(
			((FramebufferContainer*)registry->getNode(framebuffer))->getRenderPass(),
			((FramebufferContainer*)registry->getNode(framebuffer))->getFramebuffer());

		//for (auto& shaderScope : shaderScopes)
		//{
		//	shaderScope.onCommandRecord(commandbuffer, flight);
		//}

		commandbuffer->cmdEndRenderPass();
	}
}