module;
#include <vector>
module GFX.RDG.RasterPassNode;
import Core.Log;
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
import RHI.ISampler;
import RHI.IRenderPass;
import RHI.IStorageBuffer;
import RHI.IBarrier;
import Core.MemoryManager;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	RasterPassNode::RasterPassNode(
		void* graph, 
		std::vector<NodeHandle> const& ins,
		uint32_t const& constant_size)
		:ins(ins)
	{
		type = NodeDetailedType::RASTER_PASS;

		RenderGraph* rg = (RenderGraph*)graph;
		for (unsigned int i = 0; i < ins.size(); i++)
		{
			// TODO :: Dinstinguish Vertex / Fragment shaders
			//rg->getResourceNode(ios[i])->shaderStages |= (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT;
			switch (rg->getResourceNode(ins[i])->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
				rg->storageBufferDescriptorCount += rg->getMaxFrameInFlight();
				break;
			case NodeDetailedType::UNIFORM_BUFFER:
				rg->uniformBufferDescriptorCount += rg->getMaxFrameInFlight();
				break;
			case NodeDetailedType::SAMPLER:
				rg->samplerDescriptorCount += rg->getMaxFrameInFlight();
				break;
			default:
				break;
			}
		}
	}

	auto RasterPassNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;

		// vertex buffer layout
		RHI::BufferLayout vertex_buffer_layout =
		{
			{RHI::DataType::Float3, "Position"},
			{RHI::DataType::Float3, "Color"},
			{RHI::DataType::Float2, "UV"},
		};
		vertexLayout = factory->createVertexLayout(vertex_buffer_layout);

		// input assembly
		inputAssembly = factory->createInputAssembly(RHI::TopologyKind::TriangleList);

		// viewport scissors
		if (useFlights)
		{
			FramebufferContainer* framebuffer_0 = rg->getFramebufferContainerFlight(framebufferFlights, 0);
			RHI::Extend extend{ framebuffer_0->getWidth(), framebuffer_0->getHeight() };
			viewportScissors = factory->createViewportsScissors(extend, extend);
		}
		else
		{
			FramebufferContainer* framebuffer_0 = rg->getFramebufferContainer(framebuffer);
			RHI::Extend extend{ framebuffer_0->getWidth(), framebuffer_0->getHeight() };
			viewportScissors = factory->createViewportsScissors(extend, extend);
		}

		// raster
		RHI::RasterizerDesc rasterizer_desc =
		{
			RHI::PolygonMode::FILL,
			0.0f,
			RHI::CullMode::NONE,
		};
		rasterizer = factory->createRasterizer(rasterizer_desc);

		// multisample
		RHI::MultiSampleDesc multisampling_desc =
		{
			false,
		};
		multisampling = factory->createMultisampling(multisampling_desc);

		// depth stencil
		RHI::DepthStencilDesc depthstencil_desc =
		{
			true,
			false,
			RHI::CompareOp::LESS
		};
		depthstencil = factory->createDepthStencil(depthstencil_desc);

		// color blending 
		RHI::ColorBlendingDesc colorBlending_desc =
		{
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			true,
		};
		colorBlending = factory->createColorBlending(colorBlending_desc);

		// pipeline state
		std::vector<RHI::PipelineState> pipelinestates_desc =
		{
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
		};
		dynamicStates = factory->createDynamicState(pipelinestates_desc);

		//
		// create desc layout
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc =
		{ {{ 0, 1, RHI::DescriptorType::UNIFORM_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr },
		   { 1, 1, RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, (uint32_t)RHI::ShaderStageFlagBits::FRAGMENT_BIT, nullptr },
		   { 2, 1, RHI::DescriptorType::STORAGE_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT | (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr }} };
		desciptorSetLayout = factory->createDescriptorSetLayout(descriptor_set_layout_desc);

		//
		// create sets
		uint32_t MAX_FRAMES_IN_FLIGHT = rg->getMaxFrameInFlight();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		RHI::IDescriptorPool* descriptor_pool = rg->getDescriptorPool();
		RHI::DescriptorSetDesc descriptor_set_desc =
		{ descriptor_pool,
			desciptorSetLayout.get() };
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			descriptorSets[i] = factory->createDescriptorSet(descriptor_set_desc);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			int textureIdx = 0;
			for (unsigned int j = 0; j < ins.size(); j++)
			{
				ResourceNode* resource = rg->getResourceNode(ins[j]);
				switch (resource->type)
				{
				case NodeDetailedType::STORAGE_BUFFER:
					descriptorSets[i]->update(((StorageBufferNode*)resource)->getStorageBuffer(), j, 0);
					break;
				case NodeDetailedType::UNIFORM_BUFFER:
					descriptorSets[i]->update(rg->getUniformBufferFlight(ins[j], i), j, 0);
					break;
				case NodeDetailedType::SAMPLER:
					descriptorSets[i]->update(rg->getTextureBufferNode(textures[textureIdx++])->getTextureView(),
						rg->getSamplerNode(ins[j])->getSampler(), j, 0);
					break;
				default:
					SE_CORE_ERROR("GFX :: Raster Pass Node Binding Resource Type unsupported!");
					break;
				}
			}
		}

		// create pipeline layouts
		RHI::PipelineLayoutDesc pipelineLayout_desc =
		{ {desciptorSetLayout.get()} };
		pipelineLayout = factory->createPipelineLayout(pipelineLayout_desc);
		
		RHI::PipelineDesc pipeline_desc =
		{
			{ shaderVert.get(), shaderFrag.get()},
			vertexLayout.get(),
			inputAssembly.get(),
			viewportScissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			colorBlending.get(),
			dynamicStates.get(),
			pipelineLayout.get(),
			useFlights ? rg->getFramebufferContainerFlight(framebufferFlights, 0)->renderPass.get() : rg->getFramebufferContainer(framebuffer)->renderPass.get(),
		};
		pipeline = factory->createPipeline(pipeline_desc);
	}

	auto RasterPassNode::onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;

		// viewport scissors
		if (useFlights)
		{
			FramebufferContainer* framebuffer_0 = rg->getFramebufferContainerFlight(framebufferFlights, 0);
			RHI::Extend extend{ framebuffer_0->getWidth(), framebuffer_0->getHeight() };
			viewportScissors = factory->createViewportsScissors(extend, extend);
		}
		else
		{
			FramebufferContainer* framebuffer_0 = rg->getFramebufferContainer(framebuffer);
			RHI::Extend extend{ framebuffer_0->getWidth(), framebuffer_0->getHeight() };
			viewportScissors = factory->createViewportsScissors(extend, extend);
		}


		RHI::PipelineDesc pipeline_desc =
		{
			{ shaderVert.get(), shaderFrag.get()},
			vertexLayout.get(),
			inputAssembly.get(),
			viewportScissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			colorBlending.get(),
			dynamicStates.get(),
			pipelineLayout.get(),
			useFlights ? rg->getFramebufferContainerFlight(framebufferFlights, 0)->renderPass.get() : rg->getFramebufferContainer(framebuffer)->renderPass.get(),
		};
		pipeline = factory->createPipeline(pipeline_desc);
	}
}