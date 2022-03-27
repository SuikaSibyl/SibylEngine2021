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

	auto RasterPassNode::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
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
			RHI::CullMode::BACK,
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
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc;
		for (int i = 0; i < ins.size(); i++)
		{
			RHI::DescriptorType descriptorType;
			ResourceNode* resource = rg->getResourceNode(ins[i]);
			switch (resource->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
				descriptorType = RHI::DescriptorType::STORAGE_BUFFER;
				break;
			case NodeDetailedType::UNIFORM_BUFFER:
				descriptorType = RHI::DescriptorType::UNIFORM_BUFFER;
				break;
			case NodeDetailedType::SAMPLER:
				descriptorType = RHI::DescriptorType::COMBINED_IMAGE_SAMPLER;
				break;
			default:
				SE_CORE_ERROR("GFX :: Raster Pass Node Binding Resource Type unsupported!");
				break;
			}

			descriptor_set_layout_desc.perBindingDesc.emplace_back(
				i, 1, descriptorType, stageMasks[i], nullptr
			);
		}

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

		std::vector<RHI::IShader*> shader_groups;
		if (shaderVert.get() == nullptr && shaderFrag.get() != nullptr && shaderMesh.get() != nullptr)
		{
			shader_groups = { shaderMesh.get(), shaderFrag.get() };
		}
		else if (shaderVert.get() != nullptr && shaderFrag.get() != nullptr && shaderMesh.get() == nullptr)
		{
			shader_groups = { shaderVert.get(), shaderFrag.get() };
		}
		else
		{
			SE_CORE_ERROR("RDG :: Unkown Raster Pass Shader composition!");
		}

		RHI::PipelineDesc pipeline_desc =
		{
			shader_groups,
			(shaderVert.get() != nullptr) ? vertexLayout.get() : nullptr,
			(shaderVert.get() != nullptr) ? inputAssembly.get() : nullptr,
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

		// Get Indirect Draw Buffer
		if (indirectDrawBufferHandle)
		{
			indirectDrawBuffer = rg->getIndirectDrawBufferNode(indirectDrawBufferHandle)->storageBuffer.get();
		}
	}

	auto RasterPassNode::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		RenderGraph* rg = (RenderGraph*)graph;
		unsigned texture_id = 0;
		for (unsigned int i = 0; i < ins.size(); i++)
		{
			switch (rg->getResourceNode(ins[i])->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
			{
				if (hasBit(attributes, NodeAttrbutesFlagBits::ONE_TIME_SUBMIT))
					rg->getResourceNode(ins[i])->consumeHistoryOnetime.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::BUFFER_READ_WRITE });
				else
					rg->getResourceNode(ins[i])->consumeHistory.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::BUFFER_READ_WRITE });
			}
			break;
			case NodeDetailedType::UNIFORM_BUFFER:
				break;
			case NodeDetailedType::SAMPLER:
			{
				if (hasBit(attributes, NodeAttrbutesFlagBits::ONE_TIME_SUBMIT))
					rg->getTextureBufferNode(textures[texture_id++])->consumeHistoryOnetime.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::IMAGE_SAMPLE });
				else
					rg->getTextureBufferNode(textures[texture_id++])->consumeHistory.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::IMAGE_SAMPLE });
			}
			break;
			case NodeDetailedType::COLOR_TEXTURE:
			{
				if (hasBit(attributes, NodeAttrbutesFlagBits::ONE_TIME_SUBMIT))
					rg->getTextureBufferNode(ins[i])->consumeHistoryOnetime.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::IMAGE_STORAGE_READ_WRITE });
				else
					rg->getTextureBufferNode(ins[i])->consumeHistory.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::IMAGE_STORAGE_READ_WRITE });
			}
			break;
			default:
				break;
			}
		}

		// render target
		if (!useFlights)
		{
			FramebufferContainer* framebuffer_container = rg->getFramebufferContainer(framebuffer);
			for (int i = 0; i < framebuffer_container->handles.size(); i++)
			{
				if (hasBit(attributes, NodeAttrbutesFlagBits::ONE_TIME_SUBMIT))
					rg->getResourceNode(framebuffer_container->handles[i])->consumeHistoryOnetime.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::RENDER_TARGET });
				else
					rg->getResourceNode(framebuffer_container->handles[i])->consumeHistory.emplace_back
					(ConsumeHistory{ handle, ConsumeKind::RENDER_TARGET });
			}
		}
		else 
		{
			// TODO
		}

		// if there are indirect draw buffer, add consume history
		if (indirectDrawBufferHandle)
		{
			if (hasBit(attributes, NodeAttrbutesFlagBits::ONE_TIME_SUBMIT))
				rg->getResourceNode(indirectDrawBufferHandle)->consumeHistoryOnetime.emplace_back
				(ConsumeHistory{ handle, ConsumeKind::INDIRECT_DRAW });
			else
				rg->getResourceNode(indirectDrawBufferHandle)->consumeHistory.emplace_back
				(ConsumeHistory{ handle, ConsumeKind::INDIRECT_DRAW });
		}
	}

	auto RasterPassNode::rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void
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

	auto RasterPassNode::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{			
		// render pass
		commandbuffer->cmdBeginRenderPass(
			((FramebufferContainer*)registry->getNode(framebuffer))->getRenderPass(),
			((FramebufferContainer*)registry->getNode(framebuffer))->getFramebuffer());
		commandbuffer->cmdBindPipeline(pipeline.get());

		customCommandRecord(this, commandbuffer, flight);

		commandbuffer->cmdEndRenderPass();
	}
}