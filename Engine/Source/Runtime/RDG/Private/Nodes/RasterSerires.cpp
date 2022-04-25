module;
#include <vector>
#include <unordered_map>
module GFX.RDG.RasterNodes;
import Core.Log;
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
import RHI.IShaderReflection;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	RasterDrawCall::RasterDrawCall(RHI::IPipelineLayout** pipeline_layout)
		:pipelineLayout(pipeline_layout)
	{}

	auto RasterDrawCall::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		if (indexBuffer && vertexBuffer)
		{
			commandbuffer->cmdBindVertexBuffer(vertexBuffer);
			commandbuffer->cmdBindIndexBuffer(indexBuffer);
			commandbuffer->cmdPushConstants(*pipelineLayout, RHI::ShaderStage::VERTEX, sizeof(PerObjectUniformBuffer), &uniform);

			if (indirectDrawBuffer)
				commandbuffer->cmdDrawIndexedIndirect(indirectDrawBuffer, 0, 1, sizeof(unsigned int) * 5);
			else
				commandbuffer->cmdDrawIndexed(indexBuffer->getIndicesCount(), 1, 0, 0, 0);
		}
	}

	auto RasterDrawCall::clearDrawCallInfo() noexcept -> void
	{
		vertexBuffer = nullptr;
		indexBuffer = nullptr;
		uniform = PerObjectUniformBuffer();
	}

	// ==============================================================================
	// RasterMaterialScope
	// ==============================================================================

	auto RasterMaterialScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		// build descriptor sets
		RenderGraph* rg = (RenderGraph*)graph;
		renderGraph = graph;
		uint32_t MAX_FRAMES_IN_FLIGHT = rg->getMaxFrameInFlight();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		RHI::IDescriptorPool* descriptor_pool = rg->getDescriptorPool();
		RHI::DescriptorSetDesc descriptor_set_desc =
		{	descriptor_pool,
			desciptorSetLayout };
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			descriptorSets[i] = factory->createDescriptorSet(descriptor_set_desc);

		// bind descriptor sets
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			unsigned int resourceIdx = 0;
			unsigned int textureIdx = 0;
			for (unsigned int j = 0; j < totalResourceNum; j++)
			{
				if (j == hasPerFrameUniformBuffer)
				{
					descriptorSets[i]->update(rg->getUniformBufferFlight(perFrameUniformBufferFlight, i), j, 0);
				}
				else if (j == hasPerViewUniformBuffer)
				{
					descriptorSets[i]->update(rg->getUniformBufferFlight(perViewUniformBufferFlight, i), j, 0);
				}
				else
				{
					ResourceNode* resource = rg->getResourceNode(resources[resourceIdx]);
					switch (resource->type)
					{
					case NodeDetailedType::STORAGE_BUFFER:
						descriptorSets[i]->update(((StorageBufferNode*)resource)->getStorageBuffer(), j, 0);
						resourceIdx++;
						break;
					case NodeDetailedType::UNIFORM_BUFFER:
						descriptorSets[i]->update(rg->getUniformBufferFlight(resources[resourceIdx++], i), j, 0);
						break;
					case NodeDetailedType::SAMPLER:
						descriptorSets[i]->update(rg->getTextureBufferNode(sampled_textures[textureIdx++])->getTextureView(),
							rg->getSamplerNode(resources[resourceIdx++])->getSampler(), j, 0);
						break;
					default:
						SE_CORE_ERROR("GFX :: Raster Pass Node Binding Resource Type unsupported!");
						break;
					}
				}
			}
		}
	}
	
	auto RasterMaterialScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		RenderGraph* rg = (RenderGraph*)graph;
		// Add History
		unsigned textureIdx = 0;
		for (unsigned int i = 0; i < resources.size(); i++)
		{
			switch (rg->getResourceNode(resources[i])->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
			{
				rg->getResourceNode(resources[i])->getConsumeHistory().emplace_back
				(ConsumeHistory{ handle, ConsumeKind::BUFFER_READ_WRITE });
			}
			break;
			case NodeDetailedType::UNIFORM_BUFFER:
				break;
			case NodeDetailedType::SAMPLER:
			{
				rg->getTextureBufferNode(sampled_textures[textureIdx++])->getConsumeHistory().emplace_back
				(ConsumeHistory{ handle, ConsumeKind::IMAGE_SAMPLE });
			}
			break;
			case NodeDetailedType::COLOR_TEXTURE:
			{
				rg->getTextureBufferNode(resources[i])->getConsumeHistory().emplace_back
				(ConsumeHistory{ handle, ConsumeKind::IMAGE_STORAGE_READ_WRITE });
			}
			break;
			default:
				break;
			}
		}
	}

	auto RasterMaterialScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		//for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));
		RHI::IDescriptorSet* set = descriptorSets[flight].get();
		commandbuffer->cmdBindDescriptorSets(
			RHI::PipelineBintPoint::GRAPHICS,
			pipelineLayout,
			0, 1, &set, 0, nullptr);

		for (unsigned int i = 0; i < validDrawcallCount; i++)
		{
			auto handle = drawCalls[i];
			RasterDrawCall* drawcall = (RasterDrawCall*)render_graph->registry.getNode(handle);
			drawcall->onCommandRecord(commandbuffer, flight);
		}
	}

	auto RasterMaterialScope::onFrameStart(void* graph) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto handle : drawCalls)
		{
			RasterDrawCall* drawcall = (RasterDrawCall*)render_graph->registry.getNode(handle);
			drawcall->clearDrawCallInfo();
		}
		validDrawcallCount = 0;
	}

	auto RasterMaterialScope::addRasterDrawCall(std::string const& tag, void* graph) noexcept -> NodeHandle
	{
		RenderGraph* rg = (RenderGraph*)graph;
		NodeHandle handle = 0;
		if (validDrawcallCount < drawCalls.size())
		{
			handle = drawCalls[validDrawcallCount];
			RasterDrawCall* rdc = (RasterDrawCall*)rg->registry.getNode(handle);
			rdc->tag = tag;
		}
		else
		{
			MemScope<RasterDrawCall> rdc = MemNew<RasterDrawCall>(&pipelineLayout);
			rdc->tag = tag;
			handle = rg->registry.registNode(std::move(rdc));
			drawCalls.emplace_back(handle);
		}
		validDrawcallCount++;
		return handle;
	}

	// ==============================================================================
	// RasterPipelineScope
	// ==============================================================================

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
		depthstencil = factory->createDepthStencil(depthStencilDesc);
		// color blending 
		colorBlending = factory->createColorBlending(colorBlendingDesc);
		// pipeline state
		dynamicStates = factory->createDynamicState({
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
			});
		// descriptor layoit
		RHI::IShaderReflection reflection;
		if (shaderVert.get()) reflection = std::move(reflection * (*shaderVert->getReflection()));
		if (shaderFrag.get()) reflection = std::move(reflection * (*shaderFrag->getReflection()));
		if (shaderTask.get()) reflection = std::move(reflection * (*shaderTask->getReflection()));
		if (shaderMesh.get()) reflection = std::move(reflection * (*shaderMesh->getReflection()));
		for (auto& item : reflection.descriptorItems)
		{
			if (item.name == "PerViewUniformBuffer")
				hasPerViewUniformBuffer = item.binding;
			if (item.name == "PerFrameUniformBufferFlight")
				hasPerFrameUniformBuffer = item.binding;
		}
		totalResourceNum = reflection.descriptorItems.size();
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc = reflection.toDescriptorSetLayoutDesc();
		desciptorSetLayout = factory->createDescriptorSetLayout(descriptor_set_layout_desc);
		// pipeline layouts
		RHI::PipelineLayoutDesc pipelineLayout_desc = { {desciptorSetLayout.get()} };
		if (reflection.getPushConstantSize() != 0) pipelineLayout_desc.pushConstants.emplace_back(0, reflection.getPushConstantSize(), reflection.pushConstantItems[0].stageFlags);
		pipelineLayout = factory->createPipelineLayout(pipelineLayout_desc);
		// shaders
		std::vector<RHI::IShader*> shader_groups;
		if (shaderVert.get() == nullptr && shaderFrag.get() != nullptr && shaderMesh.get() != nullptr)
			shader_groups = { shaderMesh.get(), shaderFrag.get() };
		else if (shaderVert.get() != nullptr && shaderFrag.get() != nullptr && shaderMesh.get() == nullptr)
			shader_groups = { shaderVert.get(), shaderFrag.get() };
		else
			SE_CORE_ERROR("RDG :: Unkown Raster Pass Shader composition!");

		// create pipeline
		RHI::PipelineDesc pipeline_desc =
		{
			shader_groups,
			(shaderVert.get() != nullptr) ? vertexLayout.get() : nullptr, // if no vertex shader, no vertex layout then
			(shaderVert.get() != nullptr) ? inputAssembly.get() : nullptr, // if no vertex shader, no input assembly then
			viewportScissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			colorBlending.get(),
			dynamicStates.get(),
			pipelineLayout.get(),
			renderPass,
		};
		pipeline = factory->createPipeline(pipeline_desc);

		RenderGraph* render_graph = (RenderGraph*)graph;
		this->renderGraph = graph;
		for (auto handle : materialScopes)
		{
			RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
			fillRasterMaterialScopeDesc(material_scope, graph);
			material_scope->devirtualize(graph, factory);
		}
	}

	auto RasterPipelineScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		this->renderGraph = graph;
		RenderGraph* render_graph = (RenderGraph*)graph;
		for (auto handle : materialScopes)
		{
			RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
			material_scope->onCompile(graph, factory);
		}
	}

	auto RasterPipelineScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		commandbuffer->cmdBindPipeline(pipeline.get());
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto handle : materialScopes)
		{
			RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
			material_scope->onCommandRecord(commandbuffer, flight);
		}
	}

	auto RasterPipelineScope::onFrameStart(void* graph) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto handle : materialScopes)
		{
			for (auto handle : materialScopes)
			{
				RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
				material_scope->onFrameStart(graph);
			}

			RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
			material_scope->onFrameStart(graph);
		}
	}

	auto RasterPipelineScope::fillRasterMaterialScopeDesc(RasterMaterialScope* raster_material, void* graph) noexcept -> void
	{
		raster_material->totalResourceNum = totalResourceNum;
		raster_material->pipelineLayout = pipelineLayout.get();
		raster_material->desciptorSetLayout = desciptorSetLayout.get();
		raster_material->hasPerFrameUniformBuffer = hasPerFrameUniformBuffer;
		raster_material->hasPerViewUniformBuffer = hasPerViewUniformBuffer;
		raster_material->perFrameUniformBufferFlight = perFrameUniformBufferFlight;
		raster_material->perViewUniformBufferFlight = perViewUniformBufferFlight;
	}

	// ==============================================================================
	// RasterPassScope
	// ==============================================================================

	auto RasterPassScope::onRegistered(void* graph, void* render_graph_workshop) noexcept -> void
	{
		RenderGraphWorkshop* renderGraphWorkshop = (RenderGraphWorkshop*)render_graph_workshop;
		perViewUniformBufferFlight = renderGraphWorkshop->addUniformBufferFlights(sizeof(PerViewUniformBuffer), tag + " Per View Uniform Buffer");
	}

	auto RasterPassScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;
		this->renderGraph = graph;
		for (auto handle : pipelineScopes)
		{
			RasterPipelineScope* pipeline_scope = (RasterPipelineScope*)render_graph->registry.getNode(handle);
			fillRasterPipelineScopeDesc(pipeline_scope, graph);
			pipeline_scope->devirtualize(graph, factory);
		}
	}
	
	auto RasterPassScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		this->renderGraph = graph;
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		// add consume history to frame buffer
		FramebufferContainer* framebuffer_container = render_graph->getFramebufferContainer(framebuffer);
		for (int i = 0; i < framebuffer_container->handles.size(); i++)
		{
			render_graph->getResourceNode(framebuffer_container->handles[i])->getConsumeHistory().emplace_back
			(ConsumeHistory{ handle, ConsumeKind::RENDER_TARGET });
		}
		// make all pipeline_scopes to compile
		for (auto handle : pipelineScopes)
		{
			RasterPipelineScope* pipeline_scope = (RasterPipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onCompile(graph, factory);
		}
	}

	auto RasterPassScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		// push all barriers
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));

		for (auto handle : pipelineScopes)
		{
			RasterPipelineScope* pipeline_scope = (RasterPipelineScope*)render_graph->registry.getNode(handle);
			for (auto handle : pipeline_scope->materialScopes)
			{
				RasterMaterialScope* material_scope = (RasterMaterialScope*)render_graph->registry.getNode(handle);
				for (auto barrier : material_scope->barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));
			}
		}

		// begin render pass, actually bind the FrameBuffer
		commandbuffer->cmdBeginRenderPass(
			((FramebufferContainer*)registry->getNode(framebuffer))->getRenderPass(),
			((FramebufferContainer*)registry->getNode(framebuffer))->getFramebuffer());

		for (auto handle : pipelineScopes)
		{
			RasterPipelineScope* pipeline_scope = (RasterPipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onCommandRecord(commandbuffer, flight);
		}

		commandbuffer->cmdEndRenderPass();
	}
	
	auto RasterPassScope::onFrameStart(void* graph) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto handle : pipelineScopes)
		{
			RasterPipelineScope* pipeline_scope = (RasterPipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onFrameStart(graph);
		}
	}

	auto RasterPassScope::fillRasterPipelineScopeDesc(RasterPipelineScope* raster_pipeline, void* graph) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;
		FramebufferContainer* framebuffer_ptr = render_graph->getFramebufferContainer(framebuffer);
		RHI::Extend extend{ framebuffer_ptr->getWidth(), framebuffer_ptr->getHeight() };
		raster_pipeline->viewportExtend = extend;
		raster_pipeline->renderPass = framebuffer_ptr->renderPass.get();
		raster_pipeline->perFrameUniformBufferFlight = perFrameUniformBufferFlight;
		raster_pipeline->perViewUniformBufferFlight = perViewUniformBufferFlight;
	}

	auto RasterPassScope::updatePerViewUniformBuffer(PerViewUniformBuffer const& buffer, uint32_t const& current_frame) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		perViewUniformBuffer = buffer;
		Buffer ubo_proxy((void*)&perViewUniformBuffer, sizeof(PerViewUniformBuffer), 1);
		render_graph->getUniformBufferFlight(perViewUniformBufferFlight, current_frame)->updateBuffer(&ubo_proxy);
	}
}