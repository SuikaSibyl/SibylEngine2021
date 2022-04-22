module;
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <unordered_map>
module GFX.RDG.ComputeSeries;
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
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.MultiDispatchScope;

namespace SIByL::GFX::RDG
{
	auto ComputeDispatch::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		if (pushConstant)
		{
			Buffer buffer;
			pushConstant(buffer);
			commandbuffer->cmdPushConstants(*pipelineLayout, RHI::ShaderStage::COMPUTE, buffer.getSize(), buffer.getData());
		}

		uint32_t x, y, z;
		customSize(x, y, z);
		commandbuffer->cmdDispatch(x, y, z);
	}

	// ===========================================================
	// ComputePipelineScope
	// ===========================================================
	auto ComputeMaterialScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		// build descriptor sets
		RenderGraph* rg = (RenderGraph*)graph;
		renderGraph = graph;
		uint32_t MAX_FRAMES_IN_FLIGHT = rg->getMaxFrameInFlight();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		RHI::IDescriptorPool* descriptor_pool = rg->getDescriptorPool();
		RHI::DescriptorSetDesc descriptor_set_desc =
		{ descriptor_pool,
			desciptorSetLayout };
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			descriptorSets[i] = factory->createDescriptorSet(descriptor_set_desc);

		// bind descriptor sets
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			unsigned int textureIdx = 0;
			for (unsigned int j = 0; j < totalResourceNum; j++)
			{
				ResourceNode* resource = rg->getResourceNode(resources[j]);
				switch (resource->type)
				{
				case NodeDetailedType::STORAGE_BUFFER:
					descriptorSets[i]->update(((StorageBufferNode*)resource)->getStorageBuffer(), j, 0);
					break;
				case NodeDetailedType::UNIFORM_BUFFER:
					descriptorSets[i]->update(rg->getUniformBufferFlight(resources[j], i), j, 0);
					break;
				case NodeDetailedType::SAMPLER:
					descriptorSets[i]->update(rg->getTextureBufferNode(sampled_textures[textureIdx++])->getTextureView(),
						rg->getSamplerNode(resources[j])->getSampler(), j, 0);
					break;
				case NodeDetailedType::COLOR_TEXTURE:
					descriptorSets[i]->update(rg->getTextureBufferNode(resources[j])->getTextureView(), j, 0);
					break;
				default:
					SE_CORE_ERROR("GFX :: Compute Pass Node Binding Resource Type unsupported!");
					break;
				}
			}
		}

		// handle all dispatches
		for (auto handle : dispatches)
		{
			ComputeDispatch* dispatch = (ComputeDispatch*)rg->registry.getNode(handle);
			dispatch->pipelineLayout = &pipelineLayout;
			dispatch->devirtualize(graph, factory);
		}
	}

	auto ComputeMaterialScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		RenderGraph* rg = (RenderGraph*)graph;
		// Add Resource Usage
		for (unsigned int j = 0; j < resources.size(); j++)
		{
			ResourceNode* resource = rg->getResourceNode(resources[j]);
			if (resource->type == NodeDetailedType::COLOR_TEXTURE)
				rg->getColorBufferNode(resources[j])->usages |= (uint32_t)RHI::ImageUsageFlagBits::STORAGE_BIT;
		}
		// Add History
		unsigned textureIdx = 0;
		for (unsigned int i = 0; i < resources.size(); i++)
		{
			switch (rg->getResourceNode(resources[i])->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
			{
				rg->getResourceNode(resources[i])->consumeHistory.emplace_back
				(ConsumeHistory{ handle, ConsumeKind::BUFFER_READ_WRITE });
			}
			break;
			case NodeDetailedType::UNIFORM_BUFFER:
				break;
			case NodeDetailedType::SAMPLER:
			{
				rg->getTextureBufferNode(sampled_textures[textureIdx++])->consumeHistory.emplace_back
				(ConsumeHistory{ handle, ConsumeKind::IMAGE_SAMPLE });
			}
			break;
			case NodeDetailedType::COLOR_TEXTURE:
			{
				rg->getTextureBufferNode(resources[i])->consumeHistory.emplace_back
				(ConsumeHistory{ handle, ConsumeKind::IMAGE_STORAGE_READ_WRITE });
			}
			break;
			default:
				break;
			}
		}
	}

	auto ComputeMaterialScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));

		RHI::IDescriptorSet* set = descriptorSets[flight].get();
		commandbuffer->cmdBindDescriptorSets(
			RHI::PipelineBintPoint::COMPUTE,
			pipelineLayout,
			0, 1, &set, 0, nullptr);

		for (auto handle : dispatches)
		{
			ComputeDispatch* dispatch = (ComputeDispatch*)render_graph->registry.getNode(handle);
			dispatch->onCommandRecord(commandbuffer, flight);
		}
	}

	// ===========================================================
	// ComputePipelineScope
	// ===========================================================
	auto ComputePipelineScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;
		// descriptor layoit
		RHI::IShaderReflection reflection = *shaderComp->getReflection();
		totalResourceNum = reflection.descriptorItems.size();
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc = reflection.toDescriptorSetLayoutDesc();
		desciptorSetLayout = factory->createDescriptorSetLayout(descriptor_set_layout_desc);
		// pipeline layouts
		RHI::PipelineLayoutDesc pipelineLayout_desc = { {desciptorSetLayout.get()} };
		if (reflection.getPushConstantSize() != 0) pipelineLayout_desc.pushConstants.emplace_back(0, reflection.getPushConstantSize(), reflection.pushConstantItems[0].stageFlags);
		pipelineLayout = factory->createPipelineLayout(pipelineLayout_desc);
		// create comput pipeline
		pipeline = factory->createPipeline({
			pipelineLayout.get(),
			shaderComp.get(),
			});
		
		RenderGraph* render_graph = (RenderGraph*)graph;
		this->renderGraph = graph;
		for (auto handle : materialScopes)
		{
			ComputeMaterialScope* material_scope = (ComputeMaterialScope*)render_graph->registry.getNode(handle);
			fillComputeMaterialScopeDesc(material_scope, graph);
			material_scope->devirtualize(graph, factory);
		}
	}

	auto ComputePipelineScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		this->renderGraph = graph;
		RenderGraph* render_graph = (RenderGraph*)graph;
		for (auto handle : materialScopes)
		{
			ComputeMaterialScope* material_scope = (ComputeMaterialScope*)render_graph->registry.getNode(handle);
			material_scope->onCompile(graph, factory);
		}
	}

	auto ComputePipelineScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		commandbuffer->cmdBindComputePipeline(pipeline.get());
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto handle : materialScopes)
		{
			ComputeMaterialScope* material_scope = (ComputeMaterialScope*)render_graph->registry.getNode(handle);
			material_scope->onCommandRecord(commandbuffer, flight);
		}
	}

	auto ComputePipelineScope::fillComputeMaterialScopeDesc(ComputeMaterialScope* compute_material, void* graph) noexcept -> void
	{
		compute_material->totalResourceNum = totalResourceNum;
		compute_material->pipelineLayout = pipelineLayout.get();
		compute_material->desciptorSetLayout = desciptorSetLayout.get();
	}

	// ===========================================================
	// ComputePassScope
	// ===========================================================
	auto ComputePassScope::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;
		this->renderGraph = graph;
		for (auto handle : pipelineScopes)
		{
			ComputePipelineScope* pipeline_scope = (ComputePipelineScope*)render_graph->registry.getNode(handle);
			//fillRasterPipelineScopeDesc(pipeline_scope, graph);
			pipeline_scope->devirtualize(graph, factory);
		}
	}

	auto ComputePassScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		this->renderGraph = graph;
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		// make all pipeline_scopes to compile
		for (auto handle : pipelineScopes)
		{
			ComputePipelineScope* pipeline_scope = (ComputePipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onCompile(graph, factory);
		}
	}

	auto ComputePassScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		// push all barriers
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));

		for (auto handle : pipelineScopes)
		{
			ComputePipelineScope* pipeline_scope = (ComputePipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onCommandRecord(commandbuffer, flight);
		}
	}

	// ===========================================================
	// ComputePassIndefiniteScope
	// ===========================================================
	auto ComputePassIndefiniteScope::onRegistered(void* graph, void* render_graph_workshop) noexcept -> void
	{
		// Create Multi Dispatch Begin
		RenderGraph* rg = (RenderGraph*)graph;
		MemScope<MultiDispatchScope> mds = MemNew<MultiDispatchScope>();
		multiDispatchBegin = rg->registry.registNode(std::move(mds));
		rg->tag(multiDispatchBegin, "Compute Pass Indefinite Scope Begin");

		// Create Multi Dispatch End
		MemScope<PassScopeEnd> pse = MemNew<PassScopeEnd>();
		pse->scopeBeginHandle = (multiDispatchBegin);
		pse->type = NodeDetailedType::SCOPE_END;
		multiDispatchEnd = rg->registry.registNode(std::move(pse));
		rg->tag(multiDispatchEnd, "Compute Pass Indefinite Scope End");
	}

	auto ComputePassIndefiniteScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		this->renderGraph = graph;
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		// onCompile Begin Multi Dispatch
		render_graph->registry.getNode(multiDispatchBegin)->onCompile(graph, factory);
		// Compute Pass Scope
		barriers.clear();
		// make all pipeline_scopes to compile
		for (auto handle : pipelineScopes)
		{
			ComputePipelineScope* pipeline_scope = (ComputePipelineScope*)render_graph->registry.getNode(handle);
			pipeline_scope->onCompile(graph, factory);
		}
		// onCompile End Multi Dispatch
		render_graph->registry.getNode(multiDispatchEnd)->onCompile(graph, factory);
	}

	auto ComputePassIndefiniteScope::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		uint32_t dispatch_times = customDispatchCount();
		// If no dispatch, call the EndPass Barrier
		if (dispatch_times == 0)
		{
			for (auto barrier : render_graph->getPassNode(multiDispatchEnd)->barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));
		}
		// If have dispatch, call the BeginPass Barrier
		else
		{
			// BEGIN barrier
			for (auto barrier : render_graph->getPassNode(multiDispatchBegin)->barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));

			for (int i = 0; i < dispatch_times; i++)
			{
				//for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));
				for (auto handle : pipelineScopes)
				{
					ComputePipelineScope* pipeline_scope = (ComputePipelineScope*)render_graph->registry.getNode(handle);
					pipeline_scope->onCommandRecord(commandbuffer, flight);
				}
			}
		}
	}

}