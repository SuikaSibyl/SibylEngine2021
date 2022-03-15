module;
#include <vector>
module GFX.RDG.ComputePassNode;

import Core.Log;

import RHI.IEnum;
import RHI.IShader;
import RHI.IPipeline;
import RHI.IPipelineLayout;
import RHI.IDescriptorPool;
import RHI.IDescriptorSet;
import RHI.IDescriptorSetLayout;
import RHI.IFactory;
import RHI.IPipelineLayout;

import Core.MemoryManager;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;

namespace SIByL::GFX::RDG
{
	ComputePassNode::ComputePassNode(void* graph, RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size)
		: shader(shader)
		, ios(ios) 
		, constantSize(constant_size)
	{
		RenderGraph* rg = (RenderGraph*)graph;
		for (unsigned int i = 0; i < ios.size(); i++)
		{
			rg->getResourceNode(ios[i])->shaderStages |= (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT;
			switch (rg->getResourceNode(ios[i])->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
				rg->storageBufferDescriptorCount += rg->getMaxFrameInFlight();
				break;
			case NodeDetailedType::UNIFORM_BUFFER:
				rg->uniformBufferDescriptorCount += rg->getMaxFrameInFlight();
				break;
			default:
				break;
			}
		}
	}
	
	auto ComputePassNode::execute(RHI::ICommandBuffer* buffer, unsigned int x, unsigned int y, unsigned int z, unsigned int frame) noexcept -> void
	{
		RHI::IDescriptorSet* compute_tmp_set = compute_descriptorSets[frame].get();

		buffer->cmdBindComputePipeline(pipeline.get());
		buffer->cmdBindDescriptorSets(RHI::PipelineBintPoint::COMPUTE,
			compute_pipeline_layout.get(), 0, 1, &compute_tmp_set, 0, nullptr);
		buffer->cmdDispatch(x, y, z);
	}

	auto ComputePassNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;

		// create descriptor set layout
		RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc;
		descriptor_set_layout_desc.perBindingDesc.resize(ios.size());
		for (unsigned int i = 0; i < ios.size(); i++)
		{
			descriptor_set_layout_desc.perBindingDesc[i] = {
				i, 1, rg->getResourceNode(ios[i])->resourceType, rg->getResourceNode(ios[i])->shaderStages, nullptr
			};
		}
		MemScope<RHI::IDescriptorSetLayout> compute_desciptor_set_layout = factory->createDescriptorSetLayout(descriptor_set_layout_desc);

		// create pipeline layout
		RHI::PipelineLayoutDesc pipelineLayout_desc =
		{ {compute_desciptor_set_layout.get()} };
		if (constantSize > 0)
		{
			pipelineLayout_desc.pushConstants = { {0,constantSize, (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT} };
		}
		compute_pipeline_layout = factory->createPipelineLayout(pipelineLayout_desc);

		// create comput pipeline
		RHI::ComputePipelineDesc pipeline_desc =
		{
			compute_pipeline_layout.get(),
			shader
		};
		pipeline = factory->createPipeline(pipeline_desc);

		uint32_t MAX_FRAMES_IN_FLIGHT = rg->getMaxFrameInFlight();
		// create descriptor sets
		RHI::IDescriptorPool* descriptor_pool = rg->getDescriptorPool();
		compute_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		RHI::DescriptorSetDesc descriptor_set_desc =
		{	descriptor_pool,
			compute_desciptor_set_layout.get() };
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			compute_descriptorSets[i] = factory->createDescriptorSet(descriptor_set_desc);

		// configure descriptors in sets
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			for (unsigned int j = 0; j < ios.size(); j++)
			{
				ResourceNode* resource = rg->getResourceNode(ios[j]);
				switch (resource->resourceType)
				{
				case RHI::DescriptorType::STORAGE_BUFFER:
					compute_descriptorSets[i]->update(((StorageBufferNode*)resource)->getStorageBuffer(), j, 0);
					break;
				default:
					SE_CORE_ERROR("GFX :: Compute Pass Node Binding Resource Type unsupported!");
					break;
				}
			}
		}
	}
}