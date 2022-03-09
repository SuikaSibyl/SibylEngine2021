module;
#include <unordered_map>
module GFX.RDG.RenderGraph;
import Core.MemoryManager;
import RHI.IDescriptorPool;
import RHI.IFactory;
import ECS.UID;
import GFX.RDG.Common;
import GFX.RDG.PassNode;
import GFX.RDG.ResourceNode;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;
import GFX.RDG.UniformBufferNode;

namespace SIByL::GFX::RDG
{
	auto RenderGraph::getDescriptorPool() noexcept -> RHI::IDescriptorPool*
	{
		return descriptorPool.get();
	}

	auto RenderGraph::getResourceNode(NodeHandle handle) noexcept -> ResourceNode*
	{
		if (resources.find(handle) != resources.end())
		{
			return resources[handle].get();
		}
		return nullptr;
	}
	
	auto RenderGraph::getPassNode(NodeHandle handle) noexcept -> PassNode*
	{
		if (passes.find(handle) != passes.end())
		{
			return passes[handle].get();
		}
		return nullptr;
	}

	auto RenderGraph::getComputePassNode(NodeHandle handle) noexcept -> ComputePassNode*
	{
		return (ComputePassNode*)getPassNode(handle);
	}

	auto RenderGraphBuilder::addTexture() noexcept -> NodeHandle
	{

		return 0;
	}

	auto RenderGraphBuilder::addUniformBuffer(size_t size) noexcept -> NodeHandle
	{
		uniformBufferCount++;
		MemScope<UniformBufferNode> ubn = MemNew<UniformBufferNode>();
		ubn->size = size;
		ubn->resourceType = RHI::DescriptorType::UNIFORM_BUFFER;
		MemScope<ResourceNode> res = MemCast<ResourceNode>(ubn);
		NodeHandle handle = ECS::UniqueID::RequestUniqueID();
		attached.resources[handle] = std::move(res);
		return handle;
	}

	auto RenderGraphBuilder::addStorageBuffer(size_t size) noexcept -> NodeHandle
	{
		storageBufferCount++;
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = size;
		sbn->resourceType = RHI::DescriptorType::STORAGE_BUFFER;
		MemScope<ResourceNode> res = MemCast<ResourceNode>(sbn);
		NodeHandle handle = ECS::UniqueID::RequestUniqueID();
		attached.resources[handle] = std::move(res);
		return handle;
	}

	auto RenderGraphBuilder::addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<ComputePassNode> cpn = MemNew<ComputePassNode>((void*)&attached, shader, std::move(ios), constant_size);
		MemScope<PassNode> res = MemCast<PassNode>(cpn);
		NodeHandle handle = ECS::UniqueID::RequestUniqueID();
		attached.passes[handle] = std::move(res);
		return handle;
	}

	auto RenderGraphBuilder::build(RHI::IResourceFactory* factory) noexcept -> void
	{
		// build resources
		for(auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
		{
			iter->second->onBuild((void*)&attached, factory);
		}

		// create pool
		uint32_t MAX_FRAMES_IN_FLIGHT = attached.getMaxFrameInFlight();
		RHI::DescriptorPoolDesc descriptor_pool_desc =
		{ {{RHI::DescriptorType::UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT},
		   {RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT},
		   {RHI::DescriptorType::STORAGE_BUFFER, storageBufferCount * MAX_FRAMES_IN_FLIGHT}}, // set types
			attached.passes.size() * MAX_FRAMES_IN_FLIGHT}; // total sets
		attached.descriptorPool = factory->createDescriptorPool(descriptor_pool_desc);

		// build passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
		{
			iter->second->onBuild((void*)&attached, factory);
		}
	}
}