module;
#include <unordered_map>
module GFX.RDG.RenderGraph;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IDescriptorPool;
import RHI.IFactory;
import ECS.UID;
import GFX.RDG.Common;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;
import GFX.RDG.UniformBufferNode;
import GFX.RDG.IndirectDrawBufferNode;
import GFX.RDG.DepthBufferNode;
import GFX.RDG.ColorBufferNode;

namespace SIByL::GFX::RDG
{
	auto RenderGraph::reDatum(uint32_t const& width, uint32_t const& height) noexcept -> void
	{
		datumWidth = width;
		datumHeight = height;

		// reDatum resources
		for (auto iter = resources.begin(); iter != resources.end(); iter++)
		{
			registry.getNode((*iter))->onReDatum((void*)this, factory);
		}

		// reDatum passes
		for (auto iter = passes.begin(); iter != passes.end(); iter++)
		{
			registry.getNode((*iter))->onReDatum((void*)this, factory);
		}
	}

	auto RenderGraph::getDescriptorPool() noexcept -> RHI::IDescriptorPool*
	{
		return descriptorPool.get();
	}

	auto RenderGraph::getResourceNode(NodeHandle handle) noexcept -> ResourceNode*
	{
		return (ResourceNode*)registry.getNode(handle);
	}
	
	auto RenderGraph::getIndirectDrawBufferNode(NodeHandle handle) noexcept -> IndirectDrawBufferNode*
	{
		return (IndirectDrawBufferNode*)getResourceNode(handle);
	}
	
	auto RenderGraph::getPassNode(NodeHandle handle) noexcept -> PassNode*
	{
		return (PassNode*)registry.getNode(handle);
	}

	auto RenderGraph::getComputePassNode(NodeHandle handle) noexcept -> ComputePassNode*
	{
		return (ComputePassNode*)getPassNode(handle);
	}

	auto RenderGraph::getTextureBufferNode(NodeHandle handle) noexcept -> TextureBufferNode*
	{
		return (TextureBufferNode*)getResourceNode(handle);
	}

	auto RenderGraph::getTextureBufferNodeFlight(NodeHandle handle, uint32_t flight) noexcept -> TextureBufferNode*
	{
		NodeHandle flight_handle = ((FlightContainer*)getResourceNode(handle))->handleOnFlight(flight);
		return (TextureBufferNode*)getResourceNode(flight_handle);;
	}

	auto RenderGraph::getUniformBufferFlight(NodeHandle handle, uint32_t const& flight) noexcept -> RHI::IUniformBuffer*
	{
		NodeHandle flight_handle = ((FlightContainer*)registry.getNode(handle))->handleOnFlight(flight);
		return ((UniformBufferNode*)registry.getNode(flight_handle))->uniformBuffer.get();
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
		NodeHandle handle = attached.registry.registNode(std::move(ubn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addUniformBufferFlights(size_t size) noexcept -> NodeHandle
	{
		uint32_t flights_count = attached.getMaxFrameInFlight();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addUniformBuffer(size);
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addStorageBuffer(size_t size) noexcept -> NodeHandle
	{
		storageBufferCount++;
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = size;
		sbn->resourceType = RHI::DescriptorType::STORAGE_BUFFER;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		return handle;
	}
	
	auto RenderGraphBuilder::addStorageBufferExt(RHI::IStorageBuffer* external) noexcept -> NodeHandle
	{
		storageBufferCount++;
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = external->getSize();
		sbn->resourceType = RHI::DescriptorType::STORAGE_BUFFER;
		sbn->attributes |= (uint32_t) NodeAttrbutesFlagBits::PLACEHOLDER;
		sbn->externalStorageBuffer = external;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addColorBufferExt(RHI::ITexture* texture, RHI::ITextureView* view) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>();
		cbn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER);
		cbn->ext_texture = texture;
		cbn->ext_view = view;
		NodeHandle handle = attached.registry.registNode(std::move(cbn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addColorBufferFlightsExt(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle
	{
		uint32_t flights_count = textures.size();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addColorBufferExt(textures[i], views[i]);
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addDepthBuffer(float const& rel_width, float const& rel_height) noexcept -> NodeHandle
	{
		MemScope<DepthBufferNode> dbn = MemNew<DepthBufferNode>(rel_width, rel_height);
		NodeHandle handle = attached.registry.registNode(std::move(dbn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addIndirectDrawBuffer() noexcept -> NodeHandle
	{
		storageBufferCount++;
		MemScope<IndirectDrawBufferNode> sbn = MemNew<IndirectDrawBufferNode>();
		sbn->size = sizeof(unsigned int) * 5;
		sbn->resourceType = RHI::DescriptorType::STORAGE_BUFFER;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<ComputePassNode> cpn = MemNew<ComputePassNode>((void*)&attached, shader, std::move(ios), constant_size);
		NodeHandle handle = attached.registry.registNode(std::move(cpn));
		attached.passes.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::build(RHI::IResourceFactory* factory) noexcept -> void
	{
		attached.factory = factory;

		// build resources
		for(auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
		{
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
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
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
		}
	}
}