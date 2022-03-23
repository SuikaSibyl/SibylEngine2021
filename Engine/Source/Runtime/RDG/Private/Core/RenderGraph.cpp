module;
#include <unordered_map>
module GFX.RDG.RenderGraph;
import Core.Log;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IDescriptorPool;
import RHI.IFactory;
import RHI.ITexture;
import RHI.ISampler;
import ECS.UID;
import GFX.RDG.Common;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.UniformBufferNode;
import GFX.RDG.IndirectDrawBufferNode;
import GFX.RDG.DepthBufferNode;
import GFX.RDG.ColorBufferNode;
import GFX.RDG.SamplerNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.MultiDispatchScope;

namespace SIByL::GFX::RDG
{
	auto RenderGraph::print() noexcept -> void
	{
		SE_CORE_INFO("RESOURCES :: ");

		// print resources
		for (auto iter = resources.begin(); iter != resources.end(); iter++)
		{
			registry.getNode((*iter))->onPrint();
		}

		SE_CORE_INFO("");
		SE_CORE_INFO("ONETIME PASSES :: ");
		// reDatum passes
		for (auto iter = passesOnetime.begin(); iter != passesOnetime.end(); iter++)
		{
			registry.getNode((*iter))->onPrint();
		}

		SE_CORE_INFO("");
		SE_CORE_INFO("PASSES :: ");
		// reDatum passes
		for (auto iter = passes.begin(); iter != passes.end(); iter++)
		{
			registry.getNode((*iter))->onPrint();
		}
	}

	auto RenderGraph::tag(NodeHandle handle, std::string_view tag) noexcept -> void
	{
		registry.getNode(handle)->tag = tag;
	}

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

		// reDatum passes
		for (auto iter = passesOnetime.begin(); iter != passesOnetime.end(); iter++)
		{
			registry.getNode((*iter))->onReDatum((void*)this, factory);
		}
	}

	auto RenderGraph::recordCommands(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		for (auto iter = passes.begin(); iter != passes.end(); iter++)
		{
			if (getPassNode(*iter)->type == NodeDetailedType::MULTI_DISPATCH_SCOPE)
			{
				unsigned int dispatch_times = getMultiDispatchScope(*iter)->customDispatchCount();
				if (dispatch_times == 0)
				{
					while (getPassNode(*iter)->type != NodeDetailedType::SCOPE_END) iter++;
					for (auto barrier : getPassNode(*iter)->barriers) commandbuffer->cmdPipelineBarrier(barrierPool.getBarrier(barrier));
				}
				else
				{
					// BEGIN barrier
					for (auto barrier : getPassNode(*iter)->barriers) commandbuffer->cmdPipelineBarrier(barrierPool.getBarrier(barrier));

					auto loopbegin = iter + 1;
					for (int i = 0; i < dispatch_times; i++)
					{
						auto loop_iter = loopbegin;
						while (getPassNode(*loop_iter)->type != NodeDetailedType::SCOPE_END)
						{
							for (auto barrier : getPassNode(*loop_iter)->barriers) commandbuffer->cmdPipelineBarrier(barrierPool.getBarrier(barrier));
							getPassNode(*iter)->onCommandRecord(commandbuffer, flight);
							loop_iter++;
						}
					}
				}
			}
			else
			{
				for (auto barrier : getPassNode(*iter)->barriers)
				{
					commandbuffer->cmdPipelineBarrier(barrierPool.getBarrier(barrier));
				}
				getPassNode(*iter)->onCommandRecord(commandbuffer, flight);
			}
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
		return (TextureBufferNode*)getResourceNode(flight_handle);
	}

    auto RenderGraph::getColorBufferNode(NodeHandle handle) noexcept -> ColorBufferNode*
	{
		return (ColorBufferNode*)getResourceNode(handle);
	}

	auto RenderGraph::getFramebufferContainerFlight(NodeHandle handle, uint32_t flight) noexcept -> FramebufferContainer*
	{
		NodeHandle flight_handle = ((FlightContainer*)getResourceNode(handle))->handleOnFlight(flight);
		return (FramebufferContainer*)getResourceNode(flight_handle);
	}

	auto RenderGraph::getFramebufferContainer(NodeHandle handle) noexcept -> FramebufferContainer*
	{
		return (FramebufferContainer*)getResourceNode(handle);
	}

	auto RenderGraph::getUniformBufferFlight(NodeHandle handle, uint32_t const& flight) noexcept -> RHI::IUniformBuffer*
	{
		NodeHandle flight_handle = ((FlightContainer*)registry.getNode(handle))->handleOnFlight(flight);
		return ((UniformBufferNode*)registry.getNode(flight_handle))->uniformBuffer.get();
	}

	auto RenderGraph::getContainer(NodeHandle handle) noexcept -> Container*
	{
		return (Container*)registry.getNode(handle);
	}
	
	auto RenderGraph::getSamplerNode(NodeHandle handle) noexcept -> SamplerNode*
	{
		return (SamplerNode*)registry.getNode(handle);
	}
	
	auto RenderGraph::getRasterPassNode(NodeHandle handle) noexcept -> RasterPassNode*
	{
		return (RasterPassNode*)registry.getNode(handle);
	}

	auto RenderGraph::getPassScope(NodeHandle handle) noexcept -> PassScope*
	{
		return (PassScope*)registry.getNode(handle);
	}

	auto RenderGraph::getMultiDispatchScope(NodeHandle handle) noexcept -> MultiDispatchScope*
	{
		return (MultiDispatchScope*)registry.getNode(handle);
	}

	// =====================================================
	// =====================================================
	// =====================================================

	auto RenderGraphBuilder::addTexture() noexcept -> NodeHandle
	{
		return 0;
	}

	auto RenderGraphBuilder::addUniformBuffer(size_t size) noexcept -> NodeHandle
	{
		MemScope<UniformBufferNode> ubn = MemNew<UniformBufferNode>();
		ubn->size = size;
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
		fc->type = NodeDetailedType::UNIFORM_BUFFER;
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addStorageBuffer(size_t size, std::string_view name) noexcept -> NodeHandle
	{
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = size;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}
	
	auto RenderGraphBuilder::addStorageBufferExt(RHI::IStorageBuffer* external, std::string_view name) noexcept -> NodeHandle
	{
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = external->getSize();
		sbn->attributes |= (uint32_t) NodeAttrbutesFlagBits::PLACEHOLDER;
		sbn->externalStorageBuffer = external;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addColorBufferExt(RHI::ITexture* texture, RHI::ITextureView* view, std::string_view name, bool present) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>();
		cbn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER);
		cbn->texture.ref = texture;
		cbn->textureView.ref = view;
		cbn->format = texture->getDescription().format;
		if(present) cbn->attributes |= (uint32_t)NodeAttrbutesFlagBits::PRESENT;
		NodeHandle handle = attached.registry.registNode(std::move(cbn));
		attached.resources.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addColorBufferFlightsExt(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle
	{
		uint32_t flights_count = textures.size();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addColorBufferExt(textures[i], views[i], "WHAT");
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		fc->type = NodeDetailedType::COLOR_TEXTURE;
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addColorBufferFlightsExtPresent(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle
	{
		uint32_t flights_count = textures.size();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addColorBufferExt(textures[i], views[i], "WHAT", true);
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		fc->type = NodeDetailedType::COLOR_TEXTURE;
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addColorBuffer(RHI::ResourceFormat format, float const& rel_width, float const& rel_height, std::string_view name) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>(format, rel_width, rel_height);
		NodeHandle handle = attached.registry.registNode(std::move(cbn));
		attached.resources.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addDepthBuffer(float const& rel_width, float const& rel_height) noexcept -> NodeHandle
	{
		MemScope<DepthBufferNode> dbn = MemNew<DepthBufferNode>(rel_width, rel_height);
		NodeHandle handle = attached.registry.registNode(std::move(dbn));
		attached.resources.emplace_back(handle);
		return handle;
	}
	
	auto RenderGraphBuilder::addSamplerExt(RHI::ISampler* sampler) noexcept -> NodeHandle
	{
		MemScope<SamplerNode> sn = MemNew<SamplerNode>();
		sn->extSampler = sampler;
		sn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER);
		NodeHandle handle = attached.registry.registNode(std::move(sn));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addIndirectDrawBuffer(std::string_view name) noexcept -> NodeHandle
	{
		MemScope<IndirectDrawBufferNode> sbn = MemNew<IndirectDrawBufferNode>();
		sbn->size = sizeof(unsigned int) * 5;
		NodeHandle handle = attached.registry.registNode(std::move(sbn));
		attached.resources.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, std::string_view name, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<ComputePassNode> cpn = MemNew<ComputePassNode>((void*)&attached, shader, std::move(ios), constant_size);
		cpn->type = NodeDetailedType::COMPUTE_PASS;
		NodeHandle handle = attached.registry.registNode(std::move(cpn));
		attached.passes.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addComputePassOneTime(RHI::IShader* shader, std::vector<NodeHandle>&& ios, std::string_view name, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<ComputePassNode> cpn = MemNew<ComputePassNode>((void*)&attached, shader, std::move(ios), constant_size);
		cpn->type = NodeDetailedType::COMPUTE_PASS;
		cpn->attributes |= (uint32_t)NodeAttrbutesFlagBits::ONE_TIME_SUBMIT;
		NodeHandle handle = attached.registry.registNode(std::move(cpn));
		attached.passesOnetime.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addFrameBufferRef(std::vector<NodeHandle> const& color_attachments, NodeHandle depth_attachment) noexcept -> NodeHandle
	{
		MemScope<FramebufferContainer> fbc = MemNew<FramebufferContainer>();
		fbc->colorAttachCount = color_attachments.size();
		fbc->depthAttachCount = (depth_attachment == 0) ? 0 : 1;
		fbc->handles.resize(fbc->colorAttachCount + fbc->depthAttachCount);
		for (int i = 0; i < fbc->colorAttachCount; i++)
		{
			fbc->handles[i] = color_attachments[i];
			attached.getColorBufferNode(color_attachments[i])->usages |= (uint32_t)RHI::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT;
		}
		if (depth_attachment) fbc->handles[fbc->colorAttachCount] = depth_attachment;
		NodeHandle handle = attached.registry.registNode(std::move(fbc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::addFrameBufferFlightsRef(std::vector<std::pair<std::vector<NodeHandle> const&, NodeHandle>> infos) noexcept -> NodeHandle
	{
		uint32_t flights_count = infos.size();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addFrameBufferRef(infos[i].first, infos[i].second);
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		fc->type = NodeDetailedType::FRAME_BUFFER;
		NodeHandle handle = attached.registry.registNode(std::move(fc));
		attached.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphBuilder::build(RHI::IResourceFactory* factory, uint32_t const& width, uint32_t const& height) noexcept -> void
	{
		attached.factory = factory;
		attached.datumWidth = width;
		attached.datumHeight = height;

		// compile resources
		for (auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
		{
			attached.registry.getNode((*iter))->onCompile((void*)&attached, factory);
		}
		// compile passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
		{
			attached.registry.getNode((*iter))->onCompile((void*)&attached, factory);
		}
		// compile passesOnetime
		for (auto iter = attached.passesOnetime.begin(); iter != attached.passesOnetime.end(); iter++)
		{
			attached.registry.getNode((*iter))->onCompile((void*)&attached, factory);
		}

		// build resources
		for(auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
		{
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
		}

		// create pool
		uint32_t MAX_FRAMES_IN_FLIGHT = attached.getMaxFrameInFlight();
		RHI::DescriptorPoolDesc descriptor_pool_desc{ {}, (attached.passes.size() + attached.passesOnetime.size()) * MAX_FRAMES_IN_FLIGHT};
		if (attached.storageBufferDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_BUFFER, attached.storageBufferDescriptorCount);
		if (attached.uniformBufferDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::UNIFORM_BUFFER, attached.uniformBufferDescriptorCount);
		if (attached.samplerDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, attached.samplerDescriptorCount);
		if (attached.storageImageDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_IMAGE, attached.storageImageDescriptorCount);

		attached.descriptorPool = factory->createDescriptorPool(descriptor_pool_desc);

		// build passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
		{
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
		}
		// build passes
		for (auto iter = attached.passesOnetime.begin(); iter != attached.passesOnetime.end(); iter++)
		{
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
		}
	}

	auto RenderGraphBuilder::beginMultiDispatchScope(std::string_view name) noexcept -> NodeHandle
	{
		MemScope<MultiDispatchScope> mds = MemNew<MultiDispatchScope>();
		NodeHandle handle = attached.registry.registNode(std::move(mds));
		attached.passes.emplace_back(handle);
		attached.tag(handle, name);
		scopeStack.emplace_back(handle);

		return handle;
	}

	auto RenderGraphBuilder::endScope() noexcept -> NodeHandle
	{
		MemScope<PassScopeEnd> pse = MemNew<PassScopeEnd>();
		pse->scopeBeginHandle = (scopeStack.back());
		pse->type = NodeDetailedType::SCOPE_END;
		NodeHandle handle = attached.registry.registNode(std::move(pse));
		attached.passes.emplace_back(handle);
		attached.tag(handle, "Scope End");
		scopeStack.pop_back();
		return handle;
	}

	auto RenderGraphBuilder::addRasterPass(std::vector<NodeHandle> const& ins, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<RasterPassNode> rpn = MemNew<RasterPassNode>((void*)&attached, ins, constant_size);
		rpn->type = NodeDetailedType::RASTER_PASS;
		NodeHandle handle = attached.registry.registNode(std::move(rpn));
		attached.passes.emplace_back(handle);
		return handle;
	}
}