module;
#include <string>
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
import GFX.RDG.ColorBufferNode;
import GFX.RDG.SamplerNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.MultiDispatchScope;
import GFX.RDG.RasterNodes;
import GFX.RDG.ExternalAccess;
import GFX.RDG.ComputeSeries;

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
		for (auto iter = passesBackPool.begin(); iter != passesBackPool.end(); iter++)
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
		for (auto iter = passesBackPool.begin(); iter != passesBackPool.end(); iter++)
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

	auto RenderGraph::recordCommandsNEW(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		for (auto const& passHandle : passList)
		{
			getPassNode(passHandle)->onCommandRecord(commandbuffer, flight);
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

	auto RenderGraph::getComputePipelineScope(std::string const& pass, std::string const& pipeline) noexcept -> ComputePipelineScope*
	{
		if (computePassRegister.find(pass) == computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getComputePipelineScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = computePassRegister.find(pass)->second;
		ComputePassScope* pass_node = (ComputePassScope*)registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getComputePipelineScope() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		return (ComputePipelineScope*)registry.getNode(pipeline_node_handle);
	}

	auto RenderGraph::getRasterPipelineScope(std::string const& pass, std::string const& pipeline) noexcept -> RasterPipelineScope*
	{
		if (rasterPassRegister.find(pass) == rasterPassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getRasterPipelineScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = rasterPassRegister.find(pass)->second;
		RasterPassScope* pass_node = (RasterPassScope*)registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getRasterPipelineScope() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		return (RasterPipelineScope*)registry.getNode(pipeline_node_handle);
	}

	auto RenderGraph::getRasterMaterialScope(std::string const& pass, std::string const& pipeline, std::string const& mat) noexcept -> RasterMaterialScope*
	{
		if (rasterPassRegister.find(pass) == rasterPassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getRasterMaterialScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = rasterPassRegister.find(pass)->second;
		RasterPassScope* pass_node = (RasterPassScope*)registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getRasterMaterialScope() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		RasterPipelineScope* pipeline_node = (RasterPipelineScope*)registry.getNode(pipeline_node_handle);
		if (pipeline_node->materialScopesRegister.find(mat) == pipeline_node->materialScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph :: getRasterMaterialScope() material name \'{0}\' not found !", mat);
			return nullptr;
		}
		auto material_node_handle = pipeline_node->materialScopesRegister.find(mat)->second;
		RasterMaterialScope* material_node = (RasterMaterialScope*)registry.getNode(material_node_handle);
		return material_node;
	}

	auto RenderGraph::onFrameStart() noexcept -> void
	{
		for (auto iter = passList.begin(); iter != passList.end(); iter++)
			registry.getNode((*iter))->onFrameStart((void*)this);
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

	auto RenderGraph::getRasterPassScope(std::string const& pass) noexcept -> RasterPassScope*
	{
		return (RasterPassScope*)registry.getNode(rasterPassRegister[pass]);
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

	auto RenderGraphBuilder::addComputePassBackPool(RHI::IShader* shader, std::vector<NodeHandle>&& ios, std::string_view name, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<ComputePassNode> cpn = MemNew<ComputePassNode>((void*)&attached, shader, std::move(ios), constant_size);
		cpn->type = NodeDetailedType::COMPUTE_PASS;
		cpn->attributes |= (uint32_t)NodeAttrbutesFlagBits::ONE_TIME_SUBMIT;
		NodeHandle handle = attached.registry.registNode(std::move(cpn));
		attached.passesBackPool.emplace_back(handle);
		attached.tag(handle, name);
		return handle;
	}

	auto RenderGraphBuilder::addFrameBufferRef(std::vector<NodeHandle> const& color_attachments, NodeHandle depth_attachment, std::vector<NodeHandle> const& unclear) noexcept -> NodeHandle
	{
		MemScope<FramebufferContainer> fbc = MemNew<FramebufferContainer>();
		fbc->colorAttachCount = color_attachments.size();
		fbc->depthAttachCount = (depth_attachment == 0) ? 0 : 1;
		fbc->unclearHandles = unclear;
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

		// create descripotr pool
		uint32_t MAX_FRAMES_IN_FLIGHT = attached.getMaxFrameInFlight();
		RHI::DescriptorPoolDesc descriptor_pool_desc{ {}, (attached.passes.size() + attached.passesBackPool.size()) * MAX_FRAMES_IN_FLIGHT };
		if (attached.storageBufferDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_BUFFER, attached.storageBufferDescriptorCount);
		if (attached.uniformBufferDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::UNIFORM_BUFFER, attached.uniformBufferDescriptorCount);
		if (attached.samplerDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, attached.samplerDescriptorCount);
		if (attached.storageImageDescriptorCount > 0) descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_IMAGE, attached.storageImageDescriptorCount);
		attached.descriptorPool = factory->createDescriptorPool(descriptor_pool_desc);

		// devirtualize resources
		for (auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
			attached.registry.getNode((*iter))->devirtualize((void*)&attached, factory);
		// devirtualize passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
			attached.registry.getNode((*iter))->devirtualize((void*)&attached, factory);
		for (auto iter = attached.passList.begin(); iter != attached.passList.end(); iter++)
			attached.registry.getNode((*iter))->devirtualize((void*)&attached, factory);
		// devirtualize passes onetime
		for (auto iter = attached.passesBackPool.begin(); iter != attached.passesBackPool.end(); iter++)
			attached.registry.getNode((*iter))->devirtualize((void*)&attached, factory);

		// clear barriers
		attached.barrierPool.barriers.clear();
		attached.barrierPool.barriers.clear();
		attached.barrierPool.barriers.clear();

		// compile resources
		for (auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
			attached.registry.getNode((*iter))->onCompile((void*)&attached, factory);
		// compile passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
			attached.registry.getNode((*iter))->onCompile((void*)&attached, factory);

		// build resources
		for(auto iter = attached.resources.begin(); iter != attached.resources.end(); iter++)
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
		// build passes
		for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
			attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
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

	auto RenderGraphBuilder::addRasterPassBackPool(std::vector<NodeHandle> const& ins, uint32_t const& constant_size) noexcept -> NodeHandle
	{
		MemScope<RasterPassNode> rpn = MemNew<RasterPassNode>((void*)&attached, ins, constant_size);
		rpn->type = NodeDetailedType::RASTER_PASS;
		NodeHandle handle = attached.registry.registNode(std::move(rpn));
		attached.passesBackPool.emplace_back(handle);
		return handle;
	}

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================
	auto RenderGraphWorkshop::addAgency(MemScope<Agency>&& agency) noexcept -> void
	{
		agency->onRegister(this);
		renderGraph.agencies.emplace_back(std::move(agency));
	}

	auto RenderGraphWorkshop::build(RHI::IResourceFactory* factory, uint32_t const& width, uint32_t const& height) noexcept -> void
	{
		renderGraph.factory = factory;
		renderGraph.datumWidth = width;
		renderGraph.datumHeight = height;

		// create descripotr pool
		uint32_t MAX_FRAMES_IN_FLIGHT = renderGraph.getMaxFrameInFlight();
		RHI::DescriptorPoolDesc descriptor_pool_desc{ {}, 1000 };
		descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_BUFFER, 1000);
		descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::UNIFORM_BUFFER, 1000);
		descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, 1000);
		descriptor_pool_desc.typeAndCount.emplace_back(RHI::DescriptorType::STORAGE_IMAGE, 1000);
		renderGraph.descriptorPool = factory->createDescriptorPool(descriptor_pool_desc);

		// agency before compile
		for (auto& agency : renderGraph.agencies)
		{
			agency->startWorkshopBuild(this);
			agency->beforeCompile();
		}
		// clear barriers
		renderGraph.barrierPool.barriers.clear();
		// compile resources
		for (auto iter = renderGraph.resources.begin(); iter != renderGraph.resources.end(); iter++)
			renderGraph.registry.getNode((*iter))->onCompile((void*)&renderGraph, factory);
		// compile passes
		for (auto iter = renderGraph.passList.begin(); iter != renderGraph.passList.end(); iter++)
			renderGraph.registry.getNode((*iter))->onCompile((void*)&renderGraph, factory);

		// devirtualize resources
		for (auto iter = renderGraph.resources.begin(); iter != renderGraph.resources.end(); iter++)
			renderGraph.registry.getNode((*iter))->devirtualize((void*)&renderGraph, factory);
		// agency before devirtualize passes
		for (auto& agency : renderGraph.agencies)
			agency->beforeDivirtualizePasses();
		// devirtualize passes
		for (auto iter = renderGraph.passList.begin(); iter != renderGraph.passList.end(); iter++)
			renderGraph.registry.getNode((*iter))->devirtualize((void*)&renderGraph, factory);

		// build resources
		for (auto iter = renderGraph.resources.begin(); iter != renderGraph.resources.end(); iter++)
			renderGraph.registry.getNode((*iter))->onBuild((void*)&renderGraph, factory);
		//// build passes
		//for (auto iter = attached.passes.begin(); iter != attached.passes.end(); iter++)
		//	attached.registry.getNode((*iter))->onBuild((void*)&attached, factory);
	}

	auto RenderGraphWorkshop::addInternalSampler() noexcept -> void
	{
		// default
		NodeHandle default_sampler = addSampler({}, "Default Sampler");
		renderGraph.samplerRegister.emplace("Default Sampler", default_sampler);

		// repeat
		RHI::SamplerDesc repeat_desc = {};
		repeat_desc.clampModeU = RHI::AddressMode::REPEAT;
		repeat_desc.clampModeV = RHI::AddressMode::REPEAT;
		repeat_desc.clampModeW = RHI::AddressMode::REPEAT;
		NodeHandle repeat_sampler = addSampler(repeat_desc, "Repeat Sampler");
		renderGraph.samplerRegister.emplace("Repeat Sampler", repeat_sampler);

		// min pooling
		RHI::SamplerDesc min_pooling_desc = {};
		min_pooling_desc.extension = RHI::Extension::MIN_POOLING;
		NodeHandle min_pooling_sampler = addSampler(min_pooling_desc, "MinPooling Sampler");
		renderGraph.samplerRegister.emplace("MinPooling Sampler", min_pooling_sampler);
	}

	auto RenderGraphWorkshop::getInternalSampler(std::string const& name) noexcept -> NodeHandle
	{
		auto iter = renderGraph.samplerRegister.find(name);
		if (iter == renderGraph.samplerRegister.end())
		{
			SE_CORE_ERROR("RDG :: getInternalSampler() name \'{0}\'not registry", name);
			return NodeHandle{};
		}
		else
			return iter->second;
	}

	auto RenderGraphWorkshop::addUniformBuffer(size_t size, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<UniformBufferNode> ubn = MemNew<UniformBufferNode>();
		ubn->size = size;
		ubn->tag = name;
		ubn->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(ubn));
		renderGraph.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphWorkshop::addUniformBufferFlights(size_t size, std::string const& name) noexcept -> NodeHandle
	{
		uint32_t flights_count = renderGraph.getMaxFrameInFlight();
		std::vector<NodeHandle> handles(flights_count);
		for (uint32_t i = 0; i < flights_count; i++)
		{
			handles[i] = addUniformBuffer(size, name + " Sub-Flight=" + std::to_string(i));
		}
		MemScope<FlightContainer> fc = MemNew<FlightContainer>(std::move(handles));
		fc->type = NodeDetailedType::UNIFORM_BUFFER;
		fc->tag = name;
		fc->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(fc));
		renderGraph.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphWorkshop::addStorageBuffer(size_t size, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = size;
		NodeHandle handle = renderGraph.registry.registNode(std::move(sbn));
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addStorageBufferExt(RHI::IStorageBuffer* external, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<StorageBufferNode> sbn = MemNew<StorageBufferNode>();
		sbn->size = external->getSize();
		sbn->attributes |= (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER;
		sbn->externalStorageBuffer = external;
		NodeHandle handle = renderGraph.registry.registNode(std::move(sbn));
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addIndirectDrawBuffer(std::string const& name) noexcept -> NodeHandle
	{
		MemScope<IndirectDrawBufferNode> sbn = MemNew<IndirectDrawBufferNode>();
		sbn->size = sizeof(unsigned int) * 5;
		NodeHandle handle = renderGraph.registry.registNode(std::move(sbn));
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addIndirectDrawBufferExt(RHI::IStorageBuffer* external, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<IndirectDrawBufferNode> sbn = MemNew<IndirectDrawBufferNode>();
		sbn->size = sizeof(unsigned int) * 5;
		sbn->attributes |= (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER;
		sbn->externalStorageBuffer = external;
		NodeHandle handle = renderGraph.registry.registNode(std::move(sbn));
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addSampler(RHI::SamplerDesc const& desc, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<SamplerNode> sn = MemNew<SamplerNode>();
		sn->tag = name;
		sn->desc = desc;
		NodeHandle handle = renderGraph.registry.registNode(std::move(sn));
		renderGraph.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphWorkshop::addSamplerExt(RHI::ISampler* sampler, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<SamplerNode> sn = MemNew<SamplerNode>();
		sn->tag = name;
		sn->extSampler = sampler;
		sn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER);
		NodeHandle handle = renderGraph.registry.registNode(std::move(sn));
		renderGraph.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphWorkshop::addColorBuffer(RHI::ResourceFormat format, float const& rel_width, float const& rel_height, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>(format, rel_width, rel_height);
		NodeHandle handle = renderGraph.registry.registNode(std::move(cbn));
		getNode<ColorBufferNode>(handle)->signified = handle;
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addColorBufferExt(RHI::ITexture* texture, RHI::ITextureView* view, std::string const& name, bool present) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>();
		cbn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER);
		cbn->texture.ref = texture;
		cbn->textureView.ref = view;
		if(texture) cbn->format = texture->getDescription().format;
		if (present) cbn->attributes |= (uint32_t)NodeAttrbutesFlagBits::PRESENT;
		NodeHandle handle = renderGraph.registry.registNode(std::move(cbn));
		getNode<ColorBufferNode>(handle)->signified = handle;
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addColorBufferRef(RHI::ITexture* texture, RHI::ITextureView* view, NodeHandle origin, std::string const& name) noexcept -> NodeHandle
	{
		MemScope<ColorBufferNode> cbn = MemNew<ColorBufferNode>();
		ResourceNode* origin_resource = getNode<ResourceNode>(origin);
		cbn->attributes |= addBit(NodeAttrbutesFlagBits::PLACEHOLDER) | addBit(NodeAttrbutesFlagBits::REFERENCE);
		cbn->texture.ref = texture;
		cbn->textureView.ref = view;
		cbn->consumeHistoryRef = &(origin_resource->consumeHistory);
		cbn->signified = origin;
		if (texture) cbn->format = texture->getDescription().format;
		NodeHandle handle = renderGraph.registry.registNode(std::move(cbn));
		renderGraph.resources.emplace_back(handle);
		renderGraph.tag(handle, name);
		return handle;
	}

	auto RenderGraphWorkshop::addFrameBufferRef(std::vector<NodeHandle> const& color_attachments, NodeHandle depth_attachment) noexcept -> NodeHandle
	{
		MemScope<FramebufferContainer> fbc = MemNew<FramebufferContainer>();
		fbc->colorAttachCount = color_attachments.size();
		fbc->depthAttachCount = (depth_attachment == 0) ? 0 : 1;
		fbc->handles.resize(fbc->colorAttachCount + fbc->depthAttachCount);
		for (int i = 0; i < fbc->colorAttachCount; i++)
		{
			fbc->handles[i] = color_attachments[i];
			renderGraph.getColorBufferNode(color_attachments[i])->usages |= (uint32_t)RHI::ImageUsageFlagBits::COLOR_ATTACHMENT_BIT;
		}
		if (depth_attachment) fbc->handles[fbc->colorAttachCount] = depth_attachment;
		NodeHandle handle = renderGraph.registry.registNode(std::move(fbc));
		renderGraph.resources.emplace_back(handle);
		return handle;
	}

	auto RenderGraphWorkshop::addRasterPassScope(std::string const& pass, NodeHandle const& framebuffer) noexcept -> RasterPassScope*
	{
		if (renderGraph.rasterPassRegister.find(pass) != renderGraph.rasterPassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addRasterPassScope() pass name \'{0}\' duplicated !", pass);
			return nullptr;
		}
		// create RasterPassScope
		MemScope<RasterPassScope> rps = MemNew<RasterPassScope>();
		rps->tag = pass;
		rps->type = NodeDetailedType::RASTER_PASS_SCOPE;
		rps->framebuffer = framebuffer;
		rps->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(rps));
		renderGraph.rasterPassRegister.emplace(pass, handle);
		renderGraph.passList.emplace_back(handle);
		return (RasterPassScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addRasterPipelineScope(std::string const& pass, std::string const& pipeline) noexcept -> RasterPipelineScope*
	{
		if (renderGraph.rasterPassRegister.find(pass) == renderGraph.rasterPassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addRasterPipelineScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = renderGraph.rasterPassRegister.find(pass)->second;
		RasterPassScope* pass_node = (RasterPassScope*)renderGraph.registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) != pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addRasterPipelineScope() pipeline name \'{0}\' duplicated !", pipeline);
			return nullptr;
		}
		// create RasterPipelineScope
		MemScope<RasterPipelineScope> rps = MemNew<RasterPipelineScope>();
		rps->tag = pipeline;
		rps->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(rps));
		pass_node->pipelineScopesRegister.emplace(pipeline, handle);
		pass_node->pipelineScopes.emplace_back(handle);
		return (RasterPipelineScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addRasterMaterialScope(std::string const& pass, std::string const& pipeline, std::string const& mat) noexcept -> RasterMaterialScope*
	{
		if (renderGraph.rasterPassRegister.find(pass) == renderGraph.rasterPassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addRasterMaterialScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = renderGraph.rasterPassRegister.find(pass)->second;
		RasterPassScope* pass_node = (RasterPassScope*)renderGraph.registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addRasterMaterialScope() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		RasterPipelineScope* pipeline_node = (RasterPipelineScope*)renderGraph.registry.getNode(pipeline_node_handle);
		// create RasterMaterialScope
		MemScope<RasterMaterialScope> rms = MemNew<RasterMaterialScope>();
		rms->tag = mat;
		rms->onRegistered(&renderGraph, this);
		rms->type = NodeDetailedType::RASTER_MATERIAL_SCOPE;
		NodeHandle handle = renderGraph.registry.registNode(std::move(rms));
		pipeline_node->materialScopesRegister.emplace(mat, handle);
		pipeline_node->materialScopes.emplace_back(handle);
		return (RasterMaterialScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addComputePassScope(std::string const& pass) noexcept -> void
	{
		if (renderGraph.computePassRegister.find(pass) != renderGraph.computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputePassScope() pass name \'{0}\' duplicated !", pass);
			return;
		}
		// create RasterPassScope
		MemScope<ComputePassScope> cps = MemNew<ComputePassScope>();
		cps->tag = pass;
		cps->type = NodeDetailedType::COMPUTE_PASS_SCOPE;
		cps->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(cps));
		renderGraph.computePassRegister.emplace(pass, handle);
		renderGraph.passList.emplace_back(handle);
	}

	auto RenderGraphWorkshop::addComputePassIndefiniteScope(std::string const& pass) noexcept -> ComputePassIndefiniteScope*
	{
		if (renderGraph.computePassRegister.find(pass) != renderGraph.computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputePassIndefiniteScope() pass name \'{0}\' duplicated !", pass);
			return nullptr;
		}
		// create RasterPassScope
		MemScope<ComputePassIndefiniteScope> cps = MemNew<ComputePassIndefiniteScope>();
		cps->tag = pass;
		cps->type = NodeDetailedType::COMPUTE_PASS_SCOPE;
		cps->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(cps));
		renderGraph.computePassRegister.emplace(pass, handle);
		renderGraph.passList.emplace_back(handle);
		return (ComputePassIndefiniteScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addComputePipelineScope(std::string const& pass, std::string const& pipeline) noexcept -> ComputePipelineScope*
	{
		if (renderGraph.computePassRegister.find(pass) == renderGraph.computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputePipelineScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = renderGraph.computePassRegister.find(pass)->second;
		ComputePassScope* pass_node = (ComputePassScope*)renderGraph.registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) != pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputePipelineScope() pipeline name \'{0}\' duplicated !", pipeline);
			return nullptr;
		}
		// create ComputePipelineScope
		MemScope<ComputePipelineScope> cps = MemNew<ComputePipelineScope>();
		cps->tag = pipeline;
		cps->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(cps));
		pass_node->pipelineScopesRegister.emplace(pipeline, handle);
		pass_node->pipelineScopes.emplace_back(handle);
		return (ComputePipelineScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addComputeMaterialScope(std::string const& pass, std::string const& pipeline, std::string const& mat) noexcept -> ComputeMaterialScope*
	{
		if (renderGraph.computePassRegister.find(pass) == renderGraph.computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputeMaterialScope() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = renderGraph.computePassRegister.find(pass)->second;
		ComputePassScope* pass_node = (ComputePassScope*)renderGraph.registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputeMaterialScope() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		ComputePipelineScope* pipeline_node = (ComputePipelineScope*)renderGraph.registry.getNode(pipeline_node_handle);
		// create ComputeMaterialScope
		MemScope<ComputeMaterialScope> cms = MemNew<ComputeMaterialScope>();
		cms->tag = mat;
		cms->type = NodeDetailedType::COMPUTE_MATERIAL_SCOPE;
		cms->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(cms));
		pipeline_node->materialScopesRegister.emplace(mat, handle);
		pipeline_node->materialScopes.emplace_back(handle);
		return (ComputeMaterialScope*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addComputeDispatch(std::string const& pass, std::string const& pipeline, std::string const& mat, std::string const& dispatch) noexcept -> ComputeDispatch*
	{
		if (renderGraph.computePassRegister.find(pass) == renderGraph.computePassRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputeDispatch() pass name \'{0}\' not found !", pass);
			return nullptr;
		}
		auto pass_node_handle = renderGraph.computePassRegister.find(pass)->second;
		ComputePassScope* pass_node = (ComputePassScope*)renderGraph.registry.getNode(pass_node_handle);
		if (pass_node->pipelineScopesRegister.find(pipeline) == pass_node->pipelineScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputeDispatch() pipeline name \'{0}\' not found !", pipeline);
			return nullptr;
		}
		auto pipeline_node_handle = pass_node->pipelineScopesRegister.find(pipeline)->second;
		ComputePipelineScope* pipeline_node = (ComputePipelineScope*)renderGraph.registry.getNode(pipeline_node_handle);
		if (pipeline_node->materialScopesRegister.find(mat) == pipeline_node->materialScopesRegister.end())
		{
			SE_CORE_ERROR("RDG :: Render Graph Workshop :: addComputeDispatch() pipeline name \'{0}\' not found !", mat);
			return nullptr;
		}
		auto material_node_handle = pipeline_node->materialScopesRegister.find(mat)->second;
		ComputeMaterialScope* material_node = (ComputeMaterialScope*)renderGraph.registry.getNode(material_node_handle);
		// create ComputeMaterialScope
		MemScope<ComputeDispatch> dsp = MemNew<ComputeDispatch>();
		dsp->tag = dispatch;
		dsp->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(dsp));
		material_node->dispatches.emplace_back(handle);
		return (ComputeDispatch*)(renderGraph.registry.getNode(handle));
	}

	auto RenderGraphWorkshop::addExternalAccessPass(std::string const& pass) noexcept -> NodeHandle
	{
		MemScope<ExternalAccessPass> eap = MemNew<ExternalAccessPass>();
		eap->tag = pass;
		eap->type = NodeDetailedType::EXTERNAL_ACCESS_PASS;
		eap->onRegistered(&renderGraph, this);
		NodeHandle handle = renderGraph.registry.registNode(std::move(eap));
		renderGraph.passList.emplace_back(handle);
		return handle;
	}
}