module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
#include <glm/glm.hpp>
#include "entt/entt.hpp"
export module Demo.Portal;
import Core.Buffer;
import Core.Cache;
import Core.Image;
import Core.File;
import Core.Time;
import Core.MemoryManager;
import ECS.Entity;
import RHI.IEnum;
import RHI.IFactory;
import RHI.IBuffer;
import RHI.ICommandBuffer;
import RHI.IShader;
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.ITexture;
import RHI.ITextureView;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;
import GFX.RDG.ComputeSeries;
import GFX.Transform;
import GFX.BoundingBox;
import GFX.ParticleSystem;
import GFX.RDG.RenderGraph;

import ParticleSystem.ParticleSystem;
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)

namespace SIByL::Demo
{
	export class PortalSystem :public ParticleSystem::ParticleSystem
	{
	public:
		PortalSystem() = default;
		PortalSystem(RHI::IResourceFactory* factory, Timer* timer);

		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		auto registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;
		virtual auto registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;

		auto freshRenderPipeline(GFX::RDG::RenderGraph* rendergraph) noexcept -> bool;

		ECS::Entity entity = {};
		Timer* timer;

		GFX::RDG::NodeHandle hiz;

		// Resource Nodes Handles --------------------------
		GFX::RDG::NodeHandle particlePosLifetickBuffer;
		GFX::RDG::NodeHandle particleVelocityMassBuffer;
		GFX::RDG::NodeHandle particleColorBuffer;
		
		GFX::RDG::NodeHandle spriteHandle;
		GFX::RDG::NodeHandle samplerHandle;
		GFX::RDG::NodeHandle bakedCurveHandle;
		GFX::RDG::NodeHandle emitterVolumeHandle;
		GFX::RDG::NodeHandle mortonCodesHandle;
		GFX::RDG::NodeHandle doubleBufferedIndicesHandle;

		GFX::RDG::NodeHandle intermediateHistogram;
		GFX::RDG::NodeHandle intermediateHistogramLookback;
		GFX::RDG::NodeHandle globalHistogram;
		GFX::RDG::NodeHandle onesweepLookbackAggregate;
		GFX::RDG::NodeHandle onesweepLookbackPrefix;
		GFX::RDG::NodeHandle forwardPerViewUniformBufferFlight;
		GFX::RDG::NodeHandle cullingInfo;

		GFX::RDG::NodeHandle rasterColorAttachment;
		GFX::RDG::NodeHandle rasterDepthAttachment;

		// Resource ----------------------------------------
		MemScope<RHI::IStorageBuffer> torusBuffer;
		MemScope<RHI::IShader> shaderPortalInit;
		MemScope<RHI::IShader> shaderPortalEmit;
		MemScope<RHI::IShader> shaderPortalUpdate;

		MemScope<RHI::ISampler> sampler;
		MemScope<RHI::ITexture> texture;
		MemScope<RHI::ITextureView> textureView;
		MemScope<RHI::ITexture> bakedCurves;
		MemScope<RHI::ITextureView> bakedCurvesView;

		GFX::RDG::ComputeDispatch* emitDispatch = nullptr;
		GFX::RDG::ComputeDispatch* softrasterDispatch = nullptr;
		GFX::RDG::ComputeDispatch* cullInfoDispatch = nullptr;

		RHI::IResourceFactory* factory;
		GFX::RDG::RenderGraph portal_rdg;
	};

	PortalSystem::PortalSystem(RHI::IResourceFactory* factory, Timer* timer)
		:factory(factory), timer(timer)
	{
		// load precomputed samples for particle position initialization
		Buffer torusSamples;
		Buffer* samples[] = { &torusSamples };
		EmptyHeader header;
		CacheBrain::instance()->loadCache(2267996151488940154, header, samples);
		torusBuffer = factory->createStorageBuffer(&torusSamples);

		// load texture & baked curves image from file
		sampler = factory->createSampler({});
		Image image("./portal/Sparkle.tga");
		texture = factory->createTexture(&image);
		textureView = factory->createTextureView(texture.get());
		Image baked_image("./portal/portal_bake.tga");
		bakedCurves = factory->createTexture(&baked_image);
		bakedCurvesView = factory->createTextureView(bakedCurves.get());

		// init particle system
		shaderPortalInit = factory->createShaderFromBinaryFile("portal/portal_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderPortalEmit = factory->createShaderFromBinaryFile("portal/portal_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderPortalUpdate = factory->createShaderFromBinaryFile("portal/portal_update.spv", { RHI::ShaderStage::COMPUTE,"main" });
		init(sizeof(float) * 4 * 4, 100000, shaderPortalInit.get(), shaderPortalEmit.get(), shaderPortalUpdate.get());
	}

	auto PortalSystem::registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RenderGraphWorkshop local_workshop(portal_rdg);
		local_workshop.addInternalSampler();
		// Build Up Local Resources
		auto particle_buffer_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * 4 * maxParticleCount, "Particle Buffer");
		auto particle_buffer_pos_lifetick_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Pos-Lifetick");
		auto particle_buffer_vel_mass_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Velocity_Mass");
		auto particle_buffer_color_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Color");
		auto dead_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Dead Index Buffer");
		auto live_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Live Index Buffer");
		auto counter_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * 5, "Counter Buffer");
		auto indirect_draw_buffer_handle = local_workshop.addIndirectDrawBuffer("Indirect Draw Buffer");
		auto morton_codes_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Morton Codes Buffer");
		auto double_buffered_indices_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount*2, "Double-Buffered Indices Buffer");

		auto intermediate_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Block-Wise Histogram");
		auto intermediate_histogram_lookback_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256 * GRIDSIZE(maxParticleCount, (8 * 256)), "Block-Wise Sum Look Back");
		auto global_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Global Histogram");
		auto onesweep_lookback_aggregate = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(maxParticleCount, (256 * 8)), "Offset (Aggregate)");
		auto onesweep_lookback_prefix = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(maxParticleCount, (256 * 8)), "Offset (Prefix)");
		auto culling_info_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * 2 * GRIDSIZE(maxParticleCount, (16)), "Culling Info");

		// Build Up Local Init Pass
		local_workshop.addComputePassScope("Local Init");
		GFX::RDG::ComputePipelineScope* particle_init_pipeline = local_workshop.addComputePipelineScope("Local Init", "Particle Init");
		{
			particle_init_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_init_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Particle Init", "Common");
				particle_init_mat_scope->resources = { counter_buffer_handle, dead_index_buffer_handle, indirect_draw_buffer_handle };
				auto particle_init_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Particle Init", "Common", "Only");
				particle_init_dispatch_scope->pushConstant = [](Buffer& buffer) {
					uint32_t size = 100000u;
					buffer = std::move(Buffer(sizeof(size), 1));
					memcpy(buffer.getData(), &size, sizeof(size));
				};
				particle_init_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 200;
					y = 1;
					z = 1;
				};
			}
		}
		// Build Up Local Render Graph
		local_workshop.build(factory, 0, 0);
		MemScope<RHI::ICommandBuffer> transientCommandbuffer = factory->createTransientCommandBuffer();
		transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		portal_rdg.recordCommandsNEW(transientCommandbuffer.get(), 0);
		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		factory->deviceIdle();

		// Build Up Main Resources - Buffers
		particlePosLifetickBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_pos_lifetick_handle)->getStorageBuffer(), "Particle Pos Lifetick Buffer");
		particleVelocityMassBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_vel_mass_handle)->getStorageBuffer(), "Particle Velocity Mass Buffer");
		particleColorBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_color_handle)->getStorageBuffer(), "Particle Color Buffer");
		particleBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_handle)->getStorageBuffer(), "Particle Buffer");

		deadIndexBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(dead_index_buffer_handle)->getStorageBuffer(), "Dead Index Buffer");
		liveIndexBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(live_index_buffer_handle)->getStorageBuffer(), "Live Index Buffer");
		counterBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(counter_buffer_handle)->getStorageBuffer(), "Counter Buffer");
		indirectDrawBuffer = workshop->addIndirectDrawBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(indirect_draw_buffer_handle)->getStorageBuffer(), "Indirect Draw Buffer");
		mortonCodesHandle = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(morton_codes_handle)->getStorageBuffer(), "Morton Code Buffer Ref");
		doubleBufferedIndicesHandle = workshop->addIndirectDrawBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(double_buffered_indices_handle)->getStorageBuffer(), "Double-Buffered Indices Buffer");

		intermediateHistogram = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(intermediate_histogram_handle)->getStorageBuffer(), "Block-Wise Histogram Ref");
		intermediateHistogramLookback = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(intermediate_histogram_lookback_handle)->getStorageBuffer(), "Block-Wise Sum Look Back Ref");
		globalHistogram = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(global_histogram_handle)->getStorageBuffer(), "Global Histogram Ref");
		onesweepLookbackAggregate = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(onesweep_lookback_aggregate)->getStorageBuffer(), "Offset (Aggregate) Ref");
		onesweepLookbackPrefix = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(onesweep_lookback_prefix)->getStorageBuffer(), "Offset (Prefix) Ref");
		cullingInfo = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(culling_info_handle)->getStorageBuffer(), "Culling Info Ref");

		// Build Up Main Resources - Misc
		samplerHandle = workshop->getInternalSampler("Default Sampler");
		spriteHandle = workshop->addColorBufferExt(texture.get(), textureView.get(), "Sprite");
		bakedCurveHandle = workshop->addColorBufferExt(bakedCurves.get(), bakedCurvesView.get(), "Baked Curves");
		emitterVolumeHandle = workshop->addStorageBufferExt(torusBuffer.get(), "Emitter Volume Samples");
	}

	export struct EmitConstant
	{
		glm::mat4 matrix;
		unsigned int emitCount;
		float time;
	};

	export struct MortonConstant
	{
		glm::vec4 min;
		glm::vec4 max;
	};

	auto PortalSystem::registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		auto indefinite_scope = workshop->addComputePassIndefiniteScope("Particle System");
		indefinite_scope->customDispatchCount = [&timer = timer]()
			{
				static float deltaTime = 0;
				deltaTime += timer->getMsPF() > 100 ? 100 : timer->getMsPF();
				unsigned int dispatch_times = (unsigned int)(deltaTime / 20);
				deltaTime -= dispatch_times * 20;
				return dispatch_times;
			};

		// Create Emit Pass
		GFX::RDG::ComputePipelineScope* portal_emit_pipeline = workshop->addComputePipelineScope("Particle System", "Emit-Portal");
		{
			portal_emit_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_emit_mat_scope = workshop->addComputeMaterialScope("Particle System", "Emit-Portal", "Common");
				particle_emit_mat_scope->resources = { particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, emitterVolumeHandle, samplerHandle, indirectDrawBuffer };
				particle_emit_mat_scope->sampled_textures = { bakedCurveHandle };
				auto particle_emit_dispatch_scope = workshop->addComputeDispatch("Particle System", "Emit-Portal", "Common", "Only");
				emitDispatch = particle_emit_dispatch_scope;
				particle_emit_dispatch_scope->pushConstant = [&timer = timer](Buffer& buffer) {
					EmitConstant emitConstant{ glm::mat4(), 400000u / 50,(float)timer->getTotalTime()};
					buffer = std::move(Buffer(sizeof(emitConstant), 1));
					memcpy(buffer.getData(), &emitConstant, sizeof(emitConstant));
				};
				particle_emit_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 200;
					y = 1;
					z = 1;
				};
			}
		}

		// Create Update Pass
		GFX::RDG::ComputePipelineScope* portal_update_pipeline = workshop->addComputePipelineScope("Particle System", "Update-Portal");
		{
			portal_update_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_update.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_update_mat_scope = workshop->addComputeMaterialScope("Particle System", "Update-Portal", "Common");
				particle_update_mat_scope->resources = { particlePosLifetickBuffer, particleVelocityMassBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer };
				particle_update_mat_scope->sampled_textures = { bakedCurveHandle };
				auto particle_update_dispatch_scope = workshop->addComputeDispatch("Particle System", "Update-Portal", "Common", "Only");
				particle_update_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 200;
					y = 1;
					z = 1;
				};
			}
		}

		// Create Cluster Pass
		workshop->addComputePassScope("Particle System Cluster");
		// Create Morton Pipeline
		GFX::RDG::ComputePipelineScope* portal_morton_pipeline = workshop->addComputePipelineScope("Particle System Cluster", "Morton-Portal");
		{
			portal_morton_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_morton.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_morton_mat_scope = workshop->addComputeMaterialScope("Particle System Cluster", "Morton-Portal", "Common");
				particle_morton_mat_scope->resources = { mortonCodesHandle, doubleBufferedIndicesHandle, counterBuffer, particlePosLifetickBuffer, liveIndexBuffer };
				auto particle_morton_dispatch_scope = workshop->addComputeDispatch("Particle System Cluster", "Morton-Portal", "Common", "Only");

				auto& transformComponent = entity.getComponent<GFX::Transform>();
				auto& boundingBox = entity.getComponent<GFX::BoundingBox>();

				particle_morton_dispatch_scope->pushConstant = [&timer = timer, boundingBox = &boundingBox, transformComponent = &transformComponent](Buffer& buffer) {
					glm::mat4x4 transform = transformComponent->getAccumulativeTransform();
					glm::vec4 offset = { transform[3][0],transform[3][1] ,transform[3][2],0 };
					MortonConstant mortonConstant{ boundingBox->min + offset, boundingBox->max + offset };
					buffer = std::move(Buffer(sizeof(mortonConstant), 1));
					memcpy(buffer.getData(), &mortonConstant, sizeof(mortonConstant));
				};
				particle_morton_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(100000, 512);
					y = 1;
					z = 1;
				};
			}
		}
		// Create Histogram Pipeline
		GFX::RDG::ComputePipelineScope* portal_histogram_pergroup_pipeline = workshop->addComputePipelineScope("Particle System Cluster", "Histogram_PerGroup-Portal");
		{
			portal_histogram_pergroup_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_histogram_subgroup_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_histogram_01_mat_scope = workshop->addComputeMaterialScope("Particle System Cluster", "Histogram_PerGroup-Portal", "Common");
				particle_histogram_01_mat_scope->resources = { mortonCodesHandle, doubleBufferedIndicesHandle, intermediateHistogram, intermediateHistogramLookback, counterBuffer };
				auto particle_histogram_01_dispatch_scope = workshop->addComputeDispatch("Particle System Cluster", "Histogram_PerGroup-Portal", "Common", "Only");
				particle_histogram_01_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(100000, (8 * 256));
					y = 1;
					z = 1;
				};
			}
		}
		GFX::RDG::ComputePipelineScope* portal_histogram_integrate_pipeline = workshop->addComputePipelineScope("Particle System Cluster", "Histogram_Integrate-Portal");
		{
			portal_histogram_integrate_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_histogram_integrate_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_histogram_02_mat_scope = workshop->addComputeMaterialScope("Particle System Cluster", "Histogram_Integrate-Portal", "Common");
				particle_histogram_02_mat_scope->resources = { intermediateHistogram, globalHistogram, counterBuffer };
				auto particle_histogram_02_dispatch_scope = workshop->addComputeDispatch("Particle System Cluster", "Histogram_Integrate-Portal", "Common", "Only");
				particle_histogram_02_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 1;
					y = 1;
					z = 1;
				};
			}
		}
		// Create OneSweep Pass Pipeline
		GFX::RDG::ComputePipelineScope* portal_onesweep_pipeline = workshop->addComputePipelineScope("Particle System Cluster", "Histogram_Onesweep-Portal");
		{
			portal_onesweep_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_onesweep_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Onesweep-i"
			for (int i = 0; i < 4; i++)
			{
				{
					auto particle_onesweep_00_mat_scope = workshop->addComputeMaterialScope("Particle System Cluster", "Histogram_Onesweep-Portal", "Onesweep-" + std::to_string(i));
					particle_onesweep_00_mat_scope->resources = { mortonCodesHandle, doubleBufferedIndicesHandle, onesweepLookbackAggregate, onesweepLookbackPrefix, globalHistogram, counterBuffer };
					auto particle_onesweep_00_dispatch_scope = workshop->addComputeDispatch("Particle System Cluster", "Histogram_Onesweep-Portal", "Onesweep-" + std::to_string(i), "Only");
					particle_onesweep_00_dispatch_scope->pushConstant = [i = i](Buffer& buffer) {
						uint32_t pass = i;
						buffer = std::move(Buffer(sizeof(pass), 1));
						memcpy(buffer.getData(), &pass, sizeof(pass));
					};
					particle_onesweep_00_dispatch_scope->customSize = [maxParticleCount = maxParticleCount](uint32_t& x, uint32_t& y, uint32_t& z) {
						x = GRIDSIZE(maxParticleCount, (256 * 8));
						y = 1;
						z = 1;
					};
				}
			}
		}
	}
	
	auto PortalSystem::registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		auto forward_pass = workshop->renderGraph.getRasterPassScope("Forward Pass");
		forwardPerViewUniformBufferFlight = forward_pass->getPerViewUniformBufferFlightHandle();

		// Create CullingInfo Pass Pipeline
		GFX::RDG::ComputePipelineScope* portal_culling_info_pipeline = workshop->addComputePipelineScope("Particle Bounding Boxes", "CullingInfo-Portal");
		{
			portal_culling_info_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_culling_info_generation.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "CullingInfo"
			{
				auto particle_culling_info_mat_scope = workshop->addComputeMaterialScope("Particle Bounding Boxes", "CullingInfo-Portal", "Common");
				particle_culling_info_mat_scope->resources = { forwardPerViewUniformBufferFlight, particlePosLifetickBuffer, particleVelocityMassBuffer, doubleBufferedIndicesHandle, cullingInfo, counterBuffer };
				auto particle_culling_info_dispatch_scope = workshop->addComputeDispatch("Particle Bounding Boxes", "CullingInfo-Portal", "Common", "Only");
				cullInfoDispatch = particle_culling_info_dispatch_scope;
				particle_culling_info_dispatch_scope->customSize = [maxParticleCount = maxParticleCount](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(maxParticleCount, (512));
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto PortalSystem::registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RasterPipelineScope* trancparency_portal_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Portal");
		{
			trancparency_portal_pipeline->shaderVert = factory->createShaderFromBinaryFile("portal/portal_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
			trancparency_portal_pipeline->shaderFrag = factory->createShaderFromBinaryFile("portal/portal_vert_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			trancparency_portal_pipeline->cullMode = RHI::CullMode::BACK;
			trancparency_portal_pipeline->vertexBufferLayout =
			{
				{RHI::DataType::Float3, "Position"},
				{RHI::DataType::Float3, "Color"},
				{RHI::DataType::Float2, "UV"},
			};
			trancparency_portal_pipeline->colorBlendingDesc = RHI::AdditionBlending;
			trancparency_portal_pipeline->depthStencilDesc = RHI::TestLessButNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Portal", "Portal");
			portal_mat_scope->resources = { samplerHandle, particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, samplerHandle, liveIndexBuffer };
			portal_mat_scope->sampled_textures = { spriteHandle, bakedCurveHandle };
		}
		trancparency_portal_pipeline->isActive = false;

		GFX::RDG::RasterPipelineScope* trancparency_portal_mesh_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Portal Mesh");
		{
			trancparency_portal_mesh_pipeline->shaderMesh = factory->createShaderFromBinaryFile("portal/portal_mesh.spv", { RHI::ShaderStage::MESH,"main" });
			trancparency_portal_mesh_pipeline->shaderFrag = factory->createShaderFromBinaryFile("portal/portal_mesh_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			trancparency_portal_mesh_pipeline->cullMode = RHI::CullMode::NONE;

			trancparency_portal_mesh_pipeline->colorBlendingDesc = RHI::AdditionBlending;
			trancparency_portal_mesh_pipeline->depthStencilDesc = RHI::TestLessButNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Portal Mesh", "Portal");
			portal_mat_scope->resources = { samplerHandle, particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, samplerHandle, doubleBufferedIndicesHandle, indirectDrawBuffer };
			portal_mat_scope->sampled_textures = { spriteHandle, bakedCurveHandle };
		}
		trancparency_portal_mesh_pipeline->isActive = true;

		GFX::RDG::RasterPipelineScope* trancparency_portal_mesh_culling_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Portal Mesh Culling");
		{
			trancparency_portal_mesh_culling_pipeline->shaderTask = factory->createShaderFromBinaryFile("portal/portal_mesh_culling_frustrum_task.spv", { RHI::ShaderStage::TASK,"main" });
			trancparency_portal_mesh_culling_pipeline->shaderMesh = factory->createShaderFromBinaryFile("portal/portal_mesh_culling_frustrum_mesh.spv", { RHI::ShaderStage::MESH,"main" });
			trancparency_portal_mesh_culling_pipeline->shaderFrag = factory->createShaderFromBinaryFile("portal/portal_mesh_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			trancparency_portal_mesh_culling_pipeline->cullMode = RHI::CullMode::NONE;

			trancparency_portal_mesh_culling_pipeline->colorBlendingDesc = RHI::AdditionBlending;
			trancparency_portal_mesh_culling_pipeline->depthStencilDesc = RHI::TestLessButNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Portal Mesh Culling", "Portal");
			portal_mat_scope->resources = { samplerHandle, particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, samplerHandle, doubleBufferedIndicesHandle, indirectDrawBuffer, cullingInfo, workshop->getInternalSampler("HiZ Sampler") };
			portal_mat_scope->sampled_textures = { spriteHandle, bakedCurveHandle, hiz };
		}
		trancparency_portal_mesh_culling_pipeline->isActive = false;

		GFX::RDG::RasterPipelineScope* culling_aabb_vis_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Vis AABB Portal");
		{
			culling_aabb_vis_pipeline->shaderMesh = factory->createShaderFromBinaryFile("portal/portal_aabb_vis.spv", { RHI::ShaderStage::MESH,"main" });
			culling_aabb_vis_pipeline->shaderFrag = factory->createShaderFromBinaryFile("portal/portal_aabb_vis_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			culling_aabb_vis_pipeline->cullMode = RHI::CullMode::NONE;
			culling_aabb_vis_pipeline->colorBlendingDesc = RHI::AdditionBlending;
			culling_aabb_vis_pipeline->depthStencilDesc = RHI::NoTestAndNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Vis AABB Portal", "Portal");
			portal_mat_scope->resources = { cullingInfo, indirectDrawBuffer, workshop->getInternalSampler("HiZ Sampler") };
			portal_mat_scope->sampled_textures = { hiz };
		}
		culling_aabb_vis_pipeline->isActive = true;

		workshop->addComputePassScope("Particle Software Raster");
		GFX::RDG::ComputePipelineScope* portal_softraster_pipeline = workshop->addComputePipelineScope("Particle Software Raster", "SoftwareRaster-Portal");
		{
			portal_softraster_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_softraster_comp.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "CullingInfo"
			{
				auto particle_softraster_info_mat_scope = workshop->addComputeMaterialScope("Particle Software Raster", "SoftwareRaster-Portal", "Common");
				particle_softraster_info_mat_scope->resources = { rasterColorAttachment, samplerHandle, cullingInfo, indirectDrawBuffer };
				particle_softraster_info_mat_scope->sampled_textures = { rasterDepthAttachment };
				auto particle_culling_info_dispatch_scope = workshop->addComputeDispatch("Particle Software Raster", "SoftwareRaster-Portal", "Common", "Only");
				softrasterDispatch = particle_culling_info_dispatch_scope;
				particle_culling_info_dispatch_scope->customSize = [maxParticleCount = maxParticleCount](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(maxParticleCount, (16));
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto PortalSystem::freshRenderPipeline(GFX::RDG::RenderGraph* rendergraph) noexcept -> bool
	{
		GFX::ParticleSystem& ps = entity.getComponent<GFX::ParticleSystem>();
		if (ps.needRebuildPipeline)
		{
			rendergraph->getRasterPipelineScope("Forward Pass", "Vis AABB Portal")->isActive = ps.showCluster;
			ps.needRebuildPipeline = false;
			return true;
		}
		else
			return false;
	}

}