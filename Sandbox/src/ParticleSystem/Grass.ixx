module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
#include <glm/glm.hpp>
#include "entt/entt.hpp"
export module Demo.Grass;
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

import ParticleSystem.ParticleSystem;
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)

namespace SIByL::Demo
{
	export class GrassSystem :public ParticleSystem::ParticleSystem
	{
	public:
		GrassSystem() = default;
		GrassSystem(RHI::IResourceFactory* factory, Timer* timer);

		GFX::RDG::NodeHandle hiz;

		// Resource Nodes Handles --------------------------
		GFX::RDG::NodeHandle particlePosBuffer;
		GFX::RDG::NodeHandle particleColorBuffer;
		GFX::RDG::NodeHandle particleDirectionBuffer;
		GFX::RDG::NodeHandle particleVelocityBuffer;
		GFX::RDG::NodeHandle doubleBufferedIndicesHandle;

		GFX::RDG::NodeHandle spriteHandle;
		GFX::RDG::NodeHandle windNoiseHandle;
		GFX::RDG::NodeHandle cullingInfo;
		GFX::RDG::NodeHandle forwardPerViewUniformBufferFlight;

		ECS::Entity entity = {};
		RHI::IResourceFactory* factory;
		GFX::RDG::RenderGraph grass_rdg;
		Timer* timer;

		uint32_t particleMaxCount = 65536;

		GFX::RDG::ComputeDispatch* cullInfoDispatch = nullptr;

		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		auto registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;
		virtual auto registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		auto freshRenderPipeline(GFX::RDG::RenderGraph* rendergraph) noexcept -> bool;

		MemScope<RHI::ITexture> sceneDepthTexture;
		MemScope<RHI::ITextureView> sceneDepthTextureView;
		MemScope<RHI::ITexture> grassAlbedoTexture;
		MemScope<RHI::ITextureView> grassAlbedoTextureView;
		MemScope<RHI::ITexture> grassShadowTexture;
		MemScope<RHI::ITextureView> grassShadowTextureView;
		MemScope<RHI::ITexture> windNoiseTexture;
		MemScope<RHI::ITextureView> windNoiseTextureView;
	};

	GrassSystem::GrassSystem(RHI::IResourceFactory* factory, Timer* timer)
		:factory(factory), timer(timer)
	{
		Image image("./grass/scene_depth.png");
		sceneDepthTexture = factory->createTexture(&image);
		sceneDepthTextureView = factory->createTextureView(sceneDepthTexture.get());

		Image image_grass("./grass/Grass01.png");
		grassAlbedoTexture = factory->createTexture(&image_grass);
		grassAlbedoTextureView = factory->createTextureView(grassAlbedoTexture.get());

		Image image_shadow("./grass/shadowmap.png");
		grassShadowTexture = factory->createTexture(&image_shadow);
		grassShadowTextureView = factory->createTextureView(grassShadowTexture.get());

		Image image_windnoise("./grass/Noise.tga");
		windNoiseTexture = factory->createTexture(&image_windnoise);
		windNoiseTextureView = factory->createTextureView(windNoiseTexture.get());
	}
	
	struct MortonConstant
	{
		glm::vec4 min;
		glm::vec4 max;
	};

	auto GrassSystem::registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RenderGraphWorkshop local_workshop(grass_rdg);
		local_workshop.addInternalSampler();

		// Build Up Local Resources
		auto particle_buffer_pos_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * particleMaxCount, "Particle Buffer Pos-Lifetick (Grass)");
		auto particle_buffer_color_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * particleMaxCount, "Particle Buffer Color (Grass)");
		auto particle_buffer_direction_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * particleMaxCount, "Particle Buffer Direction (Grass)");
		auto particle_buffer_velocity_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * particleMaxCount, "Particle Buffer Velocity (Grass)");
		
		auto baked_scene_depth_handle = local_workshop.addColorBufferExt(sceneDepthTexture.get(), sceneDepthTextureView.get(), "Baked Scene Depth");
		auto baked_shadowmap_handle = local_workshop.addColorBufferExt(grassShadowTexture.get(), grassShadowTextureView.get(), "Baked Shadowmap");
		auto internal_sampler_handle = local_workshop.getInternalSampler("Default Sampler");

		auto morton_codes_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * particleMaxCount, "Morton Codes Buffer");
		auto double_buffered_indices_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * particleMaxCount * 2, "Double-Buffered Indices Buffer");
		auto intermediate_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Block-Wise Histogram");
		auto intermediate_histogram_lookback_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256 * GRIDSIZE(particleMaxCount, (8 * 256)), "Block-Wise Sum Look Back");
		auto global_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Global Histogram");
		auto onesweep_lookback_aggregate = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(particleMaxCount, (256 * 8)), "Offset (Aggregate)");
		auto onesweep_lookback_prefix = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(particleMaxCount, (256 * 8)), "Offset (Prefix)");
		auto culling_info_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * 2 * GRIDSIZE(particleMaxCount, (16)), "Culling Info");

		// Build Up Local Init Pass
		local_workshop.addComputePassScope("Local Init");
		GFX::RDG::ComputePipelineScope* particle_init_pipeline = local_workshop.addComputePipelineScope("Local Init", "Particle Init");
		{
			particle_init_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_init_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Particle Init", "Common");
				particle_init_mat_scope->resources = { particle_buffer_pos_handle, particle_buffer_color_handle, particle_buffer_direction_handle, particle_buffer_velocity_handle, internal_sampler_handle, internal_sampler_handle };
				particle_init_mat_scope->sampled_textures = { baked_scene_depth_handle, baked_shadowmap_handle };
				auto particle_init_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Particle Init", "Common", "Only");
				particle_init_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 65536 / 512;
					y = 1;
					z = 1;
				};
			}
		}
		// Build Up Local Cluster Pass
		GFX::RDG::ComputePipelineScope* particle_morton_pipeline = local_workshop.addComputePipelineScope("Local Init", "Morton");
		{
			particle_morton_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_morton.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_morton_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Morton", "Common");
				particle_morton_mat_scope->resources = { morton_codes_handle, double_buffered_indices_handle, particle_buffer_pos_handle };
				auto particle_morton_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Morton", "Common", "Only");
				particle_morton_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 65536 / 512;
					y = 1;
					z = 1;
				};
				particle_morton_dispatch_scope->pushConstant = [](Buffer& buffer) {
					MortonConstant mortonConstant{ glm::vec4(-160,0,-160,0), glm::vec4(160,40,160,0) };
					buffer = std::move(Buffer(sizeof(mortonConstant), 1));
					memcpy(buffer.getData(), &mortonConstant, sizeof(mortonConstant));
				};
			}
		}
		// Create Histogram Pipeline
		GFX::RDG::ComputePipelineScope* particle_histogram_pergroup_pipeline = local_workshop.addComputePipelineScope("Local Init", "Histogram_PerGroup-Grass");
		{
			particle_histogram_pergroup_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_histogram_subgroup_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_histogram_01_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Histogram_PerGroup-Grass", "Common");
				particle_histogram_01_mat_scope->resources = { morton_codes_handle, double_buffered_indices_handle, intermediate_histogram_handle, intermediate_histogram_lookback_handle };
				auto particle_histogram_01_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Histogram_PerGroup-Grass", "Common", "Only");
				particle_histogram_01_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(65536, (8 * 256));
					y = 1;
					z = 1;
				};
			}
		}
		GFX::RDG::ComputePipelineScope* portal_histogram_integrate_pipeline = local_workshop.addComputePipelineScope("Local Init", "Histogram_Integrate-Grass");
		{
			portal_histogram_integrate_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_histogram_integrate_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_histogram_02_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Histogram_Integrate-Grass", "Common");
				particle_histogram_02_mat_scope->resources = { intermediate_histogram_handle, global_histogram_handle };
				auto particle_histogram_02_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Histogram_Integrate-Grass", "Common", "Only");
				particle_histogram_02_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 1;
					y = 1;
					z = 1;
				};
			}
		}
		// Create OneSweep Pass Pipeline
		GFX::RDG::ComputePipelineScope* grass_onesweep_pipeline = local_workshop.addComputePipelineScope("Local Init", "Histogram_Onesweep-Grass");
		{
			grass_onesweep_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_onesweep_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Onesweep-i"
			for (int i = 0; i < 4; i++)
			{
				{
					auto particle_onesweep_00_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Histogram_Onesweep-Grass", "Onesweep-" + std::to_string(i));
					particle_onesweep_00_mat_scope->resources = { morton_codes_handle, double_buffered_indices_handle, onesweep_lookback_aggregate, onesweep_lookback_prefix, global_histogram_handle };
					auto particle_onesweep_00_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Histogram_Onesweep-Grass", "Onesweep-" + std::to_string(i), "Only");
					particle_onesweep_00_dispatch_scope->pushConstant = [i = i](Buffer& buffer) {
						uint32_t pass = i;
						buffer = std::move(Buffer(sizeof(pass), 1));
						memcpy(buffer.getData(), &pass, sizeof(pass));
					};
					particle_onesweep_00_dispatch_scope->customSize = [particleMaxCount = particleMaxCount](uint32_t& x, uint32_t& y, uint32_t& z) {
						x = GRIDSIZE(particleMaxCount, (256 * 8));
						y = 1;
						z = 1;
					};
				}
			}
		}
		// Build Up Local Render Graph
		local_workshop.build(factory, 0, 0);
		// Execute buffer init pass
		MemScope<RHI::ICommandBuffer> transientCommandbuffer = factory->createTransientCommandBuffer();
		transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		grass_rdg.recordCommandsNEW(transientCommandbuffer.get(), 0);
		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();

		factory->deviceIdle();
		// Build Up Main Resources - Buffers
		particlePosBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_pos_handle)->getStorageBuffer(), "Particle Pos (Grass) Buffer");
		particleColorBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_color_handle)->getStorageBuffer(), "Particle Color (Grass) Buffer");
		particleDirectionBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_direction_handle)->getStorageBuffer(), "Particle Direction (Grass) Buffer");
		particleVelocityBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_velocity_handle)->getStorageBuffer(), "Particle Velocity (Grass) Buffer");
		doubleBufferedIndicesHandle = workshop->addIndirectDrawBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(double_buffered_indices_handle)->getStorageBuffer(), "Double-Buffered Indices (Grass) Buffer");
		cullingInfo = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(culling_info_handle)->getStorageBuffer(), "Culling Info Ref");

		spriteHandle = workshop->addColorBufferExt(grassAlbedoTexture.get(), grassAlbedoTextureView.get(), "Grass Albedo");
		windNoiseHandle = workshop->addColorBufferExt(windNoiseTexture.get(), windNoiseTextureView.get(), "Grass Wind Noise");
	}

	auto GrassSystem::registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		// Create Update Pass
		GFX::RDG::ComputePipelineScope* grass_update_pipeline = workshop->addComputePipelineScope("Particle System", "Update-Grass");
		{
			grass_update_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_update.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_update_mat_scope = workshop->addComputeMaterialScope("Particle System", "Update-Grass", "Common");
				particle_update_mat_scope->resources = { particlePosBuffer, particleDirectionBuffer, particleVelocityBuffer, workshop->getInternalSampler("Repeat Sampler") };
				particle_update_mat_scope->sampled_textures = { windNoiseHandle };
				auto particle_update_dispatch_scope = workshop->addComputeDispatch("Particle System", "Update-Grass", "Common", "Only");
				particle_update_dispatch_scope->pushConstant = [timer = timer](Buffer& buffer) {
					float time = timer->getTotalTimeSeconds();
					buffer = std::move(Buffer(sizeof(time), 1));
					memcpy(buffer.getData(), &time, sizeof(time));
				};
				particle_update_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 65536 / 512;
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto GrassSystem::registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		auto forward_pass = workshop->renderGraph.getRasterPassScope("Forward Pass");
		forwardPerViewUniformBufferFlight = forward_pass->getPerViewUniformBufferFlightHandle();

		// Create CullingInfo Pass Pipeline
		GFX::RDG::ComputePipelineScope* grass_culling_info_pipeline = workshop->addComputePipelineScope("Particle Bounding Boxes", "CullingInfo-Grass");
		{
			grass_culling_info_pipeline->shaderComp = factory->createShaderFromBinaryFile("grass/grass_culling_info_generation.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "CullingInfo"
			{
				auto particle_culling_info_mat_scope = workshop->addComputeMaterialScope("Particle Bounding Boxes", "CullingInfo-Grass", "Common");
				particle_culling_info_mat_scope->resources = { forwardPerViewUniformBufferFlight, particlePosBuffer, particleDirectionBuffer, doubleBufferedIndicesHandle, cullingInfo };
				auto particle_culling_info_dispatch_scope = workshop->addComputeDispatch("Particle Bounding Boxes", "CullingInfo-Grass", "Common", "Only");
				cullInfoDispatch = particle_culling_info_dispatch_scope;
				particle_culling_info_dispatch_scope->customSize = [particleMaxCount = particleMaxCount](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(particleMaxCount, (512));
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto GrassSystem::registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RasterPipelineScope* trancparency_grass_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Grass");
		{
			trancparency_grass_pipeline->shaderVert = factory->createShaderFromBinaryFile("grass/grass_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
			trancparency_grass_pipeline->shaderFrag = factory->createShaderFromBinaryFile("grass/grass_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			trancparency_grass_pipeline->cullMode = RHI::CullMode::NONE;
			trancparency_grass_pipeline->vertexBufferLayout =
			{
				{RHI::DataType::Float3, "Position"},
				{RHI::DataType::Float3, "Color"},
				{RHI::DataType::Float2, "UV"},
			};
			trancparency_grass_pipeline->colorBlendingDesc = RHI::NoBlending;
			trancparency_grass_pipeline->depthStencilDesc = RHI::TestLessAndWrite;

			// Add Materials
			auto grass_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Grass", "Grass");
			grass_mat_scope->resources = { particlePosBuffer, particleColorBuffer, particleDirectionBuffer, workshop->getInternalSampler("Default Sampler") };
			grass_mat_scope->sampled_textures = { spriteHandle };
		}
		trancparency_grass_pipeline->isActive = false;
	
		GFX::RDG::RasterPipelineScope* grass_mesh_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Grass Mesh");
		{
			grass_mesh_pipeline->shaderMesh = factory->createShaderFromBinaryFile("grass/grass_mesh.spv", { RHI::ShaderStage::MESH,"main" });
			grass_mesh_pipeline->shaderFrag = factory->createShaderFromBinaryFile("grass/grass_mesh_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			grass_mesh_pipeline->cullMode = RHI::CullMode::NONE;

			grass_mesh_pipeline->colorBlendingDesc = RHI::NoBlending;
			grass_mesh_pipeline->depthStencilDesc = RHI::TestLessAndWrite;

			// Add Materials
			auto grass_mesh_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Grass Mesh", "Grass");
			grass_mesh_mat_scope->resources = { workshop->getInternalSampler("Default Sampler"), particlePosBuffer, particleColorBuffer, particleDirectionBuffer };
			grass_mesh_mat_scope->sampled_textures = { spriteHandle };
		}
		grass_mesh_pipeline->isActive = true;

		GFX::RDG::RasterPipelineScope* grass_mesh_culling_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Grass Mesh Culling");
		{
			grass_mesh_culling_pipeline->shaderTask = factory->createShaderFromBinaryFile("grass/grass_mesh_culling_frustrum_task.spv", { RHI::ShaderStage::TASK,"main" });
			grass_mesh_culling_pipeline->shaderMesh = factory->createShaderFromBinaryFile("grass/grass_mesh_culling_frustrum_mesh.spv", { RHI::ShaderStage::MESH,"main" });
			grass_mesh_culling_pipeline->shaderFrag = factory->createShaderFromBinaryFile("grass/grass_mesh_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			grass_mesh_culling_pipeline->cullMode = RHI::CullMode::NONE;

			grass_mesh_pipeline->colorBlendingDesc = RHI::NoBlending;
			grass_mesh_pipeline->depthStencilDesc = RHI::TestLessAndWrite;

			// Add Materials
			auto grass_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Grass Mesh Culling", "Grass");
			grass_mat_scope->resources = { workshop->getInternalSampler("Default Sampler"), particlePosBuffer, particleColorBuffer, particleDirectionBuffer, doubleBufferedIndicesHandle, cullingInfo, workshop->getInternalSampler("HiZ Sampler") };
			grass_mat_scope->sampled_textures = { spriteHandle, hiz };
		}
		grass_mesh_culling_pipeline->isActive = false;

		GFX::RDG::RasterPipelineScope* culling_aabb_vis_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Vis AABB Grass");
		{
			culling_aabb_vis_pipeline->shaderMesh = factory->createShaderFromBinaryFile("grass/grass_aabb_vis.spv", { RHI::ShaderStage::MESH,"main" });
			culling_aabb_vis_pipeline->shaderFrag = factory->createShaderFromBinaryFile("grass/grass_aabb_vis_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			culling_aabb_vis_pipeline->cullMode = RHI::CullMode::NONE;
			culling_aabb_vis_pipeline->colorBlendingDesc = RHI::AdditionBlending;
			culling_aabb_vis_pipeline->depthStencilDesc = RHI::NoTestAndNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Vis AABB Grass", "Grass");
			portal_mat_scope->resources = { cullingInfo, workshop->getInternalSampler("HiZ Sampler") };
			portal_mat_scope->sampled_textures = { hiz };
		}
		culling_aabb_vis_pipeline->isActive = true;

	}

	auto GrassSystem::freshRenderPipeline(GFX::RDG::RenderGraph* rendergraph) noexcept -> bool
	{
		GFX::ParticleSystem& ps = entity.getComponent<GFX::ParticleSystem>();
		if (ps.needRebuildPipeline)
		{
			rendergraph->getRasterPipelineScope("Forward Pass", "Vis AABB Grass")->isActive = ps.showCluster;
			ps.needRebuildPipeline = false;
			return true;
		}
		else
			return false;
	}

}