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

import ParticleSystem.ParticleSystem;
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)

namespace SIByL::Demo
{
	export class GrassSystem :public ParticleSystem::ParticleSystem
	{
	public:
		GrassSystem() = default;
		GrassSystem(RHI::IResourceFactory* factory, Timer* timer);

		// Resource Nodes Handles --------------------------
		GFX::RDG::NodeHandle particlePosBuffer;
		GFX::RDG::NodeHandle particleColorBuffer;
		GFX::RDG::NodeHandle particleDirectionBuffer;
		GFX::RDG::NodeHandle particleVelocityBuffer;

		GFX::RDG::NodeHandle spriteHandle;
		GFX::RDG::NodeHandle windNoiseHandle;


		ECS::Entity entity = {};
		RHI::IResourceFactory* factory;
		GFX::RDG::RenderGraph grass_rdg;
		Timer* timer;

		uint32_t particleMaxCount = 65536;

		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		auto registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;
		virtual auto registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;

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
				particle_update_mat_scope->resources = { particlePosBuffer, particleDirectionBuffer, particleVelocityBuffer, workshop->getInternalSampler("Default Sampler") };
				particle_update_mat_scope->sampled_textures = { windNoiseHandle };
				auto particle_update_dispatch_scope = workshop->addComputeDispatch("Particle System", "Update-Grass", "Common", "Only");
				particle_update_dispatch_scope->pushConstant = [timer = timer](Buffer& buffer) {
					float time = timer->getMsPF() * 0.001;
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
		trancparency_grass_pipeline->isActive = true;
	}

}