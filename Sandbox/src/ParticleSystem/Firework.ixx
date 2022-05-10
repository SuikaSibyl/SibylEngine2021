module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
#include <glm/glm.hpp>
#include "entt/entt.hpp"
export module Demo.Firework;
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
	export class FireworkSystem :public ParticleSystem::ParticleSystem
	{
	public:
		FireworkSystem() = default;
		FireworkSystem(RHI::IResourceFactory * factory, Timer * timer);

		ECS::Entity entity = {};
		Timer* timer;

		RHI::IResourceFactory* factory;
		GFX::RDG::RenderGraph firework_rdg;


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

		GFX::RDG::NodeHandle debugBuffer;


		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		auto registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;
		virtual auto registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;

		GFX::RDG::ComputeDispatch* emitDispatch = nullptr;
		GFX::RDG::ComputeDispatch* cullInfoDispatch = nullptr;

	};

	FireworkSystem::FireworkSystem(RHI::IResourceFactory* factory, Timer* timer)
		:factory(factory), timer(timer)
	{
		maxParticleCount = 32;
	}

	auto FireworkSystem::registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RenderGraphWorkshop local_workshop(firework_rdg);
		local_workshop.addInternalSampler();
		// Build Up Local Resources
		auto particle_buffer_pos_lifetick_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Pos-Lifetick (Firework)");
		auto particle_buffer_vel_mass_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Velocity_Mass (Firework)");
		auto particle_buffer_color_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * maxParticleCount, "Particle Buffer Color (Firework)");
		auto dead_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Dead Index Buffer (Firework)");
		auto live_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Live Index Buffer (Firework)");
		auto counter_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * 5, "Counter Buffer");
		auto indirect_draw_buffer_handle = local_workshop.addIndirectDrawBuffer("Indirect Draw Buffer");
		auto morton_codes_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Morton Codes Buffer");
		auto double_buffered_indices_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount * 2, "Double-Buffered Indices Buffer");

		auto intermediate_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Block-Wise Histogram");
		auto intermediate_histogram_lookback_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256 * GRIDSIZE(maxParticleCount, (8 * 256)), "Block-Wise Sum Look Back");
		auto global_histogram_handle = local_workshop.addStorageBuffer(sizeof(uint32_t) * 4 * 256, "Global Histogram");
		auto onesweep_lookback_aggregate = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(maxParticleCount, (256 * 8)), "Offset (Aggregate)");
		auto onesweep_lookback_prefix = local_workshop.addStorageBuffer(sizeof(uint32_t) * 256 * GRIDSIZE(maxParticleCount, (256 * 8)), "Offset (Prefix)");
		auto culling_info_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * 2 * GRIDSIZE(maxParticleCount, (16)), "Culling Info");

		auto debug_handle = local_workshop.addStorageBuffer(sizeof(float) * 4 * 32, "Debug Info");

		// Build Up Local Init Pass
		local_workshop.addComputePassScope("Local Init");
		GFX::RDG::ComputePipelineScope* particle_init_pipeline = local_workshop.addComputePipelineScope("Local Init", "Particle Init");
		{
			particle_init_pipeline->shaderComp = factory->createShaderFromBinaryFile("firework/firework_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_init_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Particle Init", "Common");
				particle_init_mat_scope->resources = { counter_buffer_handle, dead_index_buffer_handle, indirect_draw_buffer_handle };
				auto particle_init_dispatch_scope = local_workshop.addComputeDispatch("Local Init", "Particle Init", "Common", "Only");
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
		firework_rdg.recordCommandsNEW(transientCommandbuffer.get(), 0);
		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		factory->deviceIdle();

		// Build Up Main Resources - Buffers
		particlePosLifetickBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_pos_lifetick_handle)->getStorageBuffer(), "Particle Pos Lifetick Buffer");
		particleVelocityMassBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_vel_mass_handle)->getStorageBuffer(), "Particle Velocity Mass Buffer");
		particleColorBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_color_handle)->getStorageBuffer(), "Particle Color Buffer");

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

		debugBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(debug_handle)->getStorageBuffer(), "Debug Info Ref");

	}
	
	struct EmitConstant
	{
		glm::mat4 matrix;
		unsigned int emitCount;
		float time;
	};

	auto FireworkSystem::registerUpdatePasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		// Create Emit Pass
		GFX::RDG::ComputePipelineScope* firework_emit_pipeline = workshop->addComputePipelineScope("Particle System", "Emit-Firework");
		{
			firework_emit_pipeline->shaderComp = factory->createShaderFromBinaryFile("firework/firework_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_emit_mat_scope = workshop->addComputeMaterialScope("Particle System", "Emit-Firework", "Common");
				particle_emit_mat_scope->resources = { particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer };
				auto particle_emit_dispatch_scope = workshop->addComputeDispatch("Particle System", "Emit-Firework", "Common", "Only");
				emitDispatch = particle_emit_dispatch_scope;
				particle_emit_dispatch_scope->pushConstant = [&timer = timer](Buffer& buffer) {
					unsigned int num = 32u;
					buffer = std::move(Buffer(sizeof(num), 1));
					memcpy(buffer.getData(), &num, sizeof(num));
				};
				particle_emit_dispatch_scope->customSize = [](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = 1;
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto FireworkSystem::registerBoundingBoxesPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		auto forward_pass = workshop->renderGraph.getRasterPassScope("Forward Pass");
		forwardPerViewUniformBufferFlight = forward_pass->getPerViewUniformBufferFlightHandle();

		// Create CullingInfo Pass Pipeline
		GFX::RDG::ComputePipelineScope* portal_culling_info_pipeline = workshop->addComputePipelineScope("Particle Bounding Boxes", "CullingInfo-Firework");
		{
			portal_culling_info_pipeline->shaderComp = factory->createShaderFromBinaryFile("firework/firework_culling_info_generation.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "CullingInfo"
			{
				auto particle_culling_info_mat_scope = workshop->addComputeMaterialScope("Particle Bounding Boxes", "CullingInfo-Firework", "Common");
				particle_culling_info_mat_scope->resources = { forwardPerViewUniformBufferFlight, particlePosLifetickBuffer, particleVelocityMassBuffer, liveIndexBuffer, cullingInfo, counterBuffer };
				auto particle_culling_info_dispatch_scope = workshop->addComputeDispatch("Particle Bounding Boxes", "CullingInfo-Firework", "Common", "Only");
				cullInfoDispatch = particle_culling_info_dispatch_scope;
				particle_culling_info_dispatch_scope->customSize = [maxParticleCount = maxParticleCount](uint32_t& x, uint32_t& y, uint32_t& z) {
					x = GRIDSIZE(maxParticleCount, (32));
					y = 1;
					z = 1;
				};
			}
		}
	}

	auto FireworkSystem::registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		GFX::RDG::RasterPipelineScope* trancparency_portal_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Particle Firework");
		{
			trancparency_portal_pipeline->shaderVert = factory->createShaderFromBinaryFile("firework/firework_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
			trancparency_portal_pipeline->shaderFrag = factory->createShaderFromBinaryFile("firework/firework_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			trancparency_portal_pipeline->cullMode = RHI::CullMode::BACK;
			trancparency_portal_pipeline->vertexBufferLayout =
			{
				{RHI::DataType::Float3, "Position"},
				{RHI::DataType::Float3, "Color"},
				{RHI::DataType::Float2, "UV"},
			};
			trancparency_portal_pipeline->colorBlendingDesc = RHI::NoBlending;
			trancparency_portal_pipeline->depthStencilDesc = RHI::NoTestAndNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Particle Firework", "Firework");
			portal_mat_scope->resources = { particlePosLifetickBuffer, particleVelocityMassBuffer, particleColorBuffer, liveIndexBuffer };
		}
		trancparency_portal_pipeline->isActive = true;

		GFX::RDG::RasterPipelineScope* culling_aabb_vis_pipeline = workshop->addRasterPipelineScope("Forward Pass", "Vis AABB Firework");
		{
			culling_aabb_vis_pipeline->shaderMesh = factory->createShaderFromBinaryFile("firework/firework_aabb_vis.spv", { RHI::ShaderStage::MESH,"main" });
			culling_aabb_vis_pipeline->shaderFrag = factory->createShaderFromBinaryFile("portal/portal_aabb_vis_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			culling_aabb_vis_pipeline->cullMode = RHI::CullMode::NONE;
			culling_aabb_vis_pipeline->colorBlendingDesc = RHI::NoBlending;
			culling_aabb_vis_pipeline->depthStencilDesc = RHI::NoTestAndNoWrite;

			// Add Materials
			auto portal_mat_scope = workshop->addRasterMaterialScope("Forward Pass", "Vis AABB Firework", "Firework");
			portal_mat_scope->resources = { cullingInfo, indirectDrawBuffer, debugBuffer };
		}
		culling_aabb_vis_pipeline->isActive = true;

	}

}
