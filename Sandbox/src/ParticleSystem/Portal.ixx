module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
export module Demo.Portal;
import Core.Buffer;
import Core.Cache;
import Core.Image;
import Core.File;
import Core.Time;
import Core.MemoryManager;
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

import ParticleSystem.ParticleSystem;

namespace SIByL::Demo
{
	export class PortalSystem :public ParticleSystem::ParticleSystem
	{
	public:
		PortalSystem() = default;
		PortalSystem(RHI::IResourceFactory* factory, Timer* timer);

		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void override;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void override;

		Timer* timer;

		// Resource Nodes Handles --------------------------
		GFX::RDG::NodeHandle spriteHandle;
		GFX::RDG::NodeHandle samplerHandle;
		GFX::RDG::NodeHandle bakedCurveHandle;
		GFX::RDG::NodeHandle emitterVolumeHandle;

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
		auto particle_buffer_handle = local_workshop.addStorageBuffer(particleDataSize * maxParticleCount, "Particle Buffer");
		auto dead_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Dead Index Buffer");
		auto live_index_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Live Index Buffer");
		auto counter_buffer_handle = local_workshop.addStorageBuffer(sizeof(unsigned int) * 5, "Counter Buffer");
		auto indirect_draw_buffer_handle = local_workshop.addIndirectDrawBuffer("Indirect Draw Buffer");
		// Build Up Local Init Pass
		local_workshop.addComputePassScope("Local Init");
		GFX::RDG::ComputePipelineScope* particle_init_pipeline = local_workshop.addComputePipelineScope("Local Init", "Particle Init");
		{
			particle_init_pipeline->shaderComp = factory->createShaderFromBinaryFile("portal/portal_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto particle_init_mat_scope = local_workshop.addComputeMaterialScope("Local Init", "Particle Init", "Common");
				particle_init_mat_scope->resources = { particle_buffer_handle, counter_buffer_handle, live_index_buffer_handle, dead_index_buffer_handle, indirect_draw_buffer_handle };
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
		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		factory->deviceIdle();

		// Build Up Main Resources - Buffers
		particleBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(particle_buffer_handle)->getStorageBuffer(), "Particle Buffer");
		deadIndexBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(dead_index_buffer_handle)->getStorageBuffer(), "Dead Index Buffer");
		liveIndexBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(live_index_buffer_handle)->getStorageBuffer(), "Live Index Buffer");
		counterBuffer = workshop->addStorageBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(counter_buffer_handle)->getStorageBuffer(), "Counter Buffer");
		indirectDrawBuffer = workshop->addIndirectDrawBufferExt(local_workshop.getNode<GFX::RDG::StorageBufferNode>(indirect_draw_buffer_handle)->getStorageBuffer(), "Indirect Draw Buffer");

		// Build Up Main Resources - Misc
		samplerHandle = workshop->getInternalSampler("Default Sampler");
		spriteHandle = workshop->addColorBufferExt(texture.get(), textureView.get(), "Sprite");
		bakedCurveHandle = workshop->addColorBufferExt(bakedCurves.get(), bakedCurvesView.get(), "Baked Curves");
		emitterVolumeHandle = workshop->addStorageBufferExt(torusBuffer.get(), "Emitter Volume Samples");
	}

	struct EmitConstant
	{
		unsigned int emitCount;
		float time;
		float x;
		float y;
	};

	auto PortalSystem::registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		// Create Init Pass
		initPass = builder->addComputePassBackPool(initShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer }, "Particles Init", sizeof(unsigned int));

		// Create Emit Pass
		emitPass = builder->addComputePass(emitShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, emitterVolumeHandle, samplerHandle }, "Particles Emit", sizeof(unsigned int) * 4);
		builder->attached.getComputePassNode(emitPass)->customDispatch = [&timer = timer](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			EmitConstant constant_1{ 400000u / 50, (float)timer->getTotalTime(), 0, 1.07 };
			compute_pass->executeWithConstant(commandbuffer, 200, 1, 1, flight_idx, constant_1);
		};
		GFX::RDG::ComputePassNode* emitPassNode = builder->attached.getComputePassNode(emitPass);
		emitPassNode->textures = { bakedCurveHandle };

		// Create Update Pass
		updatePass = builder->addComputePass(updateShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer }, "Particles Update");
		builder->attached.getComputePassNode(updatePass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			compute_pass->execute(commandbuffer, 200, 1, 1, flight_idx);
		};
	}
}