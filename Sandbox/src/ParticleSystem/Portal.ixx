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

import ParticleSystem.ParticleSystem;

namespace SIByL::Demo
{
	export class PortalSystem :public ParticleSystem::ParticleSystem
	{
	public:
		PortalSystem() = default;
		PortalSystem(RHI::IResourceFactory* factory, Timer* timer);

		virtual auto registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void override;
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
		Image image("./assets/Sparkle.tga");
		texture = factory->createTexture(&image);
		textureView = factory->createTextureView(texture.get());
		Image baked_image("./assets/portal_bake.tga");
		bakedCurves = factory->createTexture(&baked_image);
		bakedCurvesView = factory->createTextureView(bakedCurves.get());

		// init particle system
		shaderPortalInit = factory->createShaderFromBinaryFile("portal/portal_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderPortalEmit = factory->createShaderFromBinaryFile("portal/portal_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderPortalUpdate = factory->createShaderFromBinaryFile("portal/portal_update.spv", { RHI::ShaderStage::COMPUTE,"main" });
		init(sizeof(float) * 4 * 4, 100000, shaderPortalInit.get(), shaderPortalEmit.get(), shaderPortalUpdate.get());
	}

	auto PortalSystem::registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		// Resources
		samplerHandle = builder->addSamplerExt(sampler.get());
		spriteHandle = builder->addColorBufferExt(texture.get(), textureView.get(), "Sprite");
		bakedCurveHandle = builder->addColorBufferExt(bakedCurves.get(), bakedCurvesView.get(), "Baked Curves");
		emitterVolumeHandle = builder->addStorageBufferExt(torusBuffer.get(), "Emitter Volume Samples");

		// Buffers
		particleBuffer = builder->addStorageBuffer(particleDataSize * maxParticleCount, "Particle Buffer");
		deadIndexBuffer = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Dead Index Buffer");
		liveIndexBuffer = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Live Index Buffer");
		counterBuffer = builder->addStorageBuffer(sizeof(unsigned int) * 5, "Counter Buffer");
		indirectDrawBuffer = builder->addIndirectDrawBuffer("Indirect Draw Buffer");
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