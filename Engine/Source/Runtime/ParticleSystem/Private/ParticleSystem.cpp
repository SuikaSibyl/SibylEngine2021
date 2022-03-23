module;
#include <cstdint>
#include <vector>
#include <functional>
module ParticleSystem.ParticleSystem;
import Core.Time;
import RHI.IShader;
import RHI.ICommandBuffer;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;

namespace SIByL::ParticleSystem
{
	auto ParticleSystem::init(uint32_t particleDataSize, uint32_t maxParticleCount,
		RHI::IShader* initShader,
		RHI::IShader* emitShader,
		RHI::IShader* updateShader
	) noexcept -> void
	{
		this->particleDataSize = particleDataSize;
		this->maxParticleCount = maxParticleCount;
		this->initShader = initShader;
		this->emitShader = emitShader;
		this->updateShader = updateShader;
	}

	auto ParticleSystem::registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		particleBuffer = builder->addStorageBuffer(particleDataSize * maxParticleCount, "Particle Buffer");
		deadIndexBuffer = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Dead Index Buffer");
		liveIndexBuffer = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount, "Live Index Buffer");
		counterBuffer = builder->addStorageBuffer(sizeof(unsigned int) * 5, "Counter Buffer");
		indirectDrawBuffer = builder->addIndirectDrawBuffer("Indirect Draw Buffer");
		emitterSamplerBufferExt = builder->addStorageBufferExt(emitterSamplesExt, "Emitter Volume Samples");
	}

	struct EmitConstant
	{
		unsigned int emitCount;
		float time;
		float x;
		float y;
	};

	auto ParticleSystem::registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		initPass = builder->addComputePassOneTime(initShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer }, "Particles Init", sizeof(unsigned int));
		emitPass = builder->addComputePass(emitShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, emitterSamplerBufferExt, sampler }, "Particles Emit", sizeof(unsigned int) * 4);
		builder->attached.getComputePassNode(emitPass)->customDispatch = [&timer = timer](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			EmitConstant constant_1{ 400000u / 50, (float)timer->getTotalTime(), 0, 1.07 };
			compute_pass->executeWithConstant(commandbuffer, 200, 1, 1, flight_idx, constant_1);
		};

		GFX::RDG::ComputePassNode* emitPassNode = builder->attached.getComputePassNode(emitPass);
		emitPassNode->textures = { dataBakedImage };

		updatePass = builder->addComputePass(updateShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer }, "Particles Update");
		builder->attached.getComputePassNode(updatePass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			compute_pass->execute(commandbuffer, 200, 1, 1, flight_idx);
		};
	}
}