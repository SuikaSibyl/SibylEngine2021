module;
#include <cstdint>
#include <vector>
module ParticleSystem.ParticleSystem;
import RHI.IShader;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;

namespace SIByL::ParticleSystem
{
	auto ParticleSystem::init(uint32_t particleDataSize, uint32_t maxParticleCount, RHI::IShader* initShader) noexcept -> void
	{
		this->particleDataSize = particleDataSize;
		this->maxParticleCount = maxParticleCount;
		this->initShader = initShader;
	}

	auto ParticleSystem::registerRenderGraph(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		particleBuffer = builder->addStorageBuffer(particleDataSize * maxParticleCount);
		deadIndexBuffer = builder->addStorageBuffer(sizeof(uint32_t) * maxParticleCount);
		liveIndexBufferPrimary = builder->addStorageBuffer(sizeof(uint32_t) * maxParticleCount);
		liveIndexBufferSecondary = builder->addStorageBuffer(sizeof(uint32_t) * maxParticleCount);
		counterBuffer = builder->addStorageBuffer(sizeof(uint32_t) * 4);

		emitPass = builder->addComputePass(initShader, { particleBuffer, counterBuffer, liveIndexBufferPrimary, deadIndexBuffer }, sizeof(float));
	}
}