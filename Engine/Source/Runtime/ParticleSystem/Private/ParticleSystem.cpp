module;
#include <cstdint>
#include <vector>
module ParticleSystem.ParticleSystem;
import RHI.IShader;
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

	auto ParticleSystem::registerRenderGraph(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		particleBuffer = builder->addStorageBuffer(particleDataSize * maxParticleCount);
		deadIndexBuffer = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount);
		liveIndexBufferPrimary = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount);
		liveIndexBufferSecondary = builder->addStorageBuffer(sizeof(unsigned int) * maxParticleCount);
		counterBuffer = builder->addStorageBuffer(sizeof(unsigned int) * 5);
		indirectDrawBuffer = builder->addIndirectDrawBuffer();
		emitterSamplerBufferExt = builder->addStorageBufferExt(emitterSamplesExt);

		initPass = builder->addComputePass(initShader, { particleBuffer, counterBuffer, liveIndexBufferPrimary, deadIndexBuffer, indirectDrawBuffer }, sizeof(unsigned int));
		emitPass = builder->addComputePass(emitShader, { particleBuffer, counterBuffer, liveIndexBufferPrimary, deadIndexBuffer, emitterSamplerBufferExt, sampler }, sizeof(unsigned int) * 4);
		GFX::RDG::ComputePassNode* emitPassNode = builder->attached.getComputePassNode(emitPass);
		emitPassNode->textures = { dataBakedImage };

		updatePass = builder->addComputePass(updateShader, { particleBuffer, counterBuffer, liveIndexBufferPrimary, deadIndexBuffer, indirectDrawBuffer });
	}
}