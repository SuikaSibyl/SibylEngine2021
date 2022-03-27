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
}