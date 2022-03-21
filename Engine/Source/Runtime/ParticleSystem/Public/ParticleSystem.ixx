module;
#include <cstdint>
export module ParticleSystem.ParticleSystem;
import RHI.IShader;
import RHI.IStorageBuffer;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;

namespace SIByL::ParticleSystem
{
	export class ParticleSystem
	{
	public:
		auto init(uint32_t particleDataSize, uint32_t maxParticleCount, 
			RHI::IShader* initShader,
			RHI::IShader* emitShader,
			RHI::IShader* updateShader
			) noexcept -> void;
		auto registerRenderGraph(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void;

		auto addEmitterSamples(RHI::IStorageBuffer* storageBuffer) noexcept -> void { emitterSamplesExt = storageBuffer; }

		uint32_t particleDataSize;
		uint32_t maxParticleCount;

		RHI::IShader* initShader;
		RHI::IShader* emitShader;
		RHI::IShader* updateShader;

		RHI::IStorageBuffer* emitterSamplesExt;

		GFX::RDG::NodeHandle particleBuffer;
		GFX::RDG::NodeHandle deadIndexBuffer;
		GFX::RDG::NodeHandle liveIndexBufferPrimary;
		GFX::RDG::NodeHandle liveIndexBufferSecondary;
		GFX::RDG::NodeHandle counterBuffer;
		GFX::RDG::NodeHandle indirectDrawBuffer;
		GFX::RDG::NodeHandle emitterSamplerBufferExt;

		GFX::RDG::NodeHandle initPass;
		GFX::RDG::NodeHandle emitPass;
		GFX::RDG::NodeHandle updatePass;

		GFX::RDG::NodeHandle sampler;
		GFX::RDG::NodeHandle dataBakedImage;
	};
}