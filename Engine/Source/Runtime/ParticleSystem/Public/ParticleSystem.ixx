module;
#include <cstdint>
export module ParticleSystem.ParticleSystem;
import Core.Time;
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

		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void = 0;
		virtual auto registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void = 0;

		uint32_t particleDataSize;
		uint32_t maxParticleCount;

		RHI::IShader* initShader;
		RHI::IShader* emitShader;
		RHI::IShader* updateShader;

		GFX::RDG::NodeHandle particleBuffer;
		GFX::RDG::NodeHandle deadIndexBuffer;
		GFX::RDG::NodeHandle liveIndexBuffer;
		GFX::RDG::NodeHandle counterBuffer;
		GFX::RDG::NodeHandle indirectDrawBuffer;

		GFX::RDG::NodeHandle initPass;
		GFX::RDG::NodeHandle emitPass;
		GFX::RDG::NodeHandle updatePass;
	};
}