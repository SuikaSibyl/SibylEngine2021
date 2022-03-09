module;
#include <cstdint>
export module ParticleSystem.ParticleSystem;
import RHI.IShader;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;

namespace SIByL::ParticleSystem
{
	export class ParticleSystem
	{
	public:
		auto init(uint32_t particleDataSize, uint32_t maxParticleCount, RHI::IShader* initShader) noexcept -> void;
		auto registerRenderGraph(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void;

	private:
		uint32_t particleDataSize;
		uint32_t maxParticleCount;

		RHI::IShader* initShader;

		GFX::RDG::NodeHandle particleBuffer;
		GFX::RDG::NodeHandle deadIndexBuffer;
		GFX::RDG::NodeHandle liveIndexBufferPrimary;
		GFX::RDG::NodeHandle liveIndexBufferSecondary;
		GFX::RDG::NodeHandle counterBuffer;

		GFX::RDG::NodeHandle emitPass;
	};
}