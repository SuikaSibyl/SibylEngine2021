export module GFX.PostProcessing.AcesBloom;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.ProxyUnity;

namespace SIByL::GFX::PostProcessing
{
	export struct AcesBloomProxyUnit :public RDG::ProxyUnit
	{
		virtual auto registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void override;
		virtual auto registerComputePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void override;

	};
}