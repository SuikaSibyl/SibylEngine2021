export module GFX.RDG.ProxyUnity;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	export struct ProxyUnit
	{
		virtual auto registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void {}
		virtual auto registerResources(GFX::RDG::RenderGraphWorkshop& workshop) noexcept -> void {}
		virtual auto registerRasterPasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void {}
		virtual auto registerComputePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void {}
		virtual auto registerComputePasses(GFX::RDG::RenderGraphWorkshop& workshop) noexcept -> void {}
		virtual auto postBuild() noexcept -> void {}
	};
}