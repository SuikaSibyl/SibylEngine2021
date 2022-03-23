module;
#include <vector>
#include <cstdint>
#include <functional>
module GFX.RDG.MultiDispatchScope;
import Core.BitFlag;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	MultiDispatchScope::MultiDispatchScope()
	{
		type = NodeDetailedType::MULTI_DISPATCH_SCOPE;
	}

	auto MultiDispatchScope::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;
		for (auto iter = rg->resources.begin(); iter != rg->resources.end(); iter++)
		{
			rg->getResourceNode((*iter))->consumeHistory.emplace_back(handle, ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN);
		}
	}
}