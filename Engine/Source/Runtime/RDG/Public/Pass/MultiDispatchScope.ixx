module;
#include <vector>
#include <cstdint>
#include <functional>
export module GFX.RDG.MultiDispatchScope;
import GFX.RDG.Common;
import RHI.IFactory;

namespace SIByL::GFX::RDG
{
	export struct MultiDispatchScope :public PassScope
	{
		MultiDispatchScope();
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		std::function<uint32_t(void)> customDispatchCount;
	};
}