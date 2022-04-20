module;
#include <vector>
#include <unordered_map>
export module GFX.RDG.ExternalAccess;
import GFX.RDG.Common;
import RHI.IEnum;
import RHI.IMemoryBarrier;

namespace SIByL::GFX::RDG
{
	export struct ExternalAccessItem
	{
		NodeHandle resourceHandle;
		ConsumeKind consumeKind;
	};

	export struct ExternalAccessPass :public PassNode
	{
	public:
		auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void;
		auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void;

		auto insertExternalAccessItem(ExternalAccessItem const& item) noexcept -> bool;

		std::unordered_map<NodeHandle, ExternalAccessItem> externalAccessMap;

		void* renderGraph;
	};
}