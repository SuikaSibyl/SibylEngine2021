module;
#include <cstdint>
#include <vector>
#include <string>
export module GFX.RDG.Common;
import RHI.IEnum;
import RHI.IFactory;

namespace SIByL::GFX::RDG
{
	export using NodeHandle = uint64_t;

	export struct Node
	{
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void = 0;

		std::string name;
		std::vector<NodeHandle> inputs;
		std::vector<NodeHandle> outputs;
	};

	export enum struct ResourceUsage
	{
		READ_ONLY,
		READ_WRITE,
	};

	export struct IOSocket
	{
		RHI::DescriptorType type;
		ResourceUsage access;
	};
}