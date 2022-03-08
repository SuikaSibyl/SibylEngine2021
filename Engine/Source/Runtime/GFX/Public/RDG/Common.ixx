module;
#include <cstdint>
#include <vector>
#include <string>
export module GFX.RDG.Common;
import RHI.IEnum;

namespace SIByL::GFX::RDG
{
	export using NodeHandle = uint64_t;

	export struct Node
	{
		std::string name;
		std::vector<NodeHandle> inputs;
		std::vector<NodeHandle> outputs;
	};

	export enum struct ResourceAccess
	{
		READ_ONLY,
		READ_WRITE,
	};

	export struct IOSocket
	{
		RHI::DescriptorType type;
		ResourceAccess access;
	};
}