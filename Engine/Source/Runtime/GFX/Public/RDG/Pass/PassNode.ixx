module;
#include <vector>
export module GFX.RDG.PassNode;
import RHI.ICommandBuffer;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct PassNode :public Node
	{
	public:
		virtual auto execute(RHI::ICommandBuffer* buffer, unsigned int x, unsigned int y, unsigned int z, unsigned int frame) noexcept -> void = 0;
	};
}