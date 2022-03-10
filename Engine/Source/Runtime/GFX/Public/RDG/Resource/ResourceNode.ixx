module;
#include <cstdint>
export module GFX.RDG.ResourceNode;
import RHI.IEnum;
import RHI.IFactory;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct ResourceNode :public Node
	{
	public:
		RHI::DescriptorType resourceType;
		RHI::ShaderStageFlags shaderStages;
	};
}