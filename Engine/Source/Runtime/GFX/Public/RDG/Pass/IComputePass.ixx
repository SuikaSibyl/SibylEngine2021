module;
#include <vector>
export module GFX.IComputePass;
import RHI.IPipeline;
import RHI.IDescriptorSet;
import RHI.IDescriptorSetLayout;
import Core.MemoryManager;
import GFX.RDG.IPass;

namespace SIByL::GFX::RDG
{
	export struct ComputePassInfo
	{
		RHI::DescriptorSetLayoutDesc dsLayoutDesc;

	};

	export struct IComputePass :public PassNode
	{
	public:


	private:
		MemScope<RHI::IPipeline> pipeline;
		std::vector<MemScope<RHI::IDescriptorSet>> compute_descriptorSets;
	};
}