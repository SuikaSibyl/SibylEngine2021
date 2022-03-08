module;

export module GFX.IRasterPass;
import RHI.IPipeline;
import RHI.IRenderPass;
import Core.MemoryManager;
import GFX.RDG.IPass;

namespace SIByL::GFX::RDG
{
	export struct IRasterPass :public PassNode
	{
	public:
		
	private:
		MemScope<RHI::IRenderPass> renderPass;
	};
}