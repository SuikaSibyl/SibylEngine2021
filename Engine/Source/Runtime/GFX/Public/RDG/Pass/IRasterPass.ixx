module;

export module GFX.IRasterPass;
import RHI.IPipeline;
import RHI.IRenderPass;
import Core.MemoryManager;
import GFX.RDG.PassNode;

namespace SIByL::GFX::RDG
{
	export struct RasterPassNode :public PassNode
	{
	public:
		
	private:
		MemScope<RHI::IRenderPass> renderPass;
	};
}