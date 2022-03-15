module;
#include <vector>
export module GFX.RDG.RasterPassNode;
import Core.BitFlag;
import RHI.GraphicContext;
import RHI.IPhysicalDevice;
import RHI.ILogicalDevice;
import RHI.IPipelineLayout;
import RHI.ISwapChain;
import RHI.ICompileSession;
import RHI.IEnum;
import RHI.IFactory;
import RHI.IShader;
import RHI.IFixedFunctions;
import RHI.IPipeline;
import RHI.IFramebuffer;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ISemaphore;
import RHI.IFence;
import RHI.IVertexBuffer;
import RHI.IBuffer;
import RHI.IDeviceGlobal;
import RHI.IIndexBuffer;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorPool;
import RHI.IDescriptorSet;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IRenderPass;
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.IBarrier;
import Core.MemoryManager;
import GFX.RDG.Common;
import GFX.RDG.Container;

namespace SIByL::GFX::RDG
{
	export struct RasterPassNode :public PassNode
	{
	public:
		RasterPassNode(
			void* graph,
			std::vector<NodeHandle> const& ins,
			uint32_t const& constant_size);

		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		NodeHandle framebufferFlights;
		NodeHandle framebuffer;
		bool useFlights = false;

		MemScope<RHI::IShader> shaderVert;
		MemScope<RHI::IShader> shaderFrag;
		std::vector<NodeHandle> ins;
		std::vector<NodeHandle> textures;

		MemScope<RHI::IVertexLayout> vertexLayout;
		MemScope<RHI::IInputAssembly> inputAssembly;
		MemScope<RHI::IViewportsScissors> viewportScissors;
		MemScope<RHI::IRasterizer> rasterizer;
		MemScope<RHI::IMultisampling> multisampling;
		MemScope<RHI::IDepthStencil> depthstencil;
		MemScope<RHI::IColorBlending> colorBlending;
		MemScope<RHI::IDynamicState> dynamicStates;
		MemScope<RHI::IPipelineLayout> pipelineLayout;

		MemScope<RHI::IDescriptorSetLayout> desciptorSetLayout;
		std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;
		MemScope<RHI::IPipeline> pipeline;
	};
}