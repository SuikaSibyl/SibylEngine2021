module;
#include <vector>
#include <string>
#include <unordered_map>
export module GFX.RDG.RasterNodes;
import Core.MemoryManager;
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
import RHI.ICommandBuffer;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct RasterDrawCall :public PassNode
	{
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;

		std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;
	};

	export struct RasterMaterialScope :public PassNode
	{
		std::vector<NodeHandle> drawCalls;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
	};

	export struct RasterPipelineScope :public PassNode
	{
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		RHI::BufferLayout vertexBufferLayout = {};
		RHI::TopologyKind topologyKind = RHI::TopologyKind::TriangleList;
		RHI::Extend viewportExtend;
		RHI::PolygonMode polygonMode = RHI::PolygonMode::FILL;
		RHI::CullMode cullMode = RHI::CullMode::NONE;
		float lineWidth = 0.0f;

		MemScope<RHI::IPipeline> pipeline;

		std::vector<NodeHandle> materialScopes;
		std::unordered_map<std::string, NodeHandle> materialScopesRegister;
	
	private:
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
	};

	export struct RasterPassScope :public PassNode
	{
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;

		NodeHandle framebuffer;
		std::vector<NodeHandle> pipelineScopes;
		std::unordered_map<std::string, NodeHandle> pipelineScopesRegister;
	};
}