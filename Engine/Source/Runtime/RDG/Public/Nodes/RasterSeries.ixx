module;
#include <vector>
#include <string>
#include <glm/glm.hpp>
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
	export struct PerObjectUniformBuffer {
		glm::mat4 model;
	};

	export struct PerViewUniformBuffer{
	   glm::mat4 view;
	   glm::mat4 proj;
	   glm::vec4 cameraPos;
	};

	struct RasterPassScope;
	struct RasterPipelineScope;
	struct RasterMaterialScope;

	export struct RasterDrawCall :public PassNode
	{
		RasterDrawCall(RHI::IPipelineLayout** pipeline_layout);
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		auto clearDrawCallInfo() noexcept -> void;

		RHI::IVertexBuffer* vertexBuffer;
		RHI::IIndexBuffer* indexBuffer;

		PerObjectUniformBuffer uniform;

	private:
		friend struct RasterMaterialScope;
		RHI::IPipelineLayout** pipelineLayout;
	};

	export struct RasterMaterialScope :public PassNode
	{
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		virtual auto onFrameStart(void* graph) noexcept -> void override;

		auto addRasterDrawCall(std::string const& tag, void* graph) noexcept -> NodeHandle;

		std::vector<NodeHandle> resources;
		std::vector<NodeHandle> sampled_textures;

		unsigned int validDrawcallCount = 0;
		std::vector<NodeHandle> drawCalls;

	private:
		friend struct RasterPipelineScope;
		// member
		std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;
		// Desc :: desciptor set layout
		RHI::IDescriptorSetLayout* desciptorSetLayout;
		RHI::IPipelineLayout* pipelineLayout;
		int hasPerFrameUniformBuffer = -1;
		int hasPerViewUniformBuffer = -1;
		unsigned int totalResourceNum = 0;
		NodeHandle perFrameUniformBufferFlight;
		NodeHandle perViewUniformBufferFlight;
		void* renderGraph;
	};

	export struct RasterPipelineScope :public PassNode
	{
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		virtual auto onFrameStart(void* graph) noexcept -> void override;

		auto fillRasterMaterialScopeDesc(RasterMaterialScope* raster_material, void* graph) noexcept -> void;

		// Desc :: Shaders
		MemScope<RHI::IShader> shaderVert = nullptr;
		MemScope<RHI::IShader> shaderFrag = nullptr;
		MemScope<RHI::IShader> shaderTask = nullptr;
		MemScope<RHI::IShader> shaderMesh = nullptr;
		// Desc :: Fixed Functional Paras
		RHI::BufferLayout vertexBufferLayout = {};
		RHI::TopologyKind topologyKind = RHI::TopologyKind::TriangleList;
		RHI::PolygonMode polygonMode = RHI::PolygonMode::FILL;
		RHI::CullMode cullMode = RHI::CullMode::NONE;
		RHI::ColorBlendingDesc colorBlendingDesc = RHI::NoBlending;
		RHI::DepthStencilDesc depthStencilDesc = RHI::TestLessAndWrite;

		float lineWidth = 0.0f;
		// Devirtualized :: IPipeline
		MemScope<RHI::IPipeline> pipeline;
		// Node children
		std::vector<NodeHandle> materialScopes;
		std::unordered_map<std::string, NodeHandle> materialScopesRegister;
	
	private:
		friend struct RasterPassScope;
		// Desc :: Frame buffer binded / will be filled by RasterPassScope
		RHI::IRenderPass* renderPass = nullptr;
		RHI::Extend viewportExtend;
		// Created Fixed-Function components
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
		//
		int hasPerFrameUniformBuffer = -1;
		int hasPerViewUniformBuffer = -1;
		unsigned int totalResourceNum = 0;
		NodeHandle perFrameUniformBufferFlight;
		NodeHandle perViewUniformBufferFlight;
		void* renderGraph;
	};

	export struct RasterPassScope :public PassNode
	{
		virtual auto onRegistered(void* graph, void* render_graph_workshop) noexcept -> void override;
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		virtual auto onFrameStart(void* graph) noexcept -> void override;


		auto fillRasterPipelineScopeDesc(RasterPipelineScope* raster_pipeline, void* graph) noexcept -> void;
		auto updatePerViewUniformBuffer(PerViewUniformBuffer const& buffer, uint32_t const& current_frame) noexcept -> void;

		NodeHandle framebuffer;
		std::vector<NodeHandle> pipelineScopes;
		std::unordered_map<std::string, NodeHandle> pipelineScopesRegister;

	private:
		PerViewUniformBuffer perViewUniformBuffer;
		NodeHandle perFrameUniformBufferFlight;
		NodeHandle perViewUniformBufferFlight;
		void* renderGraph;
	};
}