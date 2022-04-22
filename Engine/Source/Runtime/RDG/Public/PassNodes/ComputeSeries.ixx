module;
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <unordered_map>
#include <functional>
export module GFX.RDG.ComputeSeries;
import Core.Buffer;
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
	struct ComputeMaterialScope;
	struct ComputePipelineScope;
	struct ComputePassScope;

	export struct ComputeDispatch :public PassNode
	{
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;
		
		using DispatchSizeFn = std::function<void(uint32_t&, uint32_t&, uint32_t&)>;
		using PushConstantFn = std::function<void(Buffer&)>;

		PushConstantFn pushConstant = nullptr;
		DispatchSizeFn customSize = nullptr;
	private:
		friend struct ComputeMaterialScope;
		RHI::IPipelineLayout** pipelineLayout;
	};

	export struct ComputeMaterialScope :public PassNode
	{
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;

		std::vector<NodeHandle> resources;
		std::vector<NodeHandle> storage_textures;
		std::vector<NodeHandle> sampled_textures;

		std::vector<NodeHandle> dispatches;
	private:
		friend struct ComputePipelineScope;
		// member
		std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;
		// Desc :: desciptor set layout
		RHI::IDescriptorSetLayout* desciptorSetLayout;
		RHI::IPipelineLayout* pipelineLayout;
		unsigned int totalResourceNum = 0;
		void* renderGraph;
	};

	export struct ComputePipelineScope :public PassNode
	{
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;

		auto fillComputeMaterialScopeDesc(ComputeMaterialScope* raster_material, void* graph) noexcept -> void;

		// Desc :: Shaders
		MemScope<RHI::IShader> shaderComp = nullptr;

		// Devirtualized :: IPipeline
		MemScope<RHI::IPipeline> pipeline;
		// Node children
		std::vector<NodeHandle> materialScopes;
		std::unordered_map<std::string, NodeHandle> materialScopesRegister;

	private:
		friend struct ComputePassScope;
		MemScope<RHI::IPipelineLayout> pipelineLayout;
		MemScope<RHI::IDescriptorSetLayout> desciptorSetLayout;
		//
		unsigned int totalResourceNum = 0;
		void* renderGraph;
	};

	export struct ComputePassScope :public PassNode
	{
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void override;

		std::vector<NodeHandle> pipelineScopes;
		std::unordered_map<std::string, NodeHandle> pipelineScopesRegister;

	private:
		void* renderGraph;
	};

	export struct ComputePassIndefiniteScope :public PassNode
	{

	};

}