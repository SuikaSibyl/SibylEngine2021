module;
#include <utility>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <string_view>
export module GFX.RDG.Common;
import Core.Log;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.IFactory;
import RHI.ICommandBuffer;
import RHI.IRenderPass;
import RHI.IFramebuffer;
import ECS.UID;

namespace SIByL::GFX::RDG
{	
	// Node is the first citizen in RDG
	// The RDG is composed of many nodes with certain connection
	// A node is either a resource (ResourceNode) or a resource manipulation (PassNode)
	// ┌──────┬──────────────┬───────────────┬──────────────────┐
	// │	  │ 			 │			     │ ColorBufferNode  │
	// │	  │  			 │  TextureNode  ├──────────────────┤
	// │      │              │               │ DepthBufferNode  │
	// │	  │ ResourceNode ├───────────────┼──────────────────┼────────────────────┐
	// │      │              │               │ StorageBuffer    │ IndirectDrawBuffer │
	// │ Node │				 │  BufferNode   ├──────────────────┼────────────────────┘
	// │      │              │               │ UniformBuffer    │
	// │      ├──────────────┼───────────────┼──────────────────┘
	// │      │              │  RasterPass   │
	// │      │   PassNode   ├───────────────┤
	// │	  │              │  ComputePass  │
	// └──────┴──────────────┴───────────────┘
	// An attribute bits is used for tell attributes
	export using NodeAttributesFlags = uint32_t;
	export enum class NodeAttrbutesFlagBits: uint32_t
	{
		PLACEHOLDER = 0x00000001,
		CONTAINER   = 0x00000002,
		RESOURCE    = 0x00000004,
		FLIGHT      = 0x00000008,
		PRESENT     = 0x00000010,
	};
	export enum class NodeDetailedType :uint32_t
	{
		NONE,
		// Resources
		SAMPLER,
		COLOR_TEXTURE,
		DEPTH_TEXTURE,
		STORAGE_BUFFER,
		UNIFORM_BUFFER,
		// Resources Container
		FRAME_BUFFER,
		FRAME_BUFFER_FLIGHTS,
		UNIFORM_BUFFER_FLIGHTS,
		// Passes
		RASTER_PASS,
		COMPUTE_PASS,
		BLIT_PASS,
	};

	struct NodeRegistry;
	export struct Node
	{
		// Lifetime Virtual Funcs
		virtual auto onRegister() noexcept -> void {} // optional
		virtual auto onCompile() noexcept -> void {} // optional
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		// Debug Print
		virtual auto onPrint() noexcept -> void;
		// Members
		NodeRegistry* registry = nullptr;
		NodeDetailedType type = NodeDetailedType::NONE;
		NodeAttributesFlags attributes = 0;
		std::string_view tag;
	};

	// Node will not be exposed directly for design reason
	// We use a handle to refer to a node
	export using NodeHandle = uint64_t;

	// A registry is used to manage nodes
	export struct NodeRegistry
	{
		auto getNode(NodeHandle handle) noexcept -> Node* { return nodes[handle].get(); }

		template <class T>
		auto registNode(MemScope<T>&& node) noexcept -> NodeHandle
		{
			MemScope<T> resource = std::move(node);
			MemScope<Node> cast_node = MemCast<Node>(resource);
			cast_node->registry = this;
			cast_node->onRegister();
			NodeHandle handle = ECS::UniqueID::RequestUniqueID();
			nodes[handle] = std::move(cast_node);
			return handle;
		}
		std::unordered_map<NodeHandle, MemScope<Node>> nodes;
	};

	// Three basic Node Children struct:
	// - ResourceNode
	// - PassNode
	// - Container
	export enum class ConsumeKind :uint32_t
	{
		BUFFER_READ,
		BUFFER_WRITE,
		RENDER_TARGET,
		IMAGE_SAMPLE,
		COPY_SRC,
		COPY_DST,
	};

	export struct ConsumeHistory
	{
		NodeHandle pass;
		ConsumeKind kind;
	};

	export struct ResourceNode :public Node
	{
		ResourceNode() { attributes |= addBit(NodeAttrbutesFlagBits::RESOURCE); }
		RHI::ShaderStageFlags shaderStages = 0;
	};

	export template<class T>
	union TolerantPtr
	{
		TolerantPtr() { ref = nullptr; }
		~TolerantPtr() { scope = nullptr; }
		T* ref;
		MemScope<T> scope;
	};

	// Pass Node
	export struct PassNode :public Node
	{};

	// Framebuffers & uniform buffer flights could be a sets of resources
	// Container could also be a node, with a vector of sub-handles.
	export struct Container :public ResourceNode
	{
		Container() { attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER); }
		Container(std::initializer_list<NodeHandle> list) :handles(list) {
			attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER);}
		Container(std::vector<NodeHandle>&& handles) :handles(handles) {
			attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER); }

		auto operator[](int index) const -> NodeHandle { return handles[index]; }
		auto operator[](int index) -> NodeHandle& { return handles[index]; }
		auto size() -> size_t { return handles.size(); }
		std::vector<NodeHandle> handles;
	};

	export struct FlightContainer :public Container
	{
		FlightContainer() { attributes |= addBit(NodeAttrbutesFlagBits::FLIGHT); }
		FlightContainer(std::initializer_list<NodeHandle> list) :Container(list) { attributes |= addBit(NodeAttrbutesFlagBits::FLIGHT); }
		FlightContainer(std::vector<NodeHandle>&& handles) :Container(std::move(handles)) { attributes |= addBit(NodeAttrbutesFlagBits::FLIGHT); }

		auto handleOnFlight(uint32_t const& flight) noexcept -> NodeHandle { return handles[flight]; }
	};

	export struct FramebufferContainer :public Container
	{
		FramebufferContainer() { type = NodeDetailedType::FRAME_BUFFER; }

		auto getWidth() noexcept -> uint32_t;
		auto getHeight() noexcept -> uint32_t;
		auto getFramebuffer() noexcept -> RHI::IFramebuffer* { return framebuffer.get(); }
		auto getRenderPass() noexcept -> RHI::IRenderPass* { return renderPass.get(); }
		auto getColorAttachHandle(uint32_t idx) noexcept -> NodeHandle { return handles[idx]; }
		auto getDepthAttachHandle() noexcept -> NodeHandle { return (depthAttachCount == 1) ? handles[colorAttachCount] : 0; }

		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		uint32_t colorAttachCount = 0;
		uint32_t depthAttachCount = 0;

		MemScope<RHI::IRenderPass> renderPass;
		MemScope<RHI::IFramebuffer> framebuffer;
	};
}