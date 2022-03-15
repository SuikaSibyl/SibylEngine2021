module;
#include <utility>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
export module GFX.RDG.Common;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.IFactory;
import RHI.ICommandBuffer;
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
	};
	export enum class NodeDetailedType :uint32_t
	{
		NONE,
		STORAGE_BUFFER,
		UNIFORM_BUFFER,
		FRAME_BUFFER,
	};

	struct NodeRegistry;
	export struct Node
	{
		virtual auto onRegister() noexcept -> void {}
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		NodeRegistry* registry;
		NodeDetailedType type = NodeDetailedType::NONE;
		NodeAttributesFlags attributes = 0;
	};

	// Node will not be exposed directly for design reason
	// We use a handle to aliasing a node
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
	export struct ResourceNode :public Node
	{
	public:
		RHI::DescriptorType resourceType;
		RHI::ShaderStageFlags shaderStages;
	};

	export struct PassNode :public Node
	{
	public:

	};

	// Framebuffers & uniform buffer flights could be a sets of resources
	// Container could also be a node, with a vector of sub-handles.
	export struct Container :public Node
	{
		Container() { attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER); }
		Container(std::initializer_list<NodeHandle> list) :handles(list) {
			attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER);}
		Container(std::vector<NodeHandle>&& handles) :handles(handles) {
			attributes |= addBit(NodeAttrbutesFlagBits::CONTAINER); }

		auto size() -> size_t { return handles.size(); }
		auto operator[](int index) const -> NodeHandle { return handles[index]; }
		auto operator[](int index) -> NodeHandle& { return handles[index]; }
		auto getSubnode(uint32_t idx) ->Node* { return registry->getNode(handles[idx]); }
		std::vector<NodeHandle> handles;
	};

	export struct FlightContainer :public Container
	{
		FlightContainer() = default;
		FlightContainer(std::initializer_list<NodeHandle> list) :Container(list) {}
		FlightContainer(std::vector<NodeHandle>&& handles) :Container(std::move(handles)) {}

		auto handleOnFlight(uint32_t const& flight) noexcept -> NodeHandle { return handles[flight]; }
	};

	export struct FramebufferContainer :public Container
	{
		FramebufferContainer() { type = NodeDetailedType::FRAME_BUFFER; }
		auto getWidth() noexcept -> uint32_t { return width; }
		auto getHeight() noexcept -> uint32_t { return height; }

		uint32_t width, height;
		uint32_t colorAttachCount = 0;
		uint32_t depthAttachCount = 0;
	};
}