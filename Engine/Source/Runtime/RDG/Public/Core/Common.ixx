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
import RHI.ICommandBuffer;
import RHI.IBarrier;
import ECS.UID;

namespace SIByL::GFX::RDG
{	
	// Node is the first citizen in RDG
	// The RDG is composed of many nodes with certain connection
	// A node is either a resource (ResourceNode) or a resource manipulation (PassNode)
	// ┌──────┬──────────────┬───────────┬───────────────┬──────────────────┐
	// │	  │              │ 			 │			     │ ColorBufferNode  │
	// │	  │              │  		 │  TextureNode  ├──────────────────┤
	// │      │              │           │               │ DepthBufferNode  │
	// │	  │              │  (Atom*)  ├───────────────┼──────────────────┼────────────────────┐
	// │      │              │           │               │ StorageBuffer    │ IndirectDrawBuffer │
	// │      │	ResourceNode │ 			 │  BufferNode   ├──────────────────┼────────────────────┘
	// │      │              │           │               │ UniformBuffer    │
	// │      │              ├───────────┼───────────────┴────────┬─────────┘
	// │ Node │              │  		 │  FlightContainer       │
	// │	  │              │ Contianer ├────────────────────────┤
	// │      │	             │ 			 │  FramebufferContainer  │
	// │      ├──────────────┼───────────┼──────────────────┬─────┘
	// │	  │ 			 │			 │ RasterPass       │
	// │	  │  			 │  (Atom*)  ├──────────────────┤
	// │      │ PassNode     │           │ ComputePass      │
	// │	  │              ├───────────┼──────────────────┤
	// │      │              │  Scope    │ MultiDispatch    │
	// └──────┴──────────────┴───────────┴──────────────────┘
	// 
	// An attribute bits is used for tell attributes
	export using NodeAttributesFlags = uint32_t;
	export enum class NodeAttrbutesFlagBits: uint32_t
	{
		PLACEHOLDER		 = 0x00000001,
		CONTAINER		 = 0x00000002,
		SCOPE			 = 0x00000004,
		RESOURCE		 = 0x00000008,
		FLIGHT			 = 0x00000010,
		PRESENT			 = 0x00000020,
		ONE_TIME_SUBMIT	 = 0x00000040,
		INCLUSION		 = 0x00000080,
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
		// Pass Scope
		SCOPE,
		MULTI_DISPATCH_SCOPE,
		SCOPE_END,
	};

	// Node will not be exposed directly for design reason
	// We use a handle to refer to a node
	export using NodeHandle = uint64_t;

	struct NodeRegistry;
	export struct Node
	{
		// Lifetime Virtual Funcs
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional

		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		// Debug Print
		virtual auto onPrint() noexcept -> void;
		// Members
		NodeRegistry* registry = nullptr;
		NodeDetailedType type = NodeDetailedType::NONE;
		NodeAttributesFlags attributes = 0;
		NodeHandle handle;
		std::string tag;
	};

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
			NodeHandle handle = ECS::UniqueID::RequestUniqueID();
			cast_node->handle = handle;
			nodes[handle] = std::move(cast_node);
			return handle;
		}
		std::unordered_map<NodeHandle, MemScope<Node>> nodes;
	};

	// basic Node Children struct:
	// - ResourceNode

	// consume history will be statisticed
	// in order to generate correct barrier
	export enum class ConsumeKind :uint32_t
	{
		BUFFER_READ_WRITE,
		RENDER_TARGET,
		IMAGE_STORAGE_READ_WRITE,
		IMAGE_SAMPLE,
		COPY_SRC,
		COPY_DST,
		INDIRECT_DRAW,
		SCOPE,
		MULTI_DISPATCH_SCOPE_BEGIN,
		MULTI_DISPATCH_SCOPE_END,
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

		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override
		{
			consumeHistoryOnetime.clear();
			consumeHistory.clear();
		}

		std::vector<ConsumeHistory> consumeHistoryOnetime;
		std::vector<ConsumeHistory> consumeHistory;
	};

	// Some resource could be either external or localy owned
	export template<class T>
	union TolerantPtr
	{
		TolerantPtr() { ref = nullptr; }
		~TolerantPtr() { scope = nullptr; }
		T* ref;
		MemScope<T> scope;
	};

	// basic Node Children struct:
	// - PassNode

	// Barrier is managed together
	// There ref (by handle) will be dispatched to each pass
	export using BarrierHandle = uint64_t;
	export struct BarrierPool
	{
		auto registBarrier(MemScope<RHI::IBarrier>&& barrier) noexcept -> NodeHandle;
		auto getBarrier(NodeHandle handle) noexcept -> RHI::IBarrier*;
		std::unordered_map<BarrierHandle, MemScope<RHI::IBarrier>> barriers;
	};

	export struct PassNode :public Node
	{
		virtual auto onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void {}
		std::vector<BarrierHandle> barriers;
	};

	// - Container
	//   container is a tuple of resources
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

		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		uint32_t colorAttachCount = 0;
		uint32_t depthAttachCount = 0;

		MemScope<RHI::IRenderPass> renderPass;
		MemScope<RHI::IFramebuffer> framebuffer;
	};

	// - Scope
	//   scope is a tuple of resources
	export struct PassScope :public PassNode
	{
		PassScope() { attributes |= addBit(NodeAttrbutesFlagBits::SCOPE); }

	};
	
	export struct PassScopeEnd :public PassScope
	{
		NodeHandle scopeBeginHandle;
		virtual auto onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
	};

	export auto getString(ConsumeKind kind) noexcept -> std::string
	{
		switch (kind)
		{
		case SIByL::GFX::RDG::ConsumeKind::BUFFER_READ_WRITE:			return "BUFFER_READ_WRITE   "; break;
		case SIByL::GFX::RDG::ConsumeKind::RENDER_TARGET:				return "RENDER_TARGET "; break;
		case SIByL::GFX::RDG::ConsumeKind::IMAGE_STORAGE_READ_WRITE:	return "IMAGE_STORAGE_READ_WRITE  "; break;
		case SIByL::GFX::RDG::ConsumeKind::IMAGE_SAMPLE:				return "IMAGE_SAMPLE  "; break;
		case SIByL::GFX::RDG::ConsumeKind::COPY_SRC:					return "COPY_SRC      "; break;
		case SIByL::GFX::RDG::ConsumeKind::COPY_DST:					return "COPY_DST      "; break;
		case SIByL::GFX::RDG::ConsumeKind::INDIRECT_DRAW:				return "INDIRECT_DRAW "; break;
		case SIByL::GFX::RDG::ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN:  return "MULTI_DISPATCH_SCOPE_BEGIN "; break;
		case SIByL::GFX::RDG::ConsumeKind::MULTI_DISPATCH_SCOPE_END:	return "MULTI_DISPATCH_SCOPE_END "; break;
		default:														return "ERROR         "; break;
		}
	}
}