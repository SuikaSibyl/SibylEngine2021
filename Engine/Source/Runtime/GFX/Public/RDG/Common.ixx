module;
#include <cstdint>
#include <vector>
#include <string>
export module GFX.RDG.Common;
import RHI.IEnum;
import RHI.IFactory;

namespace SIByL::GFX::RDG
{	
	// Node is the first citizen in RDG
	// The RDG is composed of many nodes with certain connection
	// A node is either a resource (ResourceNode) or a resource manipulation (PassNode)
	// ┌──────┬──────────────┬───────────────┬──────────────────┐
	// │	  │ 			 │			     │ ColorBufferNode  │
	// │	  │  			 │  TextureNode  ├──────────────────┤
	// │      │              │               │ DepthBufferNode  │
	// │	  │ 			 ├───────────────┼──────────────────┼────────────────────┐
	// │      │ ResourceNode │               │ StorageBuffer    │ IndirectDrawBuffer │
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
	export struct Node
	{
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void {} // optional
		NodeAttributesFlags attributes;
	};

	// Node will not be exposed directly for design reason
	// We use a handle to aliasing a node
	export using NodeHandle = uint64_t;
	// Framebuffers & uniform buffer flights could be a sets of resources
	// Container could also be a node, with a vector of sub-handles.
	export struct Container
	{
		auto size() -> size_t { return handles.size(); }
		auto operator[](int index) const -> NodeHandle { return handles[index]; }
		auto operator[](int index) -> NodeHandle& { return handles[index]; }
		std::vector<NodeHandle> handles;
	};
}