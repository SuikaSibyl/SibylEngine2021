module;

export module RHI.IFixedFunctions;
import Core.SObject;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		// Fixed Functions:
		// - Vertex input
		// - Input assembly
		// - Viewportsand scissors
		// - Rasterizer
		// - Multisampling
		// - Depthand stencil testing
		// - Color blending
		//
		// ╭──────────────────────────────────────────────────────────────────────────╮
		// │  Vertex Input                                                            │
		// ├──────────────────┬───────────────────────────────────────────────────────┤
		// │  Bindings        │  spacing between data and whether the data 			  │
		// │                  │  is per-vertex or per-instance (see instancing)		  │
		// ├──────────────────┼───────────────────────────────────────────────────────┤
		// │  Attribute Desc  │  type of the attributes passed to vertex shader,	  │
		// │                  │  which binding to load them fromand at which offset	  │
		// ╰──────────────────┴───────────────────────────────────────────────────────╯
		//  API variants
		// ╭──────────────┬──────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineVertexInputStateCreateInfo   │
		// │  DirectX 12  │                                          │
		// │  OpenGL      │                                          │
		// ╰──────────────┴──────────────────────────────────────────╯
		export class IVertexLayout
		{
		public:
			IVertexLayout() = default;
			IVertexLayout(IVertexLayout&&) = default;
			virtual ~IVertexLayout() = default;
		};

		// ╭──────────────────────────────────────────────────────────────────────────────╮
		// │  Input Assembly                                                              │
		// ├────────────────────┬─────────────────────────────────────────────────────────┤
		// │  Geometry Topology │  spacing between data and whether the data              │
		// ├────────────────────┼─────────────────────────────────────────────────────────┤
		// │  Attribute Desc    │  break up lines and triangles in the _STRIP topology    │
		// │                    │  modes by using a special index of 0xFFFF or 0xFFFFFFFF │
		// ╰────────────────────┴─────────────────────────────────────────────────────────╯
		//  API variants
		// ╭──────────────┬────────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineInputAssemblyStateCreateInfo   │
		// │  DirectX 12  │                                            │
		// │  OpenGL      │                                            │
		// ╰──────────────┴────────────────────────────────────────────╯
		export class IInputAssembly
		{
		public:
			IInputAssembly(TopologyKind topology_kind);
			IInputAssembly(IInputAssembly&&) = default;
			virtual ~IInputAssembly() = default;

		protected:
			TopologyKind topologyKind;
		};

		// ╭──────────────────────────────────────────────────────────────╮
		// │  ViewportsScissors                                           │
		// ├────────────┬─────────────────────────────────────────────────┤
		// │  Viewport  │  describes the region of the framebuffer        │
		// │            │  that the output will be rendered to            │
		// │            │  almost always be (0, 0) to (width, height)     │
		// ├────────────┼─────────────────────────────────────────────────┤
		// │  scissor   │  scissor rectangles define in which regions     │
		// │			│  pixels will actually be stored                 │
		// ╰────────────┴─────────────────────────────────────────────────╯
		//  API variants
		// ╭──────────────┬────────────────────────────────────────────────╮
		// │  Vulkan	  │   VkRect2D/VkPipelineViewportStateCreateInfo   │
		// │  DirectX 12  │                                                │
		// │  OpenGL      │                                                │
		// ╰──────────────┴────────────────────────────────────────────────╯
		export class IViewportsScissors
		{
		public:
			IViewportsScissors() = default;
			IViewportsScissors(IViewportsScissors&&) = default;
			virtual ~IViewportsScissors() = default;
		};

		// ╭───────────────────────────────────────────────────────────────────╮
		// │  Rasterizer                                                       │
		// ├─────────────────┬─────────────────────────────────────────────────┤
		// │  Depth Testing  │                                                 │
		// ├─────────────────┼─────────────────────────────────────────────────┤
		// │  Face Culling   │                                                 │
		// ├─────────────────┼─────────────────────────────────────────────────┤
		// │  Scissor Test   │                                                 │
		// ├─────────────────┼─────────────────────────────────────────────────┤
		// │  Render Mode    │  output fragments that fill entire polygons or  │
		// │				 │  just the edges (wireframe rendering)           │
		// ╰─────────────────┴─────────────────────────────────────────────────╯
		//  API variants
		// ╭──────────────┬─────────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineRasterizationStateCreateInfo    │
		// │  DirectX 12  │                                             │
		// │  OpenGL      │                                             │
		// ╰──────────────┴─────────────────────────────────────────────╯
		export class IRasterizer
		{
		public:
			IRasterizer() = default;
			IRasterizer(IRasterizer&&) = default;
			virtual ~IRasterizer() = default;
		};

		// ╭─────────────────╮
		// │  Multisampling  │
		// ╰─────────────────╯
		//  API variants
		// ╭──────────────┬──────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineMultisampleStateCreateInfo   │
		// │  DirectX 12  │                                          │
		// │  OpenGL      │                                          │
		// ╰──────────────┴──────────────────────────────────────────╯
		export class IMultisampling
		{
		public:
			IMultisampling() = default;
			IMultisampling(IMultisampling&&) = default;
			virtual ~IMultisampling() = default;
		};

		// ╭────────────────╮
		// │  DepthStencil  │
		// ╰────────────────╯
		//  API variants
		// ╭──────────────┬──────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineDepthStencilStateCreateInfo  │
		// │  DirectX 12  │                                          │
		// │  OpenGL      │                                          │
		// ╰──────────────┴──────────────────────────────────────────╯
		export class IDepthStencil
		{
		public:
			IDepthStencil() = default;
			IDepthStencil(IDepthStencil&&) = default;
			virtual ~IDepthStencil() = default;
		};

		// ╭─────────────────╮
		// │  ColorBlending  │
		// ╰─────────────────╯
		//  API variants
		// ╭──────────────┬─────────────────────────────────────────╮
		// │  Vulkan	  │   VkPipelineColorBlendAttachmentState   │
		// │  DirectX 12  │                                         │
		// │  OpenGL      │                                         │
		// ╰──────────────┴─────────────────────────────────────────╯
		export class IColorBlending
		{
		public:
			IColorBlending() = default;
			IColorBlending(IColorBlending&&) = default;
			virtual ~IColorBlending() = default;
		};

		export class IDynamicState
		{
		public:
			IDynamicState() = default;
			IDynamicState(IDynamicState&&) = default;
			virtual ~IDynamicState() = default;
		};

		export class IFixedFunctions :public SObject
		{
		public:
			IFixedFunctions() = default;
			IFixedFunctions(IFixedFunctions&&) = default;
			virtual ~IFixedFunctions() = default;


		};
	}
}
