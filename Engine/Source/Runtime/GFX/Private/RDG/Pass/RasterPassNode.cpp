module;
#include <vector>
module GFX.RasterPassNode;
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
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.IBarrier;
import Core.MemoryManager;
import GFX.RDG.PassNode;

namespace SIByL::GFX::RDG
{
	RasterPassNode::RasterPassNode(
		void* graph, 
		std::vector<NodeHandle>&& ins, 
		RHI::IShader* vertex_shader, 
		RHI::IShader* fragment_shader, 
		uint32_t const& constant_size)
	{

		//MemScope<RHI::IVertexLayout> vertex_layout = resourceFactory->createVertexLayout(vertex_buffer_layout);

	}

	auto RasterPassNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		//RenderGraph* render_graph = (RenderGraph*)graph;

		RHI::BufferLayout vertex_buffer_layout =
		{
			{RHI::DataType::Float3, "Position"},
			{RHI::DataType::Float3, "Color"},
			{RHI::DataType::Float2, "UV"},
		};
		MemScope<RHI::IVertexLayout> vertex_layout = factory->createVertexLayout(vertex_buffer_layout);
		MemScope<RHI::IInputAssembly> input_assembly = factory->createInputAssembly(RHI::TopologyKind::TriangleList);
		RHI::Extend extend{ framebuffer.getWidth(), framebuffer.getHeight() };
		MemScope<RHI::IViewportsScissors> viewport_scissors = factory->createViewportsScissors(extend, extend);
		RHI::RasterizerDesc rasterizer_desc =
		{
			RHI::PolygonMode::FILL,
			0.0f,
			RHI::CullMode::NONE,
		};
		MemScope<RHI::IRasterizer> rasterizer = factory->createRasterizer(rasterizer_desc);
		RHI::MultiSampleDesc multisampling_desc =
		{
			false,
		};
		MemScope<RHI::IMultisampling> multisampling = factory->createMultisampling(multisampling_desc);
		RHI::DepthStencilDesc depthstencil_desc =
		{
			true,
			false,
			RHI::CompareOp::LESS
		};
		MemScope<RHI::IDepthStencil> depthstencil = factory->createDepthStencil(depthstencil_desc);
		RHI::ColorBlendingDesc colorBlending_desc =
		{
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ONE,
			true,
		};
		MemScope<RHI::IColorBlending> color_blending = factory->createColorBlending(colorBlending_desc);
		std::vector<RHI::PipelineState> pipelinestates_desc =
		{
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
		};
		MemScope<RHI::IDynamicState> dynamic_states = factory->createDynamicState(pipelinestates_desc);

		// create pipeline layouts
		RHI::PipelineLayoutDesc pipelineLayout_desc =
		{ {desciptorSetLayout.get()} };
		MemScope<RHI::IPipelineLayout> pipeline_layout = factory->createPipelineLayout(pipelineLayout_desc);

		RHI::RenderPassDesc renderpass_desc =
		{ {
				// color attachment
				{
					RHI::SampleCount::COUNT_1_BIT,
					RHI::ResourceFormat::FORMAT_B8G8R8A8_SRGB,
					RHI::AttachmentLoadOp::CLEAR,
					RHI::AttachmentStoreOp::STORE,
					RHI::AttachmentLoadOp::DONT_CARE,
					RHI::AttachmentStoreOp::DONT_CARE,
					RHI::ImageLayout::UNDEFINED,
					RHI::ImageLayout::PRESENT_SRC,
					{0,0,0,1}
				},
			},
			{				// depth attachment
				{
					RHI::SampleCount::COUNT_1_BIT,
					RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT,
					RHI::AttachmentLoadOp::CLEAR,
					RHI::AttachmentStoreOp::DONT_CARE,
					RHI::AttachmentLoadOp::DONT_CARE,
					RHI::AttachmentStoreOp::DONT_CARE,
					RHI::ImageLayout::UNDEFINED,
					RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA,
					{1,0}
				},
			} };
		renderPass = factory->createRenderPass(renderpass_desc);

		RHI::PipelineDesc pipeline_desc =
		{
			{ shaderVert.get(), shaderFrag.get()},
			vertex_layout.get(),
			input_assembly.get(),
			viewport_scissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			color_blending.get(),
			dynamic_states.get(),
			pipeline_layout.get(),
			renderPass.get(),
		};
		pipeline = factory->createPipeline(pipeline_desc);
	}

}