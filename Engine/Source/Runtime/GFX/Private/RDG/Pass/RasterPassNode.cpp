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
import RHI.IRenderPass;
import RHI.IStorageBuffer;
import RHI.IBarrier;
import Core.MemoryManager;

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
		// vertex buffer layout
		RHI::BufferLayout vertex_buffer_layout =
		{
			{RHI::DataType::Float3, "Position"},
			{RHI::DataType::Float3, "Color"},
			{RHI::DataType::Float2, "UV"},
		};
		vertexLayout = factory->createVertexLayout(vertex_buffer_layout);

		// input assembly
		inputAssembly = factory->createInputAssembly(RHI::TopologyKind::TriangleList);

		// viewport scissors
		RHI::Extend extend{ framebuffer.getWidth(), framebuffer.getHeight() };
		viewportScissors = factory->createViewportsScissors(extend, extend);

		// raster
		RHI::RasterizerDesc rasterizer_desc =
		{
			RHI::PolygonMode::FILL,
			0.0f,
			RHI::CullMode::NONE,
		};
		rasterizer = factory->createRasterizer(rasterizer_desc);

		// multisample
		RHI::MultiSampleDesc multisampling_desc =
		{
			false,
		};
		multisampling = factory->createMultisampling(multisampling_desc);

		// depth stencil
		RHI::DepthStencilDesc depthstencil_desc =
		{
			true,
			false,
			RHI::CompareOp::LESS
		};
		depthstencil = factory->createDepthStencil(depthstencil_desc);

		// color blending 
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
		colorBlending = factory->createColorBlending(colorBlending_desc);

		// pipeline state
		std::vector<RHI::PipelineState> pipelinestates_desc =
		{
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
		};
		dynamicStates = factory->createDynamicState(pipelinestates_desc);

		// create pipeline layouts
		RHI::PipelineLayoutDesc pipelineLayout_desc =
		{ {desciptorSetLayout.get()} };
		MemScope<RHI::IPipelineLayout> pipelineLayout = factory->createPipelineLayout(pipelineLayout_desc);

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
			vertexLayout.get(),
			inputAssembly.get(),
			viewportScissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			colorBlending.get(),
			dynamicStates.get(),
			pipelineLayout.get(),
			renderPass.get(),
		};
		pipeline = factory->createPipeline(pipeline_desc);
	}

	auto RasterPassNode::onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		// viewport scissors
		RHI::Extend extend{ framebuffer.getWidth(), framebuffer.getHeight() };
		viewportScissors = factory->createViewportsScissors(extend, extend);

		RHI::PipelineDesc pipeline_desc =
		{
			{ shaderVert.get(), shaderFrag.get()},
			vertexLayout.get(),
			inputAssembly.get(),
			viewportScissors.get(),
			rasterizer.get(),
			multisampling.get(),
			depthstencil.get(),
			colorBlending.get(),
			dynamicStates.get(),
			pipelineLayout.get(),
			renderPass.get(),
		};
		pipeline = factory->createPipeline(pipeline_desc);
	}
}