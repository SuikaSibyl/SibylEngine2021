module;
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <Macros.h>
#include <EntryPoint.h>
#include <string_view>
#include <filesystem>
module Main;

import Core.Assert;
import Core.Test;
import Core.Window;
import Core.Enums;
import Core.Event;
import Core.SObject;
import Core.SPointer;
import Core.Window;
import Core.Layer;
import Core.LayerStack;
import Core.Application;
import Core.MemoryManager;
import Core.File;
import Core.Log;

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

import UAT.IUniversalApplication;

using namespace SIByL;

class SandboxApp :public IUniversalApplication
{
public:
	struct S1
	{
		int i;
	};

	virtual void onAwake() override
	{
		WindowLayerDesc window_layer_desc = {
			SIByL::EWindowVendor::GLFW,
			1280,
			720,
			"Hello"
		};
		WindowLayer* window_layer = attachWindowLayer(window_layer_desc);

		graphicContext.reset(RHI::IFactory::createGraphicContext({ RHI::API::VULKAN }));
		graphicContext->attachWindow(window_layer->getWindow());
		physicalDevice.reset(RHI::IFactory::createPhysicalDevice({ graphicContext.get() }));
		logicalDevice.reset(RHI::IFactory::createLogicalDevice({ physicalDevice.get() }));
		swapchain.reset(RHI::IFactory::createSwapchain({ logicalDevice.get() }));
		resourceFactory = MemNew<RHI::IResourceFactory>(logicalDevice.get());

		AssetLoader shaderLoader;
		shaderLoader.addSearchPath("../Engine/Binaries/Runtime/spirv");
		Buffer shader_vert, shader_frag;
		shaderLoader.syncReadAll("vert.spv", shader_vert);
		shaderLoader.syncReadAll("frag.spv", shader_frag);
		shaderVert = resourceFactory->createShaderFromBinary(shader_vert, { RHI::ShaderStage::VERTEX,"main" });
		shaderFrag = resourceFactory->createShaderFromBinary(shader_frag, { RHI::ShaderStage::FRAGMENT,"main" });

		MemScope<RHI::IVertexLayout> vertex_layout = resourceFactory->createVertexLayout();
		MemScope<RHI::IInputAssembly> input_assembly = resourceFactory->createInputAssembly(RHI::TopologyKind::TriangleList);
		RHI::Extend extend = swapchain->getExtend();
		MemScope<RHI::IViewportsScissors> viewport_scissors = resourceFactory->createViewportsScissors(extend, extend);
		RHI::RasterizerDesc rasterizer_desc =
		{
			RHI::PolygonMode::FILL,
			0.0f,
			RHI::CullMode::BACK,
		};
		MemScope<RHI::IRasterizer> rasterizer = resourceFactory->createRasterizer(rasterizer_desc);
		RHI::MultiSampleDesc multisampling_desc =
		{
			false,
		};
		MemScope<RHI::IMultisampling> multisampling = resourceFactory->createMultisampling(multisampling_desc);
		RHI::DepthStencilDesc depthstencil_desc =
		{
			false
		};
		MemScope<RHI::IDepthStencil> depthstencil = resourceFactory->createDepthStencil(depthstencil_desc);
		RHI::ColorBlendingDesc colorBlending_desc =
		{
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ZERO,
			RHI::BlendOperator::ADD,
			RHI::BlendFactor::ONE,
			RHI::BlendFactor::ZERO,
			false,
		};
		MemScope<RHI::IColorBlending> color_blending = resourceFactory->createColorBlending(colorBlending_desc);
		std::vector<RHI::PipelineState> pipelinestates_desc =
		{
			RHI::PipelineState::VIEWPORT,
			RHI::PipelineState::LINE_WIDTH,
		};
		MemScope<RHI::IDynamicState> dynamic_states = resourceFactory->createDynamicState(pipelinestates_desc);

		RHI::PipelineLayoutDesc pipelineLayout_desc =
		{
			0,
		};
		MemScope<RHI::IPipelineLayout> pipeline_layout = resourceFactory->createPipelineLayout(pipelineLayout_desc);

		RHI::RenderPassDesc renderpass_desc =
		{
			RHI::SampleCount::COUNT_1_BIT,
			RHI::ResourceFormat::FORMAT_B8G8R8A8_SRGB,
		};
		MemScope<RHI::IRenderPass> render_pass = resourceFactory->createRenderPass(renderpass_desc);

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
			render_pass.get(),
		};
		pipeline = resourceFactory->createPipeline(pipeline_desc);

		for (int i = 0; i < swapchain->getSwapchainCount(); i++)
		{
			RHI::FramebufferDesc framebuffer_desc =
			{
				extend.width,
				extend.height,
				render_pass.get(),
				{swapchain->getITextureView(i)},
			};
			framebuffers.emplace_back(resourceFactory->createFramebuffer(framebuffer_desc));
		}

		commandPool = resourceFactory->createCommandPool(RHI::QueueType::GRAPHICS);
		for (int i = 0; i < swapchain->getSwapchainCount(); i++)
		{
			commandbuffers.emplace_back(resourceFactory->createCommandBuffer(commandPool.get()));
		}

		commandbuffers[0]->beginRecording();
		commandbuffers[0]->cmdBeginRenderPass(render_pass.get(), framebuffers[0].get());
		commandbuffers[0]->cmdBindPipeline(pipeline.get());
		commandbuffers[0]->cmdDraw(3, 1, 0, 0);
		commandbuffers[0]->cmdEndRenderPass();
		commandbuffers[0]->endRecording();
		SE_CORE_INFO("OnAwake End");
	}

	virtual void onUpdate() override
	{
		SE_CORE_INFO("Update");
	}

private:
	Scope<RHI::IGraphicContext> graphicContext;
	Scope<RHI::IPhysicalDevice> physicalDevice;
	Scope<RHI::ILogicalDevice> logicalDevice;
	Scope<RHI::ISwapChain> swapchain;

	MemScope<RHI::IResourceFactory> resourceFactory;
	MemScope<RHI::IShader> shaderVert;
	MemScope<RHI::IShader> shaderFrag;
	MemScope<RHI::IPipeline> pipeline;

	std::vector<MemScope<RHI::IFramebuffer>> framebuffers;

	MemScope<RHI::ICommandPool> commandPool;
	std::vector<MemScope<RHI::ICommandBuffer>> commandbuffers;
};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	return app;
}