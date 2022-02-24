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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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
import Core.Time;

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

import UAT.IUniversalApplication;

using namespace SIByL;

const int MAX_FRAMES_IN_FLIGHT = 2;

class SandboxApp :public IUniversalApplication
{
public:
	struct Vertex {
		glm::vec2 pos;
		glm::vec3 color;
	};

	struct UniformBufferObject {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

	virtual void onAwake() override
	{
		RHI::SLANG::ICompileSession comipeSession;
		comipeSession.loadModule("hello-world", "computeMain");

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

		resourceFactory = MemNew<RHI::IResourceFactory>(logicalDevice.get());
		// shader resources
		AssetLoader shaderLoader;
		shaderLoader.addSearchPath("../Engine/Binaries/Runtime/spirv");
		Buffer shader_vert, shader_frag;
		shaderLoader.syncReadAll("vs_ubo.spv", shader_vert);
		shaderLoader.syncReadAll("frag.spv", shader_frag);
		shaderVert = resourceFactory->createShaderFromBinary(shader_vert, { RHI::ShaderStage::VERTEX,"main" });
		shaderFrag = resourceFactory->createShaderFromBinary(shader_frag, { RHI::ShaderStage::FRAGMENT,"main" });
		// vertex buffer
		const std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
		};
		Buffer vertex_proxy((void*)vertices.data(), vertices.size() * sizeof(Vertex), 4);
		vertexBuffer = resourceFactory->createVertexBuffer(&vertex_proxy);
		// index buffer
		const std::vector<uint16_t> indices = {
			0, 1, 2, 2, 3, 0
		};
		Buffer index_proxy((void*)indices.data(), indices.size() * sizeof(uint16_t), 4);
		indexBuffer = resourceFactory->createIndexBuffer(&index_proxy, sizeof(uint16_t));
		
		// uniform process
		{				
			// uniform buffer
			uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
				uniformBuffers[i] = resourceFactory->createUniformBuffer(sizeof(UniformBufferObject));

			// create pool
			RHI::DescriptorPoolDesc descriptor_pool_desc =
			{ {{RHI::DescriptorType::UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT}}, // set types
				MAX_FRAMES_IN_FLIGHT }; // total sets
			descriptorPool = resourceFactory->createDescriptorPool(descriptor_pool_desc);

			// create desc layout
			RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc =
			{ {{ 0, 1, RHI::DescriptorType::UNIFORM_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr }} };
			desciptor_set_layout = resourceFactory->createDescriptorSetLayout(descriptor_set_layout_desc);

			// create sets
			descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
			RHI::DescriptorSetDesc descriptor_set_desc =
			{	descriptorPool.get(),
				desciptor_set_layout.get() };
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
				descriptorSets[i] = resourceFactory->createDescriptorSet(descriptor_set_desc);

			// configure descriptors in sets
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				descriptorSets[i]->update(uniformBuffers[i].get(), 0, 0);
			}

			// create pipeline layouts
			RHI::PipelineLayoutDesc pipelineLayout_desc =
			{ {desciptor_set_layout.get()} };
			pipeline_layout = resourceFactory->createPipelineLayout(pipelineLayout_desc);
		}

		// create swapchain & related ...
		swapchain = resourceFactory->createSwapchain({});
		createModifableResource();


		commandPool = resourceFactory->createCommandPool({ RHI::QueueType::GRAPHICS, (uint32_t)RHI::CommandPoolAttributeFlagBits::RESET });
		commandbuffers.resize(MAX_FRAMES_IN_FLIGHT);
		imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			commandbuffers[i] = resourceFactory->createCommandBuffer(commandPool.get());
			imageAvailableSemaphore[i] = resourceFactory->createSemaphore();
			renderFinishedSemaphore[i] = resourceFactory->createSemaphore();
			inFlightFence[i] = resourceFactory->createFence();
		}

		timer.start();

		SE_CORE_INFO("OnAwake End");
	}
	
	void createModifableResource()
	{
		RHI::Extend extend = swapchain->getExtend();

		RHI::BufferLayout vertex_buffer_layout =
		{
			{RHI::DataType::Float2, "Position"},
			{RHI::DataType::Float3, "Color"},
		};
		MemScope<RHI::IVertexLayout> vertex_layout = resourceFactory->createVertexLayout(vertex_buffer_layout);
		MemScope<RHI::IInputAssembly> input_assembly = resourceFactory->createInputAssembly(RHI::TopologyKind::TriangleList);
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


		RHI::RenderPassDesc renderpass_desc =
		{
			RHI::SampleCount::COUNT_1_BIT,
			RHI::ResourceFormat::FORMAT_B8G8R8A8_SRGB,
		};
		renderPass = resourceFactory->createRenderPass(renderpass_desc);

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
		pipeline = resourceFactory->createPipeline(pipeline_desc);

		for (unsigned int i = 0; i < swapchain->getSwapchainCount(); i++)
		{
			RHI::FramebufferDesc framebuffer_desc =
			{
				extend.width,
				extend.height,
				renderPass.get(),
				{swapchain->getITextureView(i)},
			};
			framebuffers.emplace_back(resourceFactory->createFramebuffer(framebuffer_desc));
		}
	}

	virtual auto onWindowResize(WindowResizeEvent& e) -> bool override
	{
		logicalDevice->waitIdle();

		framebuffers.clear();
		pipeline = nullptr;
		renderPass = nullptr;
		swapchain = nullptr;

		swapchain = resourceFactory->createSwapchain({ e.GetWidth(), e.GetHeight() });
		createModifableResource();

		return false;
	}

	virtual void onUpdate() override
	{
		// 1. Wait for the previous frame to finish
		{
			inFlightFence[currentFrame]->wait();
			inFlightFence[currentFrame]->reset();
		}
		// Timer update
		{
			timer.tick();
			SE_CORE_INFO("FPS: {0}", timer.getFPS());
		}
		// update uniform buffer
		{
			float time = (float)timer.getTotalTimeSeconds();
			auto [width, height] = swapchain->getExtend();
			UniformBufferObject ubo;
			ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			ubo.proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 10.0f);
			ubo.proj[1][1] *= -1;
			Buffer ubo_proxy((void*) &ubo, sizeof(UniformBufferObject), 4);
			uniformBuffers[currentFrame]->updateBuffer(&ubo_proxy);
		}
		// drawFrame
		{
			//	2. Acquire an image from the swap chain
			uint32_t imageIndex = swapchain->acquireNextImage(imageAvailableSemaphore[currentFrame].get());
			//	3. Record a command buffer which draws the scene onto that image
			commandbuffers[currentFrame]->reset();
			commandbuffers[currentFrame]->beginRecording();
			commandbuffers[currentFrame]->cmdBeginRenderPass(renderPass.get(), framebuffers[imageIndex].get());
			commandbuffers[currentFrame]->cmdBindPipeline(pipeline.get());
			commandbuffers[currentFrame]->cmdBindVertexBuffer(vertexBuffer.get());
			commandbuffers[currentFrame]->cmdBindIndexBuffer(indexBuffer.get());
			RHI::IDescriptorSet* tmp_set = descriptorSets[currentFrame].get();
			commandbuffers[currentFrame]->cmdBindDescriptorSets(RHI::PipelineBintPoint::GRAPHICS,
				pipeline_layout.get(), 0, 1, &tmp_set, 0, nullptr);
			commandbuffers[currentFrame]->cmdDrawIndexed(6, 1, 0, 0, 0);
			commandbuffers[currentFrame]->cmdEndRenderPass();
			commandbuffers[currentFrame]->endRecording();
			//	4. Submit the recorded command buffer
			commandbuffers[currentFrame]->submit(imageAvailableSemaphore[currentFrame].get(), renderFinishedSemaphore[currentFrame].get(), inFlightFence[currentFrame].get());
			//	5. Present the swap chain image
			swapchain->present(imageIndex, renderFinishedSemaphore[currentFrame].get());
		}
		// update current frame
		{
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}
	}

	virtual void onShutdown() override
	{
		logicalDevice->waitIdle();
		RHI::DeviceToGlobal::releaseGlobal();
	}

private:
	Timer timer;
	uint32_t currentFrame = 0;

	Scope<RHI::IGraphicContext> graphicContext;
	Scope<RHI::IPhysicalDevice> physicalDevice;
	Scope<RHI::ILogicalDevice> logicalDevice;

	MemScope<RHI::IResourceFactory> resourceFactory;
	MemScope<RHI::IShader> shaderVert;
	MemScope<RHI::IShader> shaderFrag;

	MemScope<RHI::IVertexBuffer> vertexBuffer;
	MemScope<RHI::IIndexBuffer> indexBuffer;
	std::vector<MemScope<RHI::IUniformBuffer>> uniformBuffers;

	MemScope<RHI::IDescriptorPool> descriptorPool;
	MemScope<RHI::IDescriptorSetLayout> desciptor_set_layout;
	MemScope<RHI::IPipelineLayout> pipeline_layout;
	std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;

	MemScope<RHI::ISwapChain> swapchain;
	MemScope<RHI::IRenderPass> renderPass;
	MemScope<RHI::IPipeline> pipeline;
	std::vector<MemScope<RHI::IFramebuffer>> framebuffers;

	MemScope<RHI::ICommandPool> commandPool;
	std::vector<MemScope<RHI::ICommandBuffer>> commandbuffers;
	std::vector<MemScope<RHI::ISemaphore>> imageAvailableSemaphore;
	std::vector<MemScope<RHI::ISemaphore>> renderFinishedSemaphore;
	std::vector<MemScope<RHI::IFence>> inFlightFence;
};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	return app;
}