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
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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
import Core.Image;
import Core.Color;

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

import UAT.IUniversalApplication;

using namespace SIByL;

const int MAX_FRAMES_IN_FLIGHT = 2;

class SandboxApp :public IUniversalApplication
{
public:
	struct Vertex {
		glm::vec3 pos;
		glm::vec3 color;
		glm::vec2 uv;
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

		// create device
		graphicContext = (RHI::IFactory::createGraphicContext({ RHI::API::VULKAN }));
		graphicContext->attachWindow(window_layer->getWindow());
		physicalDevice = (RHI::IFactory::createPhysicalDevice({ graphicContext.get() }));
		logicalDevice = (RHI::IFactory::createLogicalDevice({ physicalDevice.get() }));
		resourceFactory = MemNew<RHI::IResourceFactory>(logicalDevice.get());

		// shader resources
		AssetLoader shaderLoader;
		shaderLoader.addSearchPath("../Engine/Binaries/Runtime/spirv");
		Buffer shader_vert, shader_frag, shader_comp, shader_comp_init;
		shaderLoader.syncReadAll("vs_particle.spv", shader_vert);
		shaderLoader.syncReadAll("fs_sampler.spv", shader_frag);
		shaderLoader.syncReadAll("particle.spv", shader_comp);
		shaderLoader.syncReadAll("init.spv", shader_comp_init);
		shaderVert = resourceFactory->createShaderFromBinary(shader_vert, { RHI::ShaderStage::VERTEX,"main" });
		shaderFrag = resourceFactory->createShaderFromBinary(shader_frag, { RHI::ShaderStage::FRAGMENT,"main" });
		shaderCompute = resourceFactory->createShaderFromBinary(shader_comp, { RHI::ShaderStage::COMPUTE,"main" });
		shaderComputeInit = resourceFactory->createShaderFromBinary(shader_comp_init, { RHI::ShaderStage::COMPUTE,"main" });
		// vertex buffer
		const std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

			{ {-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
		};
		Buffer vertex_proxy((void*)vertices.data(), vertices.size() * sizeof(Vertex), 4);
		vertexBuffer = resourceFactory->createVertexBuffer(&vertex_proxy);
		// index buffer
		const std::vector<uint16_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4
		};
		Buffer index_proxy((void*)indices.data(), indices.size() * sizeof(uint16_t), 4);
		indexBuffer = resourceFactory->createIndexBuffer(&index_proxy, sizeof(uint16_t));

		// load image
		Image image("./assets/texture.jpg");
		texture = resourceFactory->createTexture(&image);
		textureView = resourceFactory->createTextureView(texture.get());
		sampler = resourceFactory->createSampler({});

		// uniform process
		{				
			// uniform buffer
			uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
				uniformBuffers[i] = resourceFactory->createUniformBuffer(sizeof(UniformBufferObject));

			// create pool
			RHI::DescriptorPoolDesc descriptor_pool_desc =
			{ {{RHI::DescriptorType::UNIFORM_BUFFER, MAX_FRAMES_IN_FLIGHT},
			   {RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT},
			   {RHI::DescriptorType::STORAGE_BUFFER, 4 * MAX_FRAMES_IN_FLIGHT}}, // set types
				2 * MAX_FRAMES_IN_FLIGHT }; // total sets
			descriptorPool = resourceFactory->createDescriptorPool(descriptor_pool_desc);

			// create desc layout
			RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc =
			{ {{ 0, 1, RHI::DescriptorType::UNIFORM_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr },
			   { 1, 1, RHI::DescriptorType::COMBINED_IMAGE_SAMPLER, (uint32_t)RHI::ShaderStageFlagBits::FRAGMENT_BIT, nullptr },
			   { 2, 1, RHI::DescriptorType::STORAGE_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT | (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr }} };
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
				descriptorSets[i]->update(textureView.get(), sampler.get(), 1, 0);
			}

			// create pipeline layouts
			RHI::PipelineLayoutDesc pipelineLayout_desc =
			{ {desciptor_set_layout.get()} };
			pipeline_layout = resourceFactory->createPipelineLayout(pipelineLayout_desc);
		}
		// compute stuff
		{
			// create storage buffers
			storageBuffer_1 = resourceFactory->createStorageBuffer(sizeof(float) * 3 * 32);
			storageBuffer_2 = resourceFactory->createStorageBuffer(sizeof(float) * 3 * 32);

			// create descriptor set layout
			RHI::DescriptorSetLayoutDesc descriptor_set_layout_desc =
			{ {{ 0, 1, RHI::DescriptorType::STORAGE_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT | (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT, nullptr },
			   { 1, 1, RHI::DescriptorType::STORAGE_BUFFER, (uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT, nullptr }} };
			compute_desciptor_set_layout = resourceFactory->createDescriptorSetLayout(descriptor_set_layout_desc);

			// create pipeline layout
			RHI::PipelineLayoutDesc pipelineLayout_desc =
			{ {compute_desciptor_set_layout.get()} };
			compute_pipeline_layout = resourceFactory->createPipelineLayout(pipelineLayout_desc);

			// create comput pipeline
			RHI::ComputePipelineDesc pipeline_desc =
			{
				compute_pipeline_layout.get(),
				shaderCompute.get()
			};
			
			// create descriptor set
			compute_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
			RHI::DescriptorSetDesc descriptor_set_desc =
			{	descriptorPool.get(),
				compute_desciptor_set_layout.get() };
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
				compute_descriptorSets[i] = resourceFactory->createDescriptorSet(descriptor_set_desc);

			// configure descriptors in sets
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				compute_descriptorSets[i]->update(storageBuffer_1.get(), 0, 0);
				compute_descriptorSets[i]->update(storageBuffer_2.get(), 1, 0);
				descriptorSets[i]->update(storageBuffer_1.get(), 2, 0);
			}

			computePipeline = resourceFactory->createPipeline(pipeline_desc);
			pipeline_desc.shader = shaderComputeInit.get();
			initcomputePipeline = resourceFactory->createPipeline(pipeline_desc);

			compute_barrier = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				});

			compute_barrier_2 = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT,
				0,

				});
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


		// init storage buffer
		RHI::ICommandPool* transientPool = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getTransientCommandPool();
		MemScope<RHI::ICommandBuffer> transientCommandbuffer = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getResourceFactory()->createCommandBuffer(transientPool);
		transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);

		RHI::IDescriptorSet* compute_tmp_set = compute_descriptorSets[0].get();
		transientCommandbuffer->cmdBindComputePipeline(initcomputePipeline.get());
		transientCommandbuffer->cmdBindDescriptorSets(RHI::PipelineBintPoint::COMPUTE,
			compute_pipeline_layout.get(), 0, 1, &compute_tmp_set, 0, nullptr);
		transientCommandbuffer->cmdDispatch(1, 1, 1);

		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		logicalDevice->waitIdle();


		timer.start();

		SE_CORE_INFO("OnAwake End");
	}
	
	void createModifableResource()
	{
		RHI::Extend extend = swapchain->getExtend();

		depthTexture = resourceFactory->createTexture(
			{
			RHI::ResourceType::Texture2D, //ResourceType type;
			RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT, //ResourceFormat format;
			RHI::ImageTiling::OPTIMAL, //ImageTiling tiling;
			(uint32_t)RHI::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT, //ImageUsageFlags usages;
			RHI::BufferShareMode::EXCLUSIVE, //BufferShareMode shareMode;
			RHI::SampleCount::COUNT_1_BIT, //SampleCount sampleCount;
			RHI::ImageLayout::UNDEFINED, //ImageLayout layout;
			extend.width, //uint32_t width;
			extend.height //uint32_t height;
			});
		depthView = resourceFactory->createTextureView(depthTexture.get());

		RHI::BufferLayout vertex_buffer_layout =
		{
			{RHI::DataType::Float3, "Position"},
			{RHI::DataType::Float3, "Color"},
			{RHI::DataType::Float2, "UV"},
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
		{{
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
				{swapchain->getITextureView(i), depthView.get()},
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
			commandbuffers[currentFrame]->cmdDrawIndexed(12, 32, 0, 0, 0);
			commandbuffers[currentFrame]->cmdEndRenderPass();

			commandbuffers[currentFrame]->cmdPipelineBarrier(compute_barrier.get());
			RHI::IDescriptorSet* compute_tmp_set = compute_descriptorSets[currentFrame].get();
			commandbuffers[currentFrame]->cmdBindComputePipeline(computePipeline.get());
			commandbuffers[currentFrame]->cmdBindDescriptorSets(RHI::PipelineBintPoint::COMPUTE,
				compute_pipeline_layout.get(), 0, 1, &compute_tmp_set, 0, nullptr);
			commandbuffers[currentFrame]->cmdDispatch(1, 1, 1);

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
		RHI::DeviceToGlobal::removeDevice(logicalDevice.get());
	}

private:
	Timer timer;
	uint32_t currentFrame = 0;

	MemScope<RHI::IGraphicContext> graphicContext;
	MemScope<RHI::IPhysicalDevice> physicalDevice;
	MemScope<RHI::ILogicalDevice> logicalDevice;

	MemScope<RHI::IResourceFactory> resourceFactory;
	MemScope<RHI::IShader> shaderVert;
	MemScope<RHI::IShader> shaderFrag;
	MemScope<RHI::IShader> shaderCompute;
	MemScope<RHI::IShader> shaderComputeInit;

	MemScope<RHI::IVertexBuffer> vertexBuffer;
	MemScope<RHI::IIndexBuffer> indexBuffer;
	MemScope<RHI::IStorageBuffer> storageBuffer_1;
	MemScope<RHI::IStorageBuffer> storageBuffer_2;
	std::vector<MemScope<RHI::IUniformBuffer>> uniformBuffers;

	MemScope<RHI::ITexture> texture;
	MemScope<RHI::ITextureView> textureView;
	MemScope<RHI::ISampler> sampler;
	MemScope<RHI::ITexture> depthTexture;
	MemScope<RHI::ITextureView> depthView;

	MemScope<RHI::IDescriptorPool> descriptorPool;
	MemScope<RHI::IDescriptorSetLayout> desciptor_set_layout;
	MemScope<RHI::IPipelineLayout> pipeline_layout;
	std::vector<MemScope<RHI::IDescriptorSet>> descriptorSets;

	MemScope<RHI::IBarrier> compute_barrier;
	MemScope<RHI::IBarrier> compute_barrier_2;
	MemScope<RHI::IPipelineLayout> compute_pipeline_layout;
	MemScope<RHI::IDescriptorSetLayout> compute_desciptor_set_layout;
	std::vector<MemScope<RHI::IDescriptorSet>> compute_descriptorSets;

	MemScope<RHI::ISwapChain> swapchain;
	MemScope<RHI::IRenderPass> renderPass;
	MemScope<RHI::IPipeline> pipeline;
	MemScope<RHI::IPipeline> computePipeline;
	MemScope<RHI::IPipeline> initcomputePipeline;
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