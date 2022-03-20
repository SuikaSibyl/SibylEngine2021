module;
#include "entt/entt.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <Macros.h>
#include <EntryPoint.h>
#include <string_view>
#include <filesystem>
#include <utility>
#include <string>
#include <unordered_map>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <math.h>

#include "imgui/imgui/imgui.h"
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
import Core.Input;
import Core.Cache;

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

import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;

import GFX.SceneTree;
import GFX.Scene;
import GFX.Mesh;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.RasterPassNode;

import ParticleSystem.ParticleSystem;
import ParticleSystem.PrecomputedSample;

import UAT.IUniversalApplication;
import Editor.ImGuiLayer;
import Editor.Viewport;
import Editor.ImFactory;
import Editor.ImImage;

using namespace SIByL;

const int MAX_FRAMES_IN_FLIGHT = 2;

class SandboxApp :public IUniversalApplication
{
public:
	struct UniformBufferObject {
		glm::vec4 cameraPos;
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

	struct EmitConstant
	{
		unsigned int emitCount;
		float time;
		float x;
		float y;
	};

	struct Empty
	{};

	struct Size
	{
		unsigned int width;
		unsigned int height;
	};


	MemScope<Editor::ImGuiLayer> imguiLayer;
	Editor::Viewport mainViewport;
	MemScope<Editor::ImImage> viewportImImage;

	virtual void onAwake() override
	{
		// create window
		WindowLayerDesc window_layer_desc = {
			SIByL::EWindowVendor::GLFW,
			1280,
			720,
			"Hello"
		};
		window_layer = attachWindowLayer(window_layer_desc);

		// create device
		graphicContext = (RHI::IFactory::createGraphicContext({ RHI::API::VULKAN }));
		graphicContext->attachWindow(window_layer->getWindow());
		physicalDevice = (RHI::IFactory::createPhysicalDevice({ graphicContext.get() }));
		logicalDevice = (RHI::IFactory::createLogicalDevice({ physicalDevice.get() }));
		resourceFactory = MemNew<RHI::IResourceFactory>(logicalDevice.get());

		// create imgui layer
		imguiLayer = MemNew<Editor::ImGuiLayer>(logicalDevice.get());
		pushLayer(imguiLayer.get());
		Editor::ImFactory imfactory(imguiLayer.get());
		//// create swapchain & related ...
		//swapchain = resourceFactory->createSwapchain({});

		// create scene
		scene.deserialize("test_scene.scene", logicalDevice.get());

		// shader resources
		AssetLoader shaderLoader;
		shaderLoader.addSearchPath("../Engine/Binaries/Runtime/spirv");

		MemScope<RHI::IShader> shaderVert = resourceFactory->createShaderFromBinaryFile("vs_particle.spv", { RHI::ShaderStage::VERTEX,"main" });
		MemScope<RHI::IShader> shaderFrag = resourceFactory->createShaderFromBinaryFile("fs_sampler.spv", { RHI::ShaderStage::FRAGMENT,"main" });
		MemScope<RHI::IShader> shaderPortalInit = resourceFactory->createShaderFromBinaryFile("portal/portal_init.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> shaderPortalEmit = resourceFactory->createShaderFromBinaryFile("portal/portal_emit.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> shaderPortalUpdate = resourceFactory->createShaderFromBinaryFile("portal/portal_update.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> aces = resourceFactory->createShaderFromBinaryFile("aces.spv", { RHI::ShaderStage::COMPUTE,"main" });

		// load image
		Image image("./assets/Sparkle.tga");
		texture = resourceFactory->createTexture(&image);
		textureView = resourceFactory->createTextureView(texture.get());
		sampler = resourceFactory->createSampler({});

		// load precomputed samples
		Buffer torusSamples;
		Buffer* samples[] = { &torusSamples };
		EmptyHeader header;
		CacheBrain::instance()->loadCache(2267996151488940154, header, samples);
		torusBuffer = resourceFactory->createStorageBuffer(&torusSamples);

		rdg.reDatum(1280, 720);
		GFX::RDG::RenderGraphBuilder rdg_builder(rdg);
		// particle system
		portal.init(sizeof(float) * 4 * 3, 100000, shaderPortalInit.get(), shaderPortalEmit.get(), shaderPortalUpdate.get());
		portal.addEmitterSamples(torusBuffer.get());
		GFX::RDG::NodeHandle external_texture = rdg_builder.addColorBufferExt(texture.get(), textureView.get());
		GFX::RDG::NodeHandle external_sampler = rdg_builder.addSamplerExt(sampler.get());
		portal.registerRenderGraph(&rdg_builder);
		// renderer
		depthBuffer = rdg_builder.addDepthBuffer(1.f, 1.f);
		uniformBufferFlights = rdg_builder.addUniformBufferFlights(sizeof(UniformBufferObject));
		//std::vector<RHI::ITexture*> swapchin_textures;
		//std::vector<RHI::ITextureView*> swapchin_texture_views;
		//for (int i = 0; i < swapchain->getSwapchainCount(); i++)
		//{
		//	swapchin_textures.emplace_back(swapchain->getITexture(i));
		//	swapchin_texture_views.emplace_back(swapchain->getITextureView(i));
		//}
		//swapchainColorBufferFlights = rdg_builder.addColorBufferFlightsExtPresent(swapchin_textures, swapchin_texture_views);

		// raster pass sono 1
		GFX::RDG::NodeHandle srgb_color_attachment = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT, 1.f, 1.f);
		GFX::RDG::NodeHandle srgb_depth_attachment = rdg_builder.addDepthBuffer(1.f, 1.f);
		srgb_framebuffer = rdg_builder.addFrameBufferRef({ srgb_color_attachment }, srgb_depth_attachment);

		// HDR raster pass
		MemScope<RHI::IShader> shaderVert2 = resourceFactory->createShaderFromBinaryFile("vs_particle.spv", { RHI::ShaderStage::VERTEX,"main" });
		MemScope<RHI::IShader> shaderFrag2 = resourceFactory->createShaderFromBinaryFile("fs_sampler.spv", { RHI::ShaderStage::FRAGMENT,"main" });
		renderPassNodeSRGB = rdg_builder.addRasterPass({ uniformBufferFlights, external_sampler, portal.particleBuffer });
		rdg.tag(renderPassNodeSRGB, "Raster HDR");
		GFX::RDG::RasterPassNode* rasterPassNodeSRGB = rdg.getRasterPassNode(renderPassNodeSRGB);
		rasterPassNodeSRGB->shaderVert = std::move(shaderVert2);
		rasterPassNodeSRGB->shaderFrag = std::move(shaderFrag2);
		rasterPassNodeSRGB->framebuffer = srgb_framebuffer;
		rasterPassNodeSRGB->textures = { external_texture };

		// ACES
		test_write_target = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, 1.f, 1.f);
		acesPass = rdg_builder.addComputePass(aces.get(), { test_write_target, srgb_color_attachment }, sizeof(unsigned int) * 2);
		rdg.tag(acesPass, "ACES");

		// building ...
		rdg_builder.build(resourceFactory.get());
		rdg.print();

		viewportImImage = imfactory.createImImage(
			rdg.getSamplerNode(external_sampler)->getSampler(),
			rdg.getTextureBufferNode(test_write_target)->getTextureView(), 
			RHI::ImageLayout::GENERAL);
		mainViewport.bindImImage(viewportImImage.get());

		//for (int i = 0; i < 3; i++)
		//{
		//	GFX::RDG::NodeHandle color_attachment_i = rdg.getContainer(swapchainColorBufferFlights)->handles[i];
		//	rdg.getTextureBufferNode(color_attachment_i)->getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, RHI::ImageLayout::PRESENT_SRC);
		//	MemScope<RHI::IImageMemoryBarrier> image_memory_barrier = resourceFactory->createImageMemoryBarrier({
		//		rdg.getTextureBufferNode(color_attachment_i)->getTexture(), //ITexture* image;
		//		RHI::ImageSubresourceRange{
		//			(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
		//			0,
		//			1,
		//			0,
		//			1
		//		},//ImageSubresourceRange subresourceRange;
		//		(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT, //AccessFlags srcAccessMask;
		//		(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT, //AccessFlags dstAccessMask;
		//		RHI::ImageLayout::TRANSFER_DST_OPTIMAL, // old Layout
		//		RHI::ImageLayout::PRESENT_SRC // new Layout
		//	});
		//	color_attach_copy2present[i] = resourceFactory->createBarrier({
		//		(uint32_t)RHI::PipelineStageFlagBits::TRANSFER_BIT,//srcStageMask
		//		(uint32_t)RHI::PipelineStageFlagBits::ALL_GRAPHICS_BIT,//dstStageMask
		//		0,
		//		{},
		//		{},
		//		{image_memory_barrier.get()}
		//	});
		//	MemScope<RHI::IImageMemoryBarrier> image_memory_barrier2 = resourceFactory->createImageMemoryBarrier({
		//		rdg.getTextureBufferNode(color_attachment_i)->getTexture(), //ITexture* image;
		//		RHI::ImageSubresourceRange{
		//			(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
		//			0,
		//			1,
		//			0,
		//			1
		//		},//ImageSubresourceRange subresourceRange;
		//		(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT, //AccessFlags srcAccessMask;
		//		(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT, //AccessFlags dstAccessMask;
		//		RHI::ImageLayout::PRESENT_SRC, // old Layout
		//		RHI::ImageLayout::TRANSFER_DST_OPTIMAL // new Layout
		//	});

		//	color_attach_present2copy[i] = resourceFactory->createBarrier({
		//		(uint32_t)RHI::PipelineStageFlagBits::ALL_GRAPHICS_BIT,//srcStageMask
		//		(uint32_t)RHI::PipelineStageFlagBits::TRANSFER_BIT,//dstStageMask
		//		0,
		//		{},
		//		{},
		//		{image_memory_barrier2.get()}
		//	});

		//}
		//// ACES Results
		//{
		//	MemScope<RHI::IImageMemoryBarrier> image_memory_barrier = resourceFactory->createImageMemoryBarrier({
		//		rdg.getTextureBufferNode(test_write_target)->getTexture(), //ITexture* image;
		//		RHI::ImageSubresourceRange{
		//			(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
		//			0,
		//			1,
		//			0,
		//			1
		//		},//ImageSubresourceRange subresourceRange;
		//		(uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT, //AccessFlags srcAccessMask;
		//		(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT, //AccessFlags dstAccessMask;
		//		RHI::ImageLayout::GENERAL, // old Layout
		//		RHI::ImageLayout::TRANSFER_SRC_OPTIMAL // new Layout
		//	});
		//	aces_res_general2copysrc = resourceFactory->createBarrier({
		//		(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,//srcStageMask
		//		(uint32_t)RHI::PipelineStageFlagBits::TRANSFER_BIT,//dstStageMask
		//		0,
		//		{},
		//		{},
		//		{image_memory_barrier.get()}
		//	});

		//	MemScope<RHI::IImageMemoryBarrier> image_memory_barrier_2 = resourceFactory->createImageMemoryBarrier({
		//		rdg.getTextureBufferNode(test_write_target)->getTexture(), //ITexture* image;
		//		RHI::ImageSubresourceRange{
		//			(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
		//			0,
		//			1,
		//			0,
		//			1
		//		},//ImageSubresourceRange subresourceRange;
		//		(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT, //AccessFlags srcAccessMask;
		//		(uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT, //AccessFlags dstAccessMask;
		//		RHI::ImageLayout::TRANSFER_SRC_OPTIMAL, // old Layout
		//		RHI::ImageLayout::GENERAL // new Layout
		//	});

		//	aces_res_copysrc2general = resourceFactory->createBarrier({
		//		(uint32_t)RHI::PipelineStageFlagBits::TRANSFER_BIT,//srcStageMask
		//		(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,//dstStageMask
		//		0,
		//		{},
		//		{},
		//		{image_memory_barrier_2.get()}
		//		});
		//}
		// compute stuff
		{		
			MemScope<RHI::IImageMemoryBarrier> image_memory_barrier = resourceFactory->createImageMemoryBarrier({
				rdg.getTextureBufferNode(srgb_color_attachment)->getTexture(), //ITexture* image;
				RHI::ImageSubresourceRange{
					(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
					0,
					1,
					0,
					1
				},//ImageSubresourceRange subresourceRange;
				0, //AccessFlags srcAccessMask;
				(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT, //AccessFlags dstAccessMask;
				RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, // old Layout
				RHI::ImageLayout::GENERAL // new Layout
			});

			attach_read_barrier = resourceFactory->createBarrier({
					(uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT,//srcStageMask
					(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,//dstStageMask
					0,
					{},
					{},
					{image_memory_barrier.get()}
				});

			compute_barrier = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				});

			compute_memory_barrier_0 = resourceFactory->createMemoryBarrier({
					(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT | (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT | (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT
				});

			compute_barrier_0 = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				0,
				{compute_memory_barrier_0.get()}
				});

			compute_drawcall_memory_barrier = resourceFactory->createMemoryBarrier({
					(uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT | (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT
				});
			compute_drawcall_barrier = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::DRAW_INDIRECT_BIT,
				0,
				{compute_drawcall_memory_barrier.get()}
				});

			compute_barrier_2 = resourceFactory->createBarrier(RHI::BarrierDesc{
				(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT,
				0,
				});
		}
		//createModifableResource();

		// comand stuffs
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

		// timer
		timer.start();

		// init storage buffer
		RHI::ICommandPool* transientPool = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getTransientCommandPool();
		MemScope<RHI::ICommandBuffer> transientCommandbuffer = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getResourceFactory()->createCommandBuffer(transientPool);
		transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);

		rdg.getComputePassNode(portal.initPass)->executeWithConstant(transientCommandbuffer.get(), 200, 1, 1, 0, 100000u);
		rdg.getTextureBufferNode(srgb_color_attachment)->getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, RHI::ImageLayout::GENERAL);
		rdg.getTextureBufferNode(test_write_target)->getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, RHI::ImageLayout::GENERAL);
		Size size = { 1280,720 };
		rdg.getComputePassNode(acesPass)->executeWithConstant(transientCommandbuffer.get(), 40, 23, 1, 0, size);

		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		logicalDevice->waitIdle();


		SE_CORE_INFO("OnAwake End");
	}
	
	void createModifableResource()
	{
		//RHI::Extend extend = swapchain->getExtend();
		//rdg.reDatum(extend.width, extend.height);
	}

	virtual auto onWindowResize(WindowResizeEvent& e) -> bool override
	{
		logicalDevice->waitIdle();

		pipeline = nullptr;
		renderPass = nullptr;
		//swapchain = nullptr;

		//swapchain = resourceFactory->createSwapchain({ e.GetWidth(), e.GetHeight() });
		//for (int i = 0; i < swapchain->getSwapchainCount(); i++)
		//{
		//	rdg.getTextureBufferNodeFlight(swapchainColorBufferFlights, i)->resetExternal(
		//		swapchain->getITexture(i),
		//		swapchain->getITextureView(i)
		//	);
		//}

		//createModifableResource();

		return false;
	}

	float test = 1.0f;

	virtual void onUpdate() override
	{
		imguiLayer->startNewFrame();

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
			//time = 0.5 * 3.1415926;
			//uint32_t width = mainViewport.getWidth();
			//uint32_t height = mainViewport.getHeight();
			uint32_t width = 1280;
			uint32_t height = 720;
			//auto [width, height] = swapchain->getExtend();

			UniformBufferObject ubo;
			ubo.cameraPos = glm::vec4(10.0f* cosf(time), 2.0f, 10.0f * sinf(time), 0.0f);
			ubo.model = glm::mat4(1.0f);
			ubo.view = glm::lookAt(glm::vec3(10.0f * cosf(time), 2.0f, 10.0f * sinf(time)), glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			ubo.proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
			ubo.proj[1][1] *= -1;
			Buffer ubo_proxy((void*) &ubo, sizeof(UniformBufferObject), 4);
			rdg.getUniformBufferFlight(uniformBufferFlights, currentFrame)->updateBuffer(&ubo_proxy);
		}
		// drawFrame
		{
			////	2. Acquire an image from the swap chain
			//uint32_t imageIndex = swapchain->acquireNextImage(imageAvailableSemaphore[currentFrame].get());
			//	3. Record a command buffer which draws the scene onto that image
			commandbuffers[currentFrame]->reset();
			commandbuffers[currentFrame]->beginRecording();

			// render pass 1
			commandbuffers[currentFrame]->cmdPipelineBarrier(compute_drawcall_barrier.get());

			// render pass 2
			commandbuffers[currentFrame]->cmdBeginRenderPass(
				rdg.getFramebufferContainer(srgb_framebuffer)->getRenderPass(),
				rdg.getFramebufferContainer(srgb_framebuffer)->getFramebuffer());
			commandbuffers[currentFrame]->cmdBindPipeline(rdg.getRasterPassNode(renderPassNodeSRGB)->pipeline.get());
			
			std::function<void(ECS::TagComponent&, GFX::Mesh&)> mesh_processor_2 = [&](ECS::TagComponent& tag, GFX::Mesh& mesh) {
				commandbuffers[currentFrame]->cmdBindVertexBuffer(mesh.vertexBuffer.get());

				commandbuffers[currentFrame]->cmdBindIndexBuffer(mesh.indexBuffer.get());
				RHI::IDescriptorSet* tmp_set = rdg.getRasterPassNode(renderPassNodeSRGB)->descriptorSets[currentFrame].get();
				commandbuffers[currentFrame]->cmdBindDescriptorSets(RHI::PipelineBintPoint::GRAPHICS,
					rdg.getRasterPassNode(renderPassNodeSRGB)->pipelineLayout.get(), 0, 1, &tmp_set, 0, nullptr);

				commandbuffers[currentFrame]->cmdDrawIndexedIndirect(rdg.getIndirectDrawBufferNode(portal.indirectDrawBuffer)->storageBuffer.get(), 0, 1, sizeof(unsigned int) * 5);
			};
			scene.tree.context.traverse<ECS::TagComponent, GFX::Mesh>(mesh_processor_2);

			commandbuffers[currentFrame]->cmdEndRenderPass();

			// begin compute pass
			commandbuffers[currentFrame]->cmdPipelineBarrier(compute_barrier.get());
			commandbuffers[currentFrame]->cmdPipelineBarrier(attach_read_barrier.get());

			Size size = { 1280,720 };
			rdg.getComputePassNode(acesPass)->executeWithConstant(commandbuffers[currentFrame].get(), 40, 23, 1, 0, size);

			static float deltaTime = 0;
			deltaTime += timer.getMsPF() > 100 ? 100 : timer.getMsPF();
			while (deltaTime > 20)
			{
				commandbuffers[currentFrame]->cmdPipelineBarrier(compute_barrier_0.get());
				EmitConstant constant_1{ 400000u / 50, (float)timer.getTotalTime(), 0, 1.07 };
				rdg.getComputePassNode(portal.emitPass)->executeWithConstant(commandbuffers[currentFrame].get(), 200, 1, 1, currentFrame, constant_1);
				commandbuffers[currentFrame]->cmdPipelineBarrier(compute_barrier_0.get());
				rdg.getComputePassNode(portal.updatePass)->execute(commandbuffers[currentFrame].get(), 200, 1, 1, currentFrame);

				deltaTime -= 20;
			}

		//	commandbuffers[currentFrame]->cmdPipelineBarrier(color_attach_present2copy[imageIndex].get());
		//	commandbuffers[currentFrame]->cmdPipelineBarrier(aces_res_general2copysrc.get());

		//	commandbuffers[currentFrame]->cmdBlitImage(
		//		rdg.getTextureBufferNode(test_write_target)->getTexture(),
		//		RHI::ImageLayout::TRANSFER_SRC_OPTIMAL,
		//		rdg.getTextureBufferNode(rdg.getContainer(swapchainColorBufferFlights)->handles[imageIndex])->getTexture(),
		//		RHI::ImageLayout::TRANSFER_DST_OPTIMAL,
		//		{ {(uint32_t)RHI::ImageAspectFlagBits::COLOR_BIT,(uint32_t)RHI::ImageAspectFlagBits::COLOR_BIT} }
		//	);

		//	commandbuffers[currentFrame]->cmdPipelineBarrier(aces_res_copysrc2general.get());
		//	commandbuffers[currentFrame]->cmdPipelineBarrier(color_attach_copy2present[imageIndex].get());

			commandbuffers[currentFrame]->endRecording();
			//	4. Submit the recorded command buffer
			commandbuffers[currentFrame]->submit(nullptr, nullptr, inFlightFence[currentFrame].get());
			//commandbuffers[currentFrame]->submit(imageAvailableSemaphore[currentFrame].get(), renderFinishedSemaphore[currentFrame].get(), inFlightFence[currentFrame].get());
			////	5. Present the swap chain image
			//swapchain->present(imageIndex, renderFinishedSemaphore[currentFrame].get());
		}
		// update current frame
		{
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}

		imguiLayer->startGuiRecording();
		// demo window
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		// s
		mainViewport.onDrawGui();

		imguiLayer->render();
	}

	virtual void onShutdown() override
	{
		logicalDevice->waitIdle();
		RHI::DeviceToGlobal::removeDevice(logicalDevice.get());
	}

private:
	Timer timer;
	uint32_t currentFrame = 0;
	WindowLayer* window_layer;

	GFX::Scene scene;
	GFX::RDG::RenderGraph rdg;
	GFX::RDG::NodeHandle depthBuffer;
	GFX::RDG::NodeHandle renderPassNodeSRGB;
	GFX::RDG::NodeHandle uniformBufferFlights;
	GFX::RDG::NodeHandle swapchainColorBufferFlights;
	GFX::RDG::NodeHandle acesPass;
	GFX::RDG::NodeHandle test_write_target;
	GFX::RDG::NodeHandle srgb_framebuffer;
	ParticleSystem::ParticleSystem portal;

	MemScope<RHI::IGraphicContext> graphicContext;
	MemScope<RHI::IPhysicalDevice> physicalDevice;
	MemScope<RHI::ILogicalDevice> logicalDevice;

	MemScope<RHI::IResourceFactory> resourceFactory;

	MemScope<RHI::IStorageBuffer> torusBuffer;

	MemScope<RHI::ITexture> texture;
	MemScope<RHI::ITextureView> textureView;
	MemScope<RHI::ISampler> sampler;

	MemScope<RHI::IMemoryBarrier> compute_memory_barrier_0;
	MemScope<RHI::IMemoryBarrier> compute_drawcall_memory_barrier;
	MemScope<RHI::IImageMemoryBarrier> general2colorattach_imb;
	MemScope<RHI::IImageMemoryBarrier> colorattach2general_imb;
	MemScope<RHI::IBarrier> general2colorattach_b;
	MemScope<RHI::IBarrier> colorattach2general_b;
	MemScope<RHI::IBarrier> compute_barrier_0;
	MemScope<RHI::IBarrier> compute_drawcall_barrier;
	MemScope<RHI::IBarrier> compute_barrier;
	MemScope<RHI::IBarrier> compute_barrier_2;
	MemScope<RHI::IBarrier> attach_read_barrier;
	MemScope<RHI::IBarrier> aces_res_general2copysrc;
	MemScope<RHI::IBarrier> aces_res_copysrc2general;
	MemScope<RHI::IBarrier> color_attach_copy2present[3];
	MemScope<RHI::IBarrier> color_attach_present2copy[3];

	MemScope<RHI::ISwapChain> swapchain;
	MemScope<RHI::IRenderPass> renderPass;
	MemScope<RHI::IPipeline> pipeline;

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