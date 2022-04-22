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
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)
#include <ctime> 
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
import Core.Profiler;

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

import Asset.AssetLayer;
import Asset.RuntimeAssetManager;

import GFX.SceneTree;
import GFX.Scene;
import GFX.Mesh;
import GFX.Transform;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.Common;
import GFX.RDG.MultiDispatchScope;
import GFX.PostProcessing.AcesBloom;
import GFX.Renderer;
import GFX.RDG.RasterNodes;
import GFX.RDG.ExternalAccess;

import UAT.IUniversalApplication;

import Editor.ImGuiLayer;
import Editor.Viewport;
import Editor.Scene;
import Editor.ImImage;
import Editor.Inspector;
import Editor.RenderPipeline;
import Editor.EditorLayer;
import Editor.RDGImImageManager;

import Demo.Portal;
import Demo.SortTest;

using namespace SIByL;

const int MAX_FRAMES_IN_FLIGHT = 2;
//#define MACRO_USE_MESH 0


void Delay(int time)//time*1000为秒数 
{
	clock_t   now = clock();
	while (clock() - now < time);
}


class SandboxApp :public IUniversalApplication
{
public:
	struct UniformBufferObject {
		glm::vec4 cameraPos;
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

	GFX::RDG::NodeHandle renderPassNodeSRGB;
	GFX::RDG::NodeHandle renderPassNodeSRGB_mesh;

	virtual void onAwake() override
	{
		PROFILE_BEGIN_SESSION("onAwake", "onAwake");

		{
			PROFILE_SCOPE("create window layer")
				// create window
				WindowLayerDesc window_layer_desc = {
					SIByL::EWindowVendor::GLFW,
					1280,
					720,
					"Hello"
			};
			window_layer = attachWindowLayer(window_layer_desc);
		}

		{
			PROFILE_SCOPE("create RHI layer")
				// create device
#ifdef MACRO_USE_MESH
				graphicContext = (RHI::IFactory::createGraphicContext({
					RHI::API::VULKAN,
					(uint32_t)RHI::GraphicContextExtensionFlagBits::MESH_SHADER }));
#else
				graphicContext = (RHI::IFactory::createGraphicContext({
					RHI::API::VULKAN }));
#endif

			graphicContext->attachWindow(window_layer->getWindow());
			physicalDevice = (RHI::IFactory::createPhysicalDevice({ graphicContext.get() }));
			logicalDevice = (RHI::IFactory::createLogicalDevice({ physicalDevice.get() }));
			resourceFactory = MemNew<RHI::IResourceFactory>(logicalDevice.get());
		}

		{
			PROFILE_SCOPE("create asset layer")
				// create asset layer
				assetLayer = MemNew<Asset::AssetLayer>(resourceFactory.get());
			pushLayer(assetLayer.get());
		}

		{
			PROFILE_SCOPE("create ImGui layer")
				// create imgui layer
				imguiLayer = MemNew<Editor::ImGuiLayer>(logicalDevice.get());
			pushLayer(imguiLayer.get());
		}

		{
			PROFILE_SCOPE("create Editor layer")
				// create editor layer
			editorLayer = MemNew<Editor::EditorLayer>(window_layer, assetLayer.get(), imguiLayer.get(), &timer, &rdg);
			pushLayer(editorLayer.get());
			editorLayer->introspectionGui.bindApplication(this);
			editorLayer->introspectionGui.bindInspector(&(editorLayer->inspectorGui));
		}

		{
			PROFILE_SCOPE("create scene layer")
				// create scene
			scene.deserialize("test_scene.scene", assetLayer.get());
			editorLayer->sceneGui.bindScene(&scene);
			editorLayer->sceneGui.bindInspector(&(editorLayer->inspectorGui));
			editorLayer->pipelineGui.bindRenderGraph(&rdg);
			editorLayer->pipelineGui.bindInspector(&(editorLayer->inspectorGui));
		}
		GFX::RDG::NodeHandle srgb_color_attachment;
		{
			PROFILE_SCOPE("create RDG")
			// Create RDG register proxy
			acesbloom = MemNew<GFX::PostProcessing::AcesBloomProxyUnit>(resourceFactory.get());
			portal = std::move(Demo::PortalSystem(resourceFactory.get(), &timer));
			sortTest = std::move(Demo::SortTest(resourceFactory.get(), 100000u));

			// New Pipeline Building
			GFX::RDG::RenderGraphWorkshop workshop(rdg);
			workshop.addInternalSampler();

			// Build Up Pipeline
			GFX::RDG::RenderGraphBuilder rdg_builder(rdg);
			// sub-pipeline building helper components
			portal.registerResources(&workshop);
			sortTest.registerResources(&rdg_builder);
			acesbloom->registerResources(&rdg_builder);
			// renderer
			GFX::RDG::NodeHandle depthBuffer = rdg_builder.addDepthBuffer(1.f, 1.f);
			uniformBufferFlights = rdg_builder.addUniformBufferFlights(sizeof(UniformBufferObject));
			// raster pass sono 1
			srgb_color_attachment = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT, 1.f, 1.f, "SRGB Color Attach");
			GFX::RDG::NodeHandle srgb_depth_attachment = rdg_builder.addDepthBuffer(1.f, 1.f);
			GFX::RDG::NodeHandle srgb_framebuffer = rdg_builder.addFrameBufferRef({ srgb_color_attachment }, srgb_depth_attachment);

//			// HDR raster pass
//			renderPassNodeSRGB = rdg_builder.addRasterPass({ uniformBufferFlights, portal.samplerHandle, portal.particleBuffer, portal.samplerHandle, portal.liveIndexBuffer, portal.indirectDrawBuffer });
//			rdg.tag(renderPassNodeSRGB, "Raster HDR");
//			GFX::RDG::RasterPassNode* rasterPassNodeSRGB = rdg.getRasterPassNode(renderPassNodeSRGB);
//			rasterPassNodeSRGB->shaderVert = resourceFactory->createShaderFromBinaryFile("portal/portal_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
//			rasterPassNodeSRGB->shaderFrag = resourceFactory->createShaderFromBinaryFile("portal/portal_vert_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
//			rasterPassNodeSRGB->framebuffer = srgb_framebuffer;
//			rasterPassNodeSRGB->indirectDrawBufferHandle = portal.indirectDrawBuffer;
//			rasterPassNodeSRGB->textures = { portal.spriteHandle, portal.bakedCurveHandle };
//			rasterPassNodeSRGB->stageMasks = {
//				(uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::FRAGMENT_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT | (uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::VERTEX_BIT
//			};
//			rasterPassNodeSRGB->customCommandRecord = [&scene = scene](GFX::RDG::RasterPassNode* raster_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
//			{
//				std::function<void(ECS::TagComponent&, GFX::Mesh&, GFX::Renderer&)> per_mesh_behavior = [&](ECS::TagComponent& tag, GFX::Mesh& mesh, GFX::Renderer& renderer) {
//					if (!renderer.hasPass(1)) return;
//					commandbuffer->cmdBindVertexBuffer(mesh.vertexBuffer);
//					commandbuffer->cmdBindIndexBuffer(mesh.indexBuffer);
//					RHI::IDescriptorSet* set = raster_pass->descriptorSets[flight_idx].get();
//					commandbuffer->cmdBindDescriptorSets(
//						RHI::PipelineBintPoint::GRAPHICS,
//						raster_pass->pipelineLayout.get(),
//						0, 1, &set, 0, nullptr);
//
//					commandbuffer->cmdDrawIndexedIndirect(raster_pass->indirectDrawBuffer, 0, 1, sizeof(unsigned int) * 5);
//				};
//				scene.tree.context.traverse<ECS::TagComponent, GFX::Mesh, GFX::Renderer>(per_mesh_behavior);
//			};
//#ifdef MACRO_USE_MESH
//			// Mesh Based Raster Pass
//			renderPassNodeSRGB_mesh = rdg_builder.addRasterPassBackPool({ uniformBufferFlights, portal.samplerHandle, portal.particleBuffer, portal.samplerHandle, portal.liveIndexBuffer, portal.indirectDrawBuffer });
//			rdg.tag(renderPassNodeSRGB_mesh, "Raster HDR Mesh");
//			GFX::RDG::RasterPassNode* rasterPassNodeSRGB_mesh = rdg.getRasterPassNode(renderPassNodeSRGB_mesh);
//			rasterPassNodeSRGB_mesh->shaderFrag = resourceFactory->createShaderFromBinaryFile("fs_sampler.spv", { RHI::ShaderStage::FRAGMENT,"main" });
//			rasterPassNodeSRGB_mesh->shaderMesh = resourceFactory->createShaderFromBinaryFile("portal/portal_mesh.spv", { RHI::ShaderStage::MESH,"main" });
//			rasterPassNodeSRGB_mesh->framebuffer = srgb_framebuffer;
//			rasterPassNodeSRGB_mesh->indirectDrawBufferHandle = portal.indirectDrawBuffer;
//			rasterPassNodeSRGB_mesh->textures = { portal.spriteHandle, portal.bakedCurveHandle };
//			rasterPassNodeSRGB_mesh->stageMasks = {
//				(uint32_t)RHI::ShaderStageFlagBits::MESH_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::FRAGMENT_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::COMPUTE_BIT | (uint32_t)RHI::ShaderStageFlagBits::MESH_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::MESH_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::MESH_BIT,
//				(uint32_t)RHI::ShaderStageFlagBits::MESH_BIT
//			};
//			rasterPassNodeSRGB_mesh->customCommandRecord = [&scene = scene](GFX::RDG::RasterPassNode* raster_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
//			{
//				std::function<void(ECS::TagComponent&, GFX::Mesh&)> per_mesh_behavior = [&](ECS::TagComponent& tag, GFX::Mesh& mesh) {
//					commandbuffer->cmdBindVertexBuffer(mesh.vertexBuffer.get());
//					commandbuffer->cmdBindIndexBuffer(mesh.indexBuffer.get());
//					RHI::IDescriptorSet* set = raster_pass->descriptorSets[flight_idx].get();
//					commandbuffer->cmdBindDescriptorSets(
//						RHI::PipelineBintPoint::GRAPHICS,
//						raster_pass->pipelineLayout.get(),
//						0, 1, &set, 0, nullptr);
//
//					commandbuffer->cmdDrawMeshTasks(100000u / 16, 0);
//				};
//				scene.tree.context.traverse<ECS::TagComponent, GFX::Mesh>(per_mesh_behavior);
//			};
//#endif

//			// particle system update pass
//			GFX::RDG::NodeHandle scope_begin = rdg_builder.beginMultiDispatchScope("Scope Begin Portal-Update Multi-Dispatch");
//			rdg.getMultiDispatchScope(scope_begin)->customDispatchCount = [&timer = timer]()
//			{
//				static float deltaTime = 0;
//				deltaTime += timer.getMsPF() > 100 ? 100 : timer.getMsPF();
//				unsigned int dispatch_times = (unsigned int)(deltaTime / 20);
//				deltaTime -= dispatch_times * 20;
//				return dispatch_times;
//			};
//			portal.registerUpdatePasses(&rdg_builder);
//			sortTest.registerUpdatePasses(&rdg_builder);
//			GFX::RDG::NodeHandle scope_end = rdg_builder.endScope();

			// Raster Pass "Opaque Pass"
			workshop.addRasterPassScope("Opaque Pass", srgb_framebuffer);
			GFX::RDG::RasterPipelineScope* opaque_phongs_pipeline = workshop.addRasterPipelineScope("Opaque Pass", "Phongs");
			{
				opaque_phongs_pipeline->shaderVert = resourceFactory->createShaderFromBinaryFile("pbr/phong_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
				opaque_phongs_pipeline->shaderFrag = resourceFactory->createShaderFromBinaryFile("pbr/phong_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
				opaque_phongs_pipeline->cullMode = RHI::CullMode::NONE;
				opaque_phongs_pipeline->vertexBufferLayout =
				{
					{RHI::DataType::Float3, "Position"},
					{RHI::DataType::Float3, "Normal"},
					{RHI::DataType::Float2, "UV"},
					{RHI::DataType::Float4, "Tangent"},
				};
				// Add Materials
				auto dust2_01_mat_scope = workshop.addRasterMaterialScope("Opaque Pass", "Phongs", "dust2_01");
			}
			// Compute Pass "Post Processing"
			workshop.addComputePassScope("PostProcessing Pass");
			// Add Pipeline "ACES + BLOOM"
			acesbloom->iHdrImage = srgb_color_attachment;
			acesbloom->iExternalSampler = workshop.getInternalSampler("Default Sampler");
			acesbloom->registerComputePasses(workshop);

			// External Access Pass "ImGui Read Pass"
			auto imGuiReadPassHandle = workshop.addExternalAccessPass("ImGui Read Pass");
			auto imGuiReadPass = workshop.getNode<GFX::RDG::ExternalAccessPass>(imGuiReadPassHandle);
			imGuiReadPass->insertExternalAccessItem({ acesbloom->bloomCombined, GFX::RDG::ConsumeKind::IMAGE_SAMPLE });

			// Pipeline Build
			workshop.build(resourceFactory.get(), 1280, 720);

			auto imimage = editorLayer->imImageManager.addImImage(acesbloom->bloomCombined, workshop.getInternalSampler("Default Sampler"));
			editorLayer->mainViewport.bindImImage(imimage);

			//// building ...
			//rdg_builder.build(resourceFactory.get(), 1280, 720);
			//rdg.print();
#ifdef MACRO_USE_MESH
			rasterPassNodeSRGB_mesh->barriers = rasterPassNodeSRGB->barriers;
#endif
		}

		{
			PROFILE_SCOPE("create Misc");

			// comand stuffs
			commandPool = resourceFactory->createCommandPool({ RHI::QueueType::GRAPHICS, (uint32_t)RHI::CommandPoolAttributeFlagBits::RESET });
			commandbuffers.resize(MAX_FRAMES_IN_FLIGHT);
			inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				commandbuffers[i] = resourceFactory->createCommandBuffer(commandPool.get());
				inFlightFence[i] = resourceFactory->createFence();
			}

			// timer
			timer.start();
		}

		//{
		//	PROFILE_SCOPE("run First Pass")

		//		MemScope<RHI::IMemoryBarrier> compute_compute_memory_barrier = resourceFactory->createMemoryBarrier({
		//			(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
		//			(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT
		//			});
		//	MemScope<RHI::IBarrier> compute_compute_barrier = resourceFactory->createBarrier({
		//		(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
		//		(uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT,
		//		0,
		//		{compute_compute_memory_barrier.get()}
		//		}
		//	);

		//	// init storage buffer
		//	RHI::ICommandPool* transientPool = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getTransientCommandPool();
		//	MemScope<RHI::ICommandBuffer> transientCommandbuffer = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getResourceFactory()->createCommandBuffer(transientPool);
		//	transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		//	rdg.getComputePassNode(portal.initPass)->executeWithConstant(transientCommandbuffer.get(), 200, 1, 1, 0, 100000u);
		//	// Test
		//	rdg.getComputePassNode(sortTest.sortInit)->executeWithConstant(transientCommandbuffer.get(), GRIDSIZE(sortTest.elementCount, 1024), 1, 1, 0, sortTest.elementCount);
		//	transientCommandbuffer->cmdPipelineBarrier(compute_compute_barrier.get());
		//	rdg.getComputePassNode(sortTest.sortHistogramSubgroup_8_4)->execute(transientCommandbuffer.get(), GRIDSIZE(sortTest.elementCount, sortTest.subgroupHistogramElementPerBlock), 1, 1, 0);
		//	transientCommandbuffer->cmdPipelineBarrier(compute_compute_barrier.get());
		//	rdg.getComputePassNode(sortTest.sortHistogramIntegrate_8_4)->execute(transientCommandbuffer.get(), 1, 1, 1, 0);
		//	for (uint32_t i = 0; i < 4; i++)
		//	{
		//		//transientCommandbuffer->cmdPipelineBarrier(compute_compute_barrier.get());
		//		//rdg.getComputePassNode(sortTest.sortPassClear)->execute(transientCommandbuffer.get(), 1, 1, 1, 0);
		//		transientCommandbuffer->cmdPipelineBarrier(compute_compute_barrier.get());
		//		//rdg.getComputePassNode(sortTest.sortPass)->executeWithConstant(transientCommandbuffer.get(), sortTest.possibleDigitValue * GRIDSIZE(sortTest.elementCount,2048), 1, 1, 0, i);
		//		rdg.getComputePassNode(sortTest.sortPass)->executeWithConstant(transientCommandbuffer.get(), GRIDSIZE(sortTest.elementCount, sortTest.elementPerBlock), 1, 1, 0, i);
		//	}
		//	transientCommandbuffer->cmdPipelineBarrier(compute_compute_barrier.get());
		//	rdg.getComputePassNode(sortTest.sortShowKeys)->execute(transientCommandbuffer.get(), GRIDSIZE(sortTest.elementCount, 1024), 1, 1, 0);
		//	// ~Test
		//	transientCommandbuffer->endRecording();
		//	transientCommandbuffer->submit();
		//	logicalDevice->waitIdle();
		//}

		SE_CORE_INFO("OnAwake End");
		PROFILE_END_SESSION();
	}

	virtual auto onWindowResize(WindowResizeEvent& e) -> bool override
	{
		logicalDevice->waitIdle();
		return false;
	}

	virtual auto onKeyPressedEvent(KeyPressedEvent& e) -> bool override
	{
		int keycode = e.getKeyCode();
		if (keycode == 32)
		{
			for (int i = 0; i < rdg.passes.size(); i++)
			{
				if (rdg.passes[i] == renderPassNodeSRGB)
				{
					for (int j = 0; j < rdg.passesBackPool.size(); j++)
					{
						if (rdg.passesBackPool[j] == renderPassNodeSRGB_mesh)
						{
							rdg.passes[i] = renderPassNodeSRGB_mesh;
							rdg.passesBackPool[j] = renderPassNodeSRGB;
							return false;
						}
					}
				}
				else if (rdg.passes[i] == renderPassNodeSRGB_mesh)
				{
					for (int j = 0; j < rdg.passesBackPool.size(); j++)
					{
						if (rdg.passesBackPool[j] == renderPassNodeSRGB_mesh)
						{
							rdg.passes[i] = renderPassNodeSRGB;
							rdg.passesBackPool[j] = renderPassNodeSRGB_mesh;
							return false;
						}
					}
				}
			}
			SE_CORE_DEBUG("Change Render Pipeline~");
		}
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
		}
		// update scene transform propagation
		{
			scene.tree.updateTransforms();
		}
		// update uniform buffer
		{
			////auto [width, height] = swapchain->getExtend();
			//float time = (float)timer.getTotalTimeSeconds();
			//float rotation = time;// 0.5 * 3.1415926;
			// get transform from viewport
			GFX::Transform camera_transform = editorLayer->mainViewport.cameraTransform;
			GFX::RDG::PerViewUniformBuffer per_view_ubo;
			per_view_ubo.cameraPos = glm::vec4(camera_transform.getTranslation(), 1);
			per_view_ubo.view = glm::lookAtLH(camera_transform.getTranslation(), camera_transform.getTranslation() + camera_transform.getRotatedForward(), glm::vec3(0.0f, 1.0f, 0.0f));
			per_view_ubo.proj = glm::perspectiveLH_NO(glm::radians(45.0f), (float)editorLayer->mainViewport.getWidth() / (float)editorLayer->mainViewport.getHeight(), 0.1f, 5000.0f);
			per_view_ubo.proj[1][1] *= -1;
			// use transform as Per View Uniform
			auto opaque_pass = rdg.getRasterPassScope("Opaque Pass");
			opaque_pass->updatePerViewUniformBuffer(per_view_ubo, currentFrame);
		}
		// update draw calls
		{
			rdg.onFrameStart();
			auto dust2_01_mat_scope = rdg.getRasterMaterialScope("Opaque Pass", "Phongs", "dust2_01");

			std::function<void(ECS::TagComponent&, GFX::Transform&, GFX::Mesh&)> per_mesh_behavior = [&](ECS::TagComponent& tag, GFX::Transform& transform, GFX::Mesh& mesh) {
				auto drawcall_handle = dust2_01_mat_scope->addRasterDrawCall(tag.Tag, &rdg);
				auto drawcall = rdg.getNode<GFX::RDG::RasterDrawCall>(drawcall_handle);

				drawcall->vertexBuffer = mesh.vertexBuffer;
				drawcall->indexBuffer = mesh.indexBuffer;
				drawcall->uniform.model = transform.getAccumulativeTransform();
			};
			scene.tree.context.traverse<ECS::TagComponent, GFX::Transform, GFX::Mesh>(per_mesh_behavior);
		}
		// drawFrame
		{
			// record a command buffer which draws the scene onto that image
			commandbuffers[currentFrame]->reset();
			commandbuffers[currentFrame]->beginRecording();

			// render pass
			//rdg.recordCommands(commandbuffers[currentFrame].get(), currentFrame);
			rdg.recordCommandsNEW(commandbuffers[currentFrame].get(), currentFrame);

			// submit the recorded command buffer
			commandbuffers[currentFrame]->endRecording();
			commandbuffers[currentFrame]->submit(nullptr, nullptr, inFlightFence[currentFrame].get());
		}
		// update current frame
		{
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}

		imguiLayer->startGuiRecording();
		// demo window
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		editorLayer->onDrawGui();
		imguiLayer->render();

		bool needReize = editorLayer->mainViewport.getNeedResize();
		if(needReize)
		{
			logicalDevice->waitIdle();
			GFX::RDG::RenderGraphWorkshop workshop(rdg);
			workshop.build(resourceFactory.get(), editorLayer->mainViewport.getWidth(), editorLayer->mainViewport.getHeight());
			editorLayer->imImageManager.invalidAll();
		}
	}

	virtual void onShutdown() override
	{
		logicalDevice->waitIdle();
		RHI::DeviceToGlobal::removeDevice(logicalDevice.get());
	}

private:
	Timer timer;
	// Window Layer
	WindowLayer* window_layer;
	// Editor Layer
	MemScope<Asset::AssetLayer> assetLayer;
	// AssetLayer
	// Editor Layer
	MemScope<Editor::ImGuiLayer> imguiLayer;
	MemScope<Editor::EditorLayer> editorLayer;
	MemScope<Editor::ImImage> viewportImImage;

	// Application
	GFX::Scene scene;
	GFX::RDG::RenderGraph rdg;


	// RDG Proxy
	Demo::PortalSystem portal;
	MemScope<GFX::PostProcessing::AcesBloomProxyUnit> acesbloom;
	Demo::SortTest sortTest;

	// Device stuff
	MemScope<RHI::IGraphicContext> graphicContext;
	MemScope<RHI::IPhysicalDevice> physicalDevice;
	MemScope<RHI::ILogicalDevice> logicalDevice;
	MemScope<RHI::IResourceFactory> resourceFactory;

	// Comands
	MemScope<RHI::ICommandPool> commandPool;
	std::vector<MemScope<RHI::ICommandBuffer>> commandbuffers;
	std::vector<MemScope<RHI::IFence>> inFlightFence;
	uint32_t currentFrame = 0;

	// Lifetime Resources
	GFX::RDG::NodeHandle uniformBufferFlights;
};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	return app;
}