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
import Core.String;

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
import RHI.IQueryPool;

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
import GFX.RDG.Agency;
import GFX.HiZAgency;
import GFX.BoundingBox;

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
import Demo.Grass;
import Utils.DepthBaker;

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
			graphicContext = (RHI::IFactory::createGraphicContext({
				RHI::API::VULKAN,
				(uint32_t)RHI::GraphicContextExtensionFlagBits::QUERY
				| (uint32_t)RHI::GraphicContextExtensionFlagBits::MESH_SHADER
				| (uint32_t)RHI::GraphicContextExtensionFlagBits::SHADER_INT8
				}));

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
			grass = std::move(Demo::GrassSystem(resourceFactory.get(), &timer));
			//depthBaker = Utils::DepthBaker(resourceFactory.get());
			portal.entity = scene.tree.getNodeEntity("particle");
			//sortTest = std::move(Demo::SortTest(resourceFactory.get(), 100000u));

			// New Pipeline Building
			GFX::RDG::RenderGraphWorkshop workshop(rdg);
			workshop.addInternalSampler();

			// Build Up Pipeline
			GFX::RDG::RenderGraphBuilder rdg_builder(rdg);
			// sub-pipeline building helper components
			portal.registerResources(&workshop);
			//depthBaker.registerResources(&workshop);
			grass.registerResources(&workshop);
			//sortTest.registerResources(&rdg_builder);
			acesbloom->registerResources(workshop);
			// raster pass sono 1
			srgb_color_attachment = workshop.addColorBuffer(RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT, 1.f, 1.f, "SRGB Color Attach");
			GFX::RDG::NodeHandle raster_atomic_r = workshop.addColorBuffer(RHI::ResourceFormat::FORMAT_R32_UINT, 1.f, 1.f, "Raster Atomic R");
			//GFX::RDG::NodeHandle srgb_depth_attachment = rdg_builder.addDepthBuffer(1.f, 1.f);
			GFX::RDG::NodeHandle srgb_depth_attachment = workshop.addColorBuffer(RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT, 1.f, 1.f, "SRGB Depth Attach");
			GFX::RDG::NodeHandle prez_framebuffer = rdg_builder.addFrameBufferRef({  }, srgb_depth_attachment);
			GFX::RDG::NodeHandle srgb_framebuffer = rdg_builder.addFrameBufferRef({ srgb_color_attachment }, srgb_depth_attachment, { srgb_depth_attachment });

			portal.rasterColorAttachment = raster_atomic_r;
			//depthBaker.registerRenderPasses(&workshop);

			// Raster Pass "Pre Z"
			workshop.addRasterPassScope("Pre-Z Pass", prez_framebuffer);
			GFX::RDG::RasterPipelineScope* prez_opaque_pipeline = workshop.addRasterPipelineScope("Pre-Z Pass", "Opaque");
			{
				prez_opaque_pipeline->shaderVert = resourceFactory->createShaderFromBinaryFile("pbr/prez_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
				prez_opaque_pipeline->shaderFrag = resourceFactory->createShaderFromBinaryFile("pbr/prez_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
				prez_opaque_pipeline->cullMode = RHI::CullMode::NONE;
				prez_opaque_pipeline->vertexBufferLayout =
				{
					{RHI::DataType::Float3, "Position"},
					{RHI::DataType::Float3, "Normal"},
					{RHI::DataType::Float2, "UV"},
					{RHI::DataType::Float4, "Tangent"},
				};
				// Add Materials
				auto only_mat_scope = workshop.addRasterMaterialScope("Pre-Z Pass", "Opaque", "Only");
			}

			// Compute HiZ Pass
			auto hiz_agency = GFX::HiZAgency::createInstance(srgb_depth_attachment, resourceFactory.get());
			workshop.addAgency(std::move(hiz_agency));
			portal.rasterDepthAttachment = ((GFX::HiZAgency*)(rdg.agencies[0].get()))->depthSampleHandle;
			
			// Create Cluster Pass
			workshop.addComputePassScope("Particle Bounding Boxes");

			// Raster Pass "Forward Pass"
			auto forward_pass = workshop.addRasterPassScope("Forward Pass", srgb_framebuffer);
			GFX::RDG::RasterPipelineScope* opaque_phongs_pipeline = workshop.addRasterPipelineScope("Forward Pass", "Phongs");
			{
				opaque_phongs_pipeline->shaderVert = resourceFactory->createShaderFromBinaryFile("pbr/phong_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
				opaque_phongs_pipeline->shaderFrag = resourceFactory->createShaderFromBinaryFile("pbr/phong_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
				opaque_phongs_pipeline->cullMode = RHI::CullMode::NONE;
				opaque_phongs_pipeline->depthStencilDesc = RHI::TestLessEqualAndWrite;
				opaque_phongs_pipeline->vertexBufferLayout =
				{
					{RHI::DataType::Float3, "Position"},
					{RHI::DataType::Float3, "Normal"},
					{RHI::DataType::Float2, "UV"},
					{RHI::DataType::Float4, "Tangent"},
				};
				// Add Materials
				auto default_sampler = workshop.getInternalSampler("Default Sampler");
				uint64_t texture_guids[] = {
					2267850664238763130,
					2267991406240189562,
					2267850670916095098,
					13249315412305628282,
					12673417609955626106,
					660628553898249338,
					11971981967992671354,
					14779131910728043642,
					2267921049340726394,
					13249385790730259578,
					12673487988380257402,
				};
				for (int i = 1; i <= 11; i++)
				{
					auto dust2_mat_scope = workshop.addRasterMaterialScope("Forward Pass", "Phongs", "dust2_" + std::to_string(i));
					Asset::Texture* texture = assetLayer->texture(texture_guids[i - 1]);
					auto dust2_01_texture_handle = workshop.addColorBufferExt(texture->texture.get(), texture->view.get(), "dust2-" + std::to_string(texture_guids[i]));
					dust2_mat_scope->resources = { default_sampler };
					dust2_mat_scope->sampled_textures = { dust2_01_texture_handle };
				}
			}
			// Raster Pass "Transparency Pass"
			grass.registerRenderPasses(&workshop);
			portal.registerBoundingBoxesPasses(&workshop);
			portal.registerRenderPasses(&workshop);

			// Compute Pass "Post Processing"
			workshop.addComputePassScope("PostProcessing Pass");
			// Add Pipeline "ACES + BLOOM"
			acesbloom->iHdrImage = srgb_color_attachment;
			acesbloom->iExternalSampler = workshop.getInternalSampler("Default Sampler");
			acesbloom->registerComputePasses(workshop);
			// Compute Pass "Portal Particle System Update"
			portal.registerUpdatePasses(&workshop);

			// External Access Pass "ImGui Read Pass"
			auto imGuiReadPassHandle = workshop.addExternalAccessPass("ImGui Read Pass");
			auto imGuiReadPass = workshop.getNode<GFX::RDG::ExternalAccessPass>(imGuiReadPassHandle);
			imGuiReadPass->insertExternalAccessItem({ acesbloom->bloomCombined, GFX::RDG::ConsumeKind::IMAGE_SAMPLE });

			// Pipeline Build
			workshop.build(resourceFactory.get(), 1280, 720);
			rdg.print();

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
			queryPools.resize(MAX_FRAMES_IN_FLIGHT);
			inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);
			for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				commandbuffers[i] = resourceFactory->createCommandBuffer(commandPool.get());
				queryPools[i] = resourceFactory->createQueryPool({ RHI::QueryType::TIMESTAMP, 4 });
				inFlightFence[i] = resourceFactory->createFence();
			}

			// timer
			timer.start();
		}

		SE_CORE_INFO("OnAwake End");
		PROFILE_END_SESSION();
	}

	virtual auto onWindowResize(WindowResizeEvent& e) -> bool override
	{
		logicalDevice->waitIdle();
		return false;
	}

	bool needChangeRasterPipeline = false;
	virtual auto onKeyPressedEvent(KeyPressedEvent& e) -> bool override
	{
		int keycode = e.getKeyCode();
		if (keycode == 32)
		{
			needChangeRasterPipeline = true;
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
			// get transform from viewport
			GFX::Transform camera_transform = editorLayer->mainViewport.cameraTransform;
			GFX::RDG::PerViewUniformBuffer per_view_ubo;
			per_view_ubo.cameraPos = glm::vec4(camera_transform.getTranslation(), 1);
			per_view_ubo.view = glm::lookAtLH(camera_transform.getTranslation(), camera_transform.getTranslation() + camera_transform.getRotatedForward(), glm::vec3(0.0f, 1.0f, 0.0f));
			per_view_ubo.proj = glm::perspectiveLH_NO(glm::radians(45.0f), (float)editorLayer->mainViewport.getWidth() / (float)editorLayer->mainViewport.getHeight(), 0.1f, 5000.0f);
			per_view_ubo.proj[1][1] *= -1;
			glm::vec4 test = per_view_ubo.proj * glm::vec4(50, 0, 5000.0f, 1);
			// use transform as Per View Uniform
			auto prez_pass = rdg.getRasterPassScope("Pre-Z Pass");
			prez_pass->updatePerViewUniformBuffer(per_view_ubo, currentFrame);
			auto opaque_pass = rdg.getRasterPassScope("Forward Pass");
			opaque_pass->updatePerViewUniformBuffer(per_view_ubo, currentFrame);

			//GFX::RDG::PerViewUniformBuffer per_view_ubo_baker;
			//per_view_ubo_baker.cameraPos = glm::vec4(0,0,0,0);
			//per_view_ubo_baker.view = glm::lookAtLH(glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			//per_view_ubo_baker.proj = glm::ortho(-160.0f, 160.0f, -160.0f, 160.0f, -20.0f, 20.0f);
			//per_view_ubo_baker.proj[1][1] *= -1;

			//glm::vec4 test_01_01 = per_view_ubo_baker.view * glm::vec4(0, 29, 0, 1);
			//glm::vec4 test_01 = per_view_ubo_baker.proj * test_01_01;
			//glm::vec4 test_02_01 = per_view_ubo_baker.view * glm::vec4(0, -5, 0, 1);
			//glm::vec4 test_02 = per_view_ubo_baker.proj * test_02_01;

			//auto baker_pass = rdg.getRasterPassScope("Depth Baker Pass");
			//baker_pass->updatePerViewUniformBuffer(per_view_ubo_baker, currentFrame);
		}
		// 
		{
			uint64_t timestamp_pairs[4];
			bool isReady = queryPools[currentFrame]->fetchResult(0, 4, timestamp_pairs);
			float time_on_drawmesh = 0.000001 * (timestamp_pairs[1] - timestamp_pairs[0]) * physicalDevice->getTimestampPeriod();
			float time_on_softraster = 0.000001 * (timestamp_pairs[3] - timestamp_pairs[2]) * physicalDevice->getTimestampPeriod();
			editorLayer->statisticsGui.portalDrawcallTime = time_on_drawmesh;
			editorLayer->statisticsGui.portalSoftDrawTime = time_on_softraster;
		}
		// update draw calls
		{
			rdg.onFrameStart();
			auto dust2_01_mat_scope = rdg.getRasterMaterialScope("Forward Pass", "Phongs", "dust2_1");
			auto portal_mat_scope = rdg.getRasterMaterialScope("Forward Pass", "Particle Portal", "Portal");
			auto portal_mat_mesh_scope = rdg.getRasterMaterialScope("Forward Pass", "Particle Portal Mesh", "Portal");
			auto grass_mat_scope = rdg.getRasterMaterialScope("Forward Pass", "Particle Grass", "Grass");
			auto portal_mat_mesh_culling_scope = rdg.getRasterMaterialScope("Forward Pass", "Particle Portal Mesh Culling", "Portal");

			{
				auto vis_mat_scope = rdg.getRasterMaterialScope("Forward Pass", "Vis AABB Portal", "Portal");
				auto drawcall_handle = vis_mat_scope->addRasterDrawCall("Vis", &rdg);
				auto drawcall = rdg.getNode<GFX::RDG::RasterDrawCall>(drawcall_handle);
				drawcall->kind = GFX::RDG::DrawCallKind::MeshTasks;
				drawcall->taskCount = GRIDSIZE(100000u, (16 * 8));
				drawcall->pushConstant = [&viewport= editorLayer->mainViewport](Buffer& buffer) {
					glm::vec2 screen_size{ 1.f/viewport.getWidth(), 1.f/viewport.getHeight() };
					buffer = std::move(Buffer(sizeof(screen_size), 1));
					memcpy(buffer.getData(), &screen_size, sizeof(screen_size));
				};

			}
			// soft raster
			{
				auto particle_culling_info_dispatch_scope = portal.softrasterDispatch;
				particle_culling_info_dispatch_scope->queryPool = queryPools[currentFrame].get();
				particle_culling_info_dispatch_scope->stageBitsQueryBeforeCmdRecord = RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT;
				particle_culling_info_dispatch_scope->stageBitsQueryAfterCmdRecord = RHI::PipelineStageFlagBits::BOTTOM_OF_PIPE_BIT;
				particle_culling_info_dispatch_scope->idxQueryBeforeCmdRecord = 2;
				particle_culling_info_dispatch_scope->idxQueryAfterCmdRecord = 3;
				particle_culling_info_dispatch_scope->pushConstant = [&viewport = editorLayer->mainViewport](Buffer& buffer) {
					glm::vec2 screen_size{ viewport.getWidth(), viewport.getHeight() };
					buffer = std::move(Buffer(sizeof(screen_size), 1));
					memcpy(buffer.getData(), &screen_size, sizeof(screen_size));
				};

				portal.cullInfoDispatch->pushConstant = [&viewport = editorLayer->mainViewport](Buffer& buffer) {
					glm::vec2 screen_size{ viewport.getWidth(), viewport.getHeight() };
					buffer = std::move(Buffer(sizeof(screen_size), 1));
					memcpy(buffer.getData(), &screen_size, sizeof(screen_size));
				};
			}
			std::function<void(ECS::TagComponent&, GFX::Transform&, GFX::Mesh&, GFX::Renderer&)> per_particle_system_behavior = [&](ECS::TagComponent& tag, GFX::Transform& transform, GFX::Mesh& mesh, GFX::Renderer& renderer) {
				for (auto& subrenderer : renderer.subRenderers)
				{
					auto mat_scope = rdg.getRasterMaterialScope(subrenderer.passName, subrenderer.pipelineName, subrenderer.materialName);
					if (mat_scope == nullptr) continue;
					auto drawcall_handle = mat_scope->addRasterDrawCall(tag.Tag, &rdg);
					auto drawcall = rdg.getNode<GFX::RDG::RasterDrawCall>(drawcall_handle);

					drawcall->vertexBuffer = mesh.vertexBuffer;
					drawcall->indexBuffer = mesh.indexBuffer;
					drawcall->uniform.model = transform.getAccumulativeTransform();

					if (mat_scope == portal_mat_mesh_scope)
					{
						drawcall->queryPool = queryPools[currentFrame].get();
						drawcall->stageBitsQueryBeforeCmdRecord = RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT;
						drawcall->stageBitsQueryAfterCmdRecord = RHI::PipelineStageFlagBits::BOTTOM_OF_PIPE_BIT;
						drawcall->idxQueryBeforeCmdRecord = 0;
						drawcall->idxQueryAfterCmdRecord = 1;

						portal.emitDispatch->pushConstant = [&timer = timer, &transform = transform](Buffer& buffer) {
							Demo::EmitConstant emitConstant = { transform.getAccumulativeTransform(), 400000u / 50,(float)timer.getTotalTime() };
							buffer = std::move(Buffer(sizeof(emitConstant), 1));
							memcpy(buffer.getData(), &emitConstant, sizeof(emitConstant));
						};
						drawcall->kind = GFX::RDG::DrawCallKind::MeshTasks;
						drawcall->taskCount = 100000u / 16;
					}
					if (mat_scope == grass_mat_scope)
					{
						drawcall->instanceCount = 65536;
					}
					if (mat_scope == portal_mat_scope)
					{
						drawcall->queryPool = queryPools[currentFrame].get();
						drawcall->stageBitsQueryBeforeCmdRecord = RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT;
						drawcall->stageBitsQueryAfterCmdRecord = RHI::PipelineStageFlagBits::BOTTOM_OF_PIPE_BIT;
						drawcall->idxQueryBeforeCmdRecord = 0;
						drawcall->idxQueryAfterCmdRecord = 1;

						portal.emitDispatch->pushConstant = [&timer = timer, &transform = transform](Buffer& buffer) {
							Demo::EmitConstant emitConstant = { transform.getAccumulativeTransform(), 400000u / 50,(float)timer.getTotalTime() };
							buffer = std::move(Buffer(sizeof(emitConstant), 1));
							memcpy(buffer.getData(), &emitConstant, sizeof(emitConstant));
						};
						drawcall->indirectDrawBuffer = rdg.getNode<GFX::RDG::StorageBufferNode>(portal.indirectDrawBuffer)->getStorageBuffer();
					}
					if (mat_scope == portal_mat_mesh_culling_scope)
					{
						drawcall->queryPool = queryPools[currentFrame].get();
						drawcall->stageBitsQueryBeforeCmdRecord = RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT;
						drawcall->stageBitsQueryAfterCmdRecord = RHI::PipelineStageFlagBits::BOTTOM_OF_PIPE_BIT;
						drawcall->idxQueryBeforeCmdRecord = 0;
						drawcall->idxQueryAfterCmdRecord = 1;

						portal.emitDispatch->pushConstant = [&timer = timer, &transform = transform](Buffer& buffer) {
							Demo::EmitConstant emitConstant = { transform.getAccumulativeTransform(), 400000u / 50,(float)timer.getTotalTime() };
							buffer = std::move(Buffer(sizeof(emitConstant), 1));
							memcpy(buffer.getData(), &emitConstant, sizeof(emitConstant));
						};
						portal.emitDispatch->pushConstant = [&timer = timer, &transform = transform](Buffer& buffer) {
							Demo::EmitConstant emitConstant = { transform.getAccumulativeTransform(), 400000u / 50,(float)timer.getTotalTime() };
							buffer = std::move(Buffer(sizeof(emitConstant), 1));
							memcpy(buffer.getData(), &emitConstant, sizeof(emitConstant));
						};
						drawcall->kind = GFX::RDG::DrawCallKind::MeshTasks;
						drawcall->taskCount = 100000u / (16 * 32);
						drawcall->pushConstant = [&viewport = editorLayer->mainViewport](Buffer& buffer) {
							glm::vec2 screen_size{ viewport.getWidth(), viewport.getHeight() };
							buffer = std::move(Buffer(sizeof(screen_size), 1));
							memcpy(buffer.getData(), &screen_size, sizeof(screen_size));
						};
					}
				}
			};
			scene.tree.context.traverse<ECS::TagComponent, GFX::Transform, GFX::Mesh, GFX::Renderer>(per_particle_system_behavior);
		}
		// drawFrame
		{
			// record a command buffer which draws the scene onto that image
			commandbuffers[currentFrame]->reset();
			commandbuffers[currentFrame]->beginRecording();
			commandbuffers[currentFrame]->cmdResetQueryPool(queryPools[currentFrame].get(), 0, 4);

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
		logicalDevice->waitIdle();

		if (needChangeRasterPipeline)
		{
			auto& renderer = portal.entity.getComponent<GFX::Renderer>();
			if (renderer.subRenderers[0].pipelineName == "Particle Portal" && renderer.subRenderers[0].materialName == "Portal")
			{
				renderer.subRenderers[0].pipelineName = "Particle Portal Mesh";
				renderer.subRenderers[0].materialName = "Portal";
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal")->isActive = false;
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal Mesh")->isActive = true;
				SE_CORE_INFO("Current Raster: Mesh!");
			}
			else if (renderer.subRenderers[0].pipelineName == "Particle Portal Mesh" && renderer.subRenderers[0].materialName == "Portal")
			{
				renderer.subRenderers[0].pipelineName = "Particle Portal Mesh Culling";
				renderer.subRenderers[0].materialName = "Portal";
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal Mesh")->isActive = false;
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal Mesh Culling")->isActive = true;
				SE_CORE_INFO("Current Raster: Mesh Culling!");
			}
			else if (renderer.subRenderers[0].pipelineName == "Particle Portal Mesh Culling" && renderer.subRenderers[0].materialName == "Portal")
			{
				renderer.subRenderers[0].pipelineName = "Particle Portal";
				renderer.subRenderers[0].materialName = "Portal";
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal Mesh Culling")->isActive = false;
				rdg.getRasterPipelineScope("Forward Pass", "Particle Portal")->isActive = true;
				SE_CORE_INFO("Current Raster: Vertex!");
			}
		}
		bool needReize = editorLayer->mainViewport.getNeedResize();
		if (needReize | needChangeRasterPipeline)
		{
			SE_CORE_DEBUG("Resize");
			logicalDevice->waitIdle();
			GFX::RDG::RenderGraphWorkshop workshop(rdg);
			workshop.build(resourceFactory.get(), editorLayer->mainViewport.getWidth(), editorLayer->mainViewport.getHeight());
			editorLayer->imImageManager.invalidAll();

			needChangeRasterPipeline = false;
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
	Demo::GrassSystem grass;
	Utils::DepthBaker depthBaker;

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
	std::vector<MemScope<RHI::IQueryPool>> queryPools;
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