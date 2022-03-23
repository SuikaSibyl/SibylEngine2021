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
import GFX.RDG.Common;
import GFX.RDG.MultiDispatchScope;

import ParticleSystem.ParticleSystem;
import ParticleSystem.PrecomputedSample;

import UAT.IUniversalApplication;
import Editor.ImGuiLayer;
import Editor.Viewport;
import Editor.ImFactory;
import Editor.ImImage;

import Interpolator.Hermite;
import Interpolator.Sampler;

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

	struct BlurPassConstants
	{
		glm::vec2 outputSize;
		glm::vec2 globalTextSize;
		glm::vec2 textureBlurInputSize;
		glm::vec2 blurDir;
	};

	struct BloomCombineConstant
	{
		glm::vec2 size;
		float para;
	};


	MemScope<Editor::ImGuiLayer> imguiLayer;
	Editor::Viewport mainViewport;
	MemScope<Editor::ImImage> viewportImImage;
	std::vector<float> samplesUniform01;
	std::vector<float> alpha_random_samplesUniform01;

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
		Image baked_image("./assets/portal_bake.tga");
		baked_texture = resourceFactory->createTexture(&baked_image);
		baked_textureView = resourceFactory->createTextureView(baked_texture.get());

		// load precomputed samples
		Buffer torusSamples;
		Buffer* samples[] = { &torusSamples };
		EmptyHeader header;
		CacheBrain::instance()->loadCache(2267996151488940154, header, samples);
		torusBuffer = resourceFactory->createStorageBuffer(&torusSamples);
		
		// Build Up Pipeline
		GFX::RDG::RenderGraphBuilder rdg_builder(rdg);
		// sub-pipeline building helper components
		// - particle system
		portal.init(sizeof(float) * 4 * 4, 100000, shaderPortalInit.get(), shaderPortalEmit.get(), shaderPortalUpdate.get());
		portal.timer = &timer;
		// - 1. Register resources
		portal.addEmitterSamples(torusBuffer.get());
		GFX::RDG::NodeHandle external_texture = rdg_builder.addColorBufferExt(texture.get(), textureView.get(), "Sparkle Sprite");
		GFX::RDG::NodeHandle external_baked_texture = rdg_builder.addColorBufferExt(baked_texture.get(), baked_textureView.get(), "Baked Curves");
		GFX::RDG::NodeHandle external_sampler = rdg_builder.addSamplerExt(sampler.get());
		portal.sampler = external_sampler;
		portal.dataBakedImage = external_baked_texture;
		portal.registerResources(&rdg_builder);
		// renderer
		depthBuffer = rdg_builder.addDepthBuffer(1.f, 1.f);
		uniformBufferFlights = rdg_builder.addUniformBufferFlights(sizeof(UniformBufferObject));

		// raster pass sono 1
		GFX::RDG::NodeHandle srgb_color_attachment = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT, 1.f, 1.f, "SRGB Color Attach");
		GFX::RDG::NodeHandle srgb_depth_attachment = rdg_builder.addDepthBuffer(1.f, 1.f);
		srgb_framebuffer = rdg_builder.addFrameBufferRef({ srgb_color_attachment }, srgb_depth_attachment);

		// HDR raster pass
		MemScope<RHI::IShader> shaderVert2 = resourceFactory->createShaderFromBinaryFile("vs_particle.spv", { RHI::ShaderStage::VERTEX,"main" });
		MemScope<RHI::IShader> shaderFrag2 = resourceFactory->createShaderFromBinaryFile("fs_sampler.spv", { RHI::ShaderStage::FRAGMENT,"main" });
		renderPassNodeSRGB = rdg_builder.addRasterPass({ uniformBufferFlights, external_sampler, portal.particleBuffer, external_sampler });
		rdg.tag(renderPassNodeSRGB, "Raster HDR");
		GFX::RDG::RasterPassNode* rasterPassNodeSRGB = rdg.getRasterPassNode(renderPassNodeSRGB);
		rasterPassNodeSRGB->shaderVert = std::move(shaderVert2);
		rasterPassNodeSRGB->shaderFrag = std::move(shaderFrag2);
		rasterPassNodeSRGB->framebuffer = srgb_framebuffer;
		rasterPassNodeSRGB->indirectDrawBufferHandle = portal.indirectDrawBuffer;
		rasterPassNodeSRGB->textures = { external_texture, external_baked_texture };
		rasterPassNodeSRGB->customCommandRecord = [&scene = scene](GFX::RDG::RasterPassNode* raster_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			std::function<void(ECS::TagComponent&, GFX::Mesh&)> per_mesh_behavior = [&](ECS::TagComponent& tag, GFX::Mesh& mesh) {
				commandbuffer->cmdBindVertexBuffer(mesh.vertexBuffer.get());
				commandbuffer->cmdBindIndexBuffer(mesh.indexBuffer.get());
				RHI::IDescriptorSet* set = raster_pass->descriptorSets[flight_idx].get();
				commandbuffer->cmdBindDescriptorSets(
					RHI::PipelineBintPoint::GRAPHICS,
					raster_pass->pipelineLayout.get(),
					0, 1, &set, 0, nullptr);
				commandbuffer->cmdDrawIndexedIndirect(raster_pass->indirectDrawBuffer, 0, 1, sizeof(unsigned int) * 5);
			};
			scene.tree.context.traverse<ECS::TagComponent, GFX::Mesh>(per_mesh_behavior);
		};

		// ACES
		GFX::RDG::NodeHandle bloomExtract = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f, 1.f, "Bloom Extract Image");
		test_write_target = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, 1.f, 1.f, "LDR Image");
		acesPass = rdg_builder.addComputePass(aces.get(), { test_write_target, srgb_color_attachment, bloomExtract }, "ACES & Bloom Extract", sizeof(unsigned int) * 2);
		rdg.getComputePassNode(acesPass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			Size size = { 1280,720 };
			compute_pass->executeWithConstant(commandbuffer, 40, 23, 1, 0, size);
		};

		// bloom
		MemScope<RHI::IShader> BlurLevel0 = resourceFactory->createShaderFromBinaryFile("bloom/BlurLevel0.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> BlurLevel1 = resourceFactory->createShaderFromBinaryFile("bloom/BlurLevel1.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> BlurLevel2 = resourceFactory->createShaderFromBinaryFile("bloom/BlurLevel2.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> BlurLevel3 = resourceFactory->createShaderFromBinaryFile("bloom/BlurLevel3.spv", { RHI::ShaderStage::COMPUTE,"main" });
		MemScope<RHI::IShader> BlurLevel4 = resourceFactory->createShaderFromBinaryFile("bloom/BlurLevel4.spv", { RHI::ShaderStage::COMPUTE,"main" });

		glm::vec2 screenSize = { 1280,720 };
		GFX::RDG::NodeHandle bloom_00 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.5f, 0.5f, "Bloom 00 Image");
		{
			blurPassConstants[0] = BlurPassConstants{ screenSize * glm::vec2{1. / 2,1. / 2}, screenSize, screenSize * glm::vec2{1,1}, {0,1} };
			blurLevel00Pass = rdg_builder.addComputePass(BlurLevel0.get(), { external_sampler, bloom_00 }, "Bloom Blur 00", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel00Pass)->textures = { bloomExtract };
			rdg.getComputePassNode(blurLevel00Pass)->customDispatch = [&pass_constant = blurPassConstants[0]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_01 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.5f, 0.5f, "Bloom 01 Image");
		{
			blurPassConstants[1] = BlurPassConstants{ screenSize * glm::vec2{1. / 2,1. / 2}, screenSize, screenSize * glm::vec2{1. / 2,1. / 2}, {1,0} };
			blurLevel01Pass = rdg_builder.addComputePass(BlurLevel0.get(), { external_sampler, bloom_01 }, "Bloom Blur 01", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel01Pass)->textures = { bloom_00 };
			rdg.getComputePassNode(blurLevel01Pass)->customDispatch = [&pass_constant = blurPassConstants[1]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_10 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.25f, 0.25f, "Bloom 10 Image");
		{
			blurPassConstants[2] = BlurPassConstants{ screenSize * glm::vec2{1. / 4,1. / 4}, screenSize, screenSize * glm::vec2{1. / 2,1. / 2}, {0,1} };
			blurLevel10Pass = rdg_builder.addComputePass(BlurLevel1.get(), { external_sampler, bloom_10 }, "Bloom Blur 10", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel10Pass)->textures = { bloom_01 };
			rdg.getComputePassNode(blurLevel10Pass)->customDispatch = [&pass_constant = blurPassConstants[2]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_11 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.25f, 0.25f, "Bloom 11 Image");
		{
			blurPassConstants[3] = BlurPassConstants{ screenSize * glm::vec2{1. / 4,1. / 4}, screenSize, screenSize * glm::vec2{1. / 4,1. / 4}, {1,0} };
			blurLevel11Pass = rdg_builder.addComputePass(BlurLevel1.get(), { external_sampler, bloom_11 }, "Bloom Blur 11", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel11Pass)->textures = { bloom_10 };
			rdg.getComputePassNode(blurLevel11Pass)->customDispatch = [&pass_constant = blurPassConstants[3]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1, 0, pass_constant);
			};
		}

		GFX::RDG::NodeHandle bloom_20 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.125f, 0.125f, "Bloom 20 Image");
		{
			blurPassConstants[4] = BlurPassConstants{ screenSize * glm::vec2{1. / 8,1. / 8}, screenSize, screenSize * glm::vec2{1. / 4,1. / 4}, {0,1} };
			blurLevel20Pass = rdg_builder.addComputePass(BlurLevel2.get(), { external_sampler, bloom_20 }, "Bloom Blur 20", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel20Pass)->textures = { bloom_11 };
			rdg.getComputePassNode(blurLevel20Pass)->customDispatch = [&pass_constant = blurPassConstants[4]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_21 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.125f, 0.125f, "Bloom 21 Image");
		{
			blurPassConstants[5] = BlurPassConstants{ screenSize * glm::vec2{1. / 8,1. / 8}, screenSize, screenSize * glm::vec2{1. / 8,1. / 8}, {1,0} };
			blurLevel21Pass = rdg_builder.addComputePass(BlurLevel2.get(), { external_sampler, bloom_21 }, "Bloom Blur 21", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel21Pass)->textures = { bloom_20 };
			rdg.getComputePassNode(blurLevel21Pass)->customDispatch = [&pass_constant = blurPassConstants[5]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1, 0, pass_constant);
			};
		}

		GFX::RDG::NodeHandle bloom_30 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 16, 1.f / 16, "Bloom 30 Image");
		{
			blurPassConstants[6] = BlurPassConstants{ screenSize * glm::vec2{1. / 16,1. / 16}, screenSize, screenSize * glm::vec2{1. / 8,1. / 8}, {0,1} };
			blurLevel30Pass = rdg_builder.addComputePass(BlurLevel3.get(), { external_sampler, bloom_30 }, "Bloom Blur 30", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel30Pass)->textures = { bloom_21 };
			rdg.getComputePassNode(blurLevel30Pass)->customDispatch = [&pass_constant = blurPassConstants[6]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_31 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 16, 1.f / 16, "Bloom 31 Image");
		{
			blurPassConstants[7] = BlurPassConstants{ screenSize * glm::vec2{1. / 16,1. / 16}, screenSize, screenSize * glm::vec2{1. / 16,1. / 16}, {1,0} };
			blurLevel31Pass = rdg_builder.addComputePass(BlurLevel3.get(), { external_sampler, bloom_31 }, "Bloom Blur 31", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel31Pass)->textures = { bloom_30 };
			rdg.getComputePassNode(blurLevel31Pass)->customDispatch = [&pass_constant = blurPassConstants[7]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1, 0, pass_constant);
			};
		}

		GFX::RDG::NodeHandle bloom_40 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 32, 1.f / 32, "Bloom 40 Image");
		{
			blurPassConstants[8] = BlurPassConstants{ screenSize * glm::vec2{1. / 32,1. / 32}, screenSize, screenSize * glm::vec2{1. / 16,1. / 16}, {0,1} };
			blurLevel40Pass = rdg_builder.addComputePass(BlurLevel4.get(), { external_sampler, bloom_40 }, "Bloom Blur 40", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel40Pass)->textures = { bloom_31 };
			rdg.getComputePassNode(blurLevel40Pass)->customDispatch = [&pass_constant = blurPassConstants[8]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1, 0, pass_constant);
			};
		}
		GFX::RDG::NodeHandle bloom_41 = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 32, 1.f / 32, "Bloom 41 Image");
		{
			blurPassConstants[9] = BlurPassConstants{ screenSize * glm::vec2{1. / 32,1. / 32}, screenSize, screenSize * glm::vec2{1. / 32,1. / 32}, {1,0} };
			blurLevel41Pass = rdg_builder.addComputePass(BlurLevel4.get(), { external_sampler, bloom_41 }, "Bloom Blur 41", sizeof(unsigned int) * 2 * 4);
			rdg.getComputePassNode(blurLevel41Pass)->textures = { bloom_40 };
			rdg.getComputePassNode(blurLevel41Pass)->customDispatch = [&pass_constant = blurPassConstants[9]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
			{
				float screenX = 1280, screenY = 720;
				compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1, 0, pass_constant);
			};
		}

		MemScope<RHI::IShader> BlurCombine = resourceFactory->createShaderFromBinaryFile("bloom/BloomCombine.spv", { RHI::ShaderStage::COMPUTE,"main" });
		GFX::RDG::NodeHandle bloomCombined = rdg_builder.addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, 1.f, 1.f, "Combined Image");
		bloomCombinedPass = rdg_builder.addComputePass(BlurCombine.get(), { bloomCombined, 
			external_sampler, external_sampler, external_sampler, 
			external_sampler, external_sampler, external_sampler }, 
			"Bloom Combine",
			sizeof(unsigned int) * 3);
		rdg.getComputePassNode(bloomCombinedPass)->textures = { test_write_target, bloom_01, bloom_11, bloom_21, bloom_31, bloom_41 };
		rdg.getComputePassNode(bloomCombinedPass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{
			float screenX = 1280, screenY = 720;
			BloomCombineConstant bloom_combine_constant = { glm::vec2{screenX, screenY}, 5.2175 };
			compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1, 0, bloom_combine_constant);
		};

		// particle system update pass
		scope_begin = rdg_builder.beginMultiDispatchScope("Scope Begin Portal-Update Multi-Dispatch");
		rdg.getMultiDispatchScope(scope_begin)->customDispatchCount = [&timer = timer]()
		{
			static float deltaTime = 0;
			deltaTime += timer.getMsPF() > 100 ? 100 : timer.getMsPF();
			unsigned int dispatch_times = (unsigned int)(deltaTime / 20);
			deltaTime -= dispatch_times * 20;
			return dispatch_times;
		};

		portal.registerUpdatePasses(&rdg_builder);
		scope_end = rdg_builder.endScope();

		// building ...
		rdg_builder.build(resourceFactory.get(), 1280, 720);
		rdg.print();

		// create ImImage for viewport
		viewportImImage = imfactory.createImImage(
			rdg.getSamplerNode(external_sampler)->getSampler(),
			rdg.getTextureBufferNode(bloomCombined)->getTextureView(),
			RHI::ImageLayout::GENERAL);
		mainViewport.bindImImage(viewportImImage.get());

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

		// init storage buffer
		RHI::ICommandPool* transientPool = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getTransientCommandPool();
		MemScope<RHI::ICommandBuffer> transientCommandbuffer = RHI::DeviceToGlobal::getGlobal(logicalDevice.get())->getResourceFactory()->createCommandBuffer(transientPool);
		transientCommandbuffer->beginRecording((uint32_t)RHI::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		rdg.getComputePassNode(portal.initPass)->executeWithConstant(transientCommandbuffer.get(), 200, 1, 1, 0, 100000u);
		transientCommandbuffer->endRecording();
		transientCommandbuffer->submit();
		logicalDevice->waitIdle();

		SE_CORE_INFO("OnAwake End");
	}

	virtual auto onWindowResize(WindowResizeEvent& e) -> bool override
	{
		logicalDevice->waitIdle();

		pipeline = nullptr;
		renderPass = nullptr;

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
			float rotation = time;// 0.5 * 3.1415926;
			uint32_t width = 1280;
			uint32_t height = 720;
			//auto [width, height] = swapchain->getExtend();

			UniformBufferObject ubo;
			ubo.cameraPos = glm::vec4(8.0f* cosf(rotation), 2.0f, 8.0f * sinf(rotation), 0.0f);
			ubo.model = glm::mat4(1.0f);
			ubo.view = glm::lookAt(glm::vec3(8.0f * cosf(rotation), 2.0f, 8.0f * sinf(rotation)), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			ubo.proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
			ubo.proj[1][1] *= -1;
			Buffer ubo_proxy((void*) &ubo, sizeof(UniformBufferObject), 4);
			rdg.getUniformBufferFlight(uniformBufferFlights, currentFrame)->updateBuffer(&ubo_proxy);
		}
		// drawFrame
		{
			// record a command buffer which draws the scene onto that image
			commandbuffers[currentFrame]->reset();
			commandbuffers[currentFrame]->beginRecording();

			// render pass
			rdg.recordCommands(commandbuffers[currentFrame].get(), currentFrame);

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

	BlurPassConstants blurPassConstants[10];
	GFX::RDG::NodeHandle depthBuffer;
	GFX::RDG::NodeHandle renderPassNodeSRGB;
	GFX::RDG::NodeHandle uniformBufferFlights;
	GFX::RDG::NodeHandle swapchainColorBufferFlights;
	GFX::RDG::NodeHandle acesPass;
	GFX::RDG::NodeHandle test_write_target;
	GFX::RDG::NodeHandle srgb_framebuffer;
	GFX::RDG::NodeHandle blurLevel00Pass;
	GFX::RDG::NodeHandle blurLevel01Pass;
	GFX::RDG::NodeHandle blurLevel10Pass;
	GFX::RDG::NodeHandle blurLevel11Pass;
	GFX::RDG::NodeHandle blurLevel20Pass;
	GFX::RDG::NodeHandle blurLevel21Pass;
	GFX::RDG::NodeHandle blurLevel30Pass;
	GFX::RDG::NodeHandle blurLevel31Pass;
	GFX::RDG::NodeHandle blurLevel40Pass;
	GFX::RDG::NodeHandle blurLevel41Pass;
	GFX::RDG::NodeHandle bloomCombinedPass;
	GFX::RDG::NodeHandle scope_begin;
	GFX::RDG::NodeHandle scope_end;
	ParticleSystem::ParticleSystem portal;

	MemScope<RHI::IGraphicContext> graphicContext;
	MemScope<RHI::IPhysicalDevice> physicalDevice;
	MemScope<RHI::ILogicalDevice> logicalDevice;

	MemScope<RHI::IResourceFactory> resourceFactory;

	MemScope<RHI::IStorageBuffer> torusBuffer;

	MemScope<RHI::ITexture> texture;
	MemScope<RHI::ITextureView> textureView;
	MemScope<RHI::ITexture> baked_texture;
	MemScope<RHI::ITextureView> baked_textureView;
	MemScope<RHI::ISampler> sampler;

	MemScope<RHI::ISwapChain> swapchain;
	MemScope<RHI::IRenderPass> renderPass;
	MemScope<RHI::IPipeline> pipeline;

	MemScope<RHI::ICommandPool> commandPool;
	std::vector<MemScope<RHI::ICommandBuffer>> commandbuffers;
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