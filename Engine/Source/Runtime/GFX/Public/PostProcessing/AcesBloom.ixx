module;
#include <string>
#include <filesystem>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)
export module GFX.PostProcessing.AcesBloom;
import RHI.IEnum;
import RHI.IShader;
import RHI.IFactory;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.ProxyUnity;
import GFX.RDG.ComputeSeries;

namespace SIByL::GFX::PostProcessing
{
	struct BlurPassConstants
	{
		glm::vec2 outputSize;
		glm::vec2 globalTextSize;
		glm::vec2 textureBlurInputSize;
		glm::vec2 blurDir;
	};

	export struct AcesBloomProxyUnit :public RDG::ProxyUnit
	{
		AcesBloomProxyUnit(RHI::IResourceFactory* factory);
		virtual auto registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void override;
		virtual auto registerComputePasses(GFX::RDG::RenderGraphWorkshop& workshop) noexcept -> void override;

		GFX::RDG::NodeHandle iHdrImage;
		GFX::RDG::NodeHandle iExternalSampler;

		GFX::RDG::NodeHandle bloomExtract;
		GFX::RDG::NodeHandle ldrImage;
		GFX::RDG::NodeHandle bloomCombined;

		GFX::RDG::NodeHandle acesPass;
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

		//MemScope<RHI::IShader> aces_extract;
		//MemScope<RHI::IShader> blurLevel0;
		//MemScope<RHI::IShader> blurLevel1;
		//MemScope<RHI::IShader> blurLevel2;
		//MemScope<RHI::IShader> blurLevel3;
		//MemScope<RHI::IShader> blurLevel4;
		//MemScope<RHI::IShader> BlurCombine;

	private:
		GFX::RDG::NodeHandle bloom_00;
		GFX::RDG::NodeHandle bloom_01;
		GFX::RDG::NodeHandle bloom_10;
		GFX::RDG::NodeHandle bloom_11;
		GFX::RDG::NodeHandle bloom_20;
		GFX::RDG::NodeHandle bloom_21;
		GFX::RDG::NodeHandle bloom_30;
		GFX::RDG::NodeHandle bloom_31;
		GFX::RDG::NodeHandle bloom_40;
		GFX::RDG::NodeHandle bloom_41;

		RHI::IResourceFactory* factory;
		BlurPassConstants blurPassConstants[10];
	};

	AcesBloomProxyUnit::AcesBloomProxyUnit(RHI::IResourceFactory* factory)
		:factory(factory)
	{
		//aces_extract = factory->createShaderFromBinaryFile("aces.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//blurLevel0 = factory->createShaderFromBinaryFile("bloom/BlurLevel0.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//blurLevel1 = factory->createShaderFromBinaryFile("bloom/BlurLevel1.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//blurLevel2 = factory->createShaderFromBinaryFile("bloom/BlurLevel2.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//blurLevel3 = factory->createShaderFromBinaryFile("bloom/BlurLevel3.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//blurLevel4 = factory->createShaderFromBinaryFile("bloom/BlurLevel4.spv", { RHI::ShaderStage::COMPUTE,"main" });
		//BlurCombine = factory->createShaderFromBinaryFile("bloom/BloomCombine.spv", { RHI::ShaderStage::COMPUTE,"main" });

	}

	auto AcesBloomProxyUnit::registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		bloomExtract = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f, 1.f, "Bloom Extract Image");
		ldrImage = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, 1.f, 1.f, "LDR Image");
		bloomCombined = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, 1.f, 1.f, "Combined Image");

		bloom_00 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.5f, 0.5f, "Bloom 00 Image");
		bloom_01 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.5f, 0.5f, "Bloom 01 Image");
		bloom_10 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.25f, 0.25f, "Bloom 10 Image");
		bloom_11 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.25f, 0.25f, "Bloom 11 Image");
		bloom_20 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.125f, 0.125f, "Bloom 20 Image");
		bloom_21 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 0.125f, 0.125f, "Bloom 21 Image");
		bloom_30 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 16, 1.f / 16, "Bloom 30 Image");
		bloom_31 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 16, 1.f / 16, "Bloom 31 Image");
		bloom_40 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 32, 1.f / 32, "Bloom 40 Image");
		bloom_41 = builder->addColorBuffer(RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32, 1.f / 32, 1.f / 32, "Bloom 41 Image");
	}

	struct Size
	{
		unsigned int width;
		unsigned int height;
	};

	struct BloomCombineConstant
	{
		glm::vec2 size;
		float para;
	};

	auto AcesBloomProxyUnit::registerComputePasses(GFX::RDG::RenderGraphWorkshop& workshop) noexcept -> void
	{
		// ACES Pipeline
		GFX::RDG::ComputePipelineScope* extract_tone_mapping_pipeline = workshop.addComputePipelineScope("PostProcessing Pass", "Extract + Tone Mapping");
		{
			extract_tone_mapping_pipeline->shaderComp = factory->createShaderFromBinaryFile("aces.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto tone_mapping_mat_scope = workshop.addComputeMaterialScope("PostProcessing Pass", "Extract + Tone Mapping", "Common");
				tone_mapping_mat_scope->resources = { ldrImage, iHdrImage, bloomExtract };
				auto tone_mapping_dispatch_scope = workshop.addComputeDispatch("PostProcessing Pass", "Extract + Tone Mapping", "Common", "Only");
				tone_mapping_dispatch_scope->pushConstant = [rdg = &(workshop.renderGraph)](Buffer& buffer) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					glm::uvec2 size = { screenX,screenY };
					buffer = std::move(Buffer(sizeof(size), 1));
					memcpy(buffer.getData(), &size, sizeof(size));
				};
				tone_mapping_dispatch_scope->customSize = [rdg = &(workshop.renderGraph)](uint32_t& x, uint32_t& y, uint32_t& z) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					x = GRIDSIZE(screenX, 32);
					y = GRIDSIZE(screenY, 32);
					z = 1;
				};
			}
		}

		// Bloom Blur Pipeline
		std::string pipelineNames[] = {
			"Bloom Blur 0",
			"Bloom Blur 1",
			"Bloom Blur 2",
			"Bloom Blur 3",
			"Bloom Blur 4",
		};

		std::string shaderPathes[] = {
			"bloom/BlurLevel0.spv",
			"bloom/BlurLevel1.spv",
			"bloom/BlurLevel2.spv",
			"bloom/BlurLevel3.spv",
			"bloom/BlurLevel4.spv",
		};

		std::vector<GFX::RDG::NodeHandle> resources[] = {
			{ iExternalSampler, bloom_00 },
			{ iExternalSampler, bloom_01 },
			{ iExternalSampler, bloom_10 },
			{ iExternalSampler, bloom_11 },
			{ iExternalSampler, bloom_20 },
			{ iExternalSampler, bloom_21 },
			{ iExternalSampler, bloom_30 },
			{ iExternalSampler, bloom_31 },
			{ iExternalSampler, bloom_40 },
			{ iExternalSampler, bloom_41 },
		};

		std::vector<GFX::RDG::NodeHandle> sampled_textures[] = {
			{ bloomExtract },
			{ bloom_00 },
			{ bloom_01 },
			{ bloom_10 },
			{ bloom_11 },
			{ bloom_20 },
			{ bloom_21 },
			{ bloom_30 },
			{ bloom_31 },
			{ bloom_40 },
		};

		int framebuffer_size = 1;
		for (int i = 0; i < 5; i++)
		{
			framebuffer_size /= 2;

			GFX::RDG::ComputePipelineScope* bloom_blur_pipeline = workshop.addComputePipelineScope("PostProcessing Pass", pipelineNames[i]);
			{
				
				bloom_blur_pipeline->shaderComp = factory->createShaderFromBinaryFile(shaderPathes[i], {RHI::ShaderStage::COMPUTE,"main"});
				// Add Materials "Horizontal"
				{
					auto horizon_mat_scope = workshop.addComputeMaterialScope("PostProcessing Pass", pipelineNames[i], "Horizontal");
					horizon_mat_scope->resources = resources[i * 2 + 0];
					horizon_mat_scope->sampled_textures = sampled_textures[i * 2 + 0];
					auto tone_mapping_dispatch_scope = workshop.addComputeDispatch("PostProcessing Pass", pipelineNames[i], "Horizontal", "Only");
					tone_mapping_dispatch_scope->pushConstant = [rdg = &(workshop.renderGraph), framebuffer_size = framebuffer_size](Buffer& buffer) {
						glm::vec2 screenSize = { rdg->datumWidth, rdg->datumHeight };
						BlurPassConstants blurPassConstant = { screenSize * glm::vec2{framebuffer_size,framebuffer_size}, screenSize, screenSize * glm::vec2{2 * framebuffer_size,2 * framebuffer_size}, {0,1} };
						buffer = std::move(Buffer(sizeof(blurPassConstant), 1));
						memcpy(buffer.getData(), &blurPassConstant, sizeof(blurPassConstant));
					};
					tone_mapping_dispatch_scope->customSize = [rdg = &(workshop.renderGraph), framebuffer_size = framebuffer_size](uint32_t& x, uint32_t& y, uint32_t& z) {
						float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
						x = GRIDSIZE(screenX * framebuffer_size, 16);
						y = GRIDSIZE(screenY * framebuffer_size, 16);
						z = 1;
					};
				}
				// Add Materials "Vertical"
				{
					auto horizon_mat_scope = workshop.addComputeMaterialScope("PostProcessing Pass", pipelineNames[i], "Vertical");
					horizon_mat_scope->resources = resources[i * 2 + 1];
					horizon_mat_scope->sampled_textures = sampled_textures[i * 2 + 1];
					auto tone_mapping_dispatch_scope = workshop.addComputeDispatch("PostProcessing Pass", pipelineNames[i], "Vertical", "Only");
					tone_mapping_dispatch_scope->pushConstant = [rdg = &(workshop.renderGraph), framebuffer_size = framebuffer_size](Buffer& buffer) {
						glm::vec2 screenSize = { rdg->datumWidth, rdg->datumHeight };
						BlurPassConstants blurPassConstant = { screenSize * glm::vec2{framebuffer_size,framebuffer_size}, screenSize, screenSize * glm::vec2{framebuffer_size,framebuffer_size}, {1,0} };
						buffer = std::move(Buffer(sizeof(blurPassConstant), 1));
						memcpy(buffer.getData(), &blurPassConstant, sizeof(blurPassConstant));
					};
					tone_mapping_dispatch_scope->customSize = [rdg = &(workshop.renderGraph), framebuffer_size = framebuffer_size](uint32_t& x, uint32_t& y, uint32_t& z) {
						float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
						x = GRIDSIZE(screenX * framebuffer_size, 16);
						y = GRIDSIZE(screenY * framebuffer_size, 16);
						z = 1;
					};
				}
			}
		}

		// Bloom Combine Pipeline
		GFX::RDG::ComputePipelineScope* bloom_combine_pipeline = workshop.addComputePipelineScope("PostProcessing Pass", "Bloom Combine");
		{
			bloom_combine_pipeline->shaderComp = factory->createShaderFromBinaryFile("bloom/BloomCombine.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto bloom_combine_mat_scope = workshop.addComputeMaterialScope("PostProcessing Pass", "Bloom Combine", "Common");
				bloom_combine_mat_scope->resources = { 
					bloomCombined,
					iExternalSampler, iExternalSampler, iExternalSampler,
					iExternalSampler, iExternalSampler, iExternalSampler 
				};
				bloom_combine_mat_scope->sampled_textures = { ldrImage, bloom_01, bloom_11, bloom_21, bloom_31, bloom_41 };

				auto tone_mapping_dispatch_scope = workshop.addComputeDispatch("PostProcessing Pass", "Bloom Combine", "Common", "Only");
				tone_mapping_dispatch_scope->pushConstant = [rdg = &(workshop.renderGraph)](Buffer& buffer) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					BloomCombineConstant bloom_combine_constant = { glm::vec2{screenX, screenY}, 5.2175 };
					buffer = std::move(Buffer(sizeof(bloom_combine_constant), 1));
					memcpy(buffer.getData(), &bloom_combine_constant, sizeof(bloom_combine_constant));
				};
				tone_mapping_dispatch_scope->customSize = [rdg = &(workshop.renderGraph)](uint32_t& x, uint32_t& y, uint32_t& z) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					x = GRIDSIZE(screenX, 16);
					y = GRIDSIZE(screenY, 16);
					z = 1;
				};
			}
		}
		
		//{
		//	blurPassConstants[0] = BlurPassConstants{ screenSize * glm::vec2{1. / 2,1. / 2}, screenSize, screenSize * glm::vec2{1,1}, {0,1} };
		//	blurLevel00Pass = builder->addComputePass(blurLevel0.get(), { iExternalSampler, bloom_00 }, "Bloom Blur 00", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel00Pass)->textures = { bloomExtract };
		//	builder->attached.getComputePassNode(blurLevel00Pass)->customDispatch = [&pass_constant = blurPassConstants[0]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[1] = BlurPassConstants{ screenSize * glm::vec2{1. / 2,1. / 2}, screenSize, screenSize * glm::vec2{1. / 2,1. / 2}, {1,0} };
		//	blurLevel01Pass = builder->addComputePass(blurLevel0.get(), { iExternalSampler, bloom_01 }, "Bloom Blur 01", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel01Pass)->textures = { bloom_00 };
		//	builder->attached.getComputePassNode(blurLevel01Pass)->customDispatch = [&pass_constant = blurPassConstants[1]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[2] = BlurPassConstants{ screenSize * glm::vec2{1. / 4,1. / 4}, screenSize, screenSize * glm::vec2{1. / 2,1. / 2}, {0,1} };
		//	blurLevel10Pass = builder->addComputePass(blurLevel1.get(), { iExternalSampler, bloom_10 }, "Bloom Blur 10", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel10Pass)->textures = { bloom_01 };
		//	builder->attached.getComputePassNode(blurLevel10Pass)->customDispatch = [&pass_constant = blurPassConstants[2]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1, 0, pass_constant);
		//	};
		//}
		// 
		//{
		//	blurPassConstants[3] = BlurPassConstants{ screenSize * glm::vec2{1. / 4,1. / 4}, screenSize, screenSize * glm::vec2{1. / 4,1. / 4}, {1,0} };
		//	blurLevel11Pass = builder->addComputePass(blurLevel1.get(), { iExternalSampler, bloom_11 }, "Bloom Blur 11", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel11Pass)->textures = { bloom_10 };
		//	builder->attached.getComputePassNode(blurLevel11Pass)->customDispatch = [&pass_constant = blurPassConstants[3]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[4] = BlurPassConstants{ screenSize * glm::vec2{1. / 8,1. / 8}, screenSize, screenSize * glm::vec2{1. / 4,1. / 4}, {0,1} };
		//	blurLevel20Pass = builder->addComputePass(blurLevel2.get(), { iExternalSampler, bloom_20 }, "Bloom Blur 20", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel20Pass)->textures = { bloom_11 };
		//	builder->attached.getComputePassNode(blurLevel20Pass)->customDispatch = [&pass_constant = blurPassConstants[4]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[5] = BlurPassConstants{ screenSize * glm::vec2{1. / 8,1. / 8}, screenSize, screenSize * glm::vec2{1. / 8,1. / 8}, {1,0} };
		//	blurLevel21Pass = builder->addComputePass(blurLevel2.get(), { iExternalSampler, bloom_21 }, "Bloom Blur 21", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel21Pass)->textures = { bloom_20 };
		//	builder->attached.getComputePassNode(blurLevel21Pass)->customDispatch = [&pass_constant = blurPassConstants[5]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[6] = BlurPassConstants{ screenSize * glm::vec2{1. / 16,1. / 16}, screenSize, screenSize * glm::vec2{1. / 8,1. / 8}, {0,1} };
		//	blurLevel30Pass = builder->addComputePass(blurLevel3.get(), { iExternalSampler, bloom_30 }, "Bloom Blur 30", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel30Pass)->textures = { bloom_21 };
		//	builder->attached.getComputePassNode(blurLevel30Pass)->customDispatch = [&pass_constant = blurPassConstants[6]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[7] = BlurPassConstants{ screenSize * glm::vec2{1. / 16,1. / 16}, screenSize, screenSize * glm::vec2{1. / 16,1. / 16}, {1,0} };
		//	blurLevel31Pass = builder->addComputePass(blurLevel3.get(), { iExternalSampler, bloom_31 }, "Bloom Blur 31", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel31Pass)->textures = { bloom_30 };
		//	builder->attached.getComputePassNode(blurLevel31Pass)->customDispatch = [&pass_constant = blurPassConstants[7]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[8] = BlurPassConstants{ screenSize * glm::vec2{1. / 32,1. / 32}, screenSize, screenSize * glm::vec2{1. / 16,1. / 16}, {0,1} };
		//	blurLevel40Pass = builder->addComputePass(blurLevel4.get(), { iExternalSampler, bloom_40 }, "Bloom Blur 40", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel40Pass)->textures = { bloom_31 };
		//	builder->attached.getComputePassNode(blurLevel40Pass)->customDispatch = [&pass_constant = blurPassConstants[8]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1, 0, pass_constant);
		//	};
		//}
		//{
		//	blurPassConstants[9] = BlurPassConstants{ screenSize * glm::vec2{1. / 32,1. / 32}, screenSize, screenSize * glm::vec2{1. / 32,1. / 32}, {1,0} };
		//	blurLevel41Pass = builder->addComputePass(blurLevel4.get(), { iExternalSampler, bloom_41 }, "Bloom Blur 41", sizeof(unsigned int) * 2 * 4);
		//	builder->attached.getComputePassNode(blurLevel41Pass)->textures = { bloom_40 };
		//	builder->attached.getComputePassNode(blurLevel41Pass)->customDispatch = [&pass_constant = blurPassConstants[9]](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//	{
		//		float screenX = 1280, screenY = 720;
		//		compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1, 0, pass_constant);
		//	};
		//}

		//bloomCombinedPass = builder->addComputePass(BlurCombine.get(), { bloomCombined,
		//	iExternalSampler, iExternalSampler, iExternalSampler,
		//	iExternalSampler, iExternalSampler, iExternalSampler },
		//	"Bloom Combine",
		//	sizeof(unsigned int) * 3);
		//builder->attached.getComputePassNode(bloomCombinedPass)->textures = { ldrImage, bloom_01, bloom_11, bloom_21, bloom_31, bloom_41 };
		//builder->attached.getComputePassNode(bloomCombinedPass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//{
		//	float screenX = 1280, screenY = 720;
		//	BloomCombineConstant bloom_combine_constant = { glm::vec2{screenX, screenY}, 5.2175 };
		//	compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1, 0, bloom_combine_constant);
		//};
	}
}