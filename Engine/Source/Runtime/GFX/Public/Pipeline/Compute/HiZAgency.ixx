module;
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <glm/glm.hpp>
export module GFX.HiZAgency;
import Core.Buffer;
import Core.MemoryManager;
import RHI.ITexture;
import RHI.IFactory;
import RHI.ITextureView;
import GFX.RDG.Common;
import GFX.RDG.Agency;
import GFX.RDG.RenderGraph;
import GFX.RDG.ColorBufferNode;

#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)

namespace SIByL::GFX
{
	export struct HiZAgency :public RDG::Agency
	{
		static auto createInstance(GFX::RDG::NodeHandle depthInputHandle, RHI::IResourceFactory* factory) noexcept -> MemScope<RDG::Agency>;
		HiZAgency(GFX::RDG::NodeHandle depthInputHandle, RHI::IResourceFactory* factory) :depthInputHandle(depthInputHandle), factory(factory) {}
		virtual auto onRegister(void* workshop) noexcept -> void override;
		virtual auto startWorkshopBuild(void* workshop) noexcept -> void override;
		virtual auto beforeCompile() noexcept -> void override;
		virtual auto beforeDivirtualizePasses() noexcept -> void override;
		virtual auto onInvalid() noexcept -> void override;

		uint32_t mipmapCount = 0;
		GFX::RDG::NodeHandle depthInputHandle;
		GFX::RDG::NodeHandle depthSampleHandle;
		GFX::RDG::NodeHandle HiZ_handle;
		RHI::IResourceFactory* factory = nullptr;
		GFX::RDG::RenderGraphWorkshop* workshop = nullptr;
		MemScope<RHI::ITextureView> depthSampleView = nullptr;
		std::vector<MemScope<RHI::ITextureView>> mipmapTextureViews;
		std::vector<GFX::RDG::NodeHandle> mipmapColorBufferExt;
	};

	auto HiZAgency::createInstance(GFX::RDG::NodeHandle depthInputHandle, RHI::IResourceFactory* factory) noexcept -> MemScope<RDG::Agency>
	{
		MemScope<HiZAgency> hiz_agency = MemNew<HiZAgency>(depthInputHandle, factory);
		MemScope<RDG::Agency> agency = MemCast<RDG::Agency>(hiz_agency);
		return agency;
	}

	auto HiZAgency::onRegister(void* i_workshop) noexcept -> void
	{
		workshop = (GFX::RDG::RenderGraphWorkshop*)i_workshop;
		// Add Pass "HiZ Pass"
		workshop->addComputePassScope("HiZ Pass");

		// Add HiZ Depth Buffer
		HiZ_handle = workshop->addColorBuffer(RHI::ResourceFormat::FORMAT_R32_SFLOAT, 1.f, 1.f, "Z-Hierachy");
		auto HiZ_node = workshop->getNode<GFX::RDG::ColorBufferNode>(HiZ_handle);
		HiZ_node->mipLevels = 0;

		// Add Depth Sample View
		depthSampleHandle = workshop->addColorBufferRef(nullptr, nullptr, depthInputHandle, "Depth Buffer Sample Ref");
		workshop->getNode<GFX::RDG::ColorBufferNode>(depthSampleHandle)->format = RHI::ResourceFormat::FORMAT_R32_SFLOAT;

		// Add Copu Pass
				// ACES Pipeline
		GFX::RDG::ComputePipelineScope* copy_pipeline = workshop->addComputePipelineScope("HiZ Pass", "Copy Pipeline");
		{
			copy_pipeline->shaderComp = factory->createShaderFromBinaryFile("hiz/copy.spv", { RHI::ShaderStage::COMPUTE,"main" });
			// Add Materials "Common"
			{
				auto copy_mat_scope = workshop->addComputeMaterialScope("HiZ Pass", "Copy Pipeline", "Common");
				copy_mat_scope->resources = { workshop->getInternalSampler("Default Sampler"), HiZ_handle };
				copy_mat_scope->sampled_textures = { depthSampleHandle };

				auto copy_dispatch_scope = workshop->addComputeDispatch("HiZ Pass", "Copy Pipeline", "Common", "Only");
				copy_dispatch_scope->pushConstant = [rdg = &(workshop->renderGraph)](Buffer& buffer) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					glm::uvec2 size = { screenX,screenY };
					buffer = std::move(Buffer(sizeof(size), 1));
					memcpy(buffer.getData(), &size, sizeof(size));
				};
				copy_dispatch_scope->customSize = [rdg = &(workshop->renderGraph)](uint32_t& x, uint32_t& y, uint32_t& z) {
					float screenX = rdg->datumWidth, screenY = rdg->datumHeight;
					x = GRIDSIZE(screenX, 32);
					y = GRIDSIZE(screenY, 32);
					z = 1;
				};
			}
		}
	}

	auto HiZAgency::startWorkshopBuild(void* i_workshop) noexcept -> void
	{
		workshop = (GFX::RDG::RenderGraphWorkshop*)i_workshop;
	}

	auto HiZAgency::beforeCompile() noexcept -> void
	{
		// get factory
		factory = workshop->renderGraph.factory;
		// get mipmap count
		mipmapCount = static_cast<uint32_t>(std::floor(std::log2(std::max(workshop->renderGraph.datumWidth, workshop->renderGraph.datumHeight)))) + 1;;
		mipmapTextureViews.clear();
	}
	
	auto HiZAgency::beforeDivirtualizePasses() noexcept -> void
	{
		// create Depth Sample View
		auto depth_input_node = workshop->getNode<GFX::RDG::ColorBufferNode>(depthInputHandle);
		RHI::ITexture* depth_input_texture = depth_input_node->getTexture();

		auto depth_sample_handle = workshop->getNode<GFX::RDG::ColorBufferNode>(depthSampleHandle);
		depth_sample_handle->texture.ref = depth_input_texture;
		depthSampleView = factory->createTextureView(depth_input_texture, {
				(uint32_t)RHI::ImageAspectFlagBits::DEPTH_BIT,
				0,
				1,
				0,
				1
			});
		depth_sample_handle->textureView.ref = depthSampleView.get();

		// create texture views
		auto HiZ_node = workshop->getNode<GFX::RDG::ColorBufferNode>(HiZ_handle);
		RHI::ITexture* texture = HiZ_node->getTexture();

		for (int i = 0; i < mipmapCount; i++)
		{
			RHI::ImageSubresourceRange range = {
				(uint32_t)RHI::ImageAspectFlagBits::COLOR_BIT,
				i,
				1,
				0,
				1
			};
			mipmapTextureViews.emplace_back(factory->createTextureView(texture, range));
			if (mipmapColorBufferExt.size() <= i)
			{
				auto ext_handle = workshop->addColorBufferExt(texture, mipmapTextureViews[i].get(), "HiZ SubResource-" + std::to_string(i));
				mipmapColorBufferExt.emplace_back(ext_handle);
			}
			else
			{
				auto ext_node = workshop->getNode<GFX::RDG::ColorBufferNode>(mipmapColorBufferExt[i]);
				ext_node->texture.ref = texture;
				ext_node->textureView.ref = mipmapTextureViews[i].get();
			}
		}

		// create copy pass

	}

	auto HiZAgency::onInvalid() noexcept -> void
	{

	}

}