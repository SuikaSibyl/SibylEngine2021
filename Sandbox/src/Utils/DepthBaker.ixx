module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
#include <glm/glm.hpp>
#include "entt/entt.hpp"
export module Utils.DepthBaker;
import Core.Buffer;
import Core.Cache;
import Core.Image;
import Core.File;
import Core.Time;
import Core.MemoryManager;
import ECS.Entity;
import RHI.IEnum;
import RHI.IFactory;
import RHI.IBuffer;
import RHI.ICommandBuffer;
import RHI.IShader;
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.ITexture;
import RHI.ITextureView;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;
import GFX.RDG.ComputeSeries;
import GFX.Transform;
import GFX.BoundingBox;

namespace SIByL::Utils
{
	export struct DepthBaker
	{
		DepthBaker() = default;
		DepthBaker(RHI::IResourceFactory* factory) :factory(factory) {}

		auto registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;
		auto registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void;

		GFX::RDG::NodeHandle color_attachment;
		GFX::RDG::NodeHandle depth_attachment;
		GFX::RDG::NodeHandle framebuffer;

		RHI::IResourceFactory* factory;
	};

	auto DepthBaker::registerResources(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		color_attachment = workshop->addColorBuffer(RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM, -1024.f, -1024.f, "Depth Baker Color Attach");
		depth_attachment = workshop->addColorBuffer(RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT, -1024.f, -1024.f, "Depth Baker Depth Attach");
		framebuffer = workshop->addFrameBufferRef({ color_attachment }, depth_attachment);
	}

	auto DepthBaker::registerRenderPasses(GFX::RDG::RenderGraphWorkshop* workshop) noexcept -> void
	{
		auto baker_pass = workshop->addRasterPassScope("Depth Baker Pass", framebuffer);
		GFX::RDG::RasterPipelineScope* baker_pipeline = workshop->addRasterPipelineScope("Depth Baker Pass", "Depth Baker");
		{
			baker_pipeline->shaderVert = factory->createShaderFromBinaryFile("baker/depth_baker_vert.spv", { RHI::ShaderStage::VERTEX,"main" });
			baker_pipeline->shaderFrag = factory->createShaderFromBinaryFile("baker/depth_baker_frag.spv", { RHI::ShaderStage::FRAGMENT,"main" });
			baker_pipeline->cullMode = RHI::CullMode::NONE;
			baker_pipeline->depthStencilDesc = RHI::TestLessEqualAndWrite;
			baker_pipeline->vertexBufferLayout =
			{
				{RHI::DataType::Float3, "Position"},
				{RHI::DataType::Float3, "Normal"},
				{RHI::DataType::Float2, "UV"},
				{RHI::DataType::Float4, "Tangent"},
			};
			// Add Materials
			auto only_mat_scope = workshop->addRasterMaterialScope("Depth Baker Pass", "Depth Baker", "Only");
		}
	}
}