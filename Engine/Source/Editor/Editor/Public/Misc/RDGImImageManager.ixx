module;
#include <unordered_map>
export module Editor.RDGImImageManager;
import Core.MemoryManager;
import RHI.ISampler;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import Editor.ImGuiLayer;
import Editor.ImImage;

namespace SIByL::Editor
{
	export struct RDGImImage
	{
		RDGImImage(GFX::RDG::NodeHandle nodeHandle,
			GFX::RDG::NodeHandle samplerHandle,
			GFX::RDG::RenderGraph* renderGraph,
			ImGuiLayer* imguiLayer);

		auto invalid() noexcept -> void;
		auto getImImage() noexcept -> Editor::ImImage* { return imImage.get(); }

	private:
		GFX::RDG::NodeHandle nodeHandle;
		GFX::RDG::NodeHandle samplerHandle;
		GFX::RDG::RenderGraph* renderGraph;
		MemScope<Editor::ImImage> imImage = nullptr;
		ImGuiLayer* imguiLayer = nullptr;
	};

	RDGImImage::RDGImImage(GFX::RDG::NodeHandle nodeHandle,
		GFX::RDG::NodeHandle samplerHandle,
		GFX::RDG::RenderGraph* renderGraph,
		ImGuiLayer* imguiLayer)
		: nodeHandle(nodeHandle)
		, samplerHandle(samplerHandle)
		, renderGraph(renderGraph)
		, imguiLayer(imguiLayer)
	{
		invalid();
	}

	auto RDGImImage::invalid() noexcept -> void
	{
		imImage = imguiLayer->createImImage(
			renderGraph->getSamplerNode(samplerHandle)->getSampler(),
			renderGraph->getTextureBufferNode(nodeHandle)->getTextureView(),
			RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
	}

	export struct RDGImImageManager
	{
		RDGImImageManager(GFX::RDG::RenderGraph* renderGraph, ImGuiLayer* imguiLayer);
		auto addImImage(GFX::RDG::NodeHandle nodeHandle, GFX::RDG::NodeHandle samplerHandle) noexcept -> RDGImImage*;
		auto invalidAll() noexcept -> void;

		std::unordered_map<GFX::RDG::NodeHandle, MemScope<RDGImImage>> registry;

	private:
		GFX::RDG::RenderGraph* renderGraph;
		ImGuiLayer* imguiLayer;
	};

	RDGImImageManager::RDGImImageManager(GFX::RDG::RenderGraph* renderGraph, ImGuiLayer* imguiLayer)
		: renderGraph(renderGraph)
		, imguiLayer(imguiLayer)
	{}

	auto RDGImImageManager::addImImage(GFX::RDG::NodeHandle nodeHandle, GFX::RDG::NodeHandle samplerHandle) noexcept -> RDGImImage*
	{
		registry.emplace(nodeHandle, MemNew<RDGImImage>(nodeHandle, samplerHandle, renderGraph, imguiLayer));
		return registry[nodeHandle].get();
	}

	auto RDGImImageManager::invalidAll() noexcept -> void
	{
		for (auto& iter : registry)
		{
			iter.second->invalid();
		}
	}

}