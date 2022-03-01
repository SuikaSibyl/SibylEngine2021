module;
#include <vector>
export module GFX.GFXLayer;

import Core.Layer;
import Core.MemoryManager;
import Core.Window;
import Core.Event;

import RHI.IEnum;
import RHI.GraphicContext;
import RHI.IPhysicalDevice;
import RHI.ILogicalDevice;
import RHI.IFactory;
import RHI.ISwapChain;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ISemaphore;
import RHI.IFence;

namespace SIByL
{
	namespace GFX
	{
		export class GFXLayer :public ILayer
		{
		public:
			GFXLayer(RHI::API api, IWindow* attached_window);
			virtual ~GFXLayer() = default;

			// Layer virtuals
			virtual auto onAwake() noexcept -> void override;
			virtual auto onShutdown() noexcept -> void override;
			virtual auto onUpdate() noexcept -> void override;
			virtual auto onEvent(Event& e) noexcept -> void override;
			virtual auto onWindowResize(WindowResizeEvent& e) noexcept -> bool;

		protected:
			// Each GFXLayer has a uniform set of device
			MemScope<RHI::IGraphicContext> graphicContext;
			MemScope<RHI::IPhysicalDevice> physicalDevice;
			MemScope<RHI::ILogicalDevice> logicalDevice;
			MemScope<RHI::IResourceFactory> factory;
			MemScope<RHI::ISwapChain> swapchain;

			// Single-Threaded Command Recording supported only currently
			MemScope<RHI::ICommandPool> commandPool;
			std::vector<MemScope<RHI::ICommandBuffer>> commandbuffers;
			std::vector<MemScope<RHI::ISemaphore>> imageAvailableSemaphore;
			std::vector<MemScope<RHI::ISemaphore>> renderFinishedSemaphore;
			std::vector<MemScope<RHI::IFence>> inFlightFence;

			// frame management
			uint32_t currentFrame = 0;
		};
	}
}