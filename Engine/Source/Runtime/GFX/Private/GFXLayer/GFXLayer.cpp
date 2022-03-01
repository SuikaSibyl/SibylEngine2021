module;
#include <vector>
#include <Macros.h>
#include <memory>
#include <functional>
module GFX.GFXLayer;

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
import RHI.IDeviceGlobal;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ISemaphore;
import RHI.IFence;

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace SIByL::GFX
{
	GFXLayer::GFXLayer(RHI::API api, IWindow* attached_window)
	{
		// create devices
		graphicContext = RHI::IFactory::createGraphicContext({ api });
		graphicContext->attachWindow(attached_window);
		physicalDevice = RHI::IFactory::createPhysicalDevice({ graphicContext.get() });
		logicalDevice = RHI::IFactory::createLogicalDevice({ physicalDevice.get() });
		factory = MemNew<RHI::IResourceFactory>(logicalDevice.get());
		swapchain = factory->createSwapchain({});

		// create command objects
		commandPool = factory->createCommandPool({ RHI::QueueType::GRAPHICS, (uint32_t)RHI::CommandPoolAttributeFlagBits::RESET });
		commandbuffers.resize(MAX_FRAMES_IN_FLIGHT);
		imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			commandbuffers[i] = factory->createCommandBuffer(commandPool.get());
			imageAvailableSemaphore[i] = factory->createSemaphore();
			renderFinishedSemaphore[i] = factory->createSemaphore();
			inFlightFence[i] = factory->createFence();
		}
	}

	auto GFXLayer::onAwake() noexcept -> void
	{

	}

	auto GFXLayer::onShutdown() noexcept -> void
	{
		logicalDevice->waitIdle();
		RHI::DeviceToGlobal::removeDevice(logicalDevice.get());
	}

	auto GFXLayer::onUpdate() noexcept -> void
	{
		// 1. Wait for the previous frame to finish
		{
			inFlightFence[currentFrame]->wait();
			inFlightFence[currentFrame]->reset();
		}

		// update current frame
		{			
			//	2. Acquire an image from the swap chain
			uint32_t imageIndex = swapchain->acquireNextImage(imageAvailableSemaphore[currentFrame].get());

			swapchain->present(imageIndex, renderFinishedSemaphore[currentFrame].get());
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}
	}

	auto GFXLayer::onEvent(Event& e) noexcept -> void
	{
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_FN(GFXLayer::onWindowResize));

	}

	auto GFXLayer::onWindowResize(WindowResizeEvent& e) noexcept -> bool
	{
		logicalDevice->waitIdle();
		// recreate swapchain
		swapchain = factory->createSwapchain({ e.GetWidth(), e.GetHeight() });

		return false;
	}
}